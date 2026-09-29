"""
RPCA multi-parameter Optuna optimization.

Usage:
    optimize_rpcaoptuna.py -o <path> (-b <batch>)... -c <key> (-p <str> | --params=<str>) [--seed=<int>] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             Batch key.
    -c <key>               Cell type label key.
    -p --params=<str>      Params for metrics as a str
    --seed=<int>           Random seed [default: 0].
"""

import ctypes
import gc
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import optuna
import pandas as pd
import psutil
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from project_utils.hvg_utils import select_hvgs

PROJECT_ROOT = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

sys.path.insert(0, f"{PROJECT_ROOT}/scib/")

os.environ["R_HOME"] = (
    "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/lib/R"
)

from scib.metrics import final_metrics

PROCESS = psutil.Process(os.getpid())
def mem_gb():
    return PROCESS.memory_info().rss / 1024**3

N_TRIALS = 100


def trim_memory():
    import gc
    gc.collect()
    try:
        # Forces glibc to return freed heap pages back to the kernel
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def save_trial_row(trial, params, metric_results, runtime_sec):
    row = {
        "trial": trial.number,
        "runtime_sec": runtime_sec,
        **params,
        **metric_results,
    }

    df = pd.DataFrame([row])

    if os.path.exists(TRIALS_CSV):
        df.to_csv(TRIALS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(TRIALS_CSV, mode="w", header=True, index=False)


def init_rpca_r_session(data_path: str, batch_codes: pd.Series, seed: int):
    import anndata2ri
    import rpy2.robjects as ro
    from rpy2.robjects import conversion, default_converter, numpy2ri
    from rpy2.robjects.packages import importr

    converter = default_converter + numpy2ri.converter + anndata2ri.converter
    with conversion.localconverter(converter):
        importr("Seurat")
        importr("anndataR")

        r = ro.r
        r.assign("batch_vec", batch_codes.values)
        r(f"set.seed({seed})")
        r(f'obj <- read_h5ad("{data_path}", as = "Seurat")')
        r("obj$batch_vec <- batch_vec")
        r('obj.list <- SplitObject(obj, split.by = "batch_vec")')

        # Clean up full object immediately
        r("rm(obj); gc()")

        r("""
        obj.list <- lapply(X = obj.list, FUN = function(x) {
            x <- NormalizeData(x, verbose = FALSE)
            FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
        })
        features <- SelectIntegrationFeatures(object.list = obj.list)
        obj.list <- lapply(X = obj.list, FUN = function(x) {
            x <- ScaleData(x, features = features, verbose = FALSE)
            RunPCA(x, features = features, verbose = FALSE)
        })
        """)

        r('''
        log_mem <- function(step = "") {
        status <- readLines("/proc/self/status")
        rss_mb <- as.numeric(gsub("[^0-9]", "", grep("^VmRSS:", status, value = TRUE))) / 1024
        message(sprintf(">>> [R-MEM: %s] RSS = %.2f MB", step, rss_mb))
        }
        ''')


def run_rpca(
    adata: ad.AnnData,
    k_anchor: int,
    k_filter: int,
    k_score: int,
    n_trees: int,
    seed: int,
):
    import anndata2ri
    import rpy2.robjects as ro
    from rpy2.robjects import conversion, default_converter, numpy2ri

    converter = default_converter + numpy2ri.converter + anndata2ri.converter
    with conversion.localconverter(converter):
        r = ro.r
        r(f"set.seed({seed})")
        r('log_mem("Pre-Anchors")')
        r(f'''
anchors <- FindIntegrationAnchors(
    object.list = obj.list,
    anchor.features = features,
    reduction = "rpca",
    k.anchor = {k_anchor},
    k.filter = {k_filter},
    k.score = {k_score},
    n.trees = {n_trees}
)''')
        r('log_mem("Post-Anchors")')
        r(f"combined <- IntegrateData(anchorset = anchors, k.weight = {K_WEIGHT})")
        r('log_mem("Post-Integrate")')

        r('integrated_mat <- combined[["integrated"]]@data')
        r('cell_names <- colnames(combined)')
        r("rm(anchors, combined); gc(full=TRUE)")
        mat = r('integrated_mat')
        obs_names = r("cell_names")
        r("rm(integrated_mat, cell_names); gc(full = TRUE)")

    new_adata = ad.AnnData(X=mat.T, obs=pd.DataFrame(index=obs_names))
    new_adata = new_adata[adata.obs_names]
    sc.pp.pca(new_adata)
    adata.obsm[EMB_KEY] = new_adata.obsm["X_pca"]
    return adata


def compute_joint_metrics(adata_raw, adata_int):
    """
    Compute selected metrics and joint score.

    Bio-conservation metrics:
        BAMI, CASW, SCGRAPHM, TRAJ, ISOLISI

    Batch correction metrics:
        BRAS, KBET, CPCR

    joint_score = 0.6 * bio_score + 0.4 * batch_score
    """
    sc.pp.neighbors(
        adata_int,
        use_rep=EMB_KEY,
    )

    batch_key = BATCH_KEY[0] if isinstance(BATCH_KEY, list) else BATCH_KEY
    label_key = LABEL_KEY
    if "." in batch_key:
        new_batch_key = batch_key.replace(".", "_")
        adata_raw.obs[new_batch_key] = adata_raw.obs[batch_key]
        adata_int.obs[new_batch_key] = adata_int.obs[batch_key]
        batch_key = new_batch_key
    if "." in label_key:
        new_label_key = LABEL_KEY.replace(".", "_")
        adata_raw.obs[new_label_key] = adata_raw.obs[label_key]
        adata_int.obs[new_label_key] = adata_int.obs[label_key]
        label_key = new_label_key

    df = final_metrics(
        adata_raw,
        adata_int,
        batch_key,
        label_key,
        EMB_KEY,
        metric_kwargs=METRIC_KWARGS
    )

    bio_score = np.nanmean([df.loc['BAMI'], df.loc['CASW'], df.loc['SCGRAPHM'], df.loc['TRAJ'], df.loc['ISOLISI']])
    batch_score = np.nanmean([df.loc['BRAS'], df.loc['KBET'], df.loc['CPCR']])
    joint_score = bio_score * .6 + batch_score * .4
    results = {}
    results["bio_score"] = bio_score
    results["batch_score"] = batch_score
    results["joint_score"] = joint_score

    return results


def objective(trial):
    start_time = time.time()

    print(f"\n[START TRIAL {trial.number}]", flush=True)

    k_anchor = trial.suggest_int("k.anchor", 2, 10)
    k_filter = trial.suggest_int("k.filter", 100, 300)
    k_score = trial.suggest_int("k.score", 20, 80)
    n_trees = trial.suggest_int("n.trees", 20, 80)

    adata = ADATA.copy()
    adata_raw = ADATA

    params = {
        "k.anchor": k_anchor,
        "k.filter": k_filter,
        "k.score": k_score,
        "n.trees": n_trees
    }

    adata_int = run_rpca(
        adata,
        k_anchor=k_anchor,
        k_filter=k_filter,
        k_score=k_score,
        n_trees=n_trees,
        seed=SEED,
    )

    metric_results = compute_joint_metrics(adata_raw, adata_int)
    score = metric_results["joint_score"]

    for k, v in metric_results.items():
        trial.set_user_attr(k, v)

    runtime_sec = time.time() - start_time
    trial.set_user_attr("runtime_sec", runtime_sec)

    save_trial_row(
        trial=trial,
        params=params,
        metric_results=metric_results,
        runtime_sec=runtime_sec,
    )

    metric_log = " ".join(
        f"{k}={v:.4f}"
        for k, v in metric_results.items()
        if k not in ["bio_score", "batch_score", "joint_score"]
    )

    print(
        f"[END TRIAL {trial.number}] "
        f"joint={score:.4f} "
        f"bio={metric_results['bio_score']:.4f} "
        f"batch={metric_results['batch_score']:.4f} "
        f"{metric_log} "
        f"k.anchor={params['k.anchor']} "
        f"k.filter={params['k.filter']} "
        f"k.score={params['k.score']} "
        f"n.trees={params['n.trees']} "
        f"runtime={runtime_sec:.1f}s",
        flush=True,
    )


    del adata, adata_int
    gc.collect()

    print(
        f"[AFTER METRICS {trial.number}] "
        f"RSS={mem_gb():.2f} GB",
        flush=True,
    )

    return score


def main():
    global DATA_PATH, RESULT_DIR, ADATA
    global STUDY_DB, TRIALS_CSV, BEST_PARAMS_JSON
    global BATCH_KEY, LABEL_KEY, SEED, METRIC_KWARGS, K_WEIGHT

    args = docopt(__doc__)

    DATA_PATH = args["<file>"]
    adata = sc.read(DATA_PATH)

    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.obs.index.name = "barcode"
    adata.var.index.name = "gene"
    ADATA = adata.copy()

    RESULT_DIR = args["-o"]
    BATCH_KEY = args["-b"]
    LABEL_KEY = args["-c"]
    SEED = int(args["--seed"])
    metric_params = args["--params"]
    metric_params = json.loads(metric_params)

    METRIC_KWARGS = {}
    if metric_params is not None:
        for p in metric_params:
            for metric_name, mp in p.items():
                METRIC_KWARGS[f'{metric_name.lower()}_kwargs'] = mp

    os.makedirs(RESULT_DIR, exist_ok=True)

    STUDY_DB = f"{RESULT_DIR}/optuna.db"
    TRIALS_CSV = f"{RESULT_DIR}/optuna_trials.csv"
    BEST_PARAMS_JSON = f"{RESULT_DIR}/best_params.json"

    batch, _ = combine_batch(adata, BATCH_KEY)
    batch_codes = batch.cat.codes
    K_WEIGHT = min(100, batch_codes.value_counts().min() // 10 - 1)
    init_rpca_r_session(DATA_PATH, batch_codes, SEED)

    study = optuna.create_study(
        direction="maximize",
        study_name="optuna",
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,
        catch=(Exception,),
    )

    print("Optimization finished")
    print("Best joint score:", study.best_value)
    print("Best params:", study.best_params)

    best_trial = study.best_trial

    print("\nBest trial metrics:")
    for k, v in best_trial.user_attrs.items():
        print(f"{k}: {v}")

    best_output = {
        "dataset": DATA_PATH,
        "batch_key": BATCH_KEY,
        "label_key": LABEL_KEY,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_metrics": best_trial.user_attrs,
    }

    with open(BEST_PARAMS_JSON, "w") as f:
        json.dump(best_output, f, indent=2)

    print(f"\nSaved trials CSV to: {TRIALS_CSV}")
    print(f"Saved best params JSON to: {BEST_PARAMS_JSON}")

    bp = study.best_params
    adata_int = run_rpca(
        adata,
        k_anchor=bp['k.anchor'],
        k_filter=bp['k.filter'],
        k_score=bp['k.score'],
        n_trees=bp['n.trees'],
        seed=SEED,
    )

    add_umap(adata_int)
    output = minimise_anndata(
        adata_int,
        obs=BATCH_KEY + [LABEL_KEY],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"],
    )
    output.write(os.path.join(RESULT_DIR, "adata.h5ad"))

if __name__ == "__main__":
    main()
