"""
LIGER multi-parameter Optuna optimization

Usage:
    run_ligeroptuna.py -o <path> (-b <batch>)... -c <key> (-p <str> | --params=<str>) [--seed=<int>] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             Batch key.
    -c <key>               Cell type label key.
    -p --params=<str>      Params for metrics as a str
    --seed=<int>           Random seed [default: 0].
"""

import gc
import os
import sys
import time
import json
import psutil
from docopt import docopt
import optuna
import anndata as ad
import scanpy as sc
import pyliger
import numpy as np
import pandas as pd

from project_utils.adata_utils import add_umap, minimise_anndata, combine_batch, split_by_batch, EMB_KEY


PROJECT_ROOT = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

sys.path.insert(0, f"{PROJECT_ROOT}/scib/")

os.environ["R_HOME"] = (
    f"{PROJECT_ROOT}/CondaEnv/harmony_optuna/lib/R"
)
PROCESS = psutil.Process(os.getpid())
def mem_gb():
    return PROCESS.memory_info().rss / 1024**3

from scib.metrics import final_metrics

# N_HYPERPARAMS = 2
# N_TRIALS_MULTIPLIER = 1
N_TRIALS = 100


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


def run_liger(
    adata,
    knn,
    lam,
    max_iterations=30,
    seed=0,
):
    adata.X = adata.layers['counts'].copy()
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    batch_info, batch_key = combine_batch(adata, BATCH_KEY)
    parts = split_by_batch(adata, batch_key)

    obj = pyliger.create_liger(parts)
    pyliger.normalize(obj)
    pyliger.select_genes(obj)
    pyliger.scale_not_center(obj)

    pyliger.optimize_ALS(
        obj,
        k=int(knn),
        value_lambda=float(lam),
        max_iters=int(max_iterations),
        rand_seed=int(seed),
    )
    pyliger.quantile_norm(obj, knn_k=knn)

    concat = ad.concat(obj.adata_list)
    concat = concat[adata.obs_names]
    embs = concat.obsm["H_norm"]

    adata.obsm[EMB_KEY] = embs
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

    k = trial.suggest_int("k", 20, 60)
    value_lambda = trial.suggest_float("value_lambda", 0.5, 10.0, log=True)

    params = {
        "k": k,
        "value_lambda": value_lambda,
    }

    adata = ADATA.copy()
    adata_raw = ADATA
    adata_int = run_liger(
        adata=adata,
        knn=params["k"],
        lam=params["value_lambda"],
        seed=SEED,
    )

    metric_results = compute_joint_metrics(adata_raw, adata_int)
    score = metric_results["joint_score"]

    for key, value in metric_results.items():
        trial.set_user_attr(key, value)

    runtime_sec = time.time() - start_time
    trial.set_user_attr("runtime_sec", runtime_sec)

    save_trial_row(
        trial=trial,
        params=params,
        metric_results=metric_results,
        runtime_sec=runtime_sec,
    )

    metric_log = " ".join(
        f"{key}={value:.4f}"
        for key, value in metric_results.items()
        if key not in ["bio_score", "batch_score", "joint_score"]
    )

    msg = (
        f"[END TRIAL {trial.number}] "
        f"joint={score:.4f} "
        f"bio={metric_results['bio_score']:.4f} "
        f"batch={metric_results['batch_score']:.4f} "
        f"{metric_log} "
        f"k={params['k']} "
        f"value_lambda={params['value_lambda']:.3g} "
    )

    print(msg, flush=True)
    del adata_int
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
    global BATCH_KEY, LABEL_KEY, SEED, METRIC_KWARGS

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

    study = optuna.create_study(
        direction="maximize",
        study_name="optuna",
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )

    remaining = max(0, N_TRIALS - len(study.trials))
    study.optimize(
        objective,
        n_trials=remaining,
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
    adata_int = run_liger(
        adata.copy(),
        knn=bp['k'],
        lam=bp['value_lambda'],
        seed=SEED
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
