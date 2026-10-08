"""
Scanorama multi-parameter Optuna optimization

Usage:
    run_scanoramaoptuna.py -o <path> (-b <batch>)... -c <key> (-p <str> | --params=<str>) [--seed=<int>] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             Batch key.
    -c <key>               Cell type label key.
    -p --params=<str>      Params for metrics as a str
    --seed=<int>           Random seed [default: 0].
"""

import os
import sys
import time
import json
from docopt import docopt
import optuna
import anndata as ad
import scanpy as sc
import scanorama
import numpy as np
import pandas as pd
import pyximport
pyximport.install(build_dir=os.path.join(sys.prefix, ".pyxbld"))

from project_utils.adata_utils import add_umap, minimise_anndata, combine_batch, EMB_KEY


PROJECT_ROOT = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

sys.path.insert(0, f"{PROJECT_ROOT}/scib/")

os.environ["R_HOME"] = (
    f"{PROJECT_ROOT}/CondaEnv/harmony_optuna/lib/R"
)

from scib.metrics import final_metrics

# N_HYPERPARAMS = 3
# N_TRIALS_MULTIPLIER = 50
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


def split_by_batch(adata, batch_key):
    batches = adata.obs[batch_key].astype("category")
    batch_names = list(batches.cat.categories)

    adatas = []
    idxs_list = []

    for batch_name in batch_names:
        idx = np.where(batches.values == batch_name)[0]
        idxs_list.append(idx)
        adatas.append(adata[idx, :].copy())

    return adatas, batch_names, idxs_list


def reassemble_scanorama_embedding(parts, idxs_list, n_obs, key="X_scanorama"):
    Z_concat = np.concatenate(
        [p.obsm[key] for p in parts],
        axis=0,
    )

    concat_indices = np.concatenate(idxs_list)

    inv = np.empty(n_obs, dtype=np.int64)
    inv[concat_indices] = np.arange(n_obs, dtype=np.int64)

    Z = Z_concat[inv, :]

    return Z


def run_scanorama(
    adata,
    knn,
    sigma,
    alpha,
    batch_size,
    seed=0,
):
    np.random.seed(seed)
    _, batch_key_final = combine_batch(adata, BATCH_KEY)
    parts, batch_names, idxs_list = split_by_batch(adata, batch_key_final)

    scanorama.integrate_scanpy(
        parts,
        batch_size=int(batch_size),
        verbose=0,
        ds_names=batch_names,
        dimred=50,
        approx=True,
        sigma=float(sigma),
        alpha=float(alpha),
        knn=int(knn),
        union=False,
        hvg=None,
        seed=int(seed),
    )

    Z = reassemble_scanorama_embedding(
        parts,
        idxs_list,
        n_obs=adata.n_obs,
        key="X_scanorama",
    )

    adata_int = adata.copy()
    adata_int.obsm[EMB_KEY] = np.asarray(Z, dtype=np.float32)
    return adata_int


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

    # Scanorama hyperparameter search space.
    knn = trial.suggest_int("knn", 5, 50)
    sigma = trial.suggest_float("sigma", 1.0, 60.0, log=True)
    alpha = trial.suggest_float("alpha", 0.0, 0.5)

    # Fixed Scanorama/downstream settings.
    batch_size = 5000
    approx = True
    union = False
    internal_dimred = 50

    adata = ADATA.copy()

    params = {
        "knn": knn,
        "sigma": sigma,
        "alpha": alpha,
        "batch_size": batch_size,
        "approx": approx,
        "union": union,
        "internal_dimred": internal_dimred,
    }

    adata_int = run_scanorama(
        adata,
        knn=knn,
        sigma=sigma,
        alpha=alpha,
        batch_size=batch_size,
        seed=SEED,
    )

    metric_results = compute_joint_metrics(adata, adata_int)
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
        f"knn={knn} "
        f"sigma={sigma:.3g} "
        f"alpha={alpha:.3g} "
        f"batch_size={batch_size} "
        f"approx={approx} "
        f"union={union} "
        f"runtime={runtime_sec:.1f}s",
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
    sc.tl.pca(
        adata,
        svd_solver="arpack",
    )
    ADATA = adata

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
    adata_int = run_scanorama(
        adata,
        knn=bp['knn'],
        sigma=bp['sigma'],
        alpha=bp['alpha'],
        batch_size=5000
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
