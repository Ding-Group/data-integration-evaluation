"""
Harmony multi-parameter Optuna optimization.

Usage:
    optimize_harmonyoptuna.py -o <path> (-b <batch>)... -c <key> (-p <str> | --params=<str>) [--seed=<int>] <file>

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
import gc
import psutil

from docopt import docopt

import optuna
import anndata as ad
import scanpy as sc
import harmonypy as hm
import numpy as np
import pandas as pd
import sys, os, pyximport
pyximport.install(build_dir=os.path.join(sys.prefix, ".pyxbld"))

from project_utils.adata_utils import add_umap, minimise_anndata, EMB_KEY

PROJECT_ROOT = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

sys.path.insert(0, f"{PROJECT_ROOT}/scib/")

os.environ["R_HOME"] = (
    f"{PROJECT_ROOT}/CondaEnv/harmony_optuna/lib/R"
)

from scib.metrics import final_metrics

# N_HYPERPARAMS = 5
# N_TRIALS_MULTIPLIER = 50
N_TRIALS = 100

PROCESS = psutil.Process(os.getpid())
def mem_gb():
    return PROCESS.memory_info().rss / 1024**3

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


def run_harmony(
    adata,
    sigma,
    theta,
    seed=0,
    lamb=None,
    nclust=None,
    tau=0.0,
    block_size=0.05,
    max_iter_harmony=10,
    max_iter_kmeans=4,
    epsilon_cluster=1e-3,
    epsilon_harmony=1e-2,
    alpha=0.2,
):
    X = adata.obsm["X_pca"]
    ho = hm.run_harmony(
        X,
        adata.obs,
        BATCH_KEY,
        theta=theta,
        lamb=lamb,
        sigma=sigma,
        nclust=nclust,
        tau=tau,
        block_size=block_size,
        max_iter_harmony=max_iter_harmony,
        max_iter_kmeans=max_iter_kmeans,
        epsilon_cluster=epsilon_cluster,
        epsilon_harmony=epsilon_harmony,
        alpha=alpha,
        verbose=False,
        random_state=SEED,
    )

    Z = np.asarray(ho.Z_corr)

    if Z.shape[0] != adata.n_obs and Z.shape[1] == adata.n_obs:
        Z = Z.T

    if not np.all(np.isfinite(Z)):
        raise optuna.exceptions.TrialPruned(
            "Harmony produced NaN or Inf in X_harmony"
        )

    adata.obsm[EMB_KEY] = Z
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

    theta = trial.suggest_float("theta", 0.0, 10.0) 
    sigma = trial.suggest_float("sigma", 0.01, 2.0, log=True)
    lamb_mode = trial.suggest_categorical("lamb_mode", ["auto", "fixed"])
    if lamb_mode == "auto":
        lamb = -1
    else:
        lamb = trial.suggest_float("lamb", 0.1, 10.0, log=True)
    tau = trial.suggest_float("tau", 0.0, 2.0)

    block_size = 0.05
    max_iter_harmony = 10
    max_iter_kmeans = 4
    epsilon_cluster = 1e-3
    epsilon_harmony = 1e-2
    alpha = 0.2

    adata = ADATA.copy()
    adata_raw = ADATA
    n_obs = adata.n_obs
    default_nclust = int(min(n_obs / 30.0, 100))

    nclust = trial.suggest_int(
        "nclust",
        max(10, int(default_nclust * 0.5)),
        min(200, int(default_nclust * 2.0)),
    )

    params = {
        "theta": theta,
        "sigma": sigma,
        "lamb_mode": lamb_mode,
        "lamb": lamb,
        "tau": tau,
        "block_size": block_size,
        "max_iter_harmony": max_iter_harmony,
        "max_iter_kmeans": max_iter_kmeans,
        "epsilon_cluster": epsilon_cluster,
        "epsilon_harmony": epsilon_harmony,
        "alpha": alpha,
        "nclust": nclust,
    }

    adata_int = run_harmony(
        adata,
        sigma=sigma,
        theta=theta,
        lamb=lamb,
        nclust=nclust,
        tau=tau,
        block_size=block_size,
        max_iter_harmony=max_iter_harmony,
        max_iter_kmeans=max_iter_kmeans,
        epsilon_cluster=epsilon_cluster,
        epsilon_harmony=epsilon_harmony,
        alpha=alpha,
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
        f"sigma={sigma:.3g} "
        f"theta={theta:.3g} "
        f"lamb_mode={lamb_mode} "
        f"lamb={lamb} "
        f"nclust={nclust} "
        f"tau={tau:.3g} "
        f"block={block_size:.3g} "
        f"iterH={max_iter_harmony} "
        f"iterK={max_iter_kmeans} "
        f"epsC={epsilon_cluster:.1e} "
        f"epsH={epsilon_harmony:.1e} "
        f"alpha={alpha:.3g} "
        f"runtime={runtime_sec:.1f}s",
        flush=True,
    )

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

    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.tl.pca(
        adata,
        svd_solver="arpack",
    )
    ADATA = adata.copy()

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
    if bp['lamb_mode'] == 'auto':
        lamb = -1
    else:
        lamb = bp['lamb']
    adata_int = run_harmony(
        adata.copy(),
        sigma=bp['sigma'],
        theta=bp['theta'],
        lamb=lamb,
        nclust=bp['nclust'],
        tau=bp['tau'],
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
