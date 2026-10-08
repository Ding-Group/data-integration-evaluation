"""
scVI multi-parameter Optuna optimization

Usage:
    run_scvioptuna.py -o <path> (-b <batch>)... -c <key> (-p <str> | --params=<str>) [--seed=<int>] <file>

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
import numpy as np
import pandas as pd
import scvi
import torch


from project_utils.adata_utils import add_umap, minimise_anndata, combine_batch, split_by_batch, EMB_KEY


PROJECT_ROOT = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

sys.path.insert(0, f"{PROJECT_ROOT}/scib/")

os.environ.setdefault(
    "R_HOME",
    f"{PROJECT_ROOT}/CondaEnv/scvi_py310/lib/R",
)
os.environ.setdefault(
    "R_LIBS_SITE",
    f"{PROJECT_ROOT}/CondaEnv/harmony_optuna/lib/R/library",
)

PROCESS = psutil.Process(os.getpid())
def mem_gb():
    return PROCESS.memory_info().rss / 1024**3
from scib.metrics import final_metrics

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


def run_scvi(
    adata,
    n_latent,
    n_hidden,
    n_layers,
    dropout_rate,
    lr,
    seed=0,
):
    np.random.seed(seed)
    scvi.settings.seed = seed

    adata_scvi = adata.copy()

    new_batch_keys= []
    for b in BATCH_KEY:
        adata_scvi.obs.rename({b: b.replace(".", "_")}, axis=1, inplace=True)
        new_batch_keys.append(b.replace(".", "_"))

    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        layer="counts",
        batch_key=new_batch_keys[0],
        categorical_covariate_keys=new_batch_keys[1:],
    )
    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=int(n_latent),
        n_hidden=int(n_hidden),
        n_layers=int(n_layers),
        dropout_rate=float(dropout_rate),
    )
    model.train(
        plan_kwargs={
            "lr": float(lr),
        },
    )

    adata.obsm[EMB_KEY] = model.get_latent_representation()

    if model is not None:
        model.deregister_manager(adata_scvi)

    del model, adata_scvi
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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

    n_latent = trial.suggest_int("n_latent", 5, 64)
    n_hidden = trial.suggest_categorical("n_hidden", [64, 128, 256])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.01, 0.2)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    

    gene_likelihood = "zinb"
    dispersion = "gene"
    latent_distribution = "normal"

    adata = ADATA
    adata_raw = ADATA

    params = {
        "n_latent": n_latent,
        "n_hidden": n_hidden,
        "n_layers": n_layers,
        "dropout_rate": dropout_rate,
        "lr": lr,
        "gene_likelihood": gene_likelihood,
        "dispersion": dispersion,
        "latent_distribution": latent_distribution
    }

    adata_int = run_scvi(
        adata,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        lr=lr,
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
        f"n_latent={n_latent} "
        f"n_hidden={n_hidden} "
        f"n_layers={n_layers} "
        f"dropout_rate={dropout_rate:.3g} "
        f"lr={lr:.3g} "
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
    adata_int = run_scvi(
        adata.copy(),
        n_latent=bp['n_latent'],
        n_hidden=bp['n_hidden'],
        n_layers=bp['n_layers'],
        dropout_rate=bp['dropout_rate'],
        lr=bp['lr'],
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
