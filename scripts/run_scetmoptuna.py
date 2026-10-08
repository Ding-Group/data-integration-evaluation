"""
scETM multi-parameter Optuna optimization

Usage:
    run_scetmoptuna.py -o <path> (-b <batch>)... -c <key> (-p <str> | --params=<str>) [--seed=<int>] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             Batch key.
    -c <key>               Cell type label key.
    -p --params=<str>      Params for metrics as a str
    --seed=<int>           Random seed [default: 0].
"""

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
import torch
from docopt import docopt
from project_utils.adata_utils import (
    EMB_KEY,
    add_umap,
    combine_batch,
    minimise_anndata
)
from project_utils.hvg_utils import select_hvgs

PROJECT_ROOT = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

sys.path.insert(0, f"{PROJECT_ROOT}/scib/")

from scETM import scETM as SCETM, UnsupervisedTrainer
from scETM.trainers import set_seed
from scib.metrics import final_metrics


PROCESS = psutil.Process(os.getpid())
def mem_gb():
    return PROCESS.memory_info().rss / 1024**3

N_TRIALS = 100
N_HIDDEN_OPTIONS = [64, 128, 256, 512]
TRAINABLE_GENE_EMB_DIM_OPTIONS = [64, 128, 256, 400, 512]


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


def run_scetm(
    adata: ad.AnnData,
    use_hvg: bool,
    hvg_method: str,
    topics: int,
    trainable_gene_emb_dim: int,
    hidden_sizes: list[int],
    lr: float,
    dropout_prob: float,
    epochs: int,
    seed: int,
) -> ad.AnnData: 
    adata = select_hvgs(adata, use_hvg, hvg_method, subset=True)

    _, batch_key = combine_batch(adata, BATCH_KEY)
    adata.obs['batch_indices'] = adata.obs[batch_key].cat.codes
    adata.obs['cell_types'] = adata.obs[LABEL_KEY]

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SCETM(
        n_trainable_genes=adata.n_vars,
        n_batches=adata.obs['batch_indices'].nunique(),
        n_topics=int(topics),
        trainable_gene_emb_dim=int(trainable_gene_emb_dim),
        hidden_sizes=list(hidden_sizes),
        dropout_prob=float(dropout_prob),
        device=device,
    )

    trainer = UnsupervisedTrainer(model, adata, init_lr=float(lr))
    trainer.train(n_epochs=int(epochs), save_model_ckpt=False, eval=False)
    
    model.get_all_embeddings_and_nll(adata)

    adata.obsm[EMB_KEY] = adata.obsm['theta']
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

    n_topics = trial.suggest_int("n_topics", 20, 200, step=10)
    trainable_gene_emb_dim = trial.suggest_categorical(
        "trainable_gene_emb_dim",
        TRAINABLE_GENE_EMB_DIM_OPTIONS,
    )
    n_hidden = trial.suggest_categorical("n_hidden", N_HIDDEN_OPTIONS)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.3, step=0.05)
    hidden_sizes = [int(n_hidden)] * int(n_layers)

    adata = ADATA.copy()
    adata_raw = ADATA

    params = {
        "n_topics": n_topics,
        "trainable_gene_emb_dim": trainable_gene_emb_dim,
        "n_hidden": n_hidden,
        "n_layers": n_layers,
        "hidden_sizes_values": hidden_sizes,
        "lr": lr,
        "dropout_prob": dropout_prob,
        "epochs": 100,
    }

    adata_int = run_scetm(
        adata,
        use_hvg=True,
        hvg_method=None,
        topics=n_topics,
        trainable_gene_emb_dim=trainable_gene_emb_dim,
        hidden_sizes=hidden_sizes,
        lr=lr,
        dropout_prob=dropout_prob,
        epochs=100,
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
        f"n_topics={n_topics} "
        f"trainable_gene_emb_dim={trainable_gene_emb_dim} "
        f"n_hidden={n_hidden} "
        f"n_layers={n_layers} "
        f"hidden_sizes={hidden_sizes} "
        f"lr={lr:.3g} "
        f"dropout_prob={dropout_prob:.3g} "
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
    best_hidden_sizes = [int(bp['n_hidden'])] * int(bp['n_layers'])
    adata_int = run_scetm(
        adata.copy(),
        use_hvg=True,
        hvg_method=None,
        topics=bp['n_topics'],
        trainable_gene_emb_dim=bp['trainable_gene_emb_dim'],
        hidden_sizes=best_hidden_sizes,
        lr=bp['lr'],
        dropout_prob=bp['dropout_prob'],
        epochs=100,
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
