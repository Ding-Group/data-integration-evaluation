"""
scVI multi-parameter Optuna optimization.

Usage:
    optimize_scvi_multi_metrics.py -o <path> (-b <batch>)... -c <key> --dataset-name=<name> [--study-name=<name>] [--seed=<int>] [--no-use-hvg] [--n-top-hvg=<int>] [--max-epochs=<int>] [--preflight-only] [--bami] [--casw] [--scgraphm] [--bras] [--kbet] [--cpcr] [--traj] [--isolisi] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys. First key is used as scVI batch_key and metric batch_key; extra keys are categorical covariates.
    -c <key>               Cell type label key.
    --dataset-name=<name>  Exact dataset identifier.
    --study-name=<name>    Optuna study name [default: scvi_joint_metrics].
    --seed=<int>           Random seed [default: 0].
    --no-use-hvg           Do not use HVG selection. [default: False]
    --n-top-hvg=<int>      Number of HVGs to use if HVG selection is enabled [default: 2000].
    --max-epochs=<int>     Maximum scVI training epochs [default: 400].
    --preflight-only       Validate configuration/data and exit before creating a study.

    --bami                 Use BAMI.
    --casw                 Use CASW.
    --scgraphm             Use SCGRAPHM.
    --bras                 Use BRAS.
    --kbet                 Use KBET.
    --cpcr                 Use CPCR.
    --traj                 Use trajectory metric.
    --isolisi              Use isolated-label LISI.
"""

import os
import sys
import time
import json
import gc

from docopt import docopt

import optuna
import anndata as ad
import scanpy as sc
import scvi
import numpy as np
import torch

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

from scib.metrics.metrics import metrics


N_NEIGHBORS = 15

CLUSTER_KEY = "leiden_opt"
EMBED_KEY = "X_scvi"
METRIC_BATCH_KEY = "scvi_metric_batch"
METRIC_LABEL_KEY = "scvi_metric_label"

SEED = 0
USE_HVG = True
N_TOP_HVG = 2000
MAX_EPOCHS = 400

USE_BAMI = False
USE_CASW = False
USE_SCGRAPHM = False
USE_BRAS = False
USE_KBET = False
USE_CPCR = False
USE_TRAJ = False
USE_ISOLISI = False

N_HYPERPARAMS = 5
N_TRIALS_MULTIPLIER = 50
N_TRIALS = N_HYPERPARAMS * N_TRIALS_MULTIPLIER


DATASET_CONFIG = {
    "human_pancreas": {
        "batch_keys": ("tech",),
        "label_key": "celltype",
        "metrics": (
            "BAMI", "CASW", "SCGRAPHM", "ISOLISI", "BRAS", "KBET", "CPCR"
        ),
    },
    "neurips_bmmc": {
        "batch_keys": ("batch",),
        "label_key": "cell_type",
        "metrics": (
            "BAMI", "CASW", "SCGRAPHM", "TRAJ", "ISOLISI",
            "BRAS", "KBET", "CPCR",
        ),
    },
    "nph_cerebellum": {
        "batch_keys": ("Sample",),
        "label_key": "Celltype",
        "metrics": ("BAMI", "CASW", "SCGRAPHM", "BRAS", "KBET", "CPCR"),
    },
    "seurat_pbmc": {
        "batch_keys": ("donor",),
        "label_key": "celltype.l2",
        "metrics": ("BAMI", "CASW", "SCGRAPHM", "BRAS", "KBET", "CPCR"),
    },
    "cellranger_versions": {
        "batch_keys": ("batch",),
        "label_key": "cell_type",
        "metrics": ("BAMI", "CASW", "SCGRAPHM", "BRAS", "KBET", "CPCR"),
    },
    "scib_sim1": {
        "batch_keys": ("Batch",),
        "label_key": "Group",
        "metrics": (
            "BAMI", "CASW", "SCGRAPHM", "ISOLISI", "BRAS", "KBET", "CPCR"
        ),
    },
    "mouse_mammary": {
        "batch_keys": ("BATCH",),
        "label_key": "celltype",
        "metrics": ("BAMI", "CASW", "SCGRAPHM", "BRAS", "KBET", "CPCR"),
    },
    "human_pbmc_sequencing_techs": {
        "batch_keys": ("Method",),
        "label_key": "CellType",
        "metrics": (
            "BAMI", "CASW", "SCGRAPHM", "ISOLISI", "BRAS", "KBET", "CPCR"
        ),
    },
    "mouse_regulatory_T": {
        "batch_keys": ("method",),
        "label_key": "cell.type",
        "metrics": ("BAMI", "CASW", "SCGRAPHM", "BRAS", "KBET", "CPCR"),
    },
}


def validate_configuration(selected_metrics):
    if DATASET_NAME not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset identifier {DATASET_NAME!r}. "
            f"Expected one of: {list(DATASET_CONFIG)}"
        )

    config = DATASET_CONFIG[DATASET_NAME]
    expected_data_path = os.path.join(
        PROJECT_ROOT,
        "data",
        f"{DATASET_NAME}.h5ad",
    )

    if os.path.normpath(DATA_PATH) != os.path.normpath(expected_data_path):
        raise ValueError(
            f"Dataset path mismatch for {DATASET_NAME}: "
            f"expected {expected_data_path!r}, got {DATA_PATH!r}"
        )

    if tuple(BATCH_KEYS) != config["batch_keys"]:
        raise ValueError(
            f"Batch keys mismatch for {DATASET_NAME}: "
            f"expected {list(config['batch_keys'])!r}, got {BATCH_KEYS!r}"
        )

    if LABEL_KEY != config["label_key"]:
        raise ValueError(
            f"Label key mismatch for {DATASET_NAME}: "
            f"expected {config['label_key']!r}, got {LABEL_KEY!r}"
        )

    expected_metrics = config["metrics"]
    if tuple(selected_metrics) != expected_metrics:
        raise ValueError(
            f"Metric policy mismatch for {DATASET_NAME}: "
            f"expected {list(expected_metrics)}, got {selected_metrics}"
        )


def validate_dataset_contents():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Dataset does not exist: {DATA_PATH}")

    adata = ad.read_h5ad(DATA_PATH, backed="r")
    try:
        obs_columns = list(adata.obs.columns)
        missing_keys = [
            key for key in (*BATCH_KEYS, LABEL_KEY)
            if key not in adata.obs
        ]
        if missing_keys:
            raise ValueError(
                f"Missing obs keys {missing_keys} in {DATASET_NAME}. "
                f"Available keys: {obs_columns}"
            )

        if "counts" not in adata.layers:
            raise ValueError(
                f"scVI requires adata.layers['counts']; missing in {DATASET_NAME}."
            )

        obs = adata.obs[[*BATCH_KEYS, LABEL_KEY]].copy()
        pseudotime = None
        if USE_TRAJ:
            if "dpt_pseudotime" not in adata.obs:
                raise ValueError(
                    f"TRAJ is enabled for {DATASET_NAME}, but "
                    "adata.obs['dpt_pseudotime'] is missing."
                )
            pseudotime = adata.obs["dpt_pseudotime"].copy()
    finally:
        adata.file.close()

    if obs.isna().any().any():
        missing_counts = obs.isna().sum().to_dict()
        raise ValueError(
            f"Missing batch/label values in {DATASET_NAME}: {missing_counts}"
        )

    n_batches = int(obs[BATCH_KEY].nunique())
    n_labels = int(obs[LABEL_KEY].nunique())
    if n_batches < 2:
        raise ValueError(
            "Batch-correction metrics require at least 2 batches; "
            f"{DATASET_NAME} has {n_batches}."
        )

    batch_coverage = obs.groupby(
        LABEL_KEY,
        observed=True,
    )[BATCH_KEY].nunique()
    label_counts = obs.groupby(LABEL_KEY, observed=True).size()

    print(
        f"Dataset preflight: n_obs={len(obs)} "
        f"n_batches={n_batches} n_labels={n_labels} counts_layer=True"
    )

    if USE_TRAJ:
        finite_pseudotime = pseudotime.dropna()
        if len(finite_pseudotime) < 2 or finite_pseudotime.nunique() < 2:
            raise ValueError(
                f"TRAJ is enabled for {DATASET_NAME}, but "
                "adata.obs['dpt_pseudotime'] has fewer than 2 usable values."
            )
        print(
            f"TRAJ preflight: usable_pseudotime={len(finite_pseudotime)} "
            f"unique_values={finite_pseudotime.nunique()}"
        )

    if USE_ISOLISI:
        min_batch_coverage = int(batch_coverage.min())
        if min_batch_coverage >= n_batches:
            raise ValueError(
                f"ISOLISI is enabled for {DATASET_NAME}, but every label "
                f"appears in all {n_batches} batches."
            )
        isolated_labels = [
            str(label)
            for label in batch_coverage.index[
                batch_coverage == min_batch_coverage
            ]
        ]
        print(
            f"ISOLISI preflight: min_batch_coverage={min_batch_coverage} "
            f"isolated_labels={isolated_labels}"
        )

    if USE_CPCR:
        eligible_labels = [
            str(label)
            for label in label_counts.index
            if int(label_counts.loc[label]) >= 100
            and int(batch_coverage.loc[label]) >= 2
        ]
        if not eligible_labels:
            raise ValueError(
                f"CPCR is enabled for {DATASET_NAME}, but no label has "
                "at least 100 cells across at least 2 batches."
            )
        print(f"CPCR preflight: eligible_labels={len(eligible_labels)}")


def check_keys(adata):
    missing = []

    for key in BATCH_KEYS:
        if key not in adata.obs:
            missing.append(key)

    if LABEL_KEY not in adata.obs:
        missing.append(LABEL_KEY)

    if missing:
        raise ValueError(
            f"Missing keys in adata.obs: {missing}. "
            f"Have: {list(adata.obs.columns)}"
        )


def check_counts_layer(adata):
    if "counts" not in adata.layers:
        raise ValueError(
            "scVI requires raw counts. Missing adata.layers['counts']. "
            "Please make sure the input h5ad contains a counts layer."
        )


def hvg_subset(adata, use_hvg=True, n_top_hvg=2000):
    if not use_hvg:
        return adata.copy()

    if n_top_hvg is None or int(n_top_hvg) <= 0:
        return adata.copy()

    if (
        "highly_variable" in adata.var
        and adata.var["highly_variable"].fillna(False).astype(bool).sum() > 0
    ):
        hvg_mask = adata.var["highly_variable"].fillna(False).astype(bool).values
    else:
        adata_hvg = adata.copy()
        sc.pp.highly_variable_genes(
            adata_hvg,
            flavor="seurat",
            n_top_genes=int(n_top_hvg),
        )
        hvg_mask = (
            adata_hvg.var["highly_variable"].fillna(False).astype(bool).values
        )
        del adata_hvg
        gc.collect()

    if hvg_mask.sum() == 0:
        raise optuna.exceptions.TrialPruned("No highly variable genes selected.")

    return adata[:, hvg_mask].copy()


def sanitize_obs_keys_for_scvi(adata, batch_keys):
    new_batch_keys = []

    for b in batch_keys:
        new_b = b.replace(".", "_")

        if new_b != b:
            if new_b in adata.obs.columns and new_b not in batch_keys:
                raise ValueError(
                    f"Cannot rename obs key {b} to {new_b}: {new_b} already exists."
                )

            adata.obs.rename({b: new_b}, axis=1, inplace=True)

        new_batch_keys.append(new_b)

    if len(set(new_batch_keys)) != len(new_batch_keys):
        raise ValueError(
            f"Duplicate batch keys after replacing dots with underscores: {new_batch_keys}"
        )

    return new_batch_keys


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

    check_keys(adata)
    check_counts_layer(adata)

    adata_scvi = hvg_subset(
        adata,
        use_hvg=USE_HVG,
        n_top_hvg=N_TOP_HVG,
    )

    new_batch_keys = sanitize_obs_keys_for_scvi(
        adata_scvi,
        BATCH_KEYS,
    )

    batch_key = new_batch_keys[0]
    categorical_covariate_keys = new_batch_keys[1:]

    if len(categorical_covariate_keys) == 0:
        categorical_covariate_keys = None

    model = None

    try:
        scvi.model.SCVI.setup_anndata(
            adata_scvi,
            layer="counts",
            batch_key=batch_key,
            categorical_covariate_keys=categorical_covariate_keys,
        )

        model = scvi.model.SCVI(
            adata_scvi,
            n_latent=int(n_latent),
            n_hidden=int(n_hidden),
            n_layers=int(n_layers),
            dropout_rate=float(dropout_rate),
            gene_likelihood="zinb",
            dispersion="gene",
            latent_distribution="normal",
        )

        model.train(
            max_epochs=int(MAX_EPOCHS),
            early_stopping=True,
            plan_kwargs={
                "lr": float(lr),
            },
        )

        Z = model.get_latent_representation()

        if Z.shape[0] != adata.n_obs:
            raise ValueError(
                f"scVI latent shape {Z.shape} unexpected "
                f"(n_obs={adata.n_obs})"
            )

        if not np.all(np.isfinite(Z)):
            raise optuna.exceptions.TrialPruned(
                "scVI produced NaN or Inf in X_scvi"
            )
    finally:
        if model is not None:
            model.deregister_manager(adata_scvi)

        del model
        del adata_scvi
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adata_int = adata.copy()
    adata_int.obsm[EMBED_KEY] = np.asarray(Z, dtype=np.float32)

    return adata_int


def metric_value(metric_series, key):
    if key not in metric_series.index:
        raise KeyError(
            f"Metric {key} not found. Available metrics: {list(metric_series.index)}"
        )

    value = metric_series[key]

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        return float(np.nanmean(arr))

    return float(value)


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
        use_rep=EMBED_KEY,
        n_neighbors=N_NEIGHBORS,
    )

    metric_key_map = {
        METRIC_BATCH_KEY: BATCH_KEY,
        METRIC_LABEL_KEY: LABEL_KEY,
    }
    for adata_name, adata_obj in (
        ("adata_raw", adata_raw),
        ("adata_int", adata_int),
    ):
        existing_aliases = [
            alias for alias in metric_key_map if alias in adata_obj.obs
        ]
        if existing_aliases:
            raise ValueError(
                f"Reserved metric obs keys already exist in {adata_name}: "
                f"{existing_aliases}"
            )
        for alias, source_key in metric_key_map.items():
            adata_obj.obs[alias] = adata_obj.obs[source_key].copy()

    metric_df = metrics(
        adata=adata_raw,
        adata_int=adata_int,
        batch_key=METRIC_BATCH_KEY,
        label_key=METRIC_LABEL_KEY,
        embed=EMBED_KEY,
        cluster_key=CLUSTER_KEY,
        verbose=False,
        bami_=USE_BAMI,
        casw_=USE_CASW,
        scgraphm_=USE_SCGRAPHM,
        bras_=USE_BRAS,
        kbet_=USE_KBET,
        cpcr_=USE_CPCR,
        traj_=USE_TRAJ,
        isolisi_=USE_ISOLISI,
    )

    metric_series = metric_df.iloc[:, 0]

    results = {}
    bio_values = []
    batch_values = []

    if USE_BAMI:
        bami = metric_value(metric_series, "BAMI")
        results["BAMI"] = bami
        bio_values.append(bami)

    if USE_CASW:
        casw = metric_value(metric_series, "CASW")
        results["CASW"] = casw
        bio_values.append(casw)

    if USE_SCGRAPHM:
        scgraphm = metric_value(metric_series, "SCGRAPHM")
        results["SCGRAPHM"] = scgraphm
        bio_values.append(scgraphm)

    if USE_TRAJ:
        traj = metric_value(metric_series, "TRAJ")
        results["TRAJ"] = traj
        bio_values.append(traj)

    if USE_ISOLISI:
        isolisi = metric_value(metric_series, "ISOLISI")
        results["ISOLISI"] = isolisi
        bio_values.append(isolisi)

    if USE_BRAS:
        bras = metric_value(metric_series, "BRAS")
        results["BRAS"] = bras
        batch_values.append(bras)

    if USE_KBET:
        kbet = metric_value(metric_series, "KBET")
        results["KBET"] = kbet
        batch_values.append(kbet)

    if USE_CPCR:
        cpcr = metric_value(metric_series, "CPCR")
        results["CPCR"] = cpcr
        batch_values.append(cpcr)

    if len(bio_values) == 0:
        raise ValueError("No bio-conservation metric selected.")

    if len(batch_values) == 0:
        raise ValueError("No batch-correction metric selected.")

    bio_score = float(np.mean(bio_values))
    batch_score = float(np.mean(batch_values))
    joint_score = float(0.6 * bio_score + 0.4 * batch_score)

    if not np.all(np.isfinite([bio_score, batch_score, joint_score])):
        raise ValueError(
            "Joint metric aggregation produced a non-finite score: "
            f"bio={bio_score}, batch={batch_score}, joint={joint_score}"
        )

    results["bio_score"] = bio_score
    results["batch_score"] = batch_score
    results["joint_score"] = joint_score

    return results


def save_trial_row(trial, params, metric_results, runtime_sec):
    import pandas as pd

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
    use_hvg = USE_HVG
    n_top_hvg = N_TOP_HVG
    max_epochs = MAX_EPOCHS

    adata = ad.read_h5ad(DATA_PATH)
    check_keys(adata)
    check_counts_layer(adata)

    params = {
        "n_latent": n_latent,
        "n_hidden": n_hidden,
        "n_layers": n_layers,
        "dropout_rate": dropout_rate,
        "lr": lr,
        "gene_likelihood": gene_likelihood,
        "dispersion": dispersion,
        "latent_distribution": latent_distribution,
        "use_hvg": use_hvg,
        "n_top_hvg": n_top_hvg,
        "max_epochs": max_epochs,
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
        f"n_latent={n_latent} "
        f"n_hidden={n_hidden} "
        f"n_layers={n_layers} "
        f"dropout_rate={dropout_rate:.3g} "
        f"lr={lr:.3g} "
        f"use_hvg={use_hvg} "
        f"n_top_hvg={n_top_hvg} "
        f"max_epochs={max_epochs} "
        f"runtime={runtime_sec:.1f}s",
        flush=True,
    )

    return score


def main():
    global DATA_PATH, DATASET_NAME, RESULT_DIR, STUDY_NAME
    global STUDY_DB, TRIALS_CSV, BEST_PARAMS_JSON
    global BATCH_KEYS, BATCH_KEY, LABEL_KEY, SEED
    global USE_HVG, N_TOP_HVG, MAX_EPOCHS
    global USE_BAMI, USE_CASW, USE_SCGRAPHM, USE_BRAS, USE_KBET
    global USE_CPCR, USE_TRAJ, USE_ISOLISI

    args = docopt(__doc__)

    DATA_PATH = args["<file>"]
    DATASET_NAME = args["--dataset-name"]
    RESULT_DIR = args["-o"]
    BATCH_KEYS = args["-b"]
    BATCH_KEY = BATCH_KEYS[0]
    LABEL_KEY = args["-c"]
    STUDY_NAME = args["--study-name"]
    SEED = int(args["--seed"])

    USE_HVG = not args["--no-use-hvg"]
    N_TOP_HVG = int(args["--n-top-hvg"])
    MAX_EPOCHS = int(args["--max-epochs"])

    USE_BAMI = args["--bami"]
    USE_CASW = args["--casw"]
    USE_SCGRAPHM = args["--scgraphm"]
    USE_BRAS = args["--bras"]
    USE_KBET = args["--kbet"]
    USE_CPCR = args["--cpcr"]
    USE_TRAJ = args["--traj"]
    USE_ISOLISI = args["--isolisi"]

    selected_metrics = []
    selected_bio_metrics = []
    selected_batch_metrics = []

    if USE_BAMI:
        selected_metrics.append("BAMI")
        selected_bio_metrics.append("BAMI")
    if USE_CASW:
        selected_metrics.append("CASW")
        selected_bio_metrics.append("CASW")
    if USE_SCGRAPHM:
        selected_metrics.append("SCGRAPHM")
        selected_bio_metrics.append("SCGRAPHM")
    if USE_TRAJ:
        selected_metrics.append("TRAJ")
        selected_bio_metrics.append("TRAJ")
    if USE_ISOLISI:
        selected_metrics.append("ISOLISI")
        selected_bio_metrics.append("ISOLISI")

    if USE_BRAS:
        selected_metrics.append("BRAS")
        selected_batch_metrics.append("BRAS")
    if USE_KBET:
        selected_metrics.append("KBET")
        selected_batch_metrics.append("KBET")
    if USE_CPCR:
        selected_metrics.append("CPCR")
        selected_batch_metrics.append("CPCR")

    if len(selected_metrics) == 0:
        raise ValueError("No metrics selected. Please pass metric flags.")

    if len(selected_bio_metrics) == 0 or len(selected_batch_metrics) == 0:
        raise ValueError(
            "Joint score requires at least one bio-conservation metric "
            "and at least one batch-correction metric."
        )

    validate_configuration(selected_metrics)
    validate_dataset_contents()

    print(f"Dataset name: {DATASET_NAME}")
    print(f"Selected metrics: {selected_metrics}")
    print(f"Bio metrics: {selected_bio_metrics}")
    print(f"Batch metrics: {selected_batch_metrics}")

    if args["--preflight-only"]:
        print("Preflight-only validation finished")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)

    STUDY_DB = f"{RESULT_DIR}/{STUDY_NAME}.db"
    TRIALS_CSV = f"{RESULT_DIR}/{STUDY_NAME}_trials.csv"
    BEST_PARAMS_JSON = f"{RESULT_DIR}/{STUDY_NAME}_best_params.json"

    print("scVI joint-metric Optuna")
    print(f"Dataset ID: {DATASET_NAME}")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Batch keys for scVI: {BATCH_KEYS}")
    print(f"Primary batch key for metrics: {BATCH_KEY}")
    print(f"Label key: {LABEL_KEY}")
    print(f"Use HVG: {USE_HVG}")
    print(f"N top HVG: {N_TOP_HVG}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Result dir: {RESULT_DIR}")
    print(f"Study DB: {STUDY_DB}")
    print(f"N trials: {N_TRIALS}")

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )

    complete_trial_count = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )
    remaining_complete_trials = max(0, N_TRIALS - complete_trial_count)

    print(f"Complete trials already in study: {complete_trial_count}")
    print(
        f"Remaining complete trials to reach {N_TRIALS}: "
        f"{remaining_complete_trials}"
    )

    while complete_trial_count < N_TRIALS:
        study.optimize(
            objective,
            n_trials=1,
            n_jobs=1,
            catch=(Exception,),
            gc_after_trial=True,
        )
        complete_trial_count = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        )
        print(f"Complete trials after attempt: {complete_trial_count}/{N_TRIALS}")

    print("Optimization finished")
    print("Best joint score:", study.best_value)
    print("Best params:", study.best_params)

    best_trial = study.best_trial

    print("\nBest trial metrics:")
    for k, v in best_trial.user_attrs.items():
        print(f"{k}: {v}")

    best_output = {
        "study_name": STUDY_NAME,
        "dataset_name": DATASET_NAME,
        "dataset": DATA_PATH,
        "batch_keys": BATCH_KEYS,
        "metric_batch_key": BATCH_KEY,
        "label_key": LABEL_KEY,
        "use_hvg": USE_HVG,
        "n_top_hvg": N_TOP_HVG,
        "max_epochs": MAX_EPOCHS,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_metrics": best_trial.user_attrs,
    }

    with open(BEST_PARAMS_JSON, "w") as f:
        json.dump(best_output, f, indent=2)

    print(f"\nSaved trials CSV to: {TRIALS_CSV}")
    print(f"Saved best params JSON to: {BEST_PARAMS_JSON}")


if __name__ == "__main__":
    main()
