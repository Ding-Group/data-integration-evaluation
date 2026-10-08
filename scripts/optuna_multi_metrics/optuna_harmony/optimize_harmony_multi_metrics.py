"""
Harmony multi-parameter Optuna optimization.

Usage:
    optimize_harmony_multi_metrics.py -o <path> -b <batch> -c <key> --dataset-name=<name> [--study-name=<name>] [--seed=<int>] [--bami] [--casw] [--scgraphm] [--bras] [--kbet] [--cpcr] [--traj] [--isolisi] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             Batch key.
    -c <key>               Cell type label key.
    --dataset-name=<name>  Exact dataset identifier.
    --study-name=<name>    Optuna study name [default: harmony_joint_metrics].
    --seed=<int>           Random seed [default: 0].

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

from docopt import docopt

import optuna
import anndata as ad
import scanpy as sc
import harmonypy as hm
import numpy as np


PROJECT_ROOT = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

sys.path.insert(0, f"{PROJECT_ROOT}/scib/")

os.environ["R_HOME"] = (
    f"{PROJECT_ROOT}/CondaEnv/harmony_optuna/lib/R"
)



from scib.metrics.metrics import metrics


N_PCS = 50
N_NEIGHBORS = 15
N_TOP_HVG = 2000

CLUSTER_KEY = "leiden_opt"
EMBED_KEY = "X_harmony"
METRIC_BATCH_KEY = "harmony_metric_batch"
METRIC_LABEL_KEY = "harmony_metric_label"
SEED = 0
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
        "batch_key": "tech",
        "label_key": "celltype",
        "metrics": (
            "BAMI", "CASW", "SCGRAPHM", "ISOLISI", "BRAS", "KBET", "CPCR"
        ),
    },
    "neurips_bmmc": {
        "batch_key": "batch",
        "label_key": "cell_type",
        "metrics": (
            "BAMI", "CASW", "SCGRAPHM", "TRAJ", "ISOLISI",
            "BRAS", "KBET", "CPCR",
        ),
    },
    "nph_cerebellum": {
        "batch_key": "Sample",
        "label_key": "Celltype",
        "metrics": ("BAMI", "CASW", "SCGRAPHM", "BRAS", "KBET", "CPCR"),
    },
    "seurat_pbmc": {
        "batch_key": "donor",
        "label_key": "celltype.l2",
        "metrics": ("BAMI", "CASW", "SCGRAPHM", "BRAS", "KBET", "CPCR"),
    },
    "cellranger_versions": {
        "batch_key": "batch",
        "label_key": "cell_type",
        "metrics": ("BAMI", "CASW", "SCGRAPHM", "BRAS", "KBET", "CPCR"),
    },
    "scib_sim1": {
        "batch_key": "Batch",
        "label_key": "Group",
        "metrics": (
            "BAMI", "CASW", "SCGRAPHM", "ISOLISI", "BRAS", "KBET", "CPCR"
        ),
    },
    "mouse_mammary": {
        "batch_key": "BATCH",
        "label_key": "celltype",
        "metrics": ("BAMI", "CASW", "SCGRAPHM", "BRAS", "KBET", "CPCR"),
    },
    "human_pbmc_sequencing_techs": {
        "batch_key": "Method",
        "label_key": "CellType",
        "metrics": (
            "BAMI", "CASW", "SCGRAPHM", "ISOLISI", "BRAS", "KBET", "CPCR"
        ),
    },
    "mouse_regulatory_T": {
        "batch_key": "method",
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

    if BATCH_KEY != config["batch_key"]:
        raise ValueError(
            f"Batch key mismatch for {DATASET_NAME}: "
            f"expected {config['batch_key']!r}, got {BATCH_KEY!r}"
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
            key for key in (BATCH_KEY, LABEL_KEY)
            if key not in adata.obs
        ]
        if missing_keys:
            raise ValueError(
                f"Missing obs keys {missing_keys} in {DATASET_NAME}. "
                f"Available keys: {obs_columns}"
            )

        obs = adata.obs[[BATCH_KEY, LABEL_KEY]].copy()
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
            f"Batch-correction metrics require at least 2 batches; "
            f"{DATASET_NAME} has {n_batches}."
        )

    batch_coverage = obs.groupby(
        LABEL_KEY,
        observed=True,
    )[BATCH_KEY].nunique()
    label_counts = obs.groupby(LABEL_KEY, observed=True).size()

    print(
        f"Dataset preflight: n_obs={len(obs)} "
        f"n_batches={n_batches} n_labels={n_labels}"
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

    if "highly_variable" in adata.var:
        adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat",
            n_top_genes=N_TOP_HVG,
        )
        adata = adata[:, adata.var.highly_variable].copy()

    sc.tl.pca(
        adata,
        n_comps=N_PCS,
        svd_solver="arpack",
    )

    X = np.asarray(adata.obsm["X_pca"], dtype=np.float32)

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
        random_state=seed,
    )

    Z = np.asarray(ho.Z_corr)

    if Z.shape[0] != adata.n_obs and Z.shape[1] == adata.n_obs:
        Z = Z.T

    if Z.shape[0] != adata.n_obs:
        raise ValueError(
            f"Harmony output shape {Z.shape} unexpected "
            f"(n_obs={adata.n_obs})"
        )

    if not np.all(np.isfinite(Z)):
        raise optuna.exceptions.TrialPruned(
            "Harmony produced NaN or Inf in X_harmony"
        )

    adata.obsm[EMBED_KEY] = Z

    return adata

def metric_value(metric_series, key):
    if key not in metric_series.index:
        raise KeyError(
            f"Metric {key} not found. "
            f"Available metrics: {list(metric_series.index)}"
        )

    value = metric_series[key]

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError(f"Metric {key} returned non-finite values: {arr}")
        return float(np.mean(arr))

    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Metric {key} returned a non-finite value: {value}")
    return value


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

    adata = ad.read_h5ad(DATA_PATH)

    if BATCH_KEY not in adata.obs or LABEL_KEY not in adata.obs:
        raise ValueError(
            f"Missing keys in adata.obs. "
            f"Have: {list(adata.obs.columns)}"
        )

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
       
    return score


def main():
    global DATA_PATH, DATASET_NAME, RESULT_DIR, STUDY_NAME
    global STUDY_DB, TRIALS_CSV, BEST_PARAMS_JSON
    global BATCH_KEY, LABEL_KEY, SEED
    global USE_BAMI, USE_CASW, USE_SCGRAPHM, USE_BRAS, USE_KBET
    global USE_CPCR, USE_TRAJ, USE_ISOLISI

    args = docopt(__doc__)

    DATA_PATH = args["<file>"]
    DATASET_NAME = args["--dataset-name"]
    RESULT_DIR = args["-o"]
    BATCH_KEY = args["-b"]
    LABEL_KEY = args["-c"]
    STUDY_NAME = args["--study-name"]
    SEED = int(args["--seed"])

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
        raise ValueError("No metrics selected. Please pass at least one metric flag.")

    if len(selected_bio_metrics) == 0:
        raise ValueError("No bio-conservation metric selected.")

    if len(selected_batch_metrics) == 0:
        raise ValueError("No batch-correction metric selected.")

    validate_configuration(selected_metrics)
    validate_dataset_contents()

    print(f"Dataset name: {DATASET_NAME}")
    print(f"Selected metrics: {selected_metrics}")
    print(f"Selected bio metrics: {selected_bio_metrics}")
    print(f"Selected batch metrics: {selected_batch_metrics}")

    os.makedirs(RESULT_DIR, exist_ok=True)

    STUDY_DB = f"{RESULT_DIR}/{STUDY_NAME}.db"
    TRIALS_CSV = f"{RESULT_DIR}/{STUDY_NAME}_trials.csv"
    BEST_PARAMS_JSON = f"{RESULT_DIR}/{STUDY_NAME}_best_params.json"

    print("Harmony joint-metric Optuna")
    print(f"Dataset ID: {DATASET_NAME}")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Batch key: {BATCH_KEY}")
    print(f"Label key: {LABEL_KEY}")
    print(f"Result dir: {RESULT_DIR}")
    print(f"Study DB: {STUDY_DB}")
    print(f"N trials: {N_TRIALS}")
    

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
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
        "study_name": STUDY_NAME,
        "dataset_name": DATASET_NAME,
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


if __name__ == "__main__":
    main()
