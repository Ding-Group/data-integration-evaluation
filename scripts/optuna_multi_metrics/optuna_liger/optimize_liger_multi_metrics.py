#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGER multi-parameter Optuna optimization.

Usage:
    optimize_liger_multi_metrics.py -o <path> -b <batch> -c <key> --dataset-name=<name> [--study-name=<name>] [--seed=<int>] [--mode=<mode>] [--layer=<layer>] [--bami] [--casw] [--scgraphm] [--bras] [--kbet] [--cpcr] [--traj] [--isolisi] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             Batch key.
    -c <key>               Cell type label key.
    --dataset-name=<name>  Exact dataset identifier.
    --study-name=<name>    Optuna study name [default: liger_joint_metrics].
    --seed=<int>           Random seed [default: 1].
    --mode=<mode>          LIGER mode: core or online [default: core].
    --layer=<layer>        Layer used as LIGER input. Use empty string to use adata.X [default: counts].

    --bami                 Use BAMI.
    --casw                 Use CASW.
    --scgraphm             Use SCGRAPHM.
    --bras                 Use BRAS.
    --kbet                 Use KBET.
    --cpcr                 Use CPCR.
    --traj                 Use trajectory metric.
    --isolisi              Use isolated-label LISI.
"""

import json
import os
import sys
import time
import types
from typing import Any

from docopt import docopt

import optuna
import anndata as ad
from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.cluster import contingency_matrix, mutual_info_score
from sklearn.metrics.cluster._expected_mutual_info_fast import (
    expected_mutual_information,
)
from sklearn.metrics.cluster._supervised import check_clusterings, entropy
from sklearn.utils import sparsefuncs


PROJECT_ROOT = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

sys.path.insert(0, f"{PROJECT_ROOT}/scib/")

os.environ.setdefault(
    "R_HOME",
    f"{PROJECT_ROOT}/CondaEnv/harmony_optuna/lib/R",
)

from scib.metrics.metrics import metrics


N_NEIGHBORS = 15
CORE_MAX_ITERS = 30

CLUSTER_KEY = "leiden_opt"
EMBED_KEY = "X_liger"
METRIC_BATCH_KEY = "liger_metric_batch"
METRIC_LABEL_KEY = "liger_metric_label"

SEED = 1
USE_BAMI = False
USE_CASW = False
USE_SCGRAPHM = False
USE_BRAS = False
USE_KBET = False
USE_CPCR = False
USE_TRAJ = False
USE_ISOLISI = False

CORE_N_HYPERPARAMS = 2
ONLINE_N_HYPERPARAMS = 4
N_TRIALS_MULTIPLIER = 50
N_TRIALS = CORE_N_HYPERPARAMS * N_TRIALS_MULTIPLIER


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


def check_keys(adata):
    missing = []
    if BATCH_KEY not in adata.obs:
        missing.append(BATCH_KEY)
    if LABEL_KEY not in adata.obs:
        missing.append(LABEL_KEY)

    if missing:
        raise ValueError(
            f"Missing keys in adata.obs: {missing}. "
            f"Have: {list(adata.obs.columns)}"
        )


def ensure_csr(X):
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)


def prepare_adata_names(adata):
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.obs.index.name = "cell"
    adata.var.index.name = "gene"
    return adata


def make_liger_object(adatas: list[AnnData]) -> Any:
    import pyliger

    return pyliger.create_liger(adatas)


def extract_embedding(lig: Any) -> np.ndarray:
    """
    Extract normalized LIGER cell factor embedding from different pyliger versions.
    """

    if hasattr(lig, "H_norm") and lig.H_norm is not None:
        hn = lig.H_norm
        if isinstance(hn, pd.DataFrame):
            return hn.values
        if isinstance(hn, np.ndarray):
            return hn

    if hasattr(lig, "H_norm_data") and lig.H_norm_data is not None:
        hn = lig.H_norm_data
        if isinstance(hn, pd.DataFrame):
            return hn.values
        if isinstance(hn, np.ndarray):
            return hn

    if hasattr(lig, "adata_list") and isinstance(lig.adata_list, list):
        mats = []

        for batch_adata in lig.adata_list:
            if isinstance(batch_adata, AnnData) and "H_norm" in batch_adata.obsm:
                mats.append(np.asarray(batch_adata.obsm["H_norm"]))

        if mats:
            return np.vstack(mats)

    raise RuntimeError("Cannot find integrated embedding H_norm in LIGER object")


def build_liger_inputs(adata, batch_key, layer):
    """
    Split AnnData by batch and create one AnnData object per batch for pyliger.
    """

    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key not found: {batch_key}")

    if layer:
        if layer in adata.layers:
            X = adata.layers[layer]
        else:
            print(
                f"[WARNING] Layer '{layer}' not found. Using adata.X for LIGER input.",
                flush=True,
            )
            X = adata.X
    else:
        X = adata.X

    X = ensure_csr(X)

    batches = adata.obs[batch_key].astype("category")
    batch_names = list(batches.cat.categories)

    adata_list = []
    idxs_list = []

    for batch_name in batch_names:
        idx = np.where(batches.values == batch_name)[0]
        idxs_list.append(idx)

        sub = adata[idx, :].copy()
        sub = prepare_adata_names(sub)

        Xb = X[idx, :]
        Xb = ensure_csr(Xb)

        sub_liger = AnnData(
            X=Xb,
            obs=sub.obs.copy(),
            var=sub.var.copy(),
        )

        sub_liger.obs_names = sub.obs_names.astype(str)
        sub_liger.var_names = sub.var_names.astype(str)
        sub_liger.obs.index.name = "cell"
        sub_liger.var.index.name = "gene"
        sub_liger.uns["sample_name"] = str(batch_name)

        adata_list.append(sub_liger)

    return adata_list, batch_names, idxs_list


def run_liger(
    adata,
    k,
    lam,
    max_iterations,
    mode,
    minibatch_size=None,
    layer="counts",
    seed=1,
):
    """
    Run pyliger and return an integrated AnnData containing X_liger.
    """

    import pyliger

    np.random.seed(int(seed))

    check_keys(adata)

    adata = adata.copy()
    adata = prepare_adata_names(adata)

    adata_list, batch_names, idxs_list = build_liger_inputs(
        adata=adata,
        batch_key=BATCH_KEY,
        layer=layer,
    )

    lig = make_liger_object(adata_list)

    pyliger.normalize(lig)
    pyliger.select_genes(lig)
    pyliger.scale_not_center(lig)

    if mode == "online":
        print(
            "[LIGER PARAMS] "
            f"function=online_iNMF k={int(k)} value_lambda={float(lam)} "
            f"max_epochs={int(max_iterations)} "
            f"miniBatch_size={int(minibatch_size)} rand_seed={int(seed)}",
            flush=True,
        )
        pyliger.online_iNMF(
            lig,
            k=int(k),
            value_lambda=float(lam),
            max_epochs=int(max_iterations),
            miniBatch_size=int(minibatch_size),
            rand_seed=int(seed),
        )

    elif mode == "core":
        print(
            "[LIGER PARAMS] "
            f"function=optimize_ALS k={int(k)} value_lambda={float(lam)} "
            f"max_iters={int(max_iterations)} rand_seed={int(seed)}",
            flush=True,
        )
        pyliger.optimize_ALS(
            lig,
            k=int(k),
            value_lambda=float(lam),
            max_iters=int(max_iterations),
            rand_seed=int(seed),
        )

    else:
        raise ValueError(f"Unsupported LIGER mode: {mode}")

    print(
        f"[LIGER PARAMS] function=quantile_norm rand_seed={int(seed)}",
        flush=True,
    )
    pyliger.quantile_norm(lig, rand_seed=int(seed))

    Z = extract_embedding(lig)
    Z = np.asarray(Z, dtype=np.float32)

    if Z.ndim != 2:
        raise ValueError(f"LIGER embedding has invalid shape: {Z.shape}")

    if Z.shape[0] != adata.n_obs and Z.shape[1] == adata.n_obs:
        Z = Z.T

    if Z.shape[0] != adata.n_obs:
        raise ValueError(
            f"LIGER embedding row count mismatch: {Z.shape[0]} vs n_obs={adata.n_obs}"
        )

    if not np.all(np.isfinite(Z)):
        raise optuna.exceptions.TrialPruned(
            "LIGER produced NaN or Inf in X_liger"
        )

    concat_indices = np.concatenate(idxs_list)

    if len(concat_indices) != adata.n_obs:
        raise ValueError(
            f"Batch index length mismatch: {len(concat_indices)} vs n_obs={adata.n_obs}"
        )

    Z_reordered = np.empty_like(Z)
    Z_reordered[concat_indices, :] = Z

    adata_int = adata.copy()
    adata_int.obsm[EMBED_KEY] = Z_reordered

    if "counts" in adata.layers:
        adata_int.layers["counts"] = adata.layers["counts"].copy()
    else:
        print(
            "[WARNING] No counts layer found in LIGER input adata. "
            "SCGRAPHM may fail if enabled.",
            flush=True,
        )

    return adata_int


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


def balanced_adjusted_mutual_info(
    labels_true,
    labels_pred,
    *,
    average_method="arithmetic",
    reweigh=True,
):
    """
    NumPy 2-compatible implementation of balanced-clustering 0.1.0 BAMI.
    """

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)

    if reweigh:
        counts_sum_per_class = np.ravel(contingency.sum(1))
        target = round(np.mean(counts_sum_per_class))
        counts_norm = counts_sum_per_class / target
        contingency = contingency.astype(np.float64)
        sparsefuncs.inplace_row_scale(contingency, 1 / counts_norm)
        contingency = contingency.astype(np.int64)

        true_sums = np.squeeze(np.asarray(sp.csc_matrix.sum(contingency, axis=1)))
        pred_sums = np.squeeze(np.asarray(sp.csc_matrix.sum(contingency, axis=0)))
        labels_true = np.repeat(np.arange(len(true_sums)), true_sums)
        labels_pred = np.repeat(np.arange(len(pred_sums)), pred_sums)

    contingency = contingency.astype(np.float64)
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    emi = expected_mutual_information(contingency, n_samples)
    h_true = entropy(labels_true)
    h_pred = entropy(labels_pred)

    if average_method == "min":
        normalizer = min(h_true, h_pred)
    elif average_method == "geometric":
        normalizer = np.sqrt(h_true * h_pred)
    elif average_method == "arithmetic":
        normalizer = np.mean([h_true, h_pred])
    elif average_method == "max":
        normalizer = max(h_true, h_pred)
    else:
        raise ValueError(
            "'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'"
        )

    denominator = normalizer - emi

    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        denominator = max(denominator, np.finfo("float64").eps)

    return (mi - emi) / denominator


def install_balanced_clustering_module():
    module = types.ModuleType("balanced_clustering")
    module.__version__ = "0.1.0-numpy2"
    module.balanced_adjusted_mutual_info = balanced_adjusted_mutual_info
    sys.modules["balanced_clustering"] = module


def validate_metric_dependencies():
    if not USE_BAMI:
        return

    install_balanced_clustering_module()

    score = balanced_adjusted_mutual_info(
        labels_true=np.array([0, 0, 1, 1]),
        labels_pred=np.array([0, 0, 1, 1]),
    )

    if not np.isclose(score, 1.0):
        raise RuntimeError(f"BAMI dependency validation returned {score}, expected 1.0.")


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


def build_search_space(trial):
    """
    LIGER hyperparameter search space.

    Keep the first general version compact:
        core mode:   k, value_lambda
        online mode: k, value_lambda, max_epochs, miniBatch_size
    """

    k = trial.suggest_int("k", 20, 60)
    value_lambda = trial.suggest_float("value_lambda", 0.5, 10.0, log=True)

    params = {
        "k": k,
        "value_lambda": value_lambda,
    }

    if MODE == "online":
        params["max_epochs"] = trial.suggest_int("max_epochs", 5, 30)
        params["miniBatch_size"] = trial.suggest_int(
            "miniBatch_size",
            1000,
            8000,
            step=1000,
        )

    return params


def objective(trial):
    start_time = time.time()

    print(f"\n[START TRIAL {trial.number}]", flush=True)

    params = build_search_space(trial)

    mode = MODE
    layer = LAYER
    n_neighbors = N_NEIGHBORS
    iteration_key = "max_epochs" if mode == "online" else "max_iters"
    max_iterations = params["max_epochs"] if mode == "online" else CORE_MAX_ITERS

    adata = ad.read_h5ad(DATA_PATH)
    check_keys(adata)

    params_for_log = {
        **params,
        "mode": mode,
        "layer": layer,
        "n_neighbors": n_neighbors,
    }
    if mode == "core":
        params_for_log["max_iters"] = CORE_MAX_ITERS

    adata_int = run_liger(
        adata=adata,
        k=params["k"],
        lam=params["value_lambda"],
        max_iterations=max_iterations,
        mode=mode,
        minibatch_size=params.get("miniBatch_size", None),
        layer=layer,
        seed=SEED,
    )

    metric_results = compute_joint_metrics(adata, adata_int)
    score = metric_results["joint_score"]

    for key, value in metric_results.items():
        trial.set_user_attr(key, value)

    runtime_sec = time.time() - start_time
    trial.set_user_attr("runtime_sec", runtime_sec)

    save_trial_row(
        trial=trial,
        params=params_for_log,
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
        f"{iteration_key}={max_iterations} "
    )

    if "miniBatch_size" in params:
        msg += f"miniBatch_size={params['miniBatch_size']} "

    msg += (
        f"mode={mode} "
        f"layer={layer} "
        f"n_neighbors={n_neighbors} "
        f"runtime={runtime_sec:.1f}s"
    )

    print(msg, flush=True)

    return score


def main():
    global DATA_PATH, DATASET_NAME, RESULT_DIR, STUDY_NAME
    global STUDY_DB, TRIALS_CSV, BEST_PARAMS_JSON
    global BATCH_KEY, LABEL_KEY, SEED, MODE, LAYER, N_TRIALS
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
    MODE = args["--mode"].lower()
    LAYER = args["--layer"]

    if SEED < 1:
        raise ValueError(
            "--seed must be at least 1 because pyliger.optimize_ALS "
            "uses rand_seed + j - 1 internally."
        )

    if LAYER.lower() in ["none", "null", "x", ""]:
        LAYER = ""

    if MODE not in ["core", "online"]:
        raise ValueError(f"Unsupported --mode={MODE}. Use core or online.")

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

    if len(selected_bio_metrics) == 0:
        raise ValueError("No bio-conservation metric selected.")

    if len(selected_batch_metrics) == 0:
        raise ValueError("No batch-correction metric selected.")

    validate_metric_dependencies()
    validate_configuration(selected_metrics)
    validate_dataset_contents()

    n_hyperparams = (
        ONLINE_N_HYPERPARAMS if MODE == "online" else CORE_N_HYPERPARAMS
    )
    N_TRIALS = n_hyperparams * N_TRIALS_MULTIPLIER

    print(f"Dataset name: {DATASET_NAME}")
    print(f"Selected metrics: {selected_metrics}")
    print(f"Selected bio metrics: {selected_bio_metrics}")
    print(f"Selected batch metrics: {selected_batch_metrics}")

    os.makedirs(RESULT_DIR, exist_ok=True)

    STUDY_DB = f"{RESULT_DIR}/{STUDY_NAME}.db"
    TRIALS_CSV = f"{RESULT_DIR}/{STUDY_NAME}_trials.csv"
    BEST_PARAMS_JSON = f"{RESULT_DIR}/{STUDY_NAME}_best_params.json"

    print("LIGER joint-metric Optuna")
    print(f"Dataset ID: {DATASET_NAME}")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Batch key: {BATCH_KEY}")
    print(f"Label key: {LABEL_KEY}")
    print(f"Mode: {MODE}")
    print(f"Layer: {LAYER if LAYER else 'adata.X'}")
    if MODE == "core":
        print(f"Fixed max_iters: {CORE_MAX_ITERS}")
    print(f"Result dir: {RESULT_DIR}")
    print(f"Study DB: {STUDY_DB}")
    print(f"N trials: {N_TRIALS}")

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,
        catch=(Exception,),
    )

    print("Optimization finished")

    complete_trials = study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE,),
    )

    if len(complete_trials) == 0:
        failed_trials = study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.FAIL,),
        )
        raise RuntimeError(
            "Optimization finished without completed trials "
            f"({len(failed_trials)} failed). Check the job error log before "
            "reading best_value."
        )

    print("Best joint score:", study.best_value)
    print("Best params:", study.best_params)

    best_trial = study.best_trial

    print("\nBest trial metrics:")
    for key, value in best_trial.user_attrs.items():
        print(f"{key}: {value}")

    best_output = {
        "study_name": STUDY_NAME,
        "dataset_name": DATASET_NAME,
        "dataset": DATA_PATH,
        "batch_key": BATCH_KEY,
        "label_key": LABEL_KEY,
        "mode": MODE,
        "layer": LAYER,
        "fixed_params": {"max_iters": CORE_MAX_ITERS} if MODE == "core" else {},
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
