#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seurat CCA multi-parameter Optuna optimization.

Usage:
    optimize_cca_multi_metrics.py -o <path> -b <batch> -c <key> --dataset-name=<name> [--study-name=<name>] [--seed=<int>] [--layer=<layer>] [--nfeatures-min=<int>] [--nfeatures-max=<int>] [--nfeatures-step=<int>] [--npcs-min=<int>] [--npcs-max=<int>] [--npcs-step=<int>] [--bami] [--casw] [--scgraphm] [--bras] [--kbet] [--cpcr] [--traj] [--isolisi] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             Batch key.
    -c <key>               Cell type label key.
    --dataset-name=<name>  Exact dataset identifier.
    --study-name=<name>    Optuna study name [default: cca_joint_metrics].
    --seed=<int>           Random seed [default: 0].
    --layer=<layer>        Layer used as Seurat counts input. Use empty string to use adata.X [default: counts].
    --nfeatures-min=<int>  Minimum variable features for FindIntegrationAnchors [default: 1000].
    --nfeatures-max=<int>  Maximum variable features for FindIntegrationAnchors [default: 5000].
    --nfeatures-step=<int> Step size for variable features [default: 500].
    --npcs-min=<int>       Minimum PCA components for FindIntegrationAnchors [default: 30].
    --npcs-max=<int>       Maximum PCA components for FindIntegrationAnchors [default: 60].
    --npcs-step=<int>      Step size for PCA components [default: 5].

    --bami                 Use BAMI.
    --casw                 Use CASW.
    --scgraphm             Use SCGRAPHM.
    --bras                 Use BRAS.
    --kbet                 Use KBET.
    --cpcr                 Use CPCR.
    --traj                 Use trajectory metric.
    --isolisi              Use isolated-label LISI.
"""

from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from docopt import docopt

import anndata as ad
import numpy as np
import optuna
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.io import mmwrite


PROJECT_ROOT = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

sys.path.insert(0, f"{PROJECT_ROOT}/scib/")

BALANCED_CLUSTERING_SITE_PACKAGES = (
    Path(PROJECT_ROOT)
    / "CondaEnv/harmony_optuna/lib/python3.10/site-packages"
)
BALANCED_CLUSTERING_PACKAGE_DIR = (
    BALANCED_CLUSTERING_SITE_PACKAGES / "balanced_clustering"
)

os.environ.setdefault(
    "R_HOME",
    f"{PROJECT_ROOT}/CondaEnv/cca_optuna/lib/R",
)

from scib.metrics.metrics import metrics


N_NEIGHBORS = 15
DIMS_USE = 30

CLUSTER_KEY = "leiden_opt"
EMBED_KEY = "X_cca"
METRIC_BATCH_KEY = "cca_metric_batch"
METRIC_LABEL_KEY = "cca_metric_label"

R_LIBS_PATH = (
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/"
    "CondaEnv/cca_optuna/lib/R/library"
)
R_SCRIPT_PATH = (
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/"
    "CondaEnv/cca_optuna/bin/Rscript"
)

SEED = 0
USE_BAMI = False
USE_CASW = False
USE_SCGRAPHM = False
USE_BRAS = False
USE_KBET = False
USE_CPCR = False
USE_TRAJ = False
USE_ISOLISI = False

N_HYPERPARAMS = 2
N_TRIALS_MULTIPLIER = 50
N_TRIALS = N_HYPERPARAMS * N_TRIALS_MULTIPLIER

DEFAULT_NFEATURES_MIN = 1000
DEFAULT_NFEATURES_MAX = 5000
DEFAULT_NFEATURES_STEP = 500
DEFAULT_NPCS_MIN = 30
DEFAULT_NPCS_MAX = 60
DEFAULT_NPCS_STEP = 5

NFEATURES_MIN = DEFAULT_NFEATURES_MIN
NFEATURES_MAX = DEFAULT_NFEATURES_MAX
NFEATURES_STEP = DEFAULT_NFEATURES_STEP
NPCS_MIN = DEFAULT_NPCS_MIN
NPCS_MAX = DEFAULT_NPCS_MAX
NPCS_STEP = DEFAULT_NPCS_STEP


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

        if LAYER and LAYER not in adata.layers:
            raise ValueError(
                f"Layer {LAYER!r} is missing in {DATASET_NAME}. "
                f"Available layers: {list(adata.layers.keys())}"
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


def ensure_balanced_clustering_importable():
    try:
        import balanced_clustering  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    init_path = BALANCED_CLUSTERING_PACKAGE_DIR / "__init__.py"

    if not init_path.exists():
        raise ModuleNotFoundError(
            "BAMI requires balanced_clustering, but it was not importable "
            "from the active Python environment and fallback package path "
            f"does not exist: {BALANCED_CLUSTERING_PACKAGE_DIR}."
        )

    spec = importlib.util.spec_from_file_location(
        "balanced_clustering",
        init_path,
        submodule_search_locations=[str(BALANCED_CLUSTERING_PACKAGE_DIR)],
    )

    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(
            "BAMI requires balanced_clustering, but Python could not load "
            f"the fallback package at {BALANCED_CLUSTERING_PACKAGE_DIR}."
        )

    fallback_site_packages = str(BALANCED_CLUSTERING_SITE_PACKAGES)
    remove_fallback_path = False

    if fallback_site_packages not in sys.path:
        sys.path.append(fallback_site_packages)
        remove_fallback_path = True

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules["balanced_clustering"] = module
        spec.loader.exec_module(module)
    finally:
        if remove_fallback_path:
            sys.path.remove(fallback_site_packages)


R_SCRIPT = r"""
R_LIBS_PATH <- "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/CondaEnv/cca_optuna/lib/R/library"
.libPaths(c(R_LIBS_PATH))

suppressPackageStartupMessages({
  library(SeuratObject, lib.loc = R_LIBS_PATH)
  library(Seurat,       lib.loc = R_LIBS_PATH)
  library(Matrix,       lib.loc = R_LIBS_PATH)
})

if (utils::packageVersion("SeuratObject") >= "5.0.0") {
  stop("FATAL: SeuratObject v5 loaded")
}

args <- commandArgs(trailingOnly = TRUE)
mtx_path   <- args[[1]]
genes_path <- args[[2]]
cells_path <- args[[3]]
meta_path  <- args[[4]]
out_csv    <- args[[5]]
batch_key  <- args[[6]]
nfeatures  <- as.integer(args[[7]])
npcs       <- as.integer(args[[8]])
dims_use   <- as.integer(args[[9]])
seed       <- as.integer(args[[10]])

set.seed(seed)

mat <- readMM(mtx_path)
genes <- readLines(genes_path)
cells <- readLines(cells_path)

rownames(mat) <- genes
colnames(mat) <- cells

meta <- read.csv(meta_path, row.names = 1, check.names = FALSE)
meta <- meta[cells, , drop = FALSE]

if (!(batch_key %in% colnames(meta))) {
  stop(paste0("batch_key not found in meta: ", batch_key))
}

if (dims_use > npcs) {
  stop(paste0("dims_use (", dims_use, ") > npcs (", npcs, ")"))
}

obj <- CreateSeuratObject(counts = mat, meta.data = meta, assay = "RNA")

split_cells <- split(colnames(obj), obj@meta.data[[batch_key]])
obj.list <- lapply(names(split_cells), function(b) {
  subset(obj, cells = split_cells[[b]])
})
names(obj.list) <- names(split_cells)

obj.list <- lapply(obj.list, function(x) {
  x <- NormalizeData(x, verbose = FALSE)
  x <- FindVariableFeatures(
    x,
    selection.method = "vst",
    nfeatures = nfeatures,
    verbose = FALSE
  )
  hvgs <- VariableFeatures(x)
  x <- ScaleData(x, features = hvgs, verbose = FALSE)
  x <- RunPCA(x, features = hvgs, npcs = npcs, verbose = FALSE)
  return(x)
})

anchors <- FindIntegrationAnchors(
  object.list = obj.list,
  normalization.method = "LogNormalize",
  anchor.features = nfeatures,
  reduction = "cca",
  dims = 1:dims_use
)

obj.int <- IntegrateData(
  anchorset = anchors,
  dims = 1:dims_use
)

DefaultAssay(obj.int) <- "integrated"
obj.int <- ScaleData(obj.int, verbose = FALSE)
obj.int <- RunPCA(obj.int, npcs = npcs, verbose = FALSE)

emb <- Embeddings(obj.int, reduction = "pca")

if (ncol(emb) < dims_use) {
  stop(paste0("integrated embedding dims=", ncol(emb), " < dims_use=", dims_use))
}

emb2 <- emb[, seq_len(dims_use), drop = FALSE]
df <- data.frame(cell_id = rownames(emb2), emb2, check.names = FALSE)
write.csv(df, out_csv, row.names = FALSE, quote = FALSE)
"""


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


def prepare_adata_names(adata):
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.obs.index.name = "cell"
    adata.var.index.name = "gene"
    return adata


def ensure_csr(X):
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(np.asarray(X))


def export_anndata_to_mtx(
    adata_obj: ad.AnnData,
    workdir: Path,
    counts_layer: str,
) -> tuple[Path, Path, Path, Path]:
    """
    Export AnnData into files readable by Seurat:
        counts.mtx: genes x cells
        genes.txt
        cells.txt
        meta.csv
    """

    workdir.mkdir(parents=True, exist_ok=True)

    if counts_layer:
        if counts_layer in adata_obj.layers:
            X = adata_obj.layers[counts_layer]
        else:
            print(
                f"[WARNING] Layer '{counts_layer}' not found. "
                "Using adata.X for Seurat CCA input.",
                flush=True,
            )
            X = adata_obj.X
    else:
        X = adata_obj.X

    X = ensure_csr(X)
    X = X.T.tocoo()

    mtx_path = workdir / "counts.mtx"
    genes_path = workdir / "genes.txt"
    cells_path = workdir / "cells.txt"
    meta_path = workdir / "meta.csv"

    mmwrite(str(mtx_path), X)

    genes_path.write_text(
        "\n".join(adata_obj.var_names.astype(str).tolist()) + "\n",
        encoding="utf-8",
    )

    cells_path.write_text(
        "\n".join(adata_obj.obs_names.astype(str).tolist()) + "\n",
        encoding="utf-8",
    )

    meta = adata_obj.obs.copy()
    meta.index = meta.index.astype(str)
    meta.to_csv(meta_path)

    return mtx_path, genes_path, cells_path, meta_path


def locked_r_env() -> dict:
    """
    Keep R package loading fixed to the cca_optuna environment.
    """

    env = os.environ.copy()

    for key in [
        "R_HOME",
        "R_LIBS_USER",
        "R_LIBS_SITE",
        "R_LIBS",
        "R_PROFILE",
        "R_PROFILE_USER",
        "R_ENVIRON",
        "R_ENVIRON_USER",
    ]:
        env.pop(key, None)

    env["R_LIBS_USER"] = R_LIBS_PATH
    env["R_LIBS_SITE"] = ""
    env["R_LIBS"] = R_LIBS_PATH

    return env


def run_seurat_cca_embedding(
    mtx_path: Path,
    genes_path: Path,
    cells_path: Path,
    meta_path: Path,
    batch_key: str,
    nfeatures: int,
    npcs: int,
    dims_use: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Seurat CCA through Rscript and return:
        cell_ids
        integrated PCA embedding
    """

    workdir = mtx_path.parent
    r_path = workdir / "run_cca.R"
    out_csv = workdir / "embedding.csv"

    r_path.write_text(R_SCRIPT, encoding="utf-8")

    cmd = [
        R_SCRIPT_PATH,
        "--vanilla",
        str(r_path),
        str(mtx_path),
        str(genes_path),
        str(cells_path),
        str(meta_path),
        str(out_csv),
        str(batch_key),
        str(nfeatures),
        str(npcs),
        str(dims_use),
        str(seed),
    ]

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=locked_r_env(),
    )

    if process.returncode != 0 or not out_csv.exists():
        raise RuntimeError(
            f"Seurat CCA failed with return code {process.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:\n{process.stdout}"
        )

    df = pd.read_csv(out_csv)

    if "cell_id" not in df.columns:
        raise RuntimeError(
            f"Seurat CCA output missing cell_id column. "
            f"Columns: {list(df.columns)}"
        )

    cell_ids = df["cell_id"].astype(str).to_numpy()
    X = df.drop(columns=["cell_id"]).to_numpy(dtype=np.float32)

    return cell_ids, X


def align_embedding_to_adata(
    adata_obj: ad.AnnData,
    cell_ids: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    Align Seurat output embedding back to AnnData obs order.
    """

    idx = {cid: i for i, cid in enumerate(cell_ids)}
    out = np.zeros((adata_obj.n_obs, X.shape[1]), dtype=np.float32)

    missing = 0

    for i, cid in enumerate(adata_obj.obs_names.astype(str)):
        j = idx.get(cid)

        if j is None:
            missing += 1
        else:
            out[i] = X[j]

    if missing > 0:
        raise ValueError(
            f"{missing} cells missing when aligning embedding to adata.obs_names"
        )

    return out


def cleanup_cca_workdir_files(workdir: Path):
    for filename in (
        "counts.mtx",
        "genes.txt",
        "cells.txt",
        "meta.csv",
        "run_cca.R",
        "embedding.csv",
    ):
        path = workdir / filename
        if path.exists():
            path.unlink()


def run_cca(
    adata_obj: ad.AnnData,
    nfeatures: int,
    npcs: int,
    dims_use: int,
    layer: str,
    seed: int = 0,
) -> ad.AnnData:
    """
    Run Seurat CCA and return integrated AnnData containing X_cca.
    """

    check_keys(adata_obj)

    adata_obj = adata_obj.copy()
    adata_obj = prepare_adata_names(adata_obj)

    if adata_obj.obs[BATCH_KEY].nunique() < 2:
        raise ValueError(
            f"Need at least 2 batches for CCA, "
            f"got {adata_obj.obs[BATCH_KEY].nunique()}"
        )

    if dims_use > npcs:
        raise optuna.exceptions.TrialPruned(
            f"dims_use={dims_use} cannot be larger than npcs={npcs}"
        )

    workdir = Path("/tmp") / f"seurat_cca_{os.getpid()}_{int(time.time() * 1000)}"

    try:
        mtx_path, genes_path, cells_path, meta_path = export_anndata_to_mtx(
            adata_obj=adata_obj,
            workdir=workdir,
            counts_layer=layer,
        )

        cell_ids, X = run_seurat_cca_embedding(
            mtx_path=mtx_path,
            genes_path=genes_path,
            cells_path=cells_path,
            meta_path=meta_path,
            batch_key=BATCH_KEY,
            nfeatures=int(nfeatures),
            npcs=int(npcs),
            dims_use=int(dims_use),
            seed=int(seed),
        )

        Z = align_embedding_to_adata(adata_obj, cell_ids, X)

        if Z.ndim != 2:
            raise ValueError(f"CCA embedding has invalid shape: {Z.shape}")

        if Z.shape[0] != adata_obj.n_obs:
            raise ValueError(
                f"CCA embedding row count mismatch: "
                f"{Z.shape[0]} vs n_obs={adata_obj.n_obs}"
            )

        if not np.all(np.isfinite(Z)):
            raise optuna.exceptions.TrialPruned(
                "Seurat CCA produced NaN or Inf in X_cca"
            )

        adata_int = adata_obj.copy()
        adata_int.obsm[EMBED_KEY] = Z

        if "counts" in adata_obj.layers:
            adata_int.layers["counts"] = adata_obj.layers["counts"].copy()
        else:
            print(
                "[WARNING] No counts layer found in CCA input adata. "
                "SCGRAPHM may fail if enabled.",
                flush=True,
            )

        return adata_int

    finally:
        cleanup_cca_workdir_files(workdir)


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
    nfeatures = trial.suggest_int(
        "nfeatures",
        NFEATURES_MIN,
        NFEATURES_MAX,
        step=NFEATURES_STEP,
    )
    npcs = trial.suggest_int(
        "npcs",
        NPCS_MIN,
        NPCS_MAX,
        step=NPCS_STEP,
    )

    return {
        "nfeatures": nfeatures,
        "npcs": npcs,
    }


def objective(trial):
    start_time = time.time()

    print(f"\n[START TRIAL {trial.number}]", flush=True)

    params = build_search_space(trial)

    layer = LAYER
    dims_use = DIMS_USE
    n_neighbors = N_NEIGHBORS

    adata_raw = ad.read_h5ad(DATA_PATH)
    check_keys(adata_raw)

    params_for_log = {
        **params,
        "dims_use": dims_use,
        "layer": layer,
        "n_neighbors": n_neighbors,
    }

    adata_int = run_cca(
        adata_obj=adata_raw,
        nfeatures=params["nfeatures"],
        npcs=params["npcs"],
        dims_use=dims_use,
        layer=layer,
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
        params=params_for_log,
        metric_results=metric_results,
        runtime_sec=runtime_sec,
    )

    metric_log = " ".join(
        f"{key}={value:.4f}"
        for key, value in metric_results.items()
        if key not in ["bio_score", "batch_score", "joint_score"]
    )

    print(
        f"[END TRIAL {trial.number}] "
        f"joint={score:.4f} "
        f"bio={metric_results['bio_score']:.4f} "
        f"batch={metric_results['batch_score']:.4f} "
        f"{metric_log} "
        f"nfeatures={params['nfeatures']} "
        f"npcs={params['npcs']} "
        f"dims_use={dims_use} "
        f"layer={layer} "
        f"n_neighbors={n_neighbors} "
        f"runtime={runtime_sec:.1f}s",
        flush=True,
    )

    return score


def main():
    global DATA_PATH, DATASET_NAME, RESULT_DIR, STUDY_NAME
    global STUDY_DB, TRIALS_CSV, BEST_PARAMS_JSON
    global BATCH_KEY, LABEL_KEY, SEED, LAYER
    global USE_BAMI, USE_CASW, USE_SCGRAPHM, USE_BRAS, USE_KBET
    global USE_CPCR, USE_TRAJ, USE_ISOLISI
    global NFEATURES_MIN, NFEATURES_MAX, NFEATURES_STEP
    global NPCS_MIN, NPCS_MAX, NPCS_STEP

    args = docopt(__doc__)

    DATA_PATH = args["<file>"]
    DATASET_NAME = args["--dataset-name"]
    RESULT_DIR = args["-o"]
    BATCH_KEY = args["-b"]
    LABEL_KEY = args["-c"]
    STUDY_NAME = args["--study-name"]
    SEED = int(args["--seed"])
    LAYER = args["--layer"]
    NFEATURES_MIN = int(
        args["--nfeatures-min"] or DEFAULT_NFEATURES_MIN
    )
    NFEATURES_MAX = int(
        args["--nfeatures-max"] or DEFAULT_NFEATURES_MAX
    )
    NFEATURES_STEP = int(
        args["--nfeatures-step"] or DEFAULT_NFEATURES_STEP
    )
    NPCS_MIN = int(args["--npcs-min"] or DEFAULT_NPCS_MIN)
    NPCS_MAX = int(args["--npcs-max"] or DEFAULT_NPCS_MAX)
    NPCS_STEP = int(args["--npcs-step"] or DEFAULT_NPCS_STEP)

    if NFEATURES_MIN > NFEATURES_MAX:
        raise ValueError("nfeatures-min cannot be larger than nfeatures-max.")
    if NPCS_MIN > NPCS_MAX:
        raise ValueError("npcs-min cannot be larger than npcs-max.")
    if NFEATURES_STEP <= 0:
        raise ValueError("nfeatures-step must be positive.")
    if NPCS_STEP <= 0:
        raise ValueError("npcs-step must be positive.")
    if NPCS_MIN < DIMS_USE:
        raise ValueError("npcs-min cannot be smaller than DIMS_USE.")

    if LAYER.lower() in ["none", "null", "x", ""]:
        LAYER = ""

    USE_BAMI = args["--bami"]
    USE_CASW = args["--casw"]
    USE_SCGRAPHM = args["--scgraphm"]
    USE_BRAS = args["--bras"]
    USE_KBET = args["--kbet"]
    USE_CPCR = args["--cpcr"]
    USE_TRAJ = args["--traj"]
    USE_ISOLISI = args["--isolisi"]

    if USE_BAMI:
        ensure_balanced_clustering_importable()

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

    print("Seurat CCA joint-metric Optuna")
    print(f"Dataset ID: {DATASET_NAME}")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Batch key: {BATCH_KEY}")
    print(f"Label key: {LABEL_KEY}")
    print(f"Layer: {LAYER if LAYER else 'adata.X'}")
    print(f"Rscript: {R_SCRIPT_PATH}")
    print(f"R libs: {R_LIBS_PATH}")
    print(f"Result dir: {RESULT_DIR}")
    print(f"Study DB: {STUDY_DB}")
    print(f"N trials: {N_TRIALS}")
    print(
        "CCA search space: "
        f"nfeatures={NFEATURES_MIN}..{NFEATURES_MAX} step {NFEATURES_STEP}; "
        f"npcs={NPCS_MIN}..{NPCS_MAX} step {NPCS_STEP}"
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )

    complete_trials_before = study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE,),
    )
    remaining_trials = max(0, N_TRIALS - len(complete_trials_before))

    print(f"Finished complete trials already in study: {len(complete_trials_before)}")
    print(f"Remaining complete trials to reach {N_TRIALS}: {remaining_trials}")

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
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
        "layer": LAYER,
        "r_script_path": R_SCRIPT_PATH,
        "r_libs_path": R_LIBS_PATH,
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
