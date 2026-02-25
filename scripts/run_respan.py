#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run ResPAN on a dataset

Usage:
    run_ResPAN.py -o <path> (-b <batch>)... -c <key> [options] <file>

Options:
    -h --help               Show this screen.
    -o <path>               Output directory.
    -b <batch>              One or more batch keys (first is main).
    -c <key>                Cell type label key.
    --seed=<int>            Random seed. [default: 0]
    --epochs=<int>          Training epochs. [default: 300]
    --batch-size=<int>      Batch size. [default: 1024]
    --reduction=<str>       Dimensionality reduction. [default: pca]
    --subsample=<int>       Per-batch subsample size. [default: 3000]
"""
import sys
sys.path.insert(0, '/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/ResPAN')

from os.path import join
from typing import List, Optional

import anndata as ad
import numpy as np
import scanpy as sc
from docopt import docopt
from ResPAN import run_respan
from scipy import sparse
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from project_utils.hvg_utils import select_hvgs


def run_respan_pipeline(
    adata: ad.AnnData,
    batch_keys: List[str],
    cell_type_key: str,
    seed: int,
    epochs: int,
    batch_size: int,
    reduction: str,
    subsample: int,
) -> ad.AnnData:
    np.random.seed(seed)

    # Must be 2000 highly variable genes
    adata_copy = adata.copy()
    sc.pp.highly_variable_genes(adata_copy, layer='counts', n_top_genes=2000, flavor='seurat_v3')
    adata_copy = adata_copy[:, adata_copy.var.highly_variable].copy()
    _, batch_key_final = combine_batch(adata_copy, batch_keys)
    adata_copy.obs['batch'] = adata_copy.obs[batch_key_final]

    adata_copy.X = adata_copy.X.toarray().astype(np.float32)
    adata_copy.obs['celltype'] = adata_copy.obs[cell_type_key]

    print(f"[Info] HVG shape={adata_copy.shape}")
    print(f"Run ResPAN: reduction={reduction}, epochs={epochs}, batch={batch_size}, subsample={subsample}")

    adata_new = run_respan(
        adata_copy,
        batch_key="batch",
        epoch=epochs,
        batch=batch_size,
        reduction=reduction,
        subsample=subsample,
        seed=seed,
    )

    sc.pp.scale(adata_new, max_value=10)
    sc.tl.pca(adata_new, 20, svd_solver='arpack')

    adata_new = adata_new[adata.obs_names]
    adata.obsm[EMB_KEY] = adata_new.obsm['X_pca']
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]

    seed = int(args["--seed"])
    epochs = int(args["--epochs"])
    batch_size = int(args["--batch-size"])
    reduction = args["--reduction"]
    subsample = int(args["--subsample"])

    print("Load data")
    adata = ad.read_h5ad(file)

    print("Start integration")
    adata = run_respan_pipeline(
        adata,
        batch_keys,
        cell_type_key,
        seed,
        epochs,
        batch_size,
        reduction,
        subsample,
    )

    print("Add UMAP")
    add_umap(adata)

    print("Save output")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
