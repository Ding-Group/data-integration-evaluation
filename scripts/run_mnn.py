#!/usr/bin/env python

"""
Run MNN (Mutual Nearest Neighbors) correction on a dataset

Usage:
    run_mnn.py --out-dir=<path> --batch-key=<key>... --cell-type-key=<key> [options] <file>

Options:
    -h --help              Show this screen.
    --out-dir=<path>       Path to output directory.
    --batch-key=<key>...   One or more batch keys.
    --cell-type-key=<key>  Cell type label key.
    --no-use-hvg           Do not use HVG selection.
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --n-pcs=<int>          Number of PCs after MNN [default: 50].
    --svd-solver=<name>    PCA SVD solver [default: arpack].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
from typing import Any, Tuple, List

import numpy as np
import anndata as ad
import scanpy as sc
import scanpy.external as sce
from docopt import docopt

from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def _ensure_combined_batch(adata: ad.AnnData, batch_keys: List[str]) -> str:
    """Combine multiple batch columns into a single categorical column if needed."""
    if isinstance(batch_keys, (list, tuple)) and len(batch_keys) > 1:
        combo = "_batch_combined"
        adata.obs[combo] = (
            adata.obs[batch_keys].astype(str).agg("|".join, axis=1).astype("category")
        )
        return combo
    return batch_keys[0]


def _extract_corrected_adata(mnn_result: Any) -> ad.AnnData:
    """
    scanpy.external.pp.mnn_correct returns different shapes across versions:
      - AnnData
      - (AnnData,)
      - ([AnnData], _)
      - ((AnnData, ...), _)
    This makes it robust.
    """
    if isinstance(mnn_result, ad.AnnData):
        return mnn_result
    if isinstance(mnn_result, (list, tuple)) and len(mnn_result) > 0:
        first = mnn_result[0]
        if isinstance(first, ad.AnnData):
            return first
        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], ad.AnnData):
            return first[0]
    raise RuntimeError("Unexpected return type from sce.pp.mnn_correct; cannot extract corrected AnnData.")


def run_mnn(
    adata: ad.AnnData,
    batch_keys: List[str],
    use_hvg: bool,
    hvg_method: str,
    n_pcs: int,
    svd_solver: str,
    seed: int,
) -> ad.AnnData:
    if use_hvg:
        if hvg_method is not None:
            layer = "counts" if ("v3" in hvg_method) else None
            sc.pp.highly_variable_genes(adata, flavor=hvg_method, layer=layer, n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable]

  
    batch_key_final = _ensure_combined_batch(adata, batch_keys)

    # Run MNN correction
    mnn_out = sce.pp.mnn_correct(adata, batch_key=batch_key_final)
    adata_corr = _extract_corrected_adata(mnn_out)

    # As in the notebook: scale + PCA on corrected data
    sc.pp.scale(adata_corr, max_value=10)
    sc.tl.pca(adata_corr, n_comps=int(n_pcs), svd_solver=svd_solver, random_state=seed)

    # Store PCA under unified embedding key
    if "X_pca" in adata_corr.obsm:
        adata_corr.obsm[EMB_KEY] = adata_corr.obsm["X_pca"]
    else:
        adata_corr.obsm[EMB_KEY] = np.zeros((adata_corr.n_obs, 2), dtype=np.float32)

    return adata_corr


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["--out-dir"]
    batch_keys = args["--batch-key"]
    cell_type_key = args["--cell-type-key"]
    seed = int(args["--seed"])
    use_hvg = not args["--use-hvg"]
    hvg_method = args["--hvg-method"]
    n_pcs = int(args["--n-pcs"])
    svd_solver = args["--svd-solver"]

    set_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running MNN")
    adata = run_mnn(
        adata=adata,
        batch_keys=batch_keys,
        use_hvg=use_hvg,
        hvg_method=hvg_method,
        n_pcs=n_pcs,
        svd_solver=svd_solver,
        seed=seed,
    )

    print("Adding UMAP")
    add_umap(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
