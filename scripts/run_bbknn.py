#!/usr/bin/env python

"""
Run BBKNN on a dataset

Usage:
    run_bbknn.py --out-dir=<path> --batch-key=<key>... --cell-type-key=<key> [options] <file>

Options:
    -h --help                        Show this screen.
    --out-dir=<path>                 Path to output directory.
    --batch-key=<key>...             One or more batch keys.
    --cell-type-key=<key>            Cell type label key.
    --no-use-hvg                     Do not use HVG selection.
    --hvg-method=<method>            HVG selection method. Default uses the pre-computed HVGs.
    --n-pcs=<int>                    Number of PCs for PCA [default: 50].
    --neighbors-within-batch=<int>   k within each batch for BBKNN [default: 3].
    --trim=<int>                     Trim parameter for BBKNN (optional).
    --metric=<str>                   Distance metric in PCA space [default: euclidean].
    --seed=<int>                     Random seed [default: 0].
"""
from os.path import join

import numpy as np
import anndata as ad
import scanpy as sc
from docopt import docopt

from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def _ensure_combined_batch(adata, batch_keys):
    """If more than one batch key is given, combine them into one column."""
    if isinstance(batch_keys, (list, tuple)) and len(batch_keys) > 1:
        combo = "_batch_combined"
        adata.obs[combo] = (
            adata.obs[batch_keys].astype(str).agg("|".join, axis=1).astype("category")
        )
        return combo
    return batch_keys[0]


def run_bbknn(adata, batch_keys, use_hvg, hvg_method, n_pcs, k_within, trim, metric, seed):
    if use_hvg:
        if hvg_method is not None:
            layer = "counts" if ("v3" in hvg_method) else None
            sc.pp.highly_variable_genes(
                adata, flavor=hvg_method, layer=layer, n_top_genes=2000
            )
        adata = adata[:, adata.var.highly_variable]

    sc.tl.pca(adata, n_comps=int(n_pcs), random_state=seed)

    batch_key_final = _ensure_combined_batch(adata, batch_keys)

    bbknn_kwargs = dict(
        batch_key=batch_key_final,
        neighbors_within_batch=int(k_within),
        metric=metric,
    )
    if trim is not None:
        bbknn_kwargs["trim"] = int(trim)
    sc.external.pp.bbknn(adata, **bbknn_kwargs)

    if "X_pca" in adata.obsm:
        adata.obsm[EMB_KEY] = adata.obsm["X_pca"]
    else:
        adata.obsm[EMB_KEY] = np.zeros((adata.n_obs, 2), dtype=np.float32)

    return adata


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
    k_within = int(args["--neighbors-within-batch"])
    trim = args["--trim"]
    metric = args["--metric"]

    set_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running BBKNN")
    adata = run_bbknn(
        adata,
        batch_keys=batch_keys,
        use_hvg=use_hvg,
        hvg_method=hvg_method,
        n_pcs=n_pcs,
        k_within=k_within,
        trim=trim,
        metric=metric,
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
