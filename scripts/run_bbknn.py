#!/usr/bin/env python

"""
Run BBKNN on a dataset

Usage:
    run_bbknn.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help                        Show this screen.
    -o <path>                        Path to output directory.
    -b <batch>                       One or more batch keys.
    -c <key>                         Cell type label key.
    --no-use-hvg                     Do not use HVG selection.
    --hvg-method=<method>            HVG selection method. Default uses the pre-computed HVGs.
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
from utils.adata_utils import add_umap, minimise_anndata, combine_batch, EMB_KEY
from utils.hvg_utils import select_hvgs


def run_bbknn(adata, batch_keys, use_hvg, hvg_method, k_within, trim, metric, seed):
    adata = select_hvgs(adata, use_hvg, hvg_method)

    batch_key_final = combine_batch(adata, batch_keys)

    bbknn_kwargs = dict(
        batch_key=batch_key_final,
        neighbors_within_batch=int(k_within),
        metric=metric,
    )
    if trim is not None:
        bbknn_kwargs["trim"] = int(trim)
    sc.external.pp.bbknn(adata, **bbknn_kwargs)

    adata.obsm[EMB_KEY] = np.zeros((adata.n_obs, 2), dtype=np.float32)

    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
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
        uns=["neighbors"],
        obsp=["connectivities", "distances"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
