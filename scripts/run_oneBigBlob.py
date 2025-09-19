#!/usr/bin/env python

"""
Make a "one big blob" baseline embedding

Usage:
    run_blob.py --out-dir=<path> --cell-type-key=<key> --batch-key=<key>... [options] <file>

Options:
    -h --help              Show this screen.
    --out-dir=<path>       Path to output directory.
    --cell-type-key=<key>  Cell type label key (kept in output).
    --batch-key=<key>...   One or more batch keys (kept in output).
    --dim=<int>            Embedding dimension [default: 10].
    --neighbors=<int>      k for KNN when building graph [default: 15].
    --leiden-res=<float>   Leiden resolution; very small gives one cluster [default: 1e-6].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
import numpy as np
import anndata as ad
import scanpy as sc
from docopt import docopt

from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def run_blob(adata, dim, neighbors, leiden_res, seed):
    rng = np.random.default_rng(seed)
    X_blob = rng.normal(0.0, 1.0, size=(adata.n_obs, int(dim))).astype("float32")
    adata.obsm[EMB_KEY] = X_blob
    add_umap(adata, n_neighbors=int(neighbors))
    sc.tl.leiden(adata, resolution=float(leiden_res), key_added="leiden_blob", random_state=seed)
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["--out-dir"]
    cell_type_key = args["--cell-type-key"]
    batch_keys = args["--batch-key"]
    dim = int(args["--dim"])
    neighbors = int(args["--neighbors"])
    leiden_res = float(args["--leiden-res"])
    seed = int(args["--seed"])

    set_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Building random embedding")
    adata = run_blob(adata, dim=dim, neighbors=neighbors, leiden_res=leiden_res, seed=seed)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key, "leiden_blob"],
        obsm=[EMB_KEY, "X_umap"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
