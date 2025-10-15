#!/usr/bin/env python

"""
Make a "one big blob" baseline embedding

Usage:
    run_blob.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --dim=<int>            Embedding dimension [default: 10].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
import numpy as np
import anndata as ad
from docopt import docopt

from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def run_blob(adata, dim, seed):
    rng = np.random.default_rng(seed)
    X_blob = rng.normal(0.0, 1.0, size=(adata.n_obs, int(dim))).astype("float32")
    adata.obsm[EMB_KEY] = X_blob
    add_umap(adata)
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    dim = int(args["--dim"])
    seed = int(args["--seed"])

    set_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Building random embedding")
    adata = run_blob(adata, dim=dim, seed=seed)

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
