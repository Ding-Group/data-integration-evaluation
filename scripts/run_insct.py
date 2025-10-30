#!/usr/bin/env python

"""
Run scVI on a dataset

Usage:
    run_scvi.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help                   Show this screen.
    -o <path>                   Path to output directory.
    -b <batch>                  One or more batch keys.
    -c <key>                    Cell type label key.
    --no-use-hvg                Do not use HVG selection. [Default: False]
    --hvg-method=<method>       HVG selection method. Default uses the pre-computed HVGs.
    --embedding-dims=<int>      Embedding dims to use. [Default: 10]
    --epochs=<int>              Number of epochs to train. [Default: 1000]
    --k=<int>                   k to use. [Default: 150]
    --batch-size=<int>          Training batch size. [Default: 64]
    --seed=<int>                Random seed. [Default: 0]
"""

import random
from os.path import join

import anndata as ad
import numpy as np
import tensorflow as tf2
from docopt import docopt
from tnn import TNN
from utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from utils.hvg_utils import select_hvgs


def run_insct(
    adata, batch_keys, use_hvg, hvg_method, embedding_dims, epochs, k, batch_size, seed
):
    np.random.seed(seed)
    random.seed(seed)
    tf2.random.set_seed(seed)
    tf2.keras.utils.set_random_seed(seed)
    adata_copy = select_hvgs(adata, use_hvg, hvg_method, subset=True, copy=True)
    _, final_batch_key = combine_batch(adata_copy, batch_keys)

    model = TNN(
        embedding_dims=embedding_dims, epochs=epochs, k=k, batch_size=batch_size
    )
    model.fit(X=adata_copy, batch_name=final_batch_key)
    embs = model.transform(X=adata_copy)
    adata.obsm[EMB_KEY] = embs

    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    embedding_dims = int(args["--embedding-dims"])
    epochs = int(args["--epochs"])
    k = int(args["--k"])
    batch_size = int(args["--batch-size"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_insct(
        adata,
        batch_keys,
        use_hvg,
        hvg_method,
        embedding_dims,
        epochs,
        k,
        batch_size,
        seed,
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
