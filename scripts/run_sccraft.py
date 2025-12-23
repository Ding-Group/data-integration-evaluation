#!/usr/bin/env python

"""
Run scCRAFT on a dataset

Usage:
    run_sccraft.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help                   Show this screen.
    -o <path>                   Path to output directory.
    -b <batch>                  One or more batch keys.
    -c <key>                    Cell type label key.
    --no-use-hvg                Do not use HVG selection. [Default: False]
    --hvg-method=<method>       HVG selection method. Default uses the pre-computed HVGs.
    --resolution1=<float>       Clustering resolution 1. [Default: 0.5]
    --resolution2=<float>       Clustering resolution 2. [Default: 7]
    --method=<str>              Clustering method. [Default: Leiden]
    --epochs=<int>              Number of epochs. [Default: 150]
    --d-coef=<float>            d_coef. [Default: 0.2]
    --kl-coef=<float>           kl_coef. [Default: 0.005]
    --seed=<int>                Random seed. [Default: 0]
"""

from os.path import join

import anndata as ad
import scCRAFT
from docopt import docopt
from scCRAFT.model import *
from utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from utils.hvg_utils import select_hvgs


def run_sccraft(
    adata,
    batch_keys,
    use_hvg,
    hvg_method,
    resolution1,
    resolution2,
    method,
    epochs,
    d_coef,
    kl_coef,
    seed,
):
    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)
    _, batch_key_final = combine_batch(adata_copy, batch_keys)

    # Their seed is pre-determined by themselves for training
    multi_resolution_cluster(
        adata_copy, resolution1=resolution1, resolution2=resolution2, method=method
    )
    VAE = train_integration_model(
        adata_copy, batch_key=batch_key_final, epochs=epochs, d_coef=d_coef, kl_coef=kl_coef
    )
    obtain_embeddings(adata_copy, VAE, seed=seed)

    adata_copy = adata_copy[adata.obs_names]

    adata.obsm[EMB_KEY] = adata_copy.obsm["X_scCRAFT"]
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    resolution1 = float(args["--resolution1"])
    resolution2 = float(args["--resolution2"])
    method = args["--method"]
    epochs = int(args["--epochs"])
    d_coef = float(args["--d-coef"])
    kl_coef = float(args["--kl-coef"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_sccraft(
        adata,
        batch_keys,
        use_hvg,
        hvg_method,
        resolution1,
        resolution2,
        method,
        epochs,
        d_coef,
        kl_coef,
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
