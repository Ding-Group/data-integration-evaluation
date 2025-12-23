#!/usr/bin/env python

"""
Run scVI on a dataset

Usage:
    run_scvi.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --n-ghosts=<int>       Number of ghost cells to use. [Default: 65]
    --dim=<int>            Dimensions of representation. [Default: 60]
    --seed=<int>           Random seed. [Default: 0]
"""

from os.path import join

import anndata as ad
import numpy as np
import OCAT
import pandas as pd
from docopt import docopt
from utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from utils.hvg_utils import select_hvgs


def run_ocat(adata, batch_keys, use_hvg, hvg_method, n_ghosts, dim, seed):
    np.random.seed(seed)
    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)
    _, batch_key_final = combine_batch(adata_copy, batch_keys)

    ref_list = [
        adata_copy[adata_copy.obs[batch_key_final] == i,]
        for i in np.unique(adata_copy.obs[batch_key_final])
    ]
    data_list = [i.layers["counts"] for i in ref_list]
    m_list = [n_ghosts] * len(data_list)
    ZW = OCAT.run_OCAT(data_list, dim=dim, m_list=m_list, random_seed=seed)

    query_latent = ad.AnnData(ZW)
    query_latent.obs = pd.concat([a.obs for a in ref_list])
    query_latent = query_latent[adata.obs_names]

    adata.obsm[EMB_KEY] = query_latent.X
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    n_ghosts = int(args["--n-ghosts"])
    dim = int(args["--dim"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_ocat(adata, batch_keys, use_hvg, hvg_method, n_ghosts, dim, seed)
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
