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
    --hvg-num=<int>        Number of highly variable genes to use. [Default: 4000]
    --search-cos           Search for optimal lambdacos. [Default: True]
    --lambda-cos=<float>   Lambdacos. [Default: 20.0]
    --training-steps=<int> Number of training steps. [Default: 2000]
    --seed=<int>           Random seed. [Default: 0]
"""

import sys
from os.path import join


import random

import anndata as ad
import numpy as np
import torch
from docopt import docopt

from utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata, split_by_batch
from utils.hvg_utils import select_hvgs

sys.path.insert(0, "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/Portal/portal/")

from util import integrate_datasets as portal_integrate
from util import preprocess_datasets as portal_preprocess



def run_portal(adata, batch_keys, hvg_num, search_cos, lambda_cos, training_steps):
    adata_copy = adata.copy()
    adata_copy.X = adata_copy.layers['counts'].copy()

    _, batch_key_final = combine_batch(adata_copy, batch_keys)
    adatas = split_by_batch(adata_copy, batch_key_final)
    lowdim_list = portal_preprocess(adatas)
    outs = portal_integrate(lowdim_list)
    new_adata = ad.AnnData(X = outs, obs= ad.concat(adatas).obs)
    new_adata = new_adata[adata.obs_names]
    adata.obsm[EMB_KEY] = new_adata.X
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    hvg_num = args["--hvg-num"]
    search_cos = args['--search-cos']
    lambda_cos = args['--lambda-cos']
    training_steps = args['--training-steps']
    seed = int(args["--seed"])

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_portal(adata, batch_keys, hvg_num, search_cos, lambda_cos, training_steps)
    print("Adding UMAP")
    add_umap(adata)
    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
