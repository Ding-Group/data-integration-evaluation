"""
Run MIRA on a dataset

Usage:
    run_mira.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --n_topics=<int>       Number of topics. [default: 10]
    --seed=<int>           Random seed. [default: 0]
"""

import os
import sys

sys.path.insert(
    0, "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/MIRA"
)
from os.path import join

import anndata as ad
import mira
import numpy as np
import scanpy as sc
import torch
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata


def run_mira(rna, atac, batch_keys, n_topics, seed):
    rna_copy = rna[:, rna.var.highly_variable].copy()
    atac = atac[:, atac.var.highly_variable].copy()

    _, final_batch_key = combine_batch(rna_copy, batch_keys)
    combine_batch(atac, batch_keys)

    model = mira.topics.make_model(
        *atac.shape,
        feature_type = 'accessibility',
        counts_layer='counts',
        categorical_covariates=final_batch_key,
        seed=seed,
    )
    model.set_learning_rates(1e-3, 0.1)
    model.set_params(num_topics = n_topics).fit(atac)
    model.predict(atac)

    model = mira.topics.make_model(
        rna_copy.n_obs, rna_copy.n_vars,
        feature_type = 'expression',
        highly_variable_key='highly_variable',
        counts_layer='counts',
        categorical_covariates=final_batch_key,
        seed=seed
    )
    model.set_learning_rates(1e-3, 0.1)
    model = model.set_params(num_topics = n_topics).fit(rna_copy)
    model.predict(rna_copy)

    rna_copy, atac = mira.utils.make_joint_representation(rna_copy, atac)
    rna.obsm[EMB_KEY] = rna_copy[rna.obs_names].obsm['X_joint_umap_features']
    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    n_topics = int(args['--n_topics'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_mira(rna_adata, atac_adata, batch_keys, n_topics, seed)
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
