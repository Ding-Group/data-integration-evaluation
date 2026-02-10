#!/usr/bin/env python

"""
Run scMoMaT on a dataset

Usage:
    run_scmomat.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --K=<int>              Embedding dimension. [Default: 30]
    --lamb=<float>         Regularization. [Default: 0.001]
    --T=<int>              Number of iterations. [Default: 1000]
    --batch_size=<float>   Batch size. [Default: 0.1]
    --seed=<int>           Random seed. [Default: 0]
"""
import os
import sys
from os.path import join

import scmomat

from docopt import docopt
import numpy as np
import anndata as ad
import scanpy as sc
import torch
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata


def run_scmomat(rna, atac, batch_keys, K, lamb, T, batch_size, seed):
    rna_copy = rna[:, rna.var.highly_variable].copy()
    atac = atac[:, atac.var.highly_variable].copy()

    _, final_batch_key = combine_batch(rna_copy, batch_keys)
    combine_batch(atac, batch_keys)

    counts_rna = []
    counts_atac = []

    for rep in rna_copy.obs[final_batch_key].unique():
        rna_sub = rna_copy[rna_copy.obs[final_batch_key] == rep]
        c_rna = rna_sub.layers['counts'].toarray()
        c_rna = scmomat.preprocess(c_rna, modality = "RNA", log = False)
        counts_rna.append(c_rna)
        
        atac_sub = atac[atac.obs[final_batch_key] == rep]
        c_atac = atac_sub.layers['counts'].toarray()
        c_atac = scmomat.preprocess(c_atac, modality = "ATAC")
        counts_atac.append(c_atac)

    feats_name = {"rna": rna_copy.var_names.to_numpy(), "atac": atac.var_names.to_numpy()}
    n_batches = rna_copy.obs[final_batch_key].nunique()
    counts = {"feats_name": feats_name, "nbatches": n_batches, "rna":counts_rna, "atac": counts_atac}

    lr = 1e-2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = scmomat.scmomat_model(counts = counts, K = K, batch_size = batch_size, interval = 1000, lr = lr, lamb = lamb, seed = seed, device = device)
    # losses = model.train_func(T = T)

    zs = model.extract_cell_factors()
    rnas = []
    for i, rep in enumerate(rna.obs[final_batch_key].unique()):
        rna_sub = rna[rna.obs[final_batch_key] == rep].copy()
        rna_sub.obsm[EMB_KEY] = zs[i]
        rnas.append(rna_sub)
    rna_copy = ad.concat(rnas)
    rna_copy = rna_copy[rna.obs.index].copy()

    return rna_copy


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    K = int(args['--K'])
    lamb = float(args['--lamb'])
    T = int(args['--T'])
    batch_size = float(args['--batch_size'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_scmomat(rna_adata, atac_adata, batch_keys, K, lamb, T, batch_size, seed)
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
