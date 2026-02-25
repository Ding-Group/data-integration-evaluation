#!/usr/bin/env python

"""
Run scMVP on a dataset

Usage:
    run_scmvp.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --latent_dim=<int>     Latent dimension. [Default: 20]
    --epochs=<int>         Number of epochs. [Default: 10]
    --seed=<int>           Random seed. [Default: 0]
"""

import os
import random
from os.path import join

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata
from scMVP.dataset import LoadData
from scMVP.inference import MultiPosterior, MultiTrainer
from scMVP.models import Multi_VAE_Attention


def run_scmvp(rna, atac, cell_type_key, latent_dim, epochs, seed):
    rna_copy = rna[:, rna.var.highly_variable].copy()
    atac = atac[:, atac.var.highly_variable].copy()

    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    input_path = os.environ['TMPDIR'] + '/'

    scipy.io.mmwrite(join(input_path, 'rna_normalize_count.mtx'), rna_copy.X.T)
    with open(join(input_path, 'rna_barcodes.txt'), 'w') as f:
        for b in rna_copy.obs_names:
            f.write(f'{b}\n')
    with open(join(input_path, 'rna_genes.txt'), 'w') as f:
        for b in rna_copy.var_names:
            f.write(f'{b}\n')

    scipy.io.mmwrite(join(input_path, 'atac_normalize_count.mtx'), atac.X.T)
    with open(join(input_path, 'atac_barcodes.txt'), 'w') as f:
        for b in atac.obs_names:
            f.write(f'{b}\n')
    with open(join(input_path, 'atac_peaks.txt'), 'w') as f:
        for b in atac.var_names:
            f.write(f'{b}\n')

    dataset = {
        "gene_names": 'rna_genes.txt',
        "gene_expression": 'rna_normalize_count.mtx',
        "gene_barcodes": 'rna_barcodes.txt',
        "atac_names": 'atac_peaks.txt',
        "atac_expression": 'atac_normalize_count.mtx',
        "atac_barcodes": 'atac_barcodes.txt'
    }

    lr = 1e-3
    use_cuda = True # False if using CPU
    n_centroids = rna_copy.obs[cell_type_key].nunique()
    n_alfa = 1.0
    del atac, rna_copy

    dataset = LoadData(dataset=dataset,data_path=input_path,
                       dense=False,gzipped=False, atac_threshold=0.001,
                       cell_threshold=1)


    multi_vae = Multi_VAE_Attention(dataset.nb_genes, len(dataset.atac_names), n_batch=0, n_latent=latent_dim, n_centroids=n_centroids, n_alfa = n_alfa, mode="mm-vae") # should provide ATAC num, alfa, mode and loss type
    trainer = MultiTrainer(
        multi_vae,
        dataset,
        train_size=0.9,
        use_cuda=use_cuda,
        frequency=5,
        seed=0
    )
    trainer.train(n_epochs=epochs, lr=lr)
    full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)),type_class=MultiPosterior)

    size = 64 if rna.n_obs % 64 != 1 else 65
    latent, latent_rna, latent_atac, cluster_gamma, cluster_index, batch_indices, labels = full.sequential(batch_size=size).get_latent()

    latent = pd.DataFrame(latent, index=dataset.barcodes)
    latent = latent.loc[rna.obs.index]
    rna.obsm[EMB_KEY] = latent.to_numpy()
    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    latent_dim = int(args['--latent_dim'])
    epochs = int(args['--epochs'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_scmvp(rna_adata, atac_adata, cell_type_key, latent_dim, epochs, seed)
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
