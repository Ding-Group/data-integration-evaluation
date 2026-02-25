#!/usr/bin/env python

"""
Run moETM on a dataset

Usage:
    run_moetm.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --n_epochs=<int>       Number of epochs. [Default: 200]
    --n_topics=<int>       Numbre of topics. [Default: 20]
    --n_latent=<int>       Number of latent dimensions. [Default: 400]
    --seed=<int>           Random seed. [Default: 0]
"""
import os
import sys
from os.path import join

sys.path.insert(0, "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/moETM/")
from dataloader import data_process_moETM
from moETM.build_model import build_moETM
from moETM.train import Train_moETM, Trainer_moETM

sys.path.pop(0)

from docopt import docopt
import numpy as np
import scanpy as sc
import torch
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata


def run_moetm(rna, atac, batch_keys, cell_type_key, n_epochs, n_topics, n_latent, seed):
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    rna = rna[:, rna.var.highly_variable].copy()
    atac = atac[:, atac.var.highly_variable].copy()

    _, final_batch_key = combine_batch(rna, batch_keys)

    batch_index = np.array(rna.obs[final_batch_key].values)
    unique_batch = list(np.unique(batch_index))
    batch_index = np.array([unique_batch.index(xs) for xs in batch_index])

    obs = rna.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)

    rna.X = rna.layers['counts'].copy()
    atac.X = atac.layers['counts'].copy()

    X_mod1_train_T, X_mod2_train_T, batch_index_train_T, train_adata_mod1 = data_process_moETM(rna, atac)
    num_batch = len(batch_index_train_T.unique())
    input_dim_mod1 = X_mod1_train_T.shape[1]
    input_dim_mod2 = X_mod2_train_T.shape[1]
    train_num = X_mod1_train_T.shape[0]

    X_mod1_train_T, X_mod2_train_T, batch_index_train_T, train_adata_mod1 = data_process_moETM(rna, atac)
    num_topic = n_topics
    emd_dim = n_latent
    encoder_mod1, encoder_mod2, decoder, optimizer = build_moETM(input_dim_mod1, input_dim_mod2, num_batch, num_topic=num_topic, emd_dim=emd_dim)

    encoder_mod1 = encoder_mod1.to('cuda')
    encoder_mod2 = encoder_mod2.to('cuda')
    decoder = decoder.to('cuda')

    trainer = Trainer_moETM(encoder_mod1, encoder_mod2, decoder, optimizer)
    Eval_kwargs = {}
    Eval_kwargs['batch_col'] = 'batch_indices'
    Eval_kwargs['plot_fname'] = 'moETM_delta'
    Eval_kwargs['cell_type_col'] = cell_type_key
    Eval_kwargs['clustering_method'] = 'louvain'
    Eval_kwargs['resolutions'] = [1.0]
    Eval_kwargs['plot_dir'] = os.environ['TMPDIR']

    Total_epoch = n_epochs
    batch_size = 2000
    Train_set = [X_mod1_train_T, X_mod2_train_T, batch_index_train_T]
    Test_set = [X_mod1_train_T, X_mod2_train_T, batch_index_train_T, train_adata_mod1]
    Train_moETM(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set, Eval_kwargs)

    trainer.encoder_mod1.to('cpu')
    trainer.encoder_mod2.to('cpu')

    res = trainer.get_embed(X_mod1_train_T, X_mod2_train_T)
    rna.obsm[EMB_KEY] = res['delta']        
    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    epochs = int(args['--n_epochs'])
    n_topics = int(args['--n_topics'])
    n_latent = int(args['--n_latent'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_moetm(rna_adata, atac_adata, batch_keys, cell_type_key, epochs, n_topics, n_latent, seed)
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
