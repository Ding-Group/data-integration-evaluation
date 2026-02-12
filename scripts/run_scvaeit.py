#!/usr/bin/env python

"""
Run scVAEIT on a dataset

Usage:
    run_scvaeit.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --n_latent=<int>       Latent dimension. [Default: 32]
    --seed=<int>           Random seed. [Default: 0]
"""

import os
import sys
from os.path import join

from docopt import docopt
import numpy as np
import anndata as ad
import scanpy as sc
import tensorflow as tf
import random
import scVAEIT
from scVAEIT.VAEIT import VAEIT
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata


def run_scvaeit(rna, atac, batch_keys, n_latent, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    chr_list = [f'chr{i}' for i in range(1, 20 if rna.uns["species"] == "mouse" else 23)] + [
        "chrX"
    ]
    if "chrY" in atac.var["chr"]:
        chr_list.append("chrY")

    rna_copy = rna[:, rna.var.highly_variable].copy()
    atac = atac[:, atac.var.chr.isin(chr_list)]
    atac = atac[:, atac.var.highly_variable].copy()

    _, final_batch_key = combine_batch(rna_copy, batch_keys)
    combine_batch(atac, batch_keys)

    atac.var_names = atac.var_names.str.replace(":", "-")

    peak_names = atac.var_names.to_numpy().astype(str)
    gene_names = rna_copy.var_names.to_numpy().astype(str)

    reorder = []
    for b in rna_copy.obs[final_batch_key].unique():
        reorder.extend(
            rna_copy[rna_copy.obs[final_batch_key] == b].obs_names.to_numpy()
        )
    rna_copy = rna_copy[reorder]
    atac = atac[reorder]

    batches = rna_copy.obs[final_batch_key].cat.codes.to_numpy()
    id_datasets = batches

    X = rna_copy.layers["counts"].copy()
    Y = atac.layers["counts"].copy()
    # Count peaks in each chromosome (assuming they are ordered)

    X = X.toarray()
    X = np.log1p(X / np.sum(X, axis=1, keepdims=True) * 1e4)

    Y[Y > 0.0] = 1.0
    Y = Y.toarray()
    data = np.c_[X, Y]

    chunk_atac = np.array(
        [
            np.sum(np.char.startswith(peak_names, f"chr{i}-"))
            for i in list(range(1, 20)) + ["X", "Y"]
        ],
        dtype=np.int32,
    )
    dim_input_arr = np.array([len(gene_names), len(peak_names)])

    assert dim_input_arr[1] == np.sum(chunk_atac)
    batches = batches[:, None]
    masks = np.zeros((len(np.unique(batches)), np.sum(dim_input_arr)), dtype=np.float32)
    sample_sizes = np.array([np.sum(batches == b) for b in np.unique(batches)], dtype=np.int32)

    config = {
    # Dimension of input features for [RNA, ADT, peaks]
        'dim_input_arr': dim_input_arr,

        # Blocks for [RNA, ADT, peaks at chr1, ... peaks at chr22]
        'dim_block': np.append([len(gene_names)], chunk_atac), # input dimension of blocks
        'dist_block':['NB'] + ['Bernoulli' for _ in chunk_atac], # distributions of blocks
        'dim_block_enc':np.array([256] + [16 for _ in chunk_atac]), # dimension of first layer of the encoder
        'dim_block_dec':np.array([256] + [16 for _ in chunk_atac]), # dimension of first layer of the decoder
        'block_names':np.array(['rna'] + ['atac' for _ in range(len(chunk_atac))]), # names of blocks
        'uni_block_names':np.array(['rna','atac']), # names for modalities
        'dim_block_embed':np.array([16] + [1 for _ in range(len(chunk_atac))])*2, # mask embedding dimension

        # Internal network structure
        'dimensions':[256], # dimension of latent layers of encoder; the reversed is used for decoder
        'dim_latent':n_latent, # the latent dimension between the encoder and decoder
        
        # Weights
        'beta_unobs':2./3., # weight for masked out observation; weight for observed values will be 1-beta_unobs.
        'beta_modal':np.array([1/3,1/3,1/3]), # weights for 3 modalities

        # Masking probability
        "p_feat" : 0.2, # probability of randomly masking out an entry
        "p_modal" : np.ones(3)/3, # probability of masking out the 3 modalities
        
        'gamma':lambda epoch: 2 * 0.8 ** (epoch / 50) # MMD loss to correct strong batch effects of multiple datasets
    }

    batches_cate = batches
    batches_cont = None


    model = VAEIT(config, data, 
                    masks, id_datasets, batches)
    hist = model.train(
        num_epoch=300, batch_size=512, save_every_epoch=150,
        verbose=True, checkpoint_dir=join(os.environ['TMPDIR'], ''.join(random.choices(['a', 'b', 'c', 'd', 'e'], k=20)))
    )
    model.update_z()
    rna.obsm[EMB_KEY] = model.adata.X
    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    n_latent = int(args['--n_latent'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_scvaeit(rna_adata, atac_adata, batch_keys, n_latent, seed)
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
