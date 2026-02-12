#!/usr/bin/env python

"""
Run UnitedNet on a dataset

Usage:
    run_unitednet.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --epochs=<int>         Number of epochs. [Default: 10]
    --seed=<int>           Random seed. [Default: 0]
"""

import os
import random
import sys
from os.path import join

sys.path.insert(0, '/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/UnitedNet/')

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch
from sklearn.preprocessing import binarize
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata
from src.interface import UnitedNet


def run_unitednet(rna, atac, cell_type_key, epochs, seed):
    rna_copy = rna[:, rna.var.highly_variable].copy()
    atac = atac[:, atac.var.highly_variable].copy()

    technique = 'atacseq'
    root_save_path = f"{os.environ['TMPDIR']}/{technique}"
    device = "cuda:0"

    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    atac.X = binarize(atac.layers['counts']).toarray()
    rna_copy.X = rna_copy.layers['counts'].toarray()
    rna_copy.obs['label'] = rna_copy.obs[cell_type_key]
    atac.obs['label'] = atac.obs[cell_type_key]

    atacseq_config = {
        "train_batch_size": 512,
        "finetune_batch_size": 5000,
        "transfer_batch_size": 512,
        "train_epochs": epochs,
        "finetune_epochs": 10,
        "transfer_epochs": 20,
        "train_task": "supervised_group_identification",
        "finetune_task": "cross_model_prediction_clas",
        "transfer_task": "supervised_group_identification",
        "train_loss_weight": None,
        "finetune_loss_weight": None,
        "transfer_loss_weight": None,
        "lr": 0.01,
        "checkpoint": 1,
        "n_head": 1,
        "noise_level":[0,0],
        "fuser_type":"WeightedMean",
        "encoders": [
            {
                "input": atac.n_vars,
                "hiddens": [64, 64],
                "output": 64,
                "use_biases": [True, True, True],
                "dropouts": [0, 0, 0],
                "activations": ["relu", "relu", "relu"],
                "use_batch_norms": [True, True, True],
                "use_layer_norms": [False, False, False],
                "is_binary_input": True,
            },
            {
                "input": rna_copy.n_vars,
                "hiddens": [64, 64],
                "output": 64,
                "use_biases": [True, True, True],
                "dropouts": [0, 0, 0],
                "activations": ["relu", "relu", "relu"],
                "use_batch_norms": [True, True, True],
                "use_layer_norms": [False, False, False],
                "is_binary_input": False,
            },
        ],
        "latent_projector": None,
        "decoders": [
            {
                "input": 64,
                "hiddens": [64, 64],
                "output": atac.n_vars,
                "use_biases": [True, True, True],
                "dropouts": [0, 0, 0],
                "activations": ["relu", "relu", "sigmoid"],
                "use_batch_norms": [False, False, False],
                "use_layer_norms": [False, False, False],
            },
            {
                "input": 64,
                "hiddens": [64, 64],
                "output": rna_copy.n_vars,
                "use_biases": [True, True, True],
                "dropouts": [0, 0, 0],
                "activations": ["relu", "relu", None],
                "use_batch_norms": [False, False, False],
                "use_layer_norms": [False, False, False],
            },
        ],
        "discriminators": [
            {
                "input": atac.n_vars,
                "hiddens": [64],
                "output": 1,
                "use_biases": [True, True],
                "dropouts": [0, 0],
                "activations": ["relu", "sigmoid"],
                "use_batch_norms": [False, False],
                "use_layer_norms": [False, True],
            },
            {
                "input": rna_copy.n_vars,
                "hiddens": [64],
                "output": 1,
                "use_biases": [True, True],
                "dropouts": [0, 0],
                "activations": ["relu", "sigmoid"],
                "use_batch_norms": [False, False],
                "use_layer_norms": [False, True],
            },
        ],
        "projectors": {
            "input": 64,
            "hiddens": [],
            "output": 100,
            "use_biases": [True],
            "dropouts": [0],
            "activations": ["relu"],
            "use_batch_norms": [False],
            "use_layer_norms": [True],
        },
        "clusters": {
            "input": 100,
            "hiddens": [],
            "output": rna_copy.obs['label'].nunique(),
            "use_biases": [False],
            "dropouts": [0],
            "activations": [None],
            "use_batch_norms": [False],
            "use_layer_norms": [False],
        },
    }

    model = UnitedNet(f"{root_save_path}/test", device=device, technique=atacseq_config)
    model.train([atac, rna_copy], verbose=False)
    adata_all = model.infer([atac, rna_copy])
    rna.obsm[EMB_KEY] = adata_all.X

    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    epochs = int(args['--epochs'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_unitednet(rna_adata, atac_adata, cell_type_key, epochs, seed)
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
