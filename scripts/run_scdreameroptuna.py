"""
Run scDREAMER on a dataset

Usage:
    run_scdreamer.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

Options:
    -h --help               Show this screen.
    -o <path>               Path to output directory.
    -b <batch>              One or more batch keys.
    -c <key>                Cell type label key.
    --params-json=<file>    JSON file with hyperparameters
    --seed=<int>            Random seed. [Default: 0]
"""

import json
import os
import random
from os.path import join

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scDREAMER.scDREAMER as model
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, combine_batch, minimise_anndata

tf.disable_v2_behavior()


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "lr_ae": float(params["lr_ae"]),
        "lr_dis": float(params["lr_dis"]),
        "lr_bc": float(params["lr_bc"]),
        "z_dim": int(params["z_dim"]),
        "epochs": int(params["epochs"]),
    }
    return hyperparams, params.get("trial_number")


def run_scdreamer(adata, batch_keys, cell_type_key, epochs, lr_ae, lr_dis, lr_bc, z_dim, seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
    tf2.random.set_seed(seed)
    tf2.keras.utils.set_random_seed(seed)

    _, batch_key_final = combine_batch(adata, batch_keys)
    adata.X = adata.layers["counts"]
    adata.write(os.path.join(os.environ["TMPDIR"], "tmp.h5ad"))
    run_config = tf.ConfigProto()

    run_config.gpu_options.per_process_gpu_memory_fraction = 0.333 / 2
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        dreamer = model(
            sess,
            epoch=epochs,
            dataset_name=os.path.join(os.environ["TMPDIR"], "tmp.h5ad"),
            batch=batch_key_final,
            cell_type=cell_type_key,
            lr_ae=lr_ae,
            lr_bc=lr_bc,
            lr_dis=lr_dis,
            z_dim=z_dim,
            name="dreamer",
        )

        dreamer.train_cluster()

    adata.obsm[EMB_KEY] = pd.read_csv(
        f"dreamerlatent_matrix_{epochs}.csv", header=None
    ).to_numpy()

    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])
    params_json_path = args["--params-json"]

    hyperparams, trial_number = load_hyperparams(params_json_path)
    if trial_number is not None:
        print(f"Running trial {trial_number}")
    print(f"Hyperparameters: {hyperparams}")

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_scdreamer(
        adata,
        batch_keys,
        cell_type_key,
        epochs=hyperparams["epochs"],
        lr_ae=hyperparams["lr_ae"],
        lr_dis=hyperparams["lr_dis"],
        lr_bc=hyperparams["lr_bc"],
        z_dim=hyperparams["z_dim"],
        seed=seed,
    )

    sc.pp.neighbors(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY],
        uns=["neighbors"],
        obsp=["connectivities", "distances"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
