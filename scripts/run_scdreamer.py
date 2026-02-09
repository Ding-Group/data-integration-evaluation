#!/usr/bin/env python

"""
Run scDREAMER on a dataset

Usage:
    run_scdreamer.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --epochs=<int>         Number of epochs. [Default: 250]
    --seed=<int>           Random seed. [Default: 0]
"""

import os
import random
from os.path import join

import anndata as ad
import numpy as np
import pandas as pd
import scDREAMER.scDREAMER as model
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata

tf.disable_v2_behavior()


def run_scdreamer(adata, batch_keys, cell_type_key, epochs, seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
    tf2.random.set_seed(seed)
    tf2.keras.utils.set_random_seed(seed)

    _, batch_key_final = combine_batch(adata, batch_keys)
    adata.X = adata.layers["counts"]
    adata.write(os.path.join(os.environ["TMPDIR"], "tmp.h5ad"))
    run_config = tf.ConfigProto()

    with tf.Session(config=run_config) as sess:
        dreamer = model(
            sess,
            epoch=epochs,
            dataset_name=os.path.join(os.environ["TMPDIR"], "tmp.h5ad"),
            batch=batch_key_final,
            cell_type=cell_type_key,
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
    epochs = int(args["--epochs"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_scdreamer(adata, batch_keys, cell_type_key, epochs, seed)
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
