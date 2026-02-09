#!/usr/bin/env python

"""
Run SAUCIE on a single-cell dataset

Usage:
    run_saucie.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help                      Show this screen.
    -o <path>                      Path to output directory.
    -b <batch>                     One or more batch keys.
    -c <key>                       Cell type label key.
    --no-use-hvg                   Do not use HVG selection. [Default: False]
    --hvg-method=<method>          HVG selection method. Default uses the pre-computed HVGs.
    --lambda-b=<float>             Lambda B. [default: 0.1]
    --steps=<int>                  Number of steps. [default: 2000]
    --seed=<int>                   Random seed. [default: 0]
"""

import os
import sys

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from utils.hvg_utils import select_hvgs

sys.path.append(
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/",
)

from SAUCIE.loader import Loader
from SAUCIE.model import SAUCIE


# Helper
def preprocess_features(X, n_comp: int = 100, random_state: int = 0) -> np.ndarray:
    """
    Reduce dimensionality and normalize features using TruncatedSVD + StandardScaler.
    Matches preprocessing done in your notebook.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    X_csr = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    n_comp = int(max(2, min(n_comp, min(X_csr.shape) - 1)))
    Z = TruncatedSVD(n_components=n_comp, random_state=random_state).fit_transform(
        X_csr
    )
    Z = StandardScaler().fit_transform(Z)
    return Z.astype(np.float32, copy=False)


def run_saucie(
    adata: ad.AnnData,
    batch_keys,
    use_hvg,
    hvg_method,
    lambda_b: float = 0.1,
    steps: int = 2000,
    seed: int = 0,
) -> ad.AnnData:
    """
    Run SAUCIE embedding on the input AnnData object and store 2D embedding in adata.obsm['X_saucie'].
    """

    # Enable TensorFlow v1 compatibility mode
    import tensorflow.compat.v1 as tf

    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)
    sc.pp.scale(adata_copy)
    labels, _ = combine_batch(adata_copy, batch_keys)
    labels = labels.cat.codes

    # Prepare data loaders
    loader_train = Loader(adata_copy.X, labels=labels, shuffle=True)
    loader_eval = Loader(adata_copy.X, labels=labels, shuffle=False)

    # Train SAUCIE and extract embeddings
    model = SAUCIE(adata_copy.X.shape[1], lambda_b=lambda_b)
    _ = model.train(loader_train, steps=int(steps))
    embedding = model.get_embedding(loader_eval)

    adata.obsm[EMB_KEY] = embedding[0]
    if np.isnan(np.sum(embedding[0])):
        raise ValueError("nan in embedding")
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    lambda_b = float(args["--lambda-b"])
    steps = int(args["--steps"])
    seed = int(args["--seed"])

    print("Loading data...")
    adata = ad.read_h5ad(file)

    print("Running SAUCIE...")
    adata = run_saucie(adata, batch_keys, use_hvg, hvg_method, lambda_b, steps, seed)

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

    output.write(os.path.join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
