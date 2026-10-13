"""
Run SAUCIE on a dataset using Optuna trial hyperparameters

Usage:
    run_saucieoptuna.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

Options:
    -h --help                      Show this screen.
    -o <path>                      Path to output directory.
    -b <batch>                     One or more batch keys.
    -c <key>                       Cell type label key.
    --no-use-hvg                   Do not use HVG selection. [default: False]
    --hvg-method=<method>          HVG selection method. Default uses the pre-computed HVGs.
    --params-json=<file>           JSON file with hyperparameters
    --seed=<int>                   Random seed. [default: 0]
"""

import json
import os
import sys

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import tensorflow as tf2
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, combine_batch, minimise_anndata
from project_utils.hvg_utils import select_hvgs

sys.path.append(
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/",
)

for gpu in tf2.config.list_physical_devices('GPU'):
    tf2.config.experimental.set_memory_growth(gpu, True)

from SAUCIE.loader import Loader
from SAUCIE.model import SAUCIE


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "lambda_b": float(params["lambda_b"]),
        "learning_rate": float(params["learning_rate"]),
        "steps": int(params["steps"]),
    }
    return hyperparams, params.get("trial_number")


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
    learning_rate: float = 0.001,
) -> ad.AnnData:
    """
    Run SAUCIE embedding on the input AnnData object and store it in adata.obsm[EMB_KEY].
    """

    # Enable TensorFlow v1 compatibility mode
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)
    sc.pp.scale(adata_copy)
    labels, _ = combine_batch(adata_copy, batch_keys)
    labels = labels.cat.codes

    # Prepare data loaders
    loader_train = Loader(adata_copy.X, labels=labels, shuffle=True)
    loader_eval = Loader(adata_copy.X, labels=labels, shuffle=False)

    # Train SAUCIE and extract embeddings
    model = None
    try:
        model = SAUCIE(
            adata_copy.X.shape[1],
            lambda_b=lambda_b,
            learning_rate=float(learning_rate),
        )
        _ = model.train(loader_train, steps=int(steps))
        embedding = model.get_embedding(loader_eval)
    finally:
        if model is not None and getattr(model, "sess", None) is not None:
            model.sess.close()
        tf.reset_default_graph()

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
    seed = int(args["--seed"])
    params_json_path = args["--params-json"]

    hyperparams, trial_number = load_hyperparams(params_json_path)
    if trial_number is not None:
        print(f"Running trial {trial_number}")
    print(f"Hyperparameters: {hyperparams}")

    print("Loading data...")
    adata = ad.read_h5ad(file)

    print("Running SAUCIE...")
    adata = run_saucie(
        adata,
        batch_keys,
        use_hvg,
        hvg_method,
        lambda_b=hyperparams["lambda_b"],
        steps=hyperparams["steps"],
        seed=seed,
        learning_rate=hyperparams["learning_rate"],
    )

    sc.pp.neighbors(adata, use_rep=EMB_KEY)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY],
        uns=["neighbors"],
        obsp=["connectivities", "distances"],
    )

    output.write(os.path.join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
