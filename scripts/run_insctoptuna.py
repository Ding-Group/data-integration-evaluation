"""
Run INSCT on a dataset

Usage:
    run_insct.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

Options:
    -h --help                   Show this screen.
    -o <path>                   Path to output directory.
    -b <batch>                  One or more batch keys.
    -c <key>                    Cell type label key.
    --no-use-hvg                Do not use HVG selection. [Default: False]
    --hvg-method=<method>       HVG selection method. Default uses the pre-computed HVGs.
    --params-json=<file>        JSON file with hyperparameters
    --epochs=<int>              Number of epochs to train. [Default: 1000]
    --batch-size=<int>          Training batch size. [Default: 64]
    --n-epochs-without-progress=<int>  Early stopping patience. [Default: 20]
    --seed=<int>                Random seed. [Default: 0]
"""

import json
import random
from os.path import join

import anndata as ad
import numpy as np
import scanpy as sc
import tensorflow as tf2
from docopt import docopt
from tnn import TNN
from project_utils.adata_utils import EMB_KEY, combine_batch, minimise_anndata
from project_utils.hvg_utils import select_hvgs

for gpu in tf2.config.list_physical_devices('GPU'):
    tf2.config.experimental.set_memory_growth(gpu, True)

def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "embedding_dims": int(params["embedding_dims"]),
        "k": int(params["k"]),
        "margin": float(params["margin"]),
        "k_to_m_ratio": float(params["k_to_m_ratio"]),
    }
    return hyperparams, params.get("trial_number")


def run_insct(
    adata,
    batch_keys,
    use_hvg,
    hvg_method,
    embedding_dims,
    epochs,
    k,
    batch_size,
    margin,
    k_to_m_ratio,
    n_epochs_without_progress,
    seed,
):
    np.random.seed(seed)
    random.seed(seed)
    tf2.random.set_seed(seed)
    tf2.keras.utils.set_random_seed(seed)
    adata_copy = select_hvgs(adata, use_hvg, hvg_method, subset=True, copy=True)
    _, final_batch_key = combine_batch(adata_copy, batch_keys)

    if k > adata_copy.obs[final_batch_key].value_counts().min() - 1:
        k = adata_copy.obs[final_batch_key].value_counts().min() - 1
        print(f"Setting k to size of smallest batch, {k}")

    model = TNN(
        embedding_dims=embedding_dims,
        epochs=epochs,
        k=k,
        batch_size=batch_size,
        margin=margin,
        k_to_m_ratio=k_to_m_ratio,
        n_epochs_without_progress=n_epochs_without_progress,
    )
    model.fit(X=adata_copy, batch_name=final_batch_key)
    embs = model.transform(X=adata_copy)
    adata.obsm[EMB_KEY] = embs

    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    epochs = int(args["--epochs"])
    batch_size = int(args["--batch-size"])
    n_epochs_without_progress = int(args["--n-epochs-without-progress"])
    seed = int(args["--seed"])
    params_json_path = args["--params-json"]

    hyperparams, trial_number = load_hyperparams(params_json_path)
    if trial_number is not None:
        print(f"Running trial {trial_number}")
    print(f"Hyperparameters: {hyperparams}")

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_insct(
        adata,
        batch_keys,
        use_hvg,
        hvg_method,
        embedding_dims=hyperparams["embedding_dims"],
        epochs=epochs,
        k=hyperparams["k"],
        batch_size=batch_size,
        margin=hyperparams["margin"],
        k_to_m_ratio=hyperparams["k_to_m_ratio"],
        n_epochs_without_progress=n_epochs_without_progress,
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
