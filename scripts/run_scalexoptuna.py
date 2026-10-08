"""
Run SCALEX on a dataset using Optuna trial hyperparameters

Usage:
    run_scalexoptuna.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

Options:
    -h --help               Show this screen.
    -o <path>               Path to output directory.
    -b <batch>              One or more batch keys.
    -c <key>                Cell type label key.
    --params-json=<file>    JSON file with hyperparameters
    --seed=<int>            Random seed. [default: 0]
"""

import json
import os
from os.path import join

import anndata as ad
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, combine_batch, minimise_anndata
from scalex.function import SCALEX


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "n_top_features": int(params["n_top_features"]),
        "batch_size": int(params["batch_size"]),
        "lr": float(params["lr"]),
        "max_iteration": int(params["max_iteration"]),
    }
    return hyperparams, params.get("trial_number")


def run_scalex(
    adata: ad.AnnData,
    batch_keys: list[str],
    n_top_features: int,
    batch_size: int,
    lr: float,
    max_iteration: int,
    seed: int,
):
    _, batch_key = combine_batch(adata, batch_keys)
    adata = SCALEX(
        adata,
        batch_name=batch_key,
        min_features=0,
        min_cells=0,
        n_top_features=n_top_features,
        seed=seed,
        batch_size=batch_size,
        lr=lr,
        max_iteration=max_iteration,
        outdir=os.environ['TMPDIR'],
        ignore_umap=True,
        num_workers=1
    )

    adata.obsm[EMB_KEY] = adata.obsm['latent']
    return adata


def main():
    args = docopt(__doc__)
    print(args)
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

    print("Running SCALEX")
    adata = run_scalex(
        adata,
        batch_keys=batch_keys,
        n_top_features=hyperparams["n_top_features"],
        batch_size=hyperparams["batch_size"],
        lr=hyperparams["lr"],
        max_iteration=hyperparams["max_iteration"],
        seed=seed,
    )

    sc.pp.neighbors(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=list(batch_keys) + [cell_type_key],
        obsm=[EMB_KEY],
        uns=["neighbors"],
        obsp=["connectivities", "distances"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
