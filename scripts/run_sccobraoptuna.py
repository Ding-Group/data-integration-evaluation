"""
Run scCobra on a dataset using Optuna trial hyperparameters

Usage:
    run_sccobraoptuna.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

Options:
    -h --help               Show this screen.
    -o <path>               Path to output directory.
    -b <batch>              One or more batch keys.
    -c <key>                Cell type label key.
    --no-use-hvg            Do not use HVG selection. [default: False]
    --hvg-method=<method>   HVG selection method. Default uses the pre-computed HVGs.
    --params-json=<file>    JSON file with hyperparameters
    --seed=<int>            Random seed. [default: 0]
"""

import json
import os
import sys
from os.path import join

sys.path.insert(
    0,
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scCobra/",
)

import anndata as ad
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, combine_batch, minimise_anndata
from project_utils.hvg_utils import select_hvgs
from scCobra import scCobra


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "batch_size": int(params["batch_size"]),
        "lr": float(params["lr"]),
        "max_iteration": int(params["max_iteration"]),
    }
    return hyperparams, params.get("trial_number")


def run_sccobra(adata, batch_keys, use_hvg, hvg_method, batch_size, lr, max_iteration, seed):
    adata_copy = select_hvgs(
        adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True, copy=True
    )

    extension = os.urandom(2).hex()
    os.makedirs(join(os.environ['TMPDIR'], extension), exist_ok=True)

    _, batch_key_final = combine_batch(adata_copy, batch_keys)
    adata_copy = scCobra(
        adata_copy,
        batch_name=batch_key_final,
        min_features=0,
        min_cells=3,
        outdir=join(os.environ['TMPDIR'], extension),
        show=False,
        processed=True,
        batch_size=batch_size,
        lr=lr,
        max_iteration=max_iteration,
        seed=seed,
    )

    adata.obsm[EMB_KEY] = adata_copy.obsm["latent"]
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

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_sccobra(
        adata,
        batch_keys,
        use_hvg,
        hvg_method,
        batch_size=hyperparams["batch_size"],
        lr=hyperparams["lr"],
        max_iteration=hyperparams["max_iteration"],
        seed=seed,
    )

    sc.pp.neighbors(adata, use_rep='X_integration')

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
