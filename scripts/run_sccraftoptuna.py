"""
Run scCRAFT on a dataset

Usage:
    run_sccraft.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

Options:
    -h --help                   Show this screen.
    -o <path>                   Path to output directory.
    -b <batch>                  One or more batch keys.
    -c <key>                    Cell type label key.
    --no-use-hvg                Do not use HVG selection. [Default: False]
    --hvg-method=<method>       HVG selection method. Default uses the pre-computed HVGs.
    --params-json=<file>        JSON file with hyperparameters
    --epochs=<int>              Number of epochs. [Default: 150]
    --seed=<int>                Random seed. [Default: 0]
"""

import json
from os.path import join

import anndata as ad
import scanpy as sc
import scCRAFT
from docopt import docopt
from scCRAFT.model import *
from project_utils.adata_utils import EMB_KEY, combine_batch, minimise_anndata
from project_utils.hvg_utils import select_hvgs


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "kl_coef": float(params["kl_coef"]),
        "d_coef": float(params["d_coef"]),
        "z_dim": int(params["z_dim"]),
        "resolution1": float(params["resolution1"]),
        "resolution2": float(params["resolution2"]),
        "method": params["method"],
    }
    return hyperparams, params.get("trial_number")


def run_sccraft(
    adata,
    batch_keys,
    use_hvg,
    hvg_method,
    resolution1,
    resolution2,
    method,
    z_dim,
    epochs,
    d_coef,
    kl_coef,
    seed,
):
    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)
    _, batch_key_final = combine_batch(adata_copy, batch_keys)

    # Their seed is pre-determined by themselves for training
    multi_resolution_cluster(
        adata_copy, resolution1=resolution1, resolution2=resolution2, method=method
    )
    VAE = train_integration_model(
        adata_copy, batch_key=batch_key_final, z_dim=z_dim, epochs=epochs, d_coef=d_coef, kl_coef=kl_coef
    )
    obtain_embeddings(adata_copy, VAE, seed=seed)

    adata_copy = adata_copy[adata.obs_names]

    adata.obsm[EMB_KEY] = adata_copy.obsm["X_scCRAFT"]
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
    seed = int(args["--seed"])
    params_json_path = args["--params-json"]

    hyperparams, trial_number = load_hyperparams(params_json_path)
    if trial_number is not None:
        print(f"Running trial {trial_number}")
    print(f"Hyperparameters: {hyperparams}")

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_sccraft(
        adata,
        batch_keys,
        use_hvg,
        hvg_method,
        resolution1=hyperparams["resolution1"],
        resolution2=hyperparams["resolution2"],
        method=hyperparams["method"],
        z_dim=hyperparams["z_dim"],
        epochs=epochs,
        d_coef=hyperparams["d_coef"],
        kl_coef=hyperparams["kl_coef"],
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
