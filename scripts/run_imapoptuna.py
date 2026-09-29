"""
Run IMAP on a dataset

Usage:
    run_imapoptuna.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

Options:
    -h --help               Show this screen.
    -o <path>               Path to output directory.
    -b <batch>              One or more batch keys.
    -c <key>                Cell type label key.
    --no-use-hvg            Do not use HVG selection. [Default: False]
    --hvg-method=<method>   HVG selection method. Default uses the pre-computed HVGs.
    --params-json=<file>    JSON file with hyperparameters
    --seed=<int>            Random seed. [Default: 0]
"""

import json
from os.path import join

import anndata as ad
import imap
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, minimise_anndata, combine_batch
from project_utils.hvg_utils import select_hvgs


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "n_epochs": int(params["n_epochs"]),
        "latent_dim": int(params["latent_dim"]),
        "lambda_co": float(params["lambda_co"]),
        "lambda_rc": float(params["lambda_rc"]),
    }
    return hyperparams, params.get("trial_number")


def run_imap(adata, batch_keys, use_hvg, hvg_method, n_epochs, latent_dim, lambda_co, lambda_rc, seed):
    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)

    _, batch_key_final = combine_batch(adata_copy, batch_keys)
    adata_copy.X = adata_copy.X.toarray()

    EC, ec_data = imap.stage1.iMAP_fast(
        adata_copy,
        key=batch_key_final,
        n_epochs=n_epochs,
        latent_dim=latent_dim,
        lambda_co=lambda_co,
        lambda_rc=lambda_rc,
        seed=seed
    )
    output_results = imap.stage2.integrate_data(adata_copy, ec_data, key=batch_key_final, n_epochs=n_epochs, seed=seed)

    adata_copy.X = output_results
    sc.pp.pca(adata_copy)

    adata.obsm[EMB_KEY] = adata_copy.obsm['X_pca']
    return adata


def main():
    args = docopt(__doc__)
    print(args)
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
    adata = run_imap(
        adata,
        batch_keys,
        use_hvg,
        hvg_method,
        n_epochs=hyperparams["n_epochs"],
        latent_dim=hyperparams["latent_dim"],
        lambda_co=hyperparams["lambda_co"],
        lambda_rc=hyperparams["lambda_rc"],
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
