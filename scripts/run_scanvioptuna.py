"""
Run scanVI on a dataset using Optuna trial hyperparameters

Usage:
    run_scanvioptuna.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

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
from os.path import join

import anndata as ad
import scanpy as sc
import scvi
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, minimise_anndata
from project_utils.hvg_utils import select_hvgs


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "n_latent": int(params["n_latent"]),
        "n_hidden": int(params["n_hidden"]),
        "n_layers": int(params["n_layers"]),
        "dropout_rate": float(params["dropout_rate"]),
        "lr": float(params["lr"]),
    }
    return hyperparams, params.get("trial_number")


def run_scanvi(
    adata,
    batch_keys,
    cell_type_key,
    use_hvg,
    hvg_method,
    seed,
    n_latent,
    n_hidden,
    n_layers,
    dropout_rate,
    lr,
):
    scvi.settings.seed = seed

    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True, copy=True)

    new_batch_keys= []
    for b in batch_keys:
        adata_copy.obs.rename({b: b.replace(".", "_")}, axis=1, inplace=True)
        new_batch_keys.append(b.replace(".", "_"))
    adata_copy.obs.rename(
        {cell_type_key: cell_type_key.replace(".", "_")}, axis=1, inplace=True
    )
    cell_type_key = cell_type_key.replace(".", "_")

    arches_params = dict(
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        dropout_rate=dropout_rate,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_latent=n_latent,
    )

    scvi.model.SCVI.setup_anndata(
        adata_copy,
        layer="counts",
        batch_key=new_batch_keys[0],
        categorical_covariate_keys=new_batch_keys[1:],
        labels_key=cell_type_key,
    )
    model = scvi.model.SCVI(adata_copy, **arches_params)
    model.train(plan_kwargs={"lr": lr})

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        model, unlabeled_category="Unknown"
    )
    scanvi_model.train(plan_kwargs={"lr": lr})

    adata.obsm[EMB_KEY] = scanvi_model.get_latent_representation()
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
    adata = run_scanvi(
        adata,
        batch_keys,
        cell_type_key,
        use_hvg,
        hvg_method,
        seed,
        n_latent=hyperparams["n_latent"],
        n_hidden=hyperparams["n_hidden"],
        n_layers=hyperparams["n_layers"],
        dropout_rate=hyperparams["dropout_rate"],
        lr=hyperparams["lr"],
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
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
