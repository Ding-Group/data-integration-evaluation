"""
Run sysVI on a dataset

Usage:
  run_sysvioptuna.py -o <path> (-b <batch>)... -c <key> --params-json=<file> [options] <file>

Options:
    -h --help                      Show this screen.
    -o <path>                      Path to output directory.
    -b <batch>                     One or more batch keys.
    -c <key>                       Cell type label key.
    --no-use-hvg                   Do not use HVG selection.
    --hvg-method=<method>          HVG selection method. Default uses the pre-computed HVGs.
    --params-json=<file>           JSON file with hyperparameters
    --max-epochs=<int>             Max training epochs [default: 200].
    --seed=<int>                   Random seed [default: 0].
    --no-embed-cat                 Disable categorical embedding.
"""

import json
from os.path import join
import numpy as np
import anndata as ad
import scanpy as sc
import scvi
from scvi.external import SysVI
from docopt import docopt

from project_utils.adata_utils import EMB_KEY, minimise_anndata
from project_utils.hvg_utils import select_hvgs


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "z_distance_cycle_weight": float(params["z_distance_cycle_weight"]),
        "n_prior_components": int(params["n_prior_components"]),
        "kl_weight": float(params["kl_weight"]),
        "n_latent": int(params["n_latent"]),
        "n_hidden": int(params["n_hidden"]),
        "n_layers": int(params["n_layers"]),
        "dropout_rate": float(params["dropout_rate"]),
    }
    return hyperparams, params.get("trial_number")


def run_sysvi(
    adata: ad.AnnData,
    batch_keys: list[str],
    use_hvg: bool,
    hvg_method: str,
    n_latent: int,
    n_hidden: int,
    n_layers: int,
    n_prior_components: int | None,
    cycle_weight: float,
    kl_weight: float,
    dropout_rate: float,
    max_epochs: int,
    embed_cat: bool,
):
    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)

    new_batch_keys= []
    for b in batch_keys:
        adata_copy.obs.rename({b: b.replace(".", "_")}, axis=1, inplace=True)
        new_batch_keys.append(b.replace(".", "_"))
    # register anndata
    SysVI.setup_anndata(
        adata=adata_copy,
        batch_key=new_batch_keys[0],
        layer="counts",
        categorical_covariate_keys=new_batch_keys[1:],
    )

    # VampPrior pseudo-inputs
    n_cells = adata_copy.n_obs
    n_pseudo = n_prior_components

    pseudo_idx = np.random.choice(n_cells, size=n_pseudo, replace=False).astype(int)

    # build model
    model = SysVI(
        adata=adata_copy,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        n_prior_components=n_pseudo,
        pseudoinputs_data_indices=pseudo_idx,
        embed_categorical_covariates=embed_cat,
        dropout_rate=dropout_rate
    )

    # train
    model.train(
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        plan_kwargs={
            "kl_weight": float(kl_weight),
            "z_distance_cycle_weight": float(cycle_weight),
        },
        accelerator="auto",
    )

    # save latent
    adata.obsm[EMB_KEY] = model.get_latent_representation()

    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    max_epochs = int(args["--max-epochs"])
    seed = int(args["--seed"])
    embed_cat = not args["--no-embed-cat"]
    params_json_path = args["--params-json"]

    hyperparams, trial_number = load_hyperparams(params_json_path)
    if trial_number is not None:
        print(f"Running trial {trial_number}")
    print(f"Hyperparameters: {hyperparams}")

    # seeds
    scvi.settings.seed = seed

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running sysVI")
    adata = run_sysvi(
        adata=adata,
        batch_keys=batch_keys,
        use_hvg=use_hvg,
        hvg_method=hvg_method,
        n_latent=hyperparams["n_latent"],
        n_hidden=hyperparams["n_hidden"],
        n_layers=hyperparams["n_layers"],
        n_prior_components=hyperparams["n_prior_components"],
        cycle_weight=hyperparams["z_distance_cycle_weight"],
        kl_weight=hyperparams["kl_weight"],
        dropout_rate=hyperparams["dropout_rate"],
        max_epochs=max_epochs,
        embed_cat=embed_cat,
    )

    sc.pp.neighbors(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))

    print("Done!")


if __name__ == "__main__":
    main()
