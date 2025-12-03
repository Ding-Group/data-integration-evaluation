#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run sysVI (scvi-tools external) on a dataset

Usage:
  run_sysvi.py -o <path> (-b <batch>)... -c <key> [options] <file>

Options:
    -h --help                      Show this screen.
    -o <path>                      Path to output directory.
    -b <batch>                     One or more batch keys.
    -c <key>                       Cell type label key.
    --n-latent=<int>               Latent dim [default: 20].
    --n-prior-components=<int>     Number of VampPrior pseudo-inputs.
    --cycle-weight=<float>         Cycle-consistency weight [default: 6.0].
    --kl-weight=<float>            KL loss weight [default: 1.0].
    --max-epochs=<int>             Max training epochs [default: 200].
    --seed=<int>                   Random seed [default: 0].
    --no-embed-cat                 Disable categorical embedding.
"""

from os.path import join
import numpy as np
import anndata as ad
import scanpy as sc
import scvi
from scvi.external import SysVI
from docopt import docopt

# utils in your repo
from utils.random import set_seed
from utils.adata_utils import EMB_KEY, minimise_anndata, add_umap


def run_sysvi(
    adata: ad.AnnData,
    batch_keys: list[str],
    n_latent: int,
    n_prior_components: int | None,
    cycle_weight: float,
    kl_weight: float,
    max_epochs: int,
    embed_cat: bool,
):
    # register anndata
    SysVI.setup_anndata(
        adata=adata,
        batch_key=batch_keys[0],
        layer="counts",
        categorical_covariate_keys=batch_keys[1:],
    )

    # VampPrior pseudo-inputs
    n_cells = adata.n_obs
    n_pseudo = n_prior_components if n_prior_components else min(64, n_cells)
    n_pseudo = min(n_pseudo, n_cells)

    pseudo_idx = np.random.choice(n_cells, size=n_pseudo, replace=False).astype(int)

    # build model
    model = SysVI(
        adata=adata,
        n_latent=n_latent,
        n_prior_components=n_pseudo,
        pseudoinputs_data_indices=pseudo_idx,
        embed_categorical_covariates=embed_cat,
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
    n_latent = int(args["--n-latent"])
    n_prior_components = (
        int(args["--n-prior-components"]) if args["--n-prior-components"] else None
    )
    cycle_weight = float(args["--cycle-weight"])
    kl_weight = float(args["--kl-weight"])
    max_epochs = int(args["--max-epochs"])
    seed = int(args["--seed"])
    embed_cat = not args["--no-embed-cat"]

    # seeds
    set_seed(seed)
    scvi.settings.seed = seed

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running sysVI")
    adata = run_sysvi(
        adata=adata,
        batch_keys=batch_keys,
        n_latent=n_latent,
        n_prior_components=n_prior_components,
        cycle_weight=cycle_weight,
        kl_weight=kl_weight,
        max_epochs=max_epochs,
        embed_cat=embed_cat,
    )

    print("Adding UMAP & neighbors")
    add_umap(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))

    print("Done!")


if __name__ == "__main__":
    main()
