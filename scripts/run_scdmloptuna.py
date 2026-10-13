"""
Run scDML on a dataset using Optuna trial hyperparameters

Usage:
    run_scdmloptuna.py -o <path> (-b <batch>)... -c <key> --params-json=<file> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys (first is main).
    -c <key>               Cell type label key.
    --params-json=<file>   JSON file with hyperparameters
    --merge-rule=<rule>    scDML merge rule: rule1 | rule2. [default: rule2]
    --seed=<int>           Random seed. [default: 0]
"""

import json
import os
from os.path import join

import anndata as ad
import numpy as np
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, combine_batch, minimise_anndata
from scDML import scDMLModel


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "resolution": float(params["resolution"]),
        "K_in": int(params["K_in"]),
        "K_bw": int(params["K_bw"]),
        "margin": float(params["margin"]),
        "num_epochs": int(params["num_epochs"]),
        "hidden_width": int(params["hidden_width"]),
    }
    return hyperparams, params.get("trial_number")


def run_scdml(
    adata,
    batch_keys,
    cell_type_key,
    resolution,
    K_in,
    K_bw,
    margin,
    num_epochs,
    hidden_width,
    merge_rule="rule2",
    seed=0,
):
    # Build neighborhood graph & init clusters on a copy (keeps original counts intact)
    adata_copy = adata.copy()
    sc.pp.scale(adata_copy, max_value=None)
    sc.tl.pca(adata_copy)
    sc.pp.neighbors(adata_copy)
    sc.tl.umap(adata_copy)
    sc.tl.louvain(adata_copy, resolution=resolution, key_added="init_cluster")

    # Run scDML on the original AnnData
    _, batch_key_final = combine_batch(adata_copy, batch_keys)

    scdml = scDMLModel(save_dir=os.environ['TMPDIR'] + '/', verbose=True)

    scdml.integrate(
        adata_copy,
        batch_key=batch_key_final,
        merge_rule=merge_rule,
        expect_num_cluster=len(np.unique(adata_copy.obs[cell_type_key])),
        ncluster_list=[len(np.unique(adata_copy.obs[cell_type_key]))],
        emb_dim=[hidden_width],
        num_epochs=num_epochs,
        K_in=K_in,
        K_bw=K_bw,
        margin=margin,
        seed=seed,
    )

    # Copy embedding to standard key
    adata.obsm[EMB_KEY] = adata_copy.obsm["X_emb"].copy()

    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]

    merge_rule = args["--merge-rule"]
    seed = int(args["--seed"])
    params_json_path = args["--params-json"]

    hyperparams, trial_number = load_hyperparams(params_json_path)
    if trial_number is not None:
        print(f"Running trial {trial_number}")
    print(f"Hyperparameters: {hyperparams}")

    print("Load data")
    adata = ad.read_h5ad(file)

    print("Run integration")
    adata = run_scdml(
        adata=adata,
        batch_keys=batch_keys,
        cell_type_key=cell_type_key,
        resolution=hyperparams["resolution"],
        K_in=hyperparams["K_in"],
        K_bw=hyperparams["K_bw"],
        margin=hyperparams["margin"],
        num_epochs=hyperparams["num_epochs"],
        hidden_width=hyperparams["hidden_width"],
        merge_rule=merge_rule,
        seed=seed,
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
