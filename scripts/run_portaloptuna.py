"""
Run Portal on a dataset

Usage:
    run_portal.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

Options:
    -h --help               Show this screen.
    -o <path>               Path to output directory.
    -b <batch>              One or more batch keys.
    -c <key>                Cell type label key.
    --params-json=<file>    JSON file with hyperparameters
    --seed=<int>            Random seed. [Default: 0]
"""

import json
import random
import sys
from os.path import join

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from docopt import docopt
from project_utils.adata_utils import (
    EMB_KEY,
    combine_batch,
    minimise_anndata,
    split_by_batch,
)

sys.path.insert(0, "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/Portal/portal/")

from util import integrate_datasets as portal_integrate
from util import preprocess_datasets as portal_preprocess


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "lambdacos": float(params["lambdacos"]),
        "training_steps": int(params["training_steps"]),
    }
    return hyperparams, params.get("trial_number")


def run_portal(adata, batch_keys, lambda_cos, training_steps):
    adata_copy = adata.copy()
    adata_copy.X = adata_copy.layers['counts'].copy()

    _, batch_key_final = combine_batch(adata_copy, batch_keys)
    adatas = split_by_batch(adata_copy, batch_key_final)
    lowdim_list = portal_preprocess(adatas)
    outs = portal_integrate(lowdim_list, lambda_cos=lambda_cos, training_steps=training_steps)
    new_adata = ad.AnnData(X = outs, obs= ad.concat(adatas).obs)
    new_adata = new_adata[adata.obs_names]
    adata.obsm[EMB_KEY] = new_adata.X
    return adata


def main():
    args = docopt(__doc__)
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

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_portal(
        adata,
        batch_keys,
        lambda_cos=hyperparams["lambdacos"],
        training_steps=hyperparams["training_steps"],
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
