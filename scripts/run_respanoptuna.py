"""
Run ResPAN on a dataset

Usage:
    run_ResPAN.py -o <path> (-b <batch>)... -c <key> --params-json=<file> [options] <file>

Options:
    -h --help               Show this screen.
    -o <path>               Output directory.
    -b <batch>              One or more batch keys (first is main).
    -c <key>                Cell type label key.
    --params-json=<file>    JSON file with hyperparameters
    --seed=<int>            Random seed. [default: 0]
"""
import sys

sys.path.insert(0, '/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/ResPAN')

import json
from os.path import join

import anndata as ad
import numpy as np
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, combine_batch, minimise_anndata
from ResPAN import run_respan


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "reduction": params["reduction"],
        "subsample": int(params["subsample"]),
        "epochs": int(params["epochs"]),
        "lambda_1": float(params["lambda_1"]),
        "lr": float(params["lr"])
    }
    return hyperparams, params.get("trial_number")


def as_dense_float32(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32, copy=False)


def run_respan_pipeline(
    adata: ad.AnnData,
    batch_keys,
    cell_type_key: str,
    seed: int,
    epochs: int,
    reduction: str,
    subsample: int,
    lambda_1: float,
    lr: float,
    batch_size: int = 1024,
) -> ad.AnnData:
    np.random.seed(seed)

    # Must be 2000 highly variable genes
    adata_copy = adata.copy()
    sc.pp.highly_variable_genes(adata_copy, layer='counts', n_top_genes=2000, flavor='seurat_v3')
    adata_copy = adata_copy[:, adata_copy.var.highly_variable].copy()
    _, batch_key_final = combine_batch(adata_copy, batch_keys)
    adata_copy.obs['batch'] = adata_copy.obs[batch_key_final]

    adata_copy.X = as_dense_float32(adata_copy.X)
    adata_copy.obs['celltype'] = adata_copy.obs[cell_type_key]

    adata_new = run_respan(
        adata_copy,
        batch_key="batch",
        epoch=epochs,
        batch=batch_size,
        lambda_1=lambda_1,
        reduction=reduction,
        subsample=subsample,
        seed=seed,
        lr=lr,
    )

    sc.pp.scale(adata_new, max_value=10)
    sc.tl.pca(adata_new, 20, svd_solver='arpack')

    adata_new = adata_new[adata.obs_names]
    adata.obsm[EMB_KEY] = adata_new.obsm['X_pca']
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

    print("Load data")
    adata = ad.read_h5ad(file)

    print("Start integration")
    adata = run_respan_pipeline(
        adata,
        batch_keys,
        cell_type_key,
        seed,
        epochs=hyperparams['epochs'],
        reduction=hyperparams['reduction'],
        subsample=hyperparams['subsample'],
        lambda_1=hyperparams['lambda_1'],
        lr=hyperparams['lr'],
    )

    sc.pp.neighbors(adata)

    print("Save output")
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
