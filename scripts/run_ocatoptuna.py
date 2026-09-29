"""
Run OCAT on a dataset

Usage:
    run_ocat.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

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
import numpy as np
import OCAT
import pandas as pd
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, combine_batch, minimise_anndata
from project_utils.hvg_utils import select_hvgs


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "n_ghosts": int(params["n_ghosts"]),
        "dim": int(params["dim"]),
        "p": float(params["p"]),
    }
    return hyperparams, params.get("trial_number")


def run_ocat(adata, batch_keys, use_hvg, hvg_method, n_ghosts, dim, p, seed):
    np.random.seed(seed)
    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)
    _, batch_key_final = combine_batch(adata_copy, batch_keys)

    ref_list = [
        adata_copy[adata_copy.obs[batch_key_final] == i,]
        for i in np.unique(adata_copy.obs[batch_key_final])
    ]
    data_list = [i.layers["counts"] for i in ref_list]
    m_list = [n_ghosts] * len(data_list)
    ZW = OCAT.run_OCAT(data_list, dim=dim, m_list=m_list, p=p, random_seed=seed)

    query_latent = ad.AnnData(ZW)
    query_latent.obs = pd.concat([a.obs for a in ref_list])
    query_latent = query_latent[adata.obs_names]

    adata.obsm[EMB_KEY] = query_latent.X
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
    adata = run_ocat(
        adata,
        batch_keys,
        use_hvg,
        hvg_method,
        n_ghosts=hyperparams["n_ghosts"],
        dim=hyperparams["dim"],
        p=hyperparams["p"],
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
