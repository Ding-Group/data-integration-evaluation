"""
Run SIMBA on a dataset using Optuna trial hyperparameters

Usage:
    run_simbaoptuna.py -o <path> (-b <batch>)...  -c <key> --params-json=<file> [options] <file>

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
import simba as si
from docopt import docopt
from project_utils.adata_utils import (
    EMB_KEY,
    combine_batch,
    minimise_anndata,
    split_by_batch,
)
from project_utils.hvg_utils import select_hvgs


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    hyperparams = {
        "n_bins": int(params["n_bins"]),
        "n_components": int(params["n_components"]),
        "k": int(params["k"]),
        "num_epochs": int(params["num_epochs"]),
    }
    return hyperparams, params.get("trial_number")


def run_simba(adata, batch_keys, use_hvg, hvg_method, n_bins, n_components, k, num_epochs, seed):
    adata_copy = adata.copy()
    adata_copy.var_names_make_unique()
    si.pp.filter_genes(adata_copy, min_n_cells=3)
    _, batch_key_final = combine_batch(adata_copy, batch_keys)

    adatas = split_by_batch(adata_copy, batch_key_final)
    for a in adatas:
        si.tl.discretize(a, n_bins=n_bins)

    largest_adata = max(adatas, key=lambda x: x.n_obs)
    cgs = []
    for a in adatas:
        if a is not largest_adata:
            cgs.append(
                si.tl.infer_edges(largest_adata, a, n_components=n_components, k=k, random_state=seed)
            )

    si.tl.gen_graph(list_CG=adatas, list_CC=cgs, copy=False, dirname="graph0")
    si.settings.pbg_params["num_epochs"] = int(num_epochs)
    si.tl.pbg_train(auto_wd=True, save_wd=True, output="model")

    # load in graph ('graph0') info
    si.load_graph_stats()
    # load in model info for ('graph0')
    si.load_pbg_config()

    dict_adata = si.read_embedding()

    for i, a in enumerate(adatas):
        if i == 0:
            a.obsm[EMB_KEY] = dict_adata["C"][a.obs_names].X
        else:
            a.obsm[EMB_KEY] = dict_adata[f"C{i + 1}"][a.obs_names].X

    a_concat = ad.concat(adatas)
    a_concat = a_concat[adata.obs_names]

    adata.obsm[EMB_KEY] = a_concat.obsm[EMB_KEY]

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
    adata = run_simba(
        adata,
        batch_keys,
        use_hvg,
        hvg_method,
        n_bins=hyperparams["n_bins"],
        n_components=hyperparams["n_components"],
        k=hyperparams["k"],
        num_epochs=hyperparams["num_epochs"],
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
