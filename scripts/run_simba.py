#!/usr/bin/env python

"""
Run SIMBA on a dataset

Usage:
    run_simba.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --n-bins=<int>         Number of bins in descretization. [Default: 5]
    --n-components=<int>   Number of components in inferring edges. [Default: 15]
    --k=<int>              k in inferring edges. [Default: 15]
    --seed=<int>           Random seed. [Default: 0]
"""

from os.path import join

import anndata as ad
import simba as si
from docopt import docopt
from utils.adata_utils import (
    EMB_KEY,
    add_umap,
    combine_batch,
    minimise_anndata,
    split_by_batch,
)
from utils.hvg_utils import select_hvgs


def run_simba(adata, batch_keys, use_hvg, hvg_method, n_bins, n_components, k, seed):
    adata_copy = adata.copy()
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
    n_bins = int(args["--n-bins"])
    n_components = int(args["--n-components"])
    k = int(args["--k"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_simba(
        adata, batch_keys, use_hvg, hvg_method, n_bins, n_components, k, seed
    )
    print("Adding UMAP")
    add_umap(adata)
    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
