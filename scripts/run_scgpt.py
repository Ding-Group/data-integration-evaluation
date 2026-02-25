#!/usr/bin/env python

"""
Run scGPT on a dataset

Usage:
    run_scgpt.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --seed=<int>           Random seed. [Default: 0]
"""

import sys
from os.path import join
from pathlib import Path

sys.path.insert(0, "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scGPT")

import anndata as ad
import scgpt as scg
from docopt import docopt
from scgpt.utils import set_seed
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata
from project_utils.hvg_utils import select_hvgs


def run_scgpt(adata, use_hvg, hvg_method, seed):
    set_seed(seed)
    if 'gene_name' in adata.var.columns:
        adata_copy = adata[:, ~adata.var['gene_name'].isna()].copy()
        adata_copy.var = adata_copy.var.set_index('gene_name')
        adata_copy.var_names = adata_copy.var_names.astype('object')
        adata_copy.var_names_make_unique()
    else:
        adata_copy = adata.copy()

    adata_copy = select_hvgs(
        adata_copy, use_hvg=use_hvg, hvg_method=hvg_method, subset=True, copy=False
    )
    adata_copy.var["Gene Symbol"] = adata_copy.var_names
    model_dir = Path("/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scGPT/scgpt_models/")
    adata_copy = scg.tasks.embed_data(
        adata_copy,
        model_dir,
        gene_col="Gene Symbol",
        batch_size=64
    )
    adata.obsm[EMB_KEY] = adata_copy.obsm['X_scGPT']
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

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_scgpt(adata, use_hvg, hvg_method, seed)
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
