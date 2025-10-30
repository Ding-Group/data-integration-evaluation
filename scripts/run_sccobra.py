#!/usr/bin/env python

"""
Run scCobra on a dataset

Usage:
    run_sccobra.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --max-iteration=<int>  Maximum number of iterations. [Default: 30000]
    --seed=<int>           Random seed. [Default: 0]
"""

from os.path import join
import sys

sys.path.insert(
    0,
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scCobra/",
)

import anndata as ad
from scCobra import scCobra

from docopt import docopt
from utils.adata_utils import add_umap, minimise_anndata, combine_batch, EMB_KEY
from utils.hvg_utils import select_hvgs


def run_sccobra(adata, batch_keys, use_hvg, hvg_method, max_iteration, seed):
    adata_copy = select_hvgs(
        adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True, copy=True
    )

    _, batch_key_final = combine_batch(adata_copy, batch_keys)
    adata_copy = scCobra(
        adata_copy,
        batch_name=batch_key_final,
        min_features=0,
        min_cells=3,
        outdir=None,
        show=False,
        processed=True,
        max_iteration=max_iteration,
        seed=seed,
    )

    adata.obsm[EMB_KEY] = adata_copy.obsm["latent"]
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    max_iteration = int(args["--max-iteration"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_sccobra(adata, batch_keys, use_hvg, hvg_method, max_iteration, seed)
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
