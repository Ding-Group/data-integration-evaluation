#!/usr/bin/env python

"""
Run LIGER on a dataset

Usage:
    run_liger.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --k=<int>              Number of factors for LIGER [default: 20].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
from typing import List

import anndata as ad
import pyliger as pyliger
from docopt import docopt
from utils.adata_utils import (
    EMB_KEY,
    add_umap,
    combine_batch,
    minimise_anndata,
    split_by_batch,
)
from utils.hvg_utils import select_hvgs


def run_liger(adata: ad.AnnData, batch_keys: List[str], use_hvg: bool, hvg_method: str, k: int, seed: int) -> ad.AnnData:
    adata = select_hvgs(adata, use_hvg, hvg_method, subset=True)

    _, batch_key = combine_batch(adata, batch_keys)
    parts = split_by_batch(adata, batch_key)

    obj = pyliger.create_liger(parts)
    pyliger.normalize(obj)
    pyliger.select_genes(obj)
    pyliger.scale_not_center(obj)

    pyliger.optimize_ALS(obj, k=int(k), rand_seed=seed)
    pyliger.quantile_norm(obj)

    concat = ad.concat(obj.adata_list)
    concat = concat[adata.obs_names]
    embs = concat.obsm['H_norm']

    adata.obsm[EMB_KEY] = embs

    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    k = int(args["--k"])

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running LIGER")
    adata = run_liger(adata, batch_keys=batch_keys, use_hvg=use_hvg, hvg_method=hvg_method, k=k, seed=seed)

    print("Adding UMAP")
    add_umap(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=list(batch_keys) + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
