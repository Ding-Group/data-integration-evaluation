#!/usr/bin/env python

"""
Run Scanorama on a dataset

Usage:
    run_scanorama.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --dimred=<int>         Latent dimension for Scanorama [default: 50].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
from typing import List

import numpy as np
import anndata as ad
import scanorama
from docopt import docopt

from project_utils.adata_utils import add_umap, combine_batch, minimise_anndata, split_by_batch, EMB_KEY
from utils.hvg_utils import select_hvgs



def run_scanorama(
    adata: ad.AnnData,
    batch_keys: List[str],
    use_hvg: bool,
    hvg_method: str,
    dimred: int,
    seed: int,
) -> ad.AnnData:
    # Optional HVG
    adata = select_hvgs(adata, use_hvg, hvg_method, subset=True, copy=True)

    _, batch_key_final = combine_batch(adata, batch_keys)

    parts = split_by_batch(adata, batch_key_final)

    scanorama.integrate_scanpy(parts, dimred=int(dimred), seed=seed)

    keys = [p.obs[batch_key_final].unique()[0] for p in parts]
    adata_out = ad.concat(parts, join="inner", label=batch_key_final, keys=keys)

    adata_out.obsm[EMB_KEY] = np.concatenate([p.obsm["X_scanorama"] for p in parts], axis=0)

    # Reorder cells back to original order
    adata_out = adata_out[adata.obs_names].copy()

    return adata_out


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    dimred = int(args["--dimred"])

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running Scanorama")
    adata = run_scanorama(
        adata=adata,
        batch_keys=batch_keys,
        use_hvg=use_hvg,
        hvg_method=hvg_method,
        dimred=dimred,
        seed=seed,
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
