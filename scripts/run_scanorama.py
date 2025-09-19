#!/usr/bin/env python

"""
Run Scanorama on a dataset

Usage:
    run_scanorama.py --out-dir=<path> --batch-key=<key>... --cell-type-key=<key> [options] <file>

Options:
    -h --help              Show this screen.
    --out-dir=<path>       Path to output directory.
    --batch-key=<key>...   One or more batch keys.
    --cell-type-key=<key>  Cell type label key.
    --no-use-hvg           Do not use HVG selection.
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --dimred=<int>         Latent dimension for Scanorama [default: 50].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
from typing import List

import numpy as np
import anndata as ad
import scanpy as sc
import scanorama
from docopt import docopt

from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def _ensure_combined_batch(adata: ad.AnnData, batch_keys: List[str]) -> str:
    """Combine multiple batch columns into a single categorical column if needed."""
    if isinstance(batch_keys, (list, tuple)) and len(batch_keys) > 1:
        combo = "_batch_combined"
        adata.obs[combo] = (
            adata.obs[batch_keys].astype(str).agg("|".join, axis=1).astype("category")
        )
        return combo
    return batch_keys[0]


def _split_by_batch(adata: ad.AnnData, batch_key: str) -> List[ad.AnnData]:
    """Split AnnData into a list of views/copies per batch level."""
    adatas = []
    for b in adata.obs[batch_key].unique():
        adatas.append(adata[adata.obs[batch_key] == b].copy())
    return adatas


def run_scanorama(
    adata: ad.AnnData,
    batch_keys: List[str],
    use_hvg: bool,
    hvg_method: str,
    dimred: int,
    seed: int,
) -> ad.AnnData:
    # Optional HVG
    if use_hvg:
        if hvg_method is not None:
            layer = "counts" if ("v3" in hvg_method) else None
            sc.pp.highly_variable_genes(adata, flavor=hvg_method, layer=layer, n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable]

    
    batch_key = _ensure_combined_batch(adata, batch_keys)

    
    parts = _split_by_batch(adata, batch_key)

    scanorama.integrate_scanpy(parts, dimred=int(dimred), seed=seed)

    keys = [p.obs[batch_key].unique()[0] for p in parts]
    adata_out = ad.concat(parts, join="inner", label=batch_key, keys=keys)

    adata_out.obsm[EMB_KEY] = np.concatenate([p.obsm["X_scanorama"] for p in parts], axis=0)

    return adata_out


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["--out-dir"]
    batch_keys = args["--batch-key"]
    cell_type_key = args["--cell-type-key"]
    seed = int(args["--seed"])
    use_hvg = not args["--use-hvg"]
    hvg_method = args["--hvg-method"]
    dimred = int(args["--dimred"])

    set_seed(seed)

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
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
