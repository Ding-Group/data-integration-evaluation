#!/usr/bin/env python

"""
Run LIGER on a dataset

Usage:
    run_liger.py --out-dir=<path> --batch-key=<key>... --cell-type-key=<key> [options] <file>

Options:
    -h --help              Show this screen.
    --out-dir=<path>       Path to output directory.
    --batch-key=<key>...   One or more batch keys.
    --cell-type-key=<key>  Cell type label key.
    --no-use-hvg           Do not use HVG selection.
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --k=<int>              Number of factors for LIGER [default: 20].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
from typing import List

import numpy as np
import anndata as ad
import scanpy as sc
import pyliger as pyliger
from docopt import docopt

from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def _ensure_combined_batch(adata: ad.AnnData, batch_keys: List[str]) -> str:
    if isinstance(batch_keys, (list, tuple)) and len(batch_keys) > 1:
        combo = "_batch_combined"
        adata.obs[combo] = (
            adata.obs[batch_keys].astype(str).agg("|".join, axis=1).astype("category")
        )
        return combo
    return batch_keys[0]


def _split_by_batch(adata: ad.AnnData, batch_key: str) -> List[ad.AnnData]:
    parts = []
    for b in adata.obs[batch_key].astype("category").cat.categories:
        a = adata[adata.obs[batch_key] == b].copy()
        a.uns["sample_name"] = str(b)
        parts.append(a)
    return parts


def _extract_liger_embeddings(obj) -> List[np.ndarray]:
    if hasattr(obj, "H_norm") and obj.H_norm is not None:
        Hs = obj.H_norm
    elif hasattr(obj, "H") and obj.H is not None:
        Hs = obj.H
    else:
        raise RuntimeError("LIGER factors not found (H_norm/H).")
    outs = []
    for H in Hs:
        arr = np.asarray(H)
        arr = arr.T if arr.shape[0] < arr.shape[1] else arr  # k x n -> n x k
        outs.append(arr.astype("float32"))
    return outs


def run_liger(adata: ad.AnnData, batch_keys: List[str], use_hvg: bool, hvg_method: str, k: int, seed: int) -> ad.AnnData:
    if use_hvg:
        if hvg_method is not None:
            layer = "counts" if ("v3" in hvg_method) else None
            sc.pp.highly_variable_genes(adata, flavor=hvg_method, layer=layer, n_top_genes=2000)
            adata = adata[:, adata.var.highly_variable].copy()
        else:
            adata = adata[:, adata.var.highly_variable].copy()

    batch_key = _ensure_combined_batch(adata, batch_keys)
    parts = _split_by_batch(adata, batch_key)

    obj = pyliger.create_liger(parts)
    pyliger.normalize(obj)
    pyliger.select_genes(obj)
    pyliger.scale_not_center(obj)

    pyliger.optimize_ALS(obj, k=int(k), seed=seed)
    pyliger.quantile_norm(obj)

    embs = _extract_liger_embeddings(obj)
    keys = [p.obs[batch_key].unique()[0] for p in parts]
    out = ad.concat(parts, join="inner", label=batch_key, keys=keys)
    out.obsm[EMB_KEY] = np.concatenate(embs, axis=0)

    return out


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["--out-dir"]
    batch_keys = args["--batch-key"]
    cell_type_key = args["--cell-type-key"]
    seed = int(args["--seed"])
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    k = int(args["--k"])

    set_seed(seed)

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
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
