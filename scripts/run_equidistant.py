#!/usr/bin/env python

"""
Make an equidistant-blobs baseline embedding

Usage:
    run_equidistant.py --out-dir=<path> --batch-key=<key>... --cell-type-key=<key> [options] <file>

Options:
    -h --help              Show this screen.
    --out-dir=<path>       Path to output directory.
    --batch-key=<key>...   One or more batch keys (kept in output).
    --cell-type-key=<key>  Cell type label key (defines the number of blobs).
    --dim=<int>            Embedding dimension d [default: 10].
    --radius=<float>       Center norm r [default: 5.0].
    --sigma=<float>        Gaussian noise std around centers [default: 0.0].
    --neighbors=<int>      k for KNN graph [default: 30].
    --leiden-res=<float>   Leiden resolution [default: 1.0].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
import numpy as np
import anndata as ad
import scanpy as sc
from scipy import linalg as la
from docopt import docopt

from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def _simplex_centers(num_classes: int, dim: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    C = num_classes
    d_eff = max(dim, C - 1)
    Q = la.null_space(np.ones((1, C)))
    V = np.eye(C) - np.ones((C, C)) / C
    centers_low = V @ Q
    centers_low = centers_low / np.linalg.norm(centers_low[0]) * radius
    centers = centers_low[:, :d_eff]
    if d_eff > centers.shape[1]:
        centers = np.pad(centers, ((0, 0), (0, d_eff - centers.shape[1])), mode="constant")
    return centers.astype("float32")


def run_eqblob(adata, label_key, dim, radius, sigma, neighbors, leiden_res, seed):
    rng = np.random.default_rng(seed)
    labels = adata.obs[label_key].astype("category")
    classes = list(labels.cat.categories)
    idx = labels.cat.codes.to_numpy()
    centers = _simplex_centers(num_classes=len(classes), dim=dim, radius=radius, rng=rng)
    X = centers[idx] + rng.normal(0.0, float(sigma), size=(adata.n_obs, centers.shape[1])).astype("float32")
    adata.obsm[EMB_KEY] = X
    add_umap(adata, n_neighbors=int(neighbors))
    sc.tl.leiden(adata, resolution=float(leiden_res), key_added="leiden_eqblob", random_state=seed)
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["--out-dir"]
    batch_keys = args["--batch-key"]
    cell_type_key = args["--cell-type-key"]
    dim = int(args["--dim"])
    radius = float(args["--radius"])
    sigma = float(args["--sigma"])
    neighbors = int(args["--neighbors"])
    leiden_res = float(args["--leiden-res"])
    seed = int(args["--seed"])

    set_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Building equidistant blobs")
    adata = run_eqblob(
        adata,
        label_key=cell_type_key,
        dim=dim,
        radius=radius,
        sigma=sigma,
        neighbors=neighbors,
        leiden_res=leiden_res,
        seed=seed,
    )

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=list(batch_keys) + [cell_type_key, "leiden_eqblob"],
        obsm=[EMB_KEY, "X_umap"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
