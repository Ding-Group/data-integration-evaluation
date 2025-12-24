# #!/usr/bin/env python

"""
Run DESC clustering on a dataset

Usage:
    run_desc.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help                      Show this screen.
    -o <path>                      Path to output directory.
    -b <batch>                     One or more batch keys.
    -c <key>                       Cell type label key.
    --no-use-hvg                   Do not use HVG selection. [Default: False]
    --hvg-method=<method>          HVG selection method. Default uses the pre-computed HVGs.
    --dims=<csv>                   Network dims (comma separated). [default: 32,16]
    --tol=<float>                  DESC tol. [default: 0.005]
    --n-neighbors=<int>            KNN neighbors. [default: 10]
    --batch-size=<int>             Batch size. [default: 256]
    --seed=<int>                   Random seed. [default: 0]
"""

import os
import sys

# Odd fix
del os.environ['DISPLAY']
sys.path.insert(0, '/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/desc/')

from os.path import join
from typing import List

import anndata as ad
import desc as DESC
import numpy as np
from docopt import docopt
from utils.adata_utils import EMB_KEY, add_umap, minimise_anndata
from utils.hvg_utils import select_hvgs


def run_desc(
    adata: ad.AnnData,
    use_hvg: bool,
    hvg_method: str,
    dims_tail: List[int],
    tol: float,
    n_neighbors: int,
    batch_size: int,
    seed: int,
) -> ad.AnnData:
    """Train DESC and return AnnData with embeddings/clusters"""
    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)
    adata_copy.X = adata_copy.X.toarray().astype(np.float32)

    dims = [adata_copy.shape[1]] + [int(d) for d in dims_tail]

    # DESC expects a list for resolutions; you already pass a list
    adata_copy = DESC.train(
        adata_copy,
        dims=dims,
        tol=float(tol),
        n_neighbors=int(n_neighbors),
        batch_size=int(batch_size),
        louvain_resolution=[0.8],
        do_tsne=False,
        do_umap=False,
        save_encoder_weights=False,
        use_GPU=True,
        save_dir=os.environ['TMPDIR']
    )

    adata_copy = adata_copy[adata.obs_names]
    adata.obsm[EMB_KEY] = adata_copy.obsm['X_Embeded_z0.8']
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]

    dims_tail = [int(x) for x in args["--dims"].split(",") if x.strip()]
    tol = float(args["--tol"])
    n_neighbors = int(args["--n-neighbors"])
    batch_size = int(args["--batch-size"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running DESC")
    adata = run_desc(
        adata,
        use_hvg,
        hvg_method,
        dims_tail=dims_tail,
        tol=tol,
        n_neighbors=n_neighbors,
        batch_size=batch_size,
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
