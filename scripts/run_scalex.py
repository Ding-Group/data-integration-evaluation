"""
Run SCALEX on a dataset

Usage:
    run_scalex.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help               Show this screen.
    -o <path>               Path to output directory.
    -b <batch>              One or more batch keys.
    -c <key>                Cell type label key.
    --n-top-features=<int>  Number of highly-variable genes to keep. [default: 2000]
    --batch-size=<int>      Number of samples per batch to load. [default: 64]
    --lr=<int>              Learning rate. [default: 2e-4]
    --max-iteration=<int>   Training iterations. [default: 30000]
    --seed=<int>            Random seed. [default: 0]
"""

from os.path import join
import os
from typing import List

import anndata as ad
from docopt import docopt
from scalex.function import SCALEX

from project_utils.adata_utils import add_umap, minimise_anndata, combine_batch, EMB_KEY


def run_scalex(
    adata: ad.AnnData,
    batch_keys: List[str],
    n_top_features: int,
    batch_size: int,
    lr: float,
    max_iteration: int,
    seed: int,
):
    _, batch_key = combine_batch(adata, batch_keys)
    adata = SCALEX(
        adata,
        batch_name=batch_key,
        min_features=0,
        min_cells=0,
        n_top_features=n_top_features,
        seed=seed,
        batch_size=batch_size,
        lr=lr,
        max_iteration=max_iteration,
        outdir=os.environ['TMPDIR'],
        ignore_umap=True,
    )

    adata.obsm[EMB_KEY] = adata.obsm['latent']
    return adata


def main():
    args = docopt(__doc__)
    print(args)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])
    n_top_features = int(args['--n-top-features'])
    batch_size = int(args['--batch-size'])
    lr = float(args['--lr'])
    max_iteration = int(args['--max-iteration'])

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running SCALEX")
    adata = run_scalex(
        adata,
        batch_keys=batch_keys,
        n_top_features=n_top_features,
        batch_size=batch_size,
        lr=lr,
        max_iteration=max_iteration,
        seed=seed,
    )

    print("Adding UMAP")
    add_umap(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=list(batch_keys) + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
