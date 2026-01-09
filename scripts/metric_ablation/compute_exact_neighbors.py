"""
Run exact nearest neighbor graph construction in Scanpy.

Usage:
    compute_exact_neighbors.py -k <neighbors> -o <path> [options] <file>

Options:
    -h --help              Show this screen.
    -k <neighbors>         Number of nearest neighbors to use.
    -o <path>              Path to output directory.
"""

from os.path import join

import anndata as ad
from docopt import docopt
import scanpy as sc


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    k = int(args["-k"])

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Computing neighbors")
    sc.pp.neighbors(
        adata, n_neighbors=k, use_rep="X_integration", transformer="sklearn"
    )
    # Metadata about the neighbors is stored in adata.uns
    adata.uns['neighbors_params'] = {'type': 'exact', 'random_state': 0, 'k': k}
    print("Saving data")
    adata.write("neighbors_adata.h5ad")
    print("Done!")


if __name__ == "__main__":
    main()
