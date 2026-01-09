"""
Run the FAISS nearest neighbor graph construction in Scanpy.

Usage:
    compute_faiss_neighbors.py -k <neighbors> -s <seed> -o <path> [options] <file>

Options:
    -h --help              Show this screen.
    -k <neighbors>         Number of nearest neighbors to use.
    -s <seed>              Random seed to use.
    -o <path>              Path to output directory.
"""

from os.path import join

import anndata as ad
import scanpy as sc
from docopt import docopt
from sklearn_ann.kneighbors.faiss import FAISSTransformer


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    k = int(args["-k"])
    seed = int(args["-s"])

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Computing neighbors")
    sc.pp.neighbors(
        adata,
        use_rep="X_integration",
        transformer=FAISSTransformer(n_neighbors=k),
    )
    # Metadata about the neighbors is stored in adata.uns
    adata.uns["neighbors_params"] = {
        "type": "faiss",
        "random_state": seed,
        "k": k,
    }
    print("Saving data")
    adata.write("neighbors_adata.h5ad")
    print("Done!")


if __name__ == "__main__":
    main()
