"""
Run the Louvain clustering algorithm in Scanpy.

Usage:
    compute_louvain.py -o <path> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Output file.
    -s <seed>              Random state. [default: 0]
    -f <flavor>            Which implementation to use. [Default: vtraag]
    -c <cluster_key>       Cluster key to add. [Default: cluster]
"""

# cluster_key is "cluster" to be consistent with the scib metrics
# so that the clustering won't be recomputed each time.
from os.path import join

import anndata as ad
from docopt import docopt
import scanpy as sc


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_file = args["-o"]
    seed = int(args["-s"])
    flavor = args["-f"]
    cluster_key = args["-c"]

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Computing clustering")
    resolutions = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    for res in resolutions:
        sc.tl.louvain(
            adata,
            resolution=res,
            random_state=seed,
            key_added=f"{cluster_key}_{res}",
            flavor=flavor,
        )
    adata.uns['clustering_params'] = {'type': 'louvain', 'random_state': seed, 'flavor': flavor}
    print("Saving data")
    adata.write(out_file)
    print("Done!")


if __name__ == "__main__":
    main()
