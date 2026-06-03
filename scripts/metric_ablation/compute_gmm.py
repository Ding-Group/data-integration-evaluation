"""
Run the Gaussian Mixture clustering algorithm in Scanpy.

Usage:
    compute_gmm.py -o <path> -t <cell_type> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Output file.
    -t <cell_type>         Cell type column.
    -s <seed>              Random state. [default: 0]
    -n <ninit>             Number of random initializations to perform [default: 1]
    -i <init>              Initialization method [default: kmeans]
    -c <cluster_key>       Cluster key to add. [default: cluster]
"""

# cluster_key is "cluster" to be consistent with the scib metrics
# so that the clustering won't be recomputed each time.
import anndata as ad
from math import ceil
from docopt import docopt
from sklearn.mixture import GaussianMixture

EMB_KEY = 'X_integration'

def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_file = args["-o"]
    cell_type_col = args["-t"]
    seed = int(args["-s"])
    n_init = int(args['-n'])
    init_params = args["-i"]
    cluster_key = args["-c"]

    print("Loading data")
    adata = ad.read_h5ad(file)
    num_cell_types = adata.obs[cell_type_col].nunique()

    # Set the number of clusters to be some range around the actual number of cell types.
    # These will correspond to the "leiden resolutions"
    # Set to -80%, -60%, ..., +60%, +80%, +100%
    n_clusters = [
        ceil(num_cell_types * 0.2),
        ceil(num_cell_types * 0.4),
        ceil(num_cell_types * 0.6),
        ceil(num_cell_types * 0.8),
        num_cell_types,
        ceil(num_cell_types * 1.2),
        ceil(num_cell_types * 1.4),
        ceil(num_cell_types * 1.6),
        ceil(num_cell_types * 1.8),
        num_cell_types * 2,
    ]

    print("Computing clustering")
    # This is just to hack it into the cluster resolutions leiden expects.
    resolutions = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    for i, res in enumerate(resolutions):
        gm = GaussianMixture(n_components=n_clusters[i], n_init=n_init, init_params=init_params, random_state=seed).fit(
            adata.obsm[EMB_KEY]
        )
        labels = gm.predict(adata.obsm[EMB_KEY])
        adata.obs[f"{cluster_key}_{res}"] = adata.obs['gm'] = labels.astype(str)

    adata.uns["clustering_params"] = {
        "type": "gmm",
        "random_state": seed,
        "init_params": init_params,
        "n_init": n_init
    }
    adata.uns['neighbors_params'] = dict()
    print("Saving data")
    adata.write(out_file)
    print("Done!")


if __name__ == "__main__":
    main()
