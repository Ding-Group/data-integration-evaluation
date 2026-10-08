"""
Make an equidistant-blobs baseline embedding

Usage:
    run_equidistant.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --radius=<float>       Center norm r [default: 10.0].
    --sigma=<float>        Gaussian noise std around centers [default: 1.0].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
import numpy as np
import anndata as ad
from scipy import linalg as la
from docopt import docopt

from project_utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def _simplex_centers(num_classes: int, radius: float) -> np.ndarray:
    C = num_classes
    d_eff = C - 1
    Q = la.null_space(np.ones((1, C)))
    V = np.eye(C) - np.ones((C, C)) / C
    centers_low = V @ Q
    centers_low = centers_low / np.linalg.norm(centers_low[0]) * radius
    centers = centers_low[:, :d_eff]
    if d_eff > centers.shape[1]:
        centers = np.pad(centers, ((0, 0), (0, d_eff - centers.shape[1])), mode="constant")
    return centers.astype("float32")


def run_eqblob(adata, label_key, radius, sigma, seed):
    rng = np.random.default_rng(seed)
    labels = adata.obs[label_key].astype("category")
    classes = list(labels.cat.categories)
    idx = labels.cat.codes.to_numpy()
    centers = _simplex_centers(num_classes=len(classes), radius=radius)
    X = centers[idx] + rng.normal(0.0, float(sigma), size=(adata.n_obs, centers.shape[1])).astype("float32")
    adata.obsm[EMB_KEY] = X
    add_umap(adata)
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    radius = float(args["--radius"])
    sigma = float(args["--sigma"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Building equidistant blobs")
    adata = run_eqblob(
        adata,
        label_key=cell_type_key,
        radius=radius,
        sigma=sigma,
        seed=seed
    )

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=list(batch_keys) + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
