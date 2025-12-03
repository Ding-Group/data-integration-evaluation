"""
Run Harmony on a dataset

Usage:
    run_harmony.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection.
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join

import scanpy as sc
import anndata as ad

from docopt import docopt
from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def run_harmony(adata, batch_keys, use_hvg, hvg_method, seed):
    if use_hvg:
        if hvg_method is not None:
            layer = "counts" if "v3" in hvg_method else None
            sc.pp.highly_variable_genes(
                adata, flavor=hvg_method, layer=layer, n_top_genes=2000
            )
        adata_copy = adata[:, adata.var.highly_variable].copy()
    sc.tl.pca(adata_copy)
    sc.external.pp.harmony_integrate(
        adata_copy, batch_keys, random_state=seed, adjusted_basis=EMB_KEY
    )
    adata.obsm[EMB_KEY] = adata_copy.obsm[EMB_KEY]
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]

    set_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_harmony(adata, batch_keys, use_hvg, hvg_method, seed)
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
