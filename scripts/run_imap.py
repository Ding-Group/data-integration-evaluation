#!/usr/bin/env python

"""
Run IMAP on a dataset

Usage:
    run_imap.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --n-epochs=<int>       Number of epochs. [Default: 150]
    --latent-dim=<int>     Number of latent dimensions. [Default: 256]
    --seed=<int>           Random seed. [Default: 0]
"""

from os.path import join

import anndata as ad
import imap
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata, combine_batch
from utils.hvg_utils import select_hvgs


def run_imap(adata, batch_keys, use_hvg, hvg_method, n_epochs, latent_dim, seed):
    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)

    _, batch_key_final = combine_batch(adata_copy, batch_keys)
    adata_copy.X = adata_copy.X.toarray()

    EC, ec_data = imap.stage1.iMAP_fast(adata_copy, key=batch_key_final, n_epochs=n_epochs, latent_dim=latent_dim, seed=seed)
    output_results = imap.stage2.integrate_data(adata_copy, ec_data, key=batch_key_final, n_epochs=n_epochs, seed=seed)

    adata_copy.X = output_results
    sc.pp.pca(adata_copy)

    adata.obsm[EMB_KEY] = adata_copy.obsm['X_pca']
    return adata


def main():
    args = docopt(__doc__)
    print(args)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    n_epochs = int(args["--n-epochs"])
    latent_dim = int(args["--latent-dim"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_imap(adata, batch_keys, use_hvg, hvg_method, n_epochs, latent_dim, seed)
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
