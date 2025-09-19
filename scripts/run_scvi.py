#!/usr/bin/env python

"""
Run scVI on a dataset

Usage:
    run_scvi.py --out-dir=<path> --batch-key=<key>... --cell-type-key=<key> [options] <file>

Options:
    -h --help              Show this screen.
    --out-dir=<path>       Path to output directory.
    --batch-key=<key>...   One or more batch keys.
    --cell-type-key=<key>  Cell type label key.
    --no-use-hvg           Do not use HVG selection.
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --seed=<int>           Random seed.
"""

from os.path import join

import scanpy as sc
import anndata as ad
import scvi

from docopt import docopt
from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def run_scvi(adata, batch_keys, use_hvg, hvg_method, seed):
    if use_hvg:
        if hvg_method is not None:
            layer = 'counts' if 'v3' in hvg_method else None
            sc.pp.highly_variable_genes(adata, flavor=hvg_method, layer=layer, n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable].copy()

    scvi.model.SCVI.setup_anndata(adata, layer='counts', categorical_covariate_keys=batch_keys)
    model = scvi.model.SCVI(adata)
    model.train()

    adata.obsm[EMB_KEY] = model.get_latent_representation()
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["--out-dir"]
    batch_keys = args['--batch-key']
    cell_type_key = args['--cell-type-key']
    seed = int(args["--seed"])
    use_hvg = not args["--use-hvg"]
    hvg_method = args["--hvg-method"]

    set_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_scvi(adata, batch_keys, use_hvg, hvg_method, seed)
    print("Adding UMAP")
    add_umap(adata)
    print("Saving data")
    output = minimise_anndata(adata, obs=batch_keys+[cell_type_key], obsm=[EMB_KEY, 'X_umap'])
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
