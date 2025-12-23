#!/usr/bin/env python

"""
Run scVI on a dataset

Usage:
    run_scvi.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --seed=<int>           Random seed. [Default: 0]
"""

from os.path import join

import anndata as ad
import scvi

from docopt import docopt
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY
from utils.hvg_utils import select_hvgs


def run_scvi(adata, batch_keys, use_hvg, hvg_method, seed):
    scvi.settings.seed = seed

    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)

    new_batch_keys= []
    for b in batch_keys:
        adata_copy.obs.rename({b: b.replace(".", "_")}, axis=1, inplace=True)
        new_batch_keys.append(b.replace(".", "_"))

    scvi.model.SCVI.setup_anndata(
        adata_copy,
        layer="counts",
        batch_key=new_batch_keys[0],
        categorical_covariate_keys=new_batch_keys[1:],
    )
    model = scvi.model.SCVI(adata_copy)
    model.train()

    adata.obsm[EMB_KEY] = model.get_latent_representation()
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_scvi(adata, batch_keys, use_hvg, hvg_method, seed)
    print("Adding UMAP")
    add_umap(adata)
    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
