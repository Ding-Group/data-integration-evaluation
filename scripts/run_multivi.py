#!/usr/bin/env python

"""
Run MultiVI on a dataset

Usage:
    run_multivi.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --n_latent=<int>       Number of latent dimensions. [Default: 10]
    --seed=<int>           Random seed. [Default: 0]
"""
import sys
from os.path import join
import muon as mu
import scvi
import scanpy as sc
import numpy as np
from docopt import docopt

from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata

def run_multivi(rna, atac, batch_keys, n_latent, seed):
    scvi.settings.seed = seed

    mdata = mu.MuData({'rna': rna, 'atac': atac})
    scvi.model.MULTIVI.setup_mudata(
    mdata,
        modalities={
            "rna_layer": "rna",
            "atac_layer": "atac",
        },
        categorical_covariate_keys=[f'rna:{bk}' for bk in batch_keys], 
        rna_layer='counts', atac_layer='counts'
    )

    model = scvi.model.MULTIVI(
        mdata,
        n_genes=len(mdata.mod["rna"].var),
        n_regions=len(mdata.mod["atac"].var),
        n_latent=n_latent
    )

    model.train(early_stopping=True)
    rna.obsm[EMB_KEY] = model.get_latent_representation()

    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    n_latent = int(args['--n_latent'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_multivi(rna_adata, atac_adata, batch_keys, n_latent, seed)
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
