#!/usr/bin/env python

"""
Run MOFA on a dataset

Usage:
    run_mofa.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --n_factors=<int>      Number of factors. [Default: 20]
    --seed=<int>           Random seed. [Default: 0]
"""

from os.path import join

import muon as mu
import scvi
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata


def run_mofa(rna, atac, batch_keys, n_factors, seed):
    scvi.settings.seed = seed
    rna_copy = rna[:, rna.var.highly_variable].copy()
    _, final_batch_key = combine_batch(rna, batch_keys)

    atac = atac[:, atac.var.highly_variable].copy()
    mdata = mu.MuData({'rna': rna_copy, 'atac': atac})
    mdata.obs['batch'] = rna_copy.obs[final_batch_key]

    mu.tl.mofa(mdata, gpu_mode=False, n_factors=n_factors, groups_label="batch", seed=seed, save_data=False)
    rna.obsm[EMB_KEY] = mdata.obsm['X_mofa']
    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    n_factors = int(args['--n_factors'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_mofa(rna_adata, atac_adata, batch_keys, n_factors, seed)
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
