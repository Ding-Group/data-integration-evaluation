#!/usr/bin/env python

"""
Run MOJITOO on a dataset

Usage:
    run_mojitoo.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --seed=<int>           Random seed. [Default: 0]
"""

import sys
from os.path import join

sys.path.insert(0, '/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/MOJITOO/pymojitoo/')

import mojitoo
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata


def run_mojitoo(rna, atac):
    atac.obsm['X_rna'] = rna.obsm['X_pca']
    mojitoo.mojitoo(atac, ["X_rna", "X_pca"])
    rna.obsm[EMB_KEY] = atac.obsm['X_mojitoo']

    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_mojitoo(rna_adata, atac_adata)
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
