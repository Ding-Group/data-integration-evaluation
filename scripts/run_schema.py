#!/usr/bin/env python

"""
Run Schema on a dataset

Usage:
    run_schema.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --min_corr=<float>     Min Corr. [Default: 0.9]
    --seed=<int>           Random seed. [Default: 0]
"""

from os.path import join

import numpy as np
import pandas as pd
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from schema import SchemaQP


def run_schema(rna, atac, batch_keys, min_corr, seed):
    np.random.seed(seed)
    rna_copy = rna[:, rna.var.highly_variable].copy()
    _, final_batch_key = combine_batch(rna_copy, batch_keys)

    sqp = SchemaQP(min_corr)
    df_gene_exp = pd.DataFrame(rna_copy.X.toarray(), index=rna_copy.obs.index, columns=rna_copy.var_names)
    atac_50d = atac.obsm['X_pca']
    batch_id = rna_copy.obs[final_batch_key]

    mod_X = sqp.fit_transform(
        df_gene_exp, # gene expression dataframe: rows=cells, cols=genes
        [ atac_50d, batch_id],  # batch_info can be a pd.Series or np.array. rows=cells
        [ 'feature_vector', 'categorical'],
        [ 1, -1]) # maximize combination of (agreement with ATAC-seq + disagreement with batch_id)

    rna.obsm[EMB_KEY] = mod_X
    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    min_corr = float(args['--min_corr'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_schema(rna_adata, atac_adata, batch_keys, min_corr, seed)
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
