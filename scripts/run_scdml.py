#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run scDML on a dataset

Usage:
    run_scdml.py -o <path> (-b <batch>)... -c <key> [options] <file>

Options:
    -h --help                       Show this screen.
    -o <path>                       Path to output directory.
    -b <batch>                      One or more batch keys (first is main).
    -c <key>                        Cell type label key.
    --resolution=<float>            Louvain resolution. [Default: 3.0]
    --merge-rule=<rule>             scDML merge rule: rule1 | rule2. [Default: rule2]
    --seed=<int>                    Random seed. [Default: 0]
"""

import os
from os.path import join

import anndata as ad
import numpy as np
import scanpy as sc
from docopt import docopt
from scDML import scDMLModel
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata


def run_scdml(
    adata,
    batch_keys,
    cell_type_key,
    resolution,
    merge_rule="rule2",
    seed=0,
):
    # Build neighborhood graph & init clusters on a copy (keeps original counts intact)
    adata_copy = adata.copy()
    sc.pp.scale(adata_copy, max_value=None)
    sc.tl.pca(adata_copy)
    sc.pp.neighbors(adata_copy)
    sc.tl.umap(adata_copy)
    sc.tl.louvain(adata_copy, resolution=resolution, key_added="init_cluster")

    # Run scDML on the original AnnData
    _, batch_key_final = combine_batch(adata_copy, batch_keys)

    scdml = scDMLModel(save_dir=os.environ['TMPDIR'] + '/', verbose=True)

    scdml.integrate(
        adata_copy,
        batch_key=batch_key_final,
        merge_rule=merge_rule,
        expect_num_cluster=len(np.unique(adata_copy.obs[cell_type_key])),
        ncluster_list=[len(np.unique(adata_copy.obs[cell_type_key]))],
        seed=seed,
    )

    # Copy embedding to standard key
    adata.obsm[EMB_KEY] = adata_copy.obsm["X_emb"].copy()

    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]

    resolution = float(args["--resolution"])
    seed = int(args["--seed"])
    merge_rule = args["--merge-rule"]

    print("Load data")
    adata = ad.read_h5ad(file)

    print("Run integration")
    adata = run_scdml(
        adata=adata,
        batch_keys=batch_keys,
        cell_type_key=cell_type_key,
        resolution=resolution,
        merge_rule=merge_rule,
        seed=seed,
    )

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
