#!/usr/bin/env python

"""
Run Geneformer on a dataset

Usage:
    run_geneformer.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --seed=<int>           Random seed. [Default: 0]
"""

import os
from os.path import join
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from docopt import docopt
from geneformer import EmbExtractor, TranscriptomeTokenizer
from transformers import set_seed
from utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from utils.hvg_utils import select_hvgs


def run_geneformer(adata, seed):
    set_seed(seed)

    mart = pd.read_csv(
        "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/data/gene_name_to_ensembl.csv",
        index_col=0,
    )
    adata_copy = adata[:, mart.index.intersection(adata.var_names)].copy()
    adata_copy.var["ensembl_id"] = mart.loc[adata_copy.var_names]["Gene stable ID"]

    adata_copy.X = adata_copy.layers["counts"].copy()
    adata_copy.obs["n_counts"] = adata_copy.X.sum(-1).astype(int)

    adata_copy.obs["barcode"] = adata.obs_names
    adata_copy.write(os.path.join(os.environ["TMPDIR"], "adata.h5ad"))

    tk = TranscriptomeTokenizer(
        {"barcode": "barcode"}, nproc=1
    )  # for V1 model, set model_version="V1"
    tk.tokenize_data(
        os.environ["TMPDIR"], os.environ["TMPDIR"], "geneformer_", file_format="h5ad"
    )

    embex = EmbExtractor(
        model_type="Pretrained",
        emb_layer=-1,
        max_ncells=None,
        emb_label=["barcode"],
        labels_to_plot=["barcode"],
        forward_batch_size=24,
        model_version="V2",  # OF NOTE: SET TO V1 MODEL, PROVIDE V1 MODEL PATH IN SUBSEQUENT CODE
        nproc=16,
    )
    embs = embex.extract_embs(
        "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/Geneformer/Geneformer-V2-104M/",  # example V1 fine-tuned model
        os.path.join(os.environ["TMPDIR"], "geneformer_.dataset"),
        os.environ["TMPDIR"],
        "geneformer_result",
    )
    embs = embs.set_index('barcode')
    embs = embs.loc[adata.obs_names]
    adata.obsm[EMB_KEY] = embs.to_numpy()
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_geneformer(adata, seed)
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
