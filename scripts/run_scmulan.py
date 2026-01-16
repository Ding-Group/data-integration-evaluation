#!/usr/bin/env python

"""
Run scMulan on a dataset

Usage:
    run_scmulan.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --seed=<int>           Random seed. [Default: 0]
"""

import os
import sys
from os.path import join

sys.path.insert(
    0,
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scMulan/Tutorials/",
)

import anndata as ad
import numpy as np
import scanpy as sc
import scMulan
import torch
from docopt import docopt
from scMulan import GeneSymbolUniform
from utils.adata_utils import EMB_KEY, add_umap, minimise_anndata


def run_scmulan(adata, seed):
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    adata_copy = adata.copy()
    adata_copy.X = adata_copy.layers["counts"].copy()
    adata_GS_uniformed = GeneSymbolUniform(
        input_adata=adata, output_dir=os.environ["TMPDIR"], output_prefix="data"
    )

    sc.pp.normalize_total(adata_GS_uniformed, target_sum=1e4)
    sc.pp.log1p(adata_GS_uniformed)

    ckp_path = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scMulan/Tutorials/ckpt/ckpt_scMulan.pt"

    scml = scMulan.model_inference(ckp_path, adata_GS_uniformed)
    base_process = scml.cuda_count()
    scml.get_cell_types_and_embds_for_adata(
        parallel=True, n_process=base_process
    )  # set parallel = False to use single GPU

    adata.obsm[EMB_KEY] = scml.adata.obsm["X_scMulan"]
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
    adata = run_scmulan(adata, seed)
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
