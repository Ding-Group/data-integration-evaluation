#!/usr/bin/env python

"""
Run UCE on a dataset

Usage:
    run_uce.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --seed=<int>           Random seed. [Default: 0]
"""

import os
import subprocess
from os.path import join

import anndata as ad
from docopt import docopt
from utils.adata_utils import EMB_KEY, add_umap, minimise_anndata


def run_uce(adata, adata_path):
    tmpdir = os.environ["TMPDIR"]
    adata_copy = adata.copy()
    adata_copy.X = adata_copy.layers['counts'].copy()

    adata_copy.write(join(tmpdir, 'adata.h5ad'))
    # Create symlink to where the model files are to avoid that bug
    subprocess.run(
        [
            "ln",
            "-s",
            "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/UCE/model_files",
            "./model_files",
        ]
    )
    subprocess.run(
        [
            "ln",
            "-s",
            "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/UCE/eval_single_anndata.py",
            "./eval_single_anndata.py",
        ]
    )
    subprocess.run(
        [
            "python",
            "eval_single_anndata.py",
            "--adata_path",
            join(tmpdir, 'adata.h5ad'),
            "--dir",
            tmpdir,
            "--batch_size",
            16,
            "--model_loc",
            "model_files/33l_8ep_1024t_1280.torch",
            "--nlayers",
            "33",
        ]
    )
    result = ad.read_h5ad(
        join(
            tmpdir, 'adata_uce_adata.h5ad'
        )
    )
    adata.obsm[EMB_KEY] = result.obsm['X_uce']
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
    adata = run_uce(adata, file)
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
