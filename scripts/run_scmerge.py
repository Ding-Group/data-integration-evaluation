#!/usr/bin/env python

"""
Run scMerge on a dataset

Usage:
    run_scmerge.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --ruvk=<int>           ruvK (number of unwanted variation factors removed). [Default: 20]
    --k-pseudobulk=<int>   k_pseudoBulk. [Default: 30]
    --k-celltype=<int>     k_celltype. [Default: 10]
    --seed=<int>           Random seed. [Default: 0]
"""

import os
from os.path import join
import subprocess

os.environ["R_HOME"] = (
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/CondaEnv/scmerge-env/lib/R"
)

import scanpy as sc
import pandas as pd
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    ruvk = int(args["--ruvk"])
    k_pseudobulk = int(args["--k-pseudobulk"])
    k_celltype = int(args["--k-celltype"])
    seed = int(args["--seed"])

    try:
        dir = os.environ["TMPDIR"]
    except KeyError:
        dir = './'

    cmd = ['Rscript', '/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/run_scmerge.R', '-o', dir]
    for b in batch_keys:
        cmd.append('-b')
        cmd.append(b)
    if not use_hvg:
        cmd.append('--no-use-hvg')
    cmd.extend([
        f"--ruvk={ruvk}",
        f"--k-pseudobulk={k_pseudobulk}",
        f"--k-celltype={k_celltype}",
        f"--seed={seed}",
        file
    ])
    subprocess.run(cmd)

    adata = sc.read(file)

    embs = pd.read_csv(f'{dir}/EMB.txt')
    adata.obsm[EMB_KEY] = embs.to_numpy()

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
