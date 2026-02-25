#!/usr/bin/env python

"""
Run CLEAR on a dataset

Usage:
    run_clear.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --epochs=<int>         Number of epochs to run. [Default: 100]
    --batch-size=<int>     Training batch size. [Default: 256]
    --low-dim=<int>        Dimension of low-dim representations. [Default: 128]
    --pcl-r=<int>          Queue size; number of negative pairs. [Default: 16384]
    --aug-prob=<float>     Augmentation probability. [Default: 0.5]
    --cluster-name=<str>   Clustering method. [Default: kmeans]
    --num-cluster=<int>    Number of clusters. [Default: -1]
    --seed=<int>           Random seed. [Default: 0]
"""

import os
import subprocess
from os.path import join

import anndata as ad
import numpy as np
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata
from project_utils.hvg_utils import select_hvgs


def run_clear(
    adata,
    file,
    use_hvg,
    hvg_method,
    epochs,
    batch_size,
    low_dim,
    pcl_r,
    aug_prob,
    cluster_name,
    num_cluster,
    seed,
):
    tmpdir = os.environ['TMPDIR']
    adata.X = adata.X.astype('float32')
    # tmpdir = './'
    if use_hvg:
        adata = select_hvgs(adata, use_hvg, hvg_method, subset=True)
        adata.write(os.path.join(tmpdir, 'adata.h5ad'))
        file = os.path.join(tmpdir, 'adata.h5ad')
    
    subprocess.run([
        'python', '/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/CLEAR/CLEAR.py',
        f'--input_h5ad_path={file}', '--epochs', str(epochs), '-b', str(batch_size), '--low_dim', str(low_dim),
        '--pcl_r', str(pcl_r), '--aug_prob', str(aug_prob), '--cluster_name', cluster_name, '--num_cluster', str(num_cluster),
        '--seed', str(seed), f'--save_dir={tmpdir}'
    ])

    _, filename = os.path.split(file)
    dataset_name, _ = os.path.splitext(filename)

    embs = np.loadtxt(os.path.join(tmpdir, f'CLEAR/feature_CLEAR_{dataset_name}.csv'), delimiter=',')
    adata.obsm[EMB_KEY] = embs
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    epochs = int(args["--epochs"])
    batch_size = int(args["--batch-size"])
    low_dim = int(args["--low-dim"])
    pcl_r = int(args["--pcl-r"])
    aug_prob = float(args["--aug-prob"])
    cluster_name = args["--cluster-name"]
    num_cluster = int(args["--num-cluster"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running integration")
    adata = run_clear(
        adata,
        file,
        use_hvg,
        hvg_method,
        epochs,
        batch_size,
        low_dim,
        pcl_r,
        aug_prob,
        cluster_name,
        num_cluster,
        seed,
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
