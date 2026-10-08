"""
Run CYCLONE on a dataset

Usage:
    run_cyclone.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --vae-z-dim=<int>      VAE z dim. [default: 32]
    --h-dim=<int>          h dim. [default: 16]
    --epochs=<int>         Number of training epochs. [default: 30]
    --seed=<int>           Random seed. [default: 0]
"""

from os.path import join
import sys

sys.path.insert(
    0,
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/CYCLONE",
)

import anndata as ad
import random
import torch
import numpy as np
from cyclone.train import cyclone

from docopt import docopt
from project_utils.adata_utils import add_umap, minimise_anndata, EMB_KEY, combine_batch
from project_utils.hvg_utils import select_hvgs


def run_cyclone(adata, batch_keys, cell_type_key, use_hvg, hvg_method, vae_z_dim, h_dim, epochs, seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    adata_copy = select_hvgs(
        adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True, copy=True
    )
    batch, _ = combine_batch(adata_copy, batch_keys)
    adata_copy.obs["Batch"] = batch
    adata_copy.obs["celltype"] = adata.obs[cell_type_key]

    adata_copy.X = adata_copy.X.toarray()
    adata_copy = cyclone(
        adata_copy,
        batch_size=256,
        vae_z_dim=vae_z_dim,
        h_dim=h_dim,
        encode_layers=[1024, 512, 256],
        decode_layers=[256, 512, 1024],
        lr_vae=0.0002,
        train_epoch=epochs,
    )

    adata.obsm[EMB_KEY] = adata_copy.obsm['cyclone_emb']

    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    vae_z_dim = int(args["--vae-z-dim"])
    h_dim = int(args["--h-dim"])
    epochs = int(args["--epochs"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_cyclone(
        adata, batch_keys, cell_type_key, use_hvg, hvg_method, vae_z_dim, h_dim, epochs, seed
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
