#!/usr/bin/env python

"""
Run scPairing on a dataset

Usage:
    run_scpairing.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --emb_dim=<int>        Embedding dimension. [Default: 10]
    --epochs=<int>         Number of epochs. [Default: 300]
    --seed=<int>           Random seed. [Default: 0]
"""

import os
from os.path import join

import numpy as np
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata
from scPairing import scPairing


def run_scpairing(rna, atac, batch_keys, emb_dim, epochs, seed):
    rna_copy = rna.copy()
    sc.external.pp.harmony_integrate(rna_copy, batch_keys)
    sc.external.pp.harmony_integrate(atac, batch_keys)
    model = scPairing(
        rna_copy, atac,
        "rna", "atac",
        batch_col=batch_keys[0],
        transformed_obsm=['X_pca_harmony', 'X_pca_harmony'],
        use_decoder=False,
        emb_dim=emb_dim,
        seed=seed
    )

    model.train(
        epochs=epochs,
        ckpt_dir=None
    )

    latents = model.get_latent_representation()
    rna.obsm[EMB_KEY] = np.concatenate(latents, axis=1)

    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    emb_dim = int(args['--emb_dim'])
    epochs = int(args['--epochs'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_scpairing(rna_adata, atac_adata, batch_keys, emb_dim, epochs, seed)
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
