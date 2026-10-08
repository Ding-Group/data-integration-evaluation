"""
Run Multigrate on a dataset

Usage:
    run_multigrate.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --z_dim=<int>          Latent dimension. [default: 15]
    --seed=<int>           Random seed. [default: 0]
"""

from os.path import join

import multigrate as mtg
import scvi
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata


def run_multigrate(rna, atac, batch_keys, seed):
    scvi.settings.seed = seed
    rna_copy = rna[:, rna.var.highly_variable].copy()
    atac = atac[:, atac.var.highly_variable].copy()

    adata = mtg.data._preprocessing.organize_multiome_anndatas(
        adatas=[[rna_copy], [atac]],  # a list of anndata objects per modality, RNA-seq always goes first
        layers=[["counts"], [None]],  # if need to use data from .layers, if None use .X
    )
    rna_indices_end = rna.shape[1]
    mtg.model.MultiVAE.setup_anndata(
        adata,
        categorical_covariate_keys=batch_keys,
        rna_indices_end=rna_indices_end,
    )
    vae = mtg.model.MultiVAE(
        adata,
        losses=["nb", "mse"],
    )
    vae.train()
    vae.get_latent_representation()
    latent = adata.obsm['latent'].copy()
    rna.obsm[EMB_KEY] = latent
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
    adata = run_multigrate(rna_adata, atac_adata, batch_keys, seed)
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
