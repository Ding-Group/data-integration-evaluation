"""
Run GenePT-W on a dataset

Usage:
    run_geneptw.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --seed=<int>           Random seed. [default: 0]
"""

import pickle
from os.path import join

import anndata as ad
import pandas as pd
from docopt import docopt
from sklearn.preprocessing import normalize
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata


def run_genept(adata):
    if 'gene_name' in adata.var.columns:
        adata_copy = adata[:, ~adata.var['gene_name'].isna()].copy()
        adata_copy.var = adata_copy.var.set_index('gene_name')
        adata_copy.var_names = adata_copy.var_names.astype('object')
        adata_copy.var_names_make_unique()
    else:
        adata_copy = adata.copy()

    with open('/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/GenePT/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle', 'rb') as f:
        r = pickle.load(f)

    r = pd.DataFrame(r)
    adata_copy = adata_copy[:, adata_copy.var_names.intersection(r.columns)].copy()
    r = r[adata_copy.var_names.intersection(r.columns)].to_numpy()
    embs = adata_copy.X @ r.T
    embs = normalize(embs, norm='l2')

    adata.obsm[EMB_KEY] = embs
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
    adata = run_genept(adata)
    print("Adding UMAP")
    add_umap(adata)
    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
