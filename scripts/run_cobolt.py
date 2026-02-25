#!/usr/bin/env python

"""
Run Cobolt on a dataset

Usage:
    run_cobolt.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --n_latent=<int>       Number of latent dimensions. [Default: 10]
    --n_epochs=<int>       Number of epochs. [Default: 100]
    --seed=<int>           Random seed. [Default: 0]
"""
from os.path import join

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch
from cobolt.model import Cobolt
from cobolt.utils import MultiomicDataset, SingleData
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata


def run_cobolt(rna, atac, batch_keys, n_latent, n_epochs, seed):
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    _, final_batch_key = combine_batch(rna, batch_keys)
    combine_batch(atac, batch_keys)

    datas = []
    for rep in rna.obs[final_batch_key].unique():
        rna_sub = rna[rna.obs[final_batch_key] == rep]
        rna_data = SingleData(
            "GeneExpr",
            "10X",
            rna_sub.var.index.to_numpy(),
            rna_sub.layers['counts'].toarray(),
            rna_sub.obs.index.to_numpy()
        )
        rna_data.filter_features(upper_quantile=0.99, lower_quantile=0.7)
        rna_data.count = scipy.sparse.csr_matrix(rna_data.count)
        datas.append(rna_data)
        
        atac_sub = atac[atac.obs[final_batch_key] == rep]
        atac_data = SingleData(
            "ChromAccess",
            "10X",
            atac_sub.var.index.to_numpy(),
            atac_sub.layers['counts'].toarray(),
            atac_sub.obs.index.to_numpy()
        )
        atac_data.filter_features(upper_quantile=0.99, lower_quantile=0.7)
        atac_data.count = scipy.sparse.csr_matrix(atac_data.count)
        datas.append(atac_data)
    
    multi_dt = MultiomicDataset.from_singledata(*datas)
    batch_size = 128 if rna.n_obs % 128 != 1 else 129
    model = Cobolt(dataset=multi_dt, lr=1e-4, n_latent=n_latent, batch_size=batch_size)
    model.train(num_epochs=n_epochs)

    model.calc_all_latent()
    latent = model.get_all_latent()
    latent_raw = model.get_all_latent(correction=False)

    embs = pd.DataFrame(latent_raw[0][latent_raw[2] == 'joint'], index=latent_raw[1][latent_raw[2]=='joint'])
    embs.index = embs.index.str.split('~', n=1).str[1]
    embs = embs.loc[rna.obs.index]
    rna.obsm[EMB_KEY] = embs.to_numpy()

    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    n_latent = int(args['--n_latent'])
    n_epochs = int(args['--n_epochs'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_cobolt(rna_adata, atac_adata, batch_keys, n_latent, n_epochs, seed)
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
