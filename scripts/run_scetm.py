#!/usr/bin/env python

"""
Run scETM on a dataset (notebook-style)

Usage:
    run_scetm.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [Default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --topics=<int>         Number of topics (q_topics) [default: 100].
    --epochs=<int>         Training epochs [default: 100].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
from typing import List

import anndata as ad
from docopt import docopt

from project_utils.adata_utils import add_umap, minimise_anndata, combine_batch, EMB_KEY
from utils.hvg_utils import select_hvgs

from scETM import scETM as SCETM, UnsupervisedTrainer
from scETM.trainers import set_seed

def run_scetm(
    adata: ad.AnnData,
    batch_keys: List[str],
    cell_type_key: str,
    use_hvg: bool,
    hvg_method: str,
    topics: int,
    epochs: int,
    seed: int,
) -> ad.AnnData: 
    adata = select_hvgs(adata, use_hvg, hvg_method, subset=True)

    _, batch_key = combine_batch(adata, batch_keys)
    adata.obs['batch_indices'] = adata.obs[batch_key].cat.codes
    adata.obs['cell_types'] = adata.obs[cell_type_key]

    set_seed(seed)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SCETM(
        n_trainable_genes=adata.n_vars,
        n_batches=adata.obs['batch_indices'].nunique(),
        n_topics=int(topics),
        device=device,
    )

    trainer = UnsupervisedTrainer(model, adata)
    trainer.train(n_epochs=int(epochs), save_model_ckpt=False, eval=False)
    
    model.get_all_embeddings_and_nll(adata)

    adata.obsm[EMB_KEY] = adata.obsm['theta']
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    topics = int(args["--topics"])
    epochs = int(args["--epochs"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running scETM")
    adata = run_scetm(
        adata=adata,
        batch_keys=batch_keys,
        cell_type_key=cell_type_key,
        use_hvg=use_hvg,
        hvg_method=hvg_method,
        topics=topics,
        epochs=epochs,
        seed=seed,
    )

    print("Adding UMAP")
    add_umap(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=list(batch_keys) + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
