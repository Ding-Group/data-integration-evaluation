"""
Run scETM on a dataset

Usage:
    run_scetm.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help                       Show this screen.
    -o <path>                       Path to output directory.
    -b <batch>                      One or more batch keys.
    -c <key>                        Cell type label key.
    --no-use-hvg                    Do not use HVG selection. [default: False]
    --hvg-method=<method>           HVG selection method. Default uses the pre-computed HVGs.
    --topics=<int>                  Number of topics (n_topics) [default: 50].
    --trainable-gene-emb-dim=<int>  Trainable dimensions of the gene embedding [default: 400].
    --hidden-sizes=<ints>           Comma-separated hidden sizes of the encoder [default: 128].
    --lr=<float>                    Initial learning rate [default: 0.005].
    --dropout-prob=<float>          Dropout probability in the cell encoder [default: 0.1].
    --epochs=<int>                  Training epochs [default: 100].
    --seed=<int>                    Random seed [default: 0].
"""

from os.path import join
from typing import List

import anndata as ad
from docopt import docopt

from project_utils.adata_utils import add_umap, minimise_anndata, combine_batch, EMB_KEY
from project_utils.hvg_utils import select_hvgs

from scETM import scETM as SCETM, UnsupervisedTrainer
from scETM.trainers import set_seed


def parse_hidden_sizes(value: str) -> List[int]:
    hidden_sizes = [int(x) for x in value.split(",") if x]
    if not hidden_sizes:
        raise ValueError("--hidden-sizes must contain at least one integer")
    return hidden_sizes


def run_scetm(
    adata: ad.AnnData,
    batch_keys: list[str],
    cell_type_key: str,
    use_hvg: bool,
    hvg_method: str,
    topics: int,
    trainable_gene_emb_dim: int,
    hidden_sizes: list[int],
    lr: float,
    dropout_prob: float,
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
        trainable_gene_emb_dim=int(trainable_gene_emb_dim),
        hidden_sizes=list(hidden_sizes),
        dropout_prob=float(dropout_prob),
        device=device,
    )

    trainer = UnsupervisedTrainer(model, adata, init_lr=float(lr))
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
    trainable_gene_emb_dim = int(args["--trainable-gene-emb-dim"])
    hidden_sizes = parse_hidden_sizes(args["--hidden-sizes"])
    lr = float(args["--lr"])
    dropout_prob = float(args["--dropout-prob"])
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
        trainable_gene_emb_dim=trainable_gene_emb_dim,
        hidden_sizes=hidden_sizes,
        lr=lr,
        dropout_prob=dropout_prob,
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
