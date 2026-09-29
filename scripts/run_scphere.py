"""
Run scPhere (SCPHERE VAE) on a dataset

Usage:
    run_scphere.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help                   Show this screen.
    -o <path>                   Path to output directory.
    -b <batch>                  One or more batch keys.
    -c <key>                    Cell type label key.
    --no-use-hvg                Do not use HVG selection. [Default: False]
    --hvg-method=<method>       HVG selection method. Default uses the pre-computed HVGs.
    --hvg-n=<int>               Number of HVGs if computing [default: 2000].
    --epochs=<int>              Training epochs [default: 100].
    --batch-size=<int>          Minibatch size [default: 128].
    --lr=<float>                Learning rate [default: 1e-3].
    --emb-dim=<int>             Embedding dimension (latent dim) [default: 10].
    --encoder-layer=<ints>      Comma-separated encoder hidden sizes.
    --decoder-layer=<ints>      Comma-separated decoder hidden sizes.
    --seed=<int>                Random seed [default: 0].
"""

import sys
sys.path.append('/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scPhere')

import random
from os.path import join

import anndata as ad
import numpy as np
import torch
from docopt import docopt
from scphere.model.vae import SCPHERE
from scphere.util.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
from project_utils.adata_utils import EMB_KEY, add_umap, dense_counts, minimise_anndata
from project_utils.hvg_utils import select_hvgs


def parse_layer_arg(value):
    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return None

    return [int(item.strip()) for item in text.split(",") if item.strip()]


def run_scphere(
    adata: ad.AnnData,
    batch_keys: list[str],
    use_hvg: bool,
    hvg_method,
    hvg_n: int,
    epochs: int,
    batch_size: int,
    lr: float,
    emb_dim: int,
    encoder_layer,
    decoder_layer,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed: int = 0
) -> ad.AnnData:
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    adata = select_hvgs(
        adata, use_hvg=use_hvg, hvg_method=hvg_method, hvg_n=hvg_n, subset=True
    )

    X = dense_counts(adata)  # (n_cells, n_genes)

    if len(batch_keys) >= 1:
        batch_vec = np.vstack(
            [adata.obs[col].astype("category").cat.codes for col in batch_keys]
        ).T
        batch_vec = torch.Tensor(batch_vec).long()
        n_batch = [len(np.unique(adata.obs[col])) for col in batch_keys]
    # else:
    #     batch_vec = adata.obs[batch_keys[0]].astype("category").cat.codes.values
    #     n_batch = len(np.unique(adata.obs[batch_keys[0]]))
    else:
        batch_vec = torch.zeros((adata.n_obs, 1))
        n_batch = 0

    model = SCPHERE(
        n_gene=X.shape[1],
        n_batch=n_batch,  # 0 (no batch) or K
        z_dim=int(emb_dim),  # latent dim
        encoder_layer=encoder_layer,
        decoder_layer=decoder_layer
    )
    model.to(device)
    model.train()

    trainer = Trainer(
        model=model,
        x=X,
        batch_id=batch_vec,
        max_epoch=int(epochs),
        mb_size=int(batch_size),
        learning_rate=float(lr),
    )
    trainer.train()

    model.eval()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(batch_vec))
    loader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=False,      # shuffle every epoch
        drop_last=False    # keep final smaller batch if not divisible
    )

    embs = []
    for _, (data, batch) in enumerate(loader):
        model.forward(data.to(device), batch.to(device))
        z_mu = model.z_mu.detach().cpu().numpy()
        embs.append(z_mu)
    z = np.concatenate(embs)

    adata.obsm[EMB_KEY] = z
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    hvg_n = int(args["--hvg-n"])
    epochs = int(args["--epochs"])
    batch_size = int(args["--batch-size"])
    lr = float(args["--lr"])
    emb_dim = int(args["--emb-dim"])
    encoder_layer = parse_layer_arg(args["--encoder-layer"])
    decoder_layer = parse_layer_arg(args["--decoder-layer"])
    seed = int(args["--seed"])

    print(torch.cuda.is_available())

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running scPhere")
    adata = run_scphere(
        adata=adata,
        batch_keys=batch_keys,
        use_hvg=use_hvg,
        hvg_method=hvg_method,
        hvg_n=hvg_n,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        emb_dim=emb_dim,
        encoder_layer=encoder_layer,
        decoder_layer=decoder_layer,
        seed=seed
    )

    print("Adding UMAP")
    add_umap(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=(list(batch_keys) if batch_keys else []) + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
