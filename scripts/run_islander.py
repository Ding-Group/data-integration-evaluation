#!/usr/bin/env python

"""
Run Islander (supervised MLP with mixup) on a dataset.
Source: https://github.com/Genentech/Islander/tree/main

Usage:
    run_islander.py -o <path> (-b <batch>)... -c <key> [options] <file>

Options:
    -h --help                   Show this screen.
    -o <path>                   Output directory.
    -b <batch>                  One or more batch keys (first is main).
    -c <key>                    Cell type label key.
    --no-use-hvg                Do not use HVG selection.
    --hvg-method=<method>       HVG selection method. Default uses the pre-computed HVGs.
    --epochs=<int>              Training epochs [default: 10].
    --batch-size=<int>          Training batch size [default: 256].
    --lr=<float>                Learning rate [default: 1e-3].
    --mixup-alpha=<float>       Mixup alpha [default: 0.4].
    --emb-dim=<int>             Embedding dimension [default: 16].
    --seed=<int>                Random seed [default: 0].
"""

import random
from os.path import join
from typing import List

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from docopt import docopt
from torch.utils.data import DataLoader, Dataset
from utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from utils.hvg_utils import select_hvgs


class NPDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class IslanderMLP(nn.Module):
    def __init__(self, in_dim, num_classes, emb_dim=16):
        super().__init__()
        self.fcl = nn.Linear(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fch = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fce = nn.Linear(128, emb_dim)
        self.fco = nn.Linear(emb_dim, num_classes)
    def forward(self, x, return_embedding=False):
        x = F.relu(self.bn1(self.fcl(x)))
        x = F.relu(self.bn2(self.fch(x)))
        emb = self.fce(x)
        logits = self.fco(emb)
        return emb if return_embedding else logits


def run_islander(
    adata: ad.AnnData,
    cell_type_key: str,
    use_hvg: bool,
    hvg_method: str,
    epochs: int,
    batch_size: int,
    lr: float,
    mixup_alpha: float,
    emb_dim: int,
) -> ad.AnnData:
    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)

    X = adata_copy.X.astype('float32').toarray()
    y = adata_copy.obs[cell_type_key].astype('category').cat.codes.to_numpy()
    X = torch.from_numpy(X)
    y = torch.from_numpy(y).to(torch.long)
    print(y)
    classes = np.unique(y)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = NPDataset(X, y)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True, num_workers=0)

    model = IslanderMLP(in_dim=X.shape[1], num_classes=len(classes), emb_dim=int(emb_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs))
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(int(epochs)):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            xb_mixed, ya, yb2, lam = mixup_data(xb, yb, alpha=float(mixup_alpha))
            preds = model(xb_mixed)
            loss = mixup_criterion(criterion, preds, ya, yb2, lam)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    model.eval()
    all_emb = []
    eval_dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False, num_workers=0)
    with torch.no_grad():
        for xb, _ in eval_dl:
            xb = xb.to(device)
            emb = model(xb, return_embedding=True)
            all_emb.append(emb.detach().cpu())

    emb_mat = torch.cat(all_emb, dim=0).numpy().astype("float32")
    adata.obsm[EMB_KEY] = emb_mat
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]

    # short parameters 
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]

    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    epochs = int(args["--epochs"])
    batch_size = int(args["--batch-size"])
    lr = float(args["--lr"])
    mixup_alpha = float(args["--mixup-alpha"])
    emb_dim = int(args["--emb-dim"])
    seed = int(args["--seed"])

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running Islander")
    adata = run_islander(
        adata=adata,
        cell_type_key=cell_type_key,
        use_hvg=use_hvg,
        hvg_method=hvg_method,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        mixup_alpha=mixup_alpha,
        emb_dim=emb_dim,
    )

    print("Adding UMAP")
    add_umap(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=list(batch_keys) + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
