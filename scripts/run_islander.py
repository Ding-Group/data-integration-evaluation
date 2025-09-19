#!/usr/bin/env python

"""
Run Islander (supervised MLP with mixup) on a dataset

Usage:
    run_islander.py --out-dir=<path> --batch-key=<key>... --cell-type-key=<key> [options] <file>

Options:
    -h --help                   Show this screen.
    --out-dir=<path>            Path to output directory.
    --batch-key=<key>...        One or more batch keys.
    --cell-type-key=<key>       Cell type label key (supervision).
    --no-use-hvg                Do not use HVG selection.
    --hvg-method=<method>       HVG selection method. Default uses the pre-computed HVGs.
    --hvg-n=<int>               Number of HVGs if computing [default: 1000].
    --epochs=<int>              Training epochs [default: 10].
    --batch-size=<int>          Training batch size [default: 256].
    --infer-batch-size=<int>    Batch size for embedding extraction [default: 2048].
    --lr=<float>                Learning rate [default: 1e-3].
    --mixup-alpha=<float>       Mixup alpha [default: 0.4].
    --emb-dim=<int>             Embedding dimension [default: 16].
    --seed=<int>                Random seed [default: 0].
"""

from os.path import join
from typing import List

import numpy as np
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from docopt import docopt

from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


class NPDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]


def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
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


def _ensure_combined_batch(adata: ad.AnnData, batch_keys: List[str]) -> str:
    if isinstance(batch_keys, (list, tuple)) and len(batch_keys) > 1:
        combo = "_batch_combined"
        adata.obs[combo] = (
            adata.obs[batch_keys].astype(str).agg("|".join, axis=1).astype("category")
        )
        return combo
    return batch_keys[0]


def _to_dense_counts(adata: ad.AnnData) -> np.ndarray:
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32, copy=False)


def run_islander(
    adata: ad.AnnData,
    batch_keys: List[str],
    cell_type_key: str,
    use_hvg: bool,
    hvg_method: str,
    hvg_n: int,
    epochs: int,
    batch_size: int,
    infer_batch_size: int,
    lr: float,
    mixup_alpha: float,
    emb_dim: int,
    seed: int,
) -> ad.AnnData:
    if use_hvg:
        if hvg_method is not None:
            layer = "counts" if ("v3" in hvg_method) else None
            sc.pp.highly_variable_genes(adata, flavor=hvg_method, layer=layer, n_top_genes=int(hvg_n), subset=True)
        else:
            adata = adata[:, adata.var.highly_variable].copy()
    batch_key = _ensure_combined_batch(adata, batch_keys)
    assert cell_type_key in adata.obs.columns
    ad_prep = adata.copy()
    if "counts" in ad_prep.layers:
        ad_prep.X = ad_prep.layers["counts"].copy()
    sc.pp.normalize_total(ad_prep, target_sum=1e4)
    sc.pp.log1p(ad_prep)
    X = _to_dense_counts(ad_prep)
    labels_str = ad_prep.obs[cell_type_key].astype(str).values
    classes = np.unique(labels_str)
    cls2id = {c: i for i, c in enumerate(classes)}
    y = np.array([cls2id[s] for s in labels_str], dtype=np.int64)
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
    with torch.no_grad():
        for i in range(0, X.shape[0], int(infer_batch_size)):
            xb = torch.from_numpy(X[i : i + int(infer_batch_size)]).to(device)
            emb = model(xb, return_embedding=True)
            all_emb.append(emb.detach().cpu())
    emb_mat = torch.cat(all_emb, dim=0).numpy().astype("float32")
    adata.obsm[EMB_KEY] = emb_mat
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["--out-dir"]
    batch_keys = args["--batch-key"]
    cell_type_key = args["--cell-type-key"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    hvg_n = int(args["--hvg-n"])
    epochs = int(args["--epochs"])
    batch_size = int(args["--batch-size"])
    infer_batch_size = int(args["--infer-batch-size"])
    lr = float(args["--lr"])
    mixup_alpha = float(args["--mixup-alpha"])
    emb_dim = int(args["--emb-dim"])
    seed = int(args["--seed"])

    set_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running Islander")
    adata = run_islander(
        adata=adata,
        batch_keys=batch_keys,
        cell_type_key=cell_type_key,
        use_hvg=use_hvg,
        hvg_method=hvg_method,
        hvg_n=hvg_n,
        epochs=epochs,
        batch_size=batch_size,
        infer_batch_size=infer_batch_size,
        lr=lr,
        mixup_alpha=mixup_alpha,
        emb_dim=emb_dim,
        seed=seed,
    )

    print("Adding UMAP")
    add_umap(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=list(batch_keys) + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
