#!/usr/bin/env python

"""
Run scETM on a dataset (notebook-style)

Usage:
    run_scetm.py --out-dir=<path> --batch-key=<key>... --cell-type-key=<key> [options] <file>

Options:
    -h --help              Show this screen.
    --out-dir=<path>       Path to output directory.
    --batch-key=<key>...   One or more batch keys (will be combined if multiple).
    --cell-type-key=<key>  Cell type label key.
    --no-use-hvg           Do not use HVG selection.
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --topics=<int>         Number of topics (q_topics) [default: 100].
    --epochs=<int>         Training epochs [default: 100].
    --batch-size=<int>     Mini-batch size for embedding extraction [default: 128].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
from typing import List

import numpy as np
import anndata as ad
import scanpy as sc
from docopt import docopt

from utils.random import set_seed
from utils.adata_utils import add_umap, minimise_anndata, EMB_KEY


def _ensure_combined_batch(adata: ad.AnnData, batch_keys: List[str]) -> str:
    """Combine multiple batch columns into one categorical column if needed."""
    if isinstance(batch_keys, (list, tuple)) and len(batch_keys) > 1:
        combo = "_batch_combined"
        adata.obs[combo] = (
            adata.obs[batch_keys].astype(str).agg("|".join, axis=1).astype("category")
        )
        return combo
    return batch_keys[0]


def _ensure_batch_indices(adata: ad.AnnData, batch_key: str) -> str:
    """Make an integer-coded column for scETM (like in the notebook)."""
    col = "_batch_indices"
    if not np.issubdtype(adata.obs[batch_key].dtype, np.integer):
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")
        adata.obs[col] = adata.obs[batch_key].cat.codes.astype(int)
    else:
        adata.obs[col] = adata.obs[batch_key].astype(int)
    return col


def _to_dense_x(adata: ad.AnnData):
    """Return dense float32 matrix using counts layer if present."""
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32, copy=False)


def _get_all_theta(model, adata: ad.AnnData, device: str, batch_size: int) -> np.ndarray:
    """Match the notebook: run model in eval mode and collect theta in batches."""
    import torch

    model.eval()
    X = _to_dense_x(adata)
    X_t = torch.tensor(X, device=device)

    thetas = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            x_batch = X_t[i : i + batch_size]
            q_delta = model.q_delta(x_batch)
            theta_b = model.q_theta(q_delta)
            thetas.append(theta_b.detach().cpu())
    return torch.cat(thetas, dim=0).numpy()


def run_scetm(
    adata: ad.AnnData,
    batch_keys: List[str],
    use_hvg: bool,
    hvg_method: str,
    topics: int,
    epochs: int,
    extract_batch_size: int,
    seed: int,
) -> ad.AnnData: 
    if use_hvg:
        if hvg_method is not None:
            layer = "counts" if ("v3" in hvg_method) else None
            sc.pp.highly_variable_genes(adata, flavor=hvg_method, layer=layer, n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable].copy()

    batch_key = _ensure_combined_batch(adata, batch_keys)
    batch_idx_key = _ensure_batch_indices(adata, batch_key)

    from scETM import scETM as SCETM, UnsupervisedTrainer

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SCETM(
        n_trainable_genes=adata.n_vars,
        n_batches=adata.obs[batch_key].nunique(),
        q_topics=int(topics),
        device=device,
    )

    
    adata.obs["cell_types"] = adata.obs.get("cell_type", adata.obs.get("CellType", None))
    adata.obs["batch_indices"] = adata.obs[batch_idx_key]

    trainer = UnsupervisedTrainer(model, adata)
    trainer.train(n_epochs=int(epochs))

    
    theta = _get_all_theta(model, adata, device=device, batch_size=int(extract_batch_size))

    adata.obsm[EMB_KEY] = theta
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["--out-dir"]
    batch_keys = args["--batch-key"]
    cell_type_key = args["--cell-type-key"]
    use_hvg = not args["--no-use-hvg"] if "--no-use-hvg" in args else not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    topics = int(args["--topics"])
    epochs = int(args["--epochs"])
    extract_bs = int(args["--batch-size"])
    seed = int(args["--seed"])

    set_seed(seed)

    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running scETM")
    adata = run_scetm(
        adata=adata,
        batch_keys=batch_keys,
        use_hvg=use_hvg,
        hvg_method=hvg_method,
        topics=topics,
        epochs=epochs,
        extract_batch_size=extract_bs,
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
