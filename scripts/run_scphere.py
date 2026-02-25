#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 用 docstring 定义了命令行用法和参数
# --batch-key 允许多次出现，后面我会把它们合并处理
# 目标让同一条 pipeline 可以跑不同数据集无需改代码
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
    --batch-size=<int>          Minibatch size [default: 512].
    --lr=<float>                Learning rate [default: 1e-3].
    --emb-dim=<int>             Embedding dimension (latent dim) [default: 10].
    --seed=<int>                Random seed [default: 0].
"""

# 导入 scPhere 的核心两个东西：模型和训练器
#   - SCPHERE: 变分自编码器，用于训练 embedding
#   - Trainer: 封装训练循环和优化过程
# 但是考虑到可能环境里没有装好，所以 try 一下
# 失败的话就会把报错信息留在 _IMPORT_ERROR
import sys

sys.path.append('/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scPhere')
import random
from os.path import join
from typing import List

import anndata as ad
import numpy as np
import torch
from docopt import docopt
from scphere.model.vae import SCPHERE
from scphere.util.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
from project_utils.adata_utils import EMB_KEY, add_umap, dense_counts, minimise_anndata
from project_utils.hvg_utils import select_hvgs


# run_scphere
# 把 notebook 里那一大堆步骤收进来，打包成一个函数
# 步骤：HVG -> counts -> batch -> build model -> train -> embedding
def run_scphere(
    adata: ad.AnnData,
    batch_keys: List[str],
    use_hvg: bool,
    hvg_method,
    hvg_n: int,
    epochs: int,
    batch_size: int,
    lr: float,
    emb_dim: int,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
) -> ad.AnnData:
    # 先决定要不要裁到 HVG
    # HVG
    adata = select_hvgs(
        adata, use_hvg=use_hvg, hvg_method=hvg_method, hvg_n=hvg_n, subset=True
    )

    # 取原始 counts 矩阵
    # Use raw counts
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

    # 构建模型
    # n_gene 基因数，n_batch 批次数，z_dim 潜在维度
    # latent_dist="vmf" 表示球面分布
    # observation_dist="nb" 用负二项分布建模 count 数据

    # Train (align arg names)
    # Build model (align with notebook/official names)
    model = SCPHERE(
        n_gene=X.shape[1],
        n_batch=n_batch,  # 0 (no batch) or K
        z_dim=int(emb_dim),  # latent dim
        batch_invariant=True,
        latent_dist="vmf",  # spherical
        observation_dist="nb"  # counts ~ NB
    )
    model.to(device)
    model.train()

    # 创建 Trainer，传入模型、数据、batch
    # 里面设置最大迭代数epochs、batch_size、lr等

    trainer = Trainer(
        model=model,
        x=X,
        batch_id=batch_vec,
        max_epoch=int(epochs),
        mb_size=int(batch_size),
        learning_rate=float(lr),
    )
    trainer.train()

    # 训练好之后调用 model.encode 得到 embedding
    # 把它存到 adata.obsm，方便后面统一评估

    # 调用 scPhere 的 VAE 模型，
    # 把输入的基因表达矩阵 X（dense counts）和批次信息 batch_vec 喂进 encoder 网络。
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


# 解析命令行参数，跑 run_scphere，然后保存结果
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
    seed = int(args["--seed"])

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

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
        emb_dim=emb_dim
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
