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
    --epochs=<int>              Training epochs [default: 200].
    --batch-size=<int>          Minibatch size [default: 128].
    --lr=<float>                Learning rate [default: 1e-3].
    --emb-dim=<int>             Embedding dimension (latent dim) [default: 16].
    --seed=<int>                Random seed [default: 0].
"""


# 导入 scPhere 的核心两个东西：模型和训练器
#   - SCPHERE: 变分自编码器，用于训练 embedding
#   - Trainer: 封装训练循环和优化过程
# 但是考虑到可能环境里没有装好，所以 try 一下
# 失败的话就会把报错信息留在 _IMPORT_ERROR
from os.path import join
from typing import List

import numpy as np
import anndata as ad
import scanpy as sc
from docopt import docopt

from utils.adata_utils import add_umap, minimise_anndata, dense_counts, EMB_KEY
from utils.hvg_utils import select_hvgs


from scphere.model.vae import SCPHERE
from scphere.util.trainer import Trainer


# 这个函数主要是处理 batch 列的情况
# 如果用户传了多个 batch key，比如病人+医院，就把它们拼接成一列
# 否则只返回单个 batch key
# 拼接的时候用 "|" 这种方式，最后转成分类类型
# def _ensure_combined_batch(adata: ad.AnnData, batch_keys: List[str]) -> str:
#     if isinstance(batch_keys, (list, tuple)) and len(batch_keys) > 1:
#         combo = "_batch_combined"
#         adata.obs[combo] = (
#             adata.obs[batch_keys].astype(str).agg("|".join, axis=1).astype("category")
#         )
#         return combo
#     return batch_keys[0]
def _ensure_combined_batch(adata: ad.AnnData, batch_keys: List[str]):
    """
    Ensure batch key(s) exist in adata.obs.
    - If single key, return that key (string).
    - If multiple keys, return list of keys (tutorial style: keep multi-column batch table).
    """

    # 如果 batch_keys 是字符串（用户只传了一个，比如 "patient"）
    # 就把它转成列表形式 ["patient"]，方便后面统一处理。
    if isinstance(batch_keys, str):
        batch_keys = [batch_keys]
    #  检查每个传入的 key 是否都在 adata.obs 里存在
    missing = [k for k in batch_keys if k not in adata.obs.columns]
    if missing:
        raise KeyError(f"Batch key(s) not found in adata.obs: {missing}")
    # 遍历所有的 batch 列，把它们的数据类型转成 pandas 的 "category"
    for k in batch_keys:
        adata.obs[k] = adata.obs[k].astype("category")
    
    # 如果只有一个 key，就返回这个 key字符串
    # 如果有多个 key，就返回它们的名字列表
    return batch_keys[0] if len(batch_keys) == 1 else batch_keys


# 函数确保我们拿到的是真实的原始 counts
# 有的 AnnData 会把原始 count 放在 adata.layers["counts"]
# 最后把稀疏矩阵转成 dense，因为后面训练用不到稀疏
# 数据类型也转成 float32，这样和 TensorFlow/PyTorch 一致，避免报错
def _dense_counts(adata: ad.AnnData) -> np.ndarray:
    """Return dense float32 raw counts from adata.layers['counts']."""
    if "counts" not in adata.layers:
        raise KeyError(
            "Expected raw counts in adata.layers['counts'], but this layer was not found."
        )
    X = adata.layers["counts"]
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32, copy=False)


# 就是决定要不要用 HVG
# 如果不用，就直接返回原数据
# 如果用，就根据不同的方法来选，eg. seurat_v3
# 最常见的就是挑 top 2000 HVGs，加速训练，也能去掉噪音
# 如果 adata 本身已经有 highly_variable 这一列了，就直接用它来裁剪
# def _select_hvgs_inplace(
#     adata: ad.AnnData,
#     use_hvg: bool,
#     hvg_method: str | None,
#     hvg_n: int,
# ) -> ad.AnnData:
#     if not use_hvg:
#         return adata
#     if hvg_method is not None:
#         layer = "counts" if ("v3" in hvg_method) else None
#         sc.pp.highly_variable_genes(
#             adata, flavor=hvg_method, layer=layer, n_top_genes=int(hvg_n), subset=True
#         )
#     else:
#         if "highly_variable" in adata.var and adata.var["highly_variable"].any():
#             adata = adata[:, adata.var["highly_variable"]].copy()
#     return adata

# at the top of the file





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
    seed: int,
) -> ad.AnnData:
    
    # 先决定要不要裁到 HVG
    # HVG 
    adata = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, hvg_n=hvg_n, subset=True)
    
    # 取原始 counts 矩阵
    # Use raw counts
    X = dense_counts(adata)  # (n_cells, n_genes)

    # 接下来要准备 batch 信息
    # 如果没传 batch 或者数据里没有，就设 n_batch=0，batch_vec 全是 -1
    # Batch vector
    batch_key = _ensure_combined_batch(adata, batch_keys) if batch_keys else None
    if (not batch_key) or (batch_key not in adata.obs.columns):
        # no batch: official/no-book recommended way
        batch_vec = np.zeros(adata.n_obs) * -1
        n_batch = 0
    else:
        cats = adata.obs[batch_key].astype("category")
        adata.obs[batch_key] = cats
        batch_vec = cats.cat.codes.to_numpy()
        n_batch = len(cats.cat.categories)

    # 构建模型
    # n_gene 基因数，n_batch 批次数，z_dim 潜在维度
    # latent_dist="vmf" 表示球面分布
    # observation_dist="nb" 用负二项分布建模 count 数据

    # Train (align arg names)
    # Build model (align with notebook/official names)
    model = SCPHERE(
        n_gene=X.shape[1],
        n_batch=n_batch,                 # 0 (no batch) or K
        z_dim=int(emb_dim),              # latent dim
        latent_dist="vmf",               # spherical
        observation_dist="nb",           # counts ~ NB
        seed=int(seed),
    )

    
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
    z = model.encode(X, batch_vec)
    #确保 embedding 是 NumPy 数组，类型是 float32
    z = np.asarray(z, dtype=np.float32)
    adata.obsm[EMB_KEY] = z
    return adata

# 解析命令行参数，跑 run_scphere，然后保存结果
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
    lr = float(args["--lr"])
    emb_dim = int(args["--emb-dim"])
    seed = int(args["--seed"])

    # set_seed(seed)

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
        seed=seed,
    )

    print("Adding UMAP")
    add_umap(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=(list(batch_keys) if batch_keys else []) + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
