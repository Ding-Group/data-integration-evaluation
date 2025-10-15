from typing import Optional, Union, List
import numpy as np
import anndata as ad
import scanpy as sc


# Unified HVG 
def select_hvgs(
    adata: ad.AnnData,
    use_hvg: bool = True,
    hvg_method: Optional[str] = None,  # e.g. "seurat" / "seurat_v3"
    hvg_n: int = 2000,
    counts_layer: str = "counts",
    subset: bool = False,
    copy: bool = False,
) -> ad.AnnData:
    """
    Unified HVG selection:
    if use_hvg:
        if hvg_method is not None:
            compute HVGs with Scanpy (uses counts layer if 'v3' in name)
        then require 'highly_variable' and subset by it (if subset=True)
    If hvg_method is None, assumes HVGs are already present in adata.var['highly_variable'].
    """
    A = adata.copy() if copy else adata
    if not use_hvg:
        return A
    # 计算 HVGs
    if hvg_method is not None:
        layer = counts_layer if "v3" in hvg_method else None
        sc.pp.highly_variable_genes(
            A, flavor=hvg_method, layer=layer, n_top_genes=int(hvg_n),
            subset=False, inplace=True
        )
    # 没有 HVG 信息 → 抛错，避免 silent fail
    if "highly_variable" not in A.var:
        raise ValueError(
            "HVG selection requested but no 'highly_variable' column is present. "
            "Either set hvg_method (e.g. 'seurat' or 'seurat_v3') or precompute HVGs."
        )

    if subset:
        A = A[:, A.var["highly_variable"]].copy()
    return A
