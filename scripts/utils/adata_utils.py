import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

EMB_KEY = 'X_integration'


def add_umap(adata):
    """
    Add UMAP to the AnnData.
    """
    sc.pp.neighbors(adata, use_rep=EMB_KEY)
    sc.tl.umap(adata)


# https://github.com/theislab/atlas-feature-selection-benchmark/blob/main/bin/functions/anndata.py
def minimise_anndata(
    adata, X=False, obs=None, layers=None, var=None, uns=None, obsm=None, varm=None, obsp=None
):
    """
    Minimise an AnnData object by removing unnecessary data

    Parameters
    ----------
    adata
        AnnData object to minimise
    X
        Whether to keep the X matrix
    layers
        List of keys in adata.layers to keep
    obs
        List of columns in adata.obs to keep
    var
        List of columns in adata.var to keep
    uns
        List of keys in adata.uns to keep
    obsm
        List of keys in adata.obsm to keep
    varm
        List of keys in adata.varm to keep

    Returns
    -------
    Minimised AnnData object
    """

    from anndata import AnnData

    adata_min = AnnData(shape=adata.shape)
    adata_min.var_names = adata.var_names.copy()
    adata_min.obs_names = adata.obs_names.copy()

    if X:
        adata_min.X = adata.X.copy()

    if layers is not None:
        for key in layers:
            adata_min.layers[key] = adata.layers[key].copy()

    if obs is not None:
        for key in obs:
            adata_min.obs[key] = adata.obs[key]

    if var is not None:
        for key in var:
            adata_min.var[key] = adata.var[key]

    if uns is not None:
        for key in uns:
            adata_min.uns[key] = adata.uns[key]

    if obsm is not None:
        for key in obsm:
            adata_min.obsm[key] = adata.obsm[key]

    if varm is not None:
        for key in varm:
            adata_min.varm[key] = adata.varm[key]

    if obsp is not None:
        for key in obsp:
            adata_min.obsp[key] = adata.obsp[key]

    adata_min = adata_min.copy()

    return adata_min


def dense_counts(adata: ad.AnnData, counts_layer: str = "counts") -> np.ndarray:
    X = adata.layers[counts_layer]
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32, copy=False)


def get_unique_colname(df, base_name="combined_batch"):
    name = base_name
    i = 1
    while name in df.columns:
        name = f"{base_name}_{i}"
        i += 1
    return name


#  Batch helper 
def combine_batch(
    adata: ad.AnnData,
    batch_keys,
):
    keys = [batch_keys] if isinstance(batch_keys, str) else list(batch_keys)
    if len(keys) == 1:
        return adata.obs[keys[0]], keys[0]
    new_col = get_unique_colname(adata.obs)
    adata.obs[new_col] = adata.obs[keys].apply(tuple, axis=1).astype('category')
    return adata.obs[new_col], new_col


def split_by_batch(adata: ad.AnnData, batch_key: str):
    """Split AnnData into a list of views/copies per batch level."""
    adatas = []
    for b in adata.obs[batch_key].unique():
        adatas.append(adata[adata.obs[batch_key] == b].copy())
    return adatas
