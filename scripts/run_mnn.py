"""
Run MNN (Mutual Nearest Neighbors) correction on a dataset

Usage:
    run_mnn.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection. [default: False]
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --n-pcs=<int>          Number of PCs after MNN [default: 50].
    --k=<int>              Number of nearest neighbors for fastMNN [default: 20].
    --ndist=<float>        Tricube bandwidth multiplier for fastMNN [default: 3].
    --cos-norm             Whether fastMNN performs cosine normalization [default: true].
    --seed=<int>           Random seed [default: 0].
"""

import os
import sys
from os.path import join

import numpy as np
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from project_utils.hvg_utils import select_hvgs

os.environ["R_HOME"] = "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/lib/R" 

def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    n_pcs = int(args["--n-pcs"])
    k = int(args["--k"])
    ndist = float(args["--ndist"])
    cos_norm = args["--cos-norm"]

    adata = sc.read(file)
    batch, batch_key_final = combine_batch(adata, batch_keys)

    adata_copy = select_hvgs(adata, use_hvg, hvg_method, subset=True, copy=True)

    try:
        import anndata2ri
        import rpy2.robjects as ro
        from rpy2.robjects import conversion, default_converter, numpy2ri
        from rpy2.robjects.packages import importr
    except Exception as e:
        raise RuntimeError(
            "rpy2 is required but not available. Install with `pip install rpy2`."
        ) from e

    converter = default_converter + numpy2ri.converter + anndata2ri.converter
    with conversion.localconverter(converter):
        del adata_copy.uns
        try:
            importr("SingleCellExperiment")
            importr("scran")
            importr("scater")
            importr("batchelor")
            importr("anndataR")

            r = ro.r
            r(f'sce <- read_h5ad("{file}", as = "SingleCellExperiment")')
            r.assign("hvg_genes", ro.StrVector(list(adata_copy.var_names)))
            r.assign("n_pcs", n_pcs)
            r.assign("mnn_k", k)
            r.assign("mnn_ndist", ndist)
            r.assign("mnn_cos_norm", cos_norm)
            r.assign("batch_vec", ro.StrVector(batch.astype(str).tolist()))
            r("sce$batch_vec <- factor(batch_vec)")
            r(f'set.seed({seed})')
            r('logcounts(sce) <- assays(sce)[["X"]]')
            r("sce <- sce[hvg_genes, , drop=FALSE]")
            r("sce_pre_list <- list()")
            r("""
for (s in unique(colData(sce)$batch_vec)) {
    sce_pre_list[[s]] <- sce[, colData(sce)$batch_vec == s]
}""")
            r("""
mnn_out <- fastMNN(
    sce_pre_list,
    d = n_pcs,
    k = mnn_k,
    ndist = mnn_ndist,
    cos.norm = mnn_cos_norm,
    BNPARAM = BiocNeighbors::AnnoyParam()
)
""")
            obj = np.asarray(r("reducedDims(mnn_out)$corrected"), dtype=np.float32)
            if obj.shape[0] != adata.n_obs and obj.shape[1] == adata.n_obs:
                obj = obj.T
            adata.obsm[EMB_KEY] = obj

        except Exception as e:
            raise RuntimeError("Running MNN failed.") from e

    print("Adding UMAP")
    add_umap(adata)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"],
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
