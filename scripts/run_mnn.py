#!/usr/bin/env python

"""
Run MNN (Mutual Nearest Neighbors) correction on a dataset

Usage:
    run_mnn.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --no-use-hvg           Do not use HVG selection.
    --hvg-method=<method>  HVG selection method. Default uses the pre-computed HVGs.
    --n-pcs=<int>          Number of PCs after MNN [default: 50].
    --seed=<int>           Random seed [default: 0].
"""

from os.path import join
import os

import scanpy as sc
from docopt import docopt
from utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata
from utils.hvg_utils import select_hvgs

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

    adata = sc.read(file)
    _, batch_key_final = combine_batch(adata, batch_keys)

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

            r = ro.r
            r.assign("sce", adata_copy)
            r.assign("n_pcs", n_pcs)
            r('logcounts(sce) <- assays(sce)[["X"]]')
            r("sce_pre_list <- list()")
            r(f"""
for (s in unique(colData(sce)${batch_key_final})) {{
    sce_pre_list[[s]] <- sce[, colData(sce)${batch_key_final} == s]
}}""")
            # Don't need to subset because the subset is done above.
            r("mnn_out <- fastMNN(sce_pre_list, d = n_pcs, BNPARAM = BiocNeighbors::AnnoyParam())")
            obj = r("reducedDims(mnn_out)$corrected")
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
