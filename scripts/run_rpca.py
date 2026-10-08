"""
Run RPCA on a dataset

Usage:
    run_rpca.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --seed=<int>           Random seed. [default: 0]
"""

import os
from os.path import join

os.environ["R_HOME"] = (
    "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/lib/R"
)

import anndata as ad
import scanpy as sc
import pandas as pd
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])

    adata = sc.read(file)
    batch, batch_key_final = combine_batch(adata, batch_keys)
    batch = batch.cat.codes

    k_weight = min(100, batch.value_counts().min() // 10 - 1)

    import anndata2ri
    import rpy2.robjects as ro
    from rpy2.robjects import conversion, default_converter, numpy2ri
    from rpy2.robjects.packages import importr

    converter = default_converter + numpy2ri.converter + anndata2ri.converter
    with conversion.localconverter(converter):
        importr("Seurat")
        importr("anndataR")

        r = ro.r
        r.assign("batch_vec", batch)
        r(f"set.seed({seed})")
        r(f'obj <- read_h5ad("{file}", as = "Seurat")')
        r("obj$batch_vec <- batch_vec")
        r('obj.list <- SplitObject(obj, split.by = "batch_vec")')
        r("""
obj.list <- lapply(X = obj.list, FUN = function(x) {
    x <- NormalizeData(x)
    x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
})
          """)
        r("""
features <- SelectIntegrationFeatures(object.list = obj.list)
obj.list <- lapply(X = obj.list, FUN = function(x) {
    x <- ScaleData(x, features = features, verbose = FALSE)
    x <- RunPCA(x, features = features, verbose = FALSE)
})
          """)
        r('anchors <- FindIntegrationAnchors(object.list = obj.list, anchor.features = features, reduction = "rpca")')
        r(f"combined <- IntegrateData(anchorset = anchors, k.weight = {k_weight})")

        mat = r('combined[["integrated"]]@data')
        obs_names = r("colnames(combined)")

    new_adata = ad.AnnData(X=mat.T, obs=pd.DataFrame(index=obs_names))
    new_adata = new_adata[adata.obs_names]
    sc.pp.pca(new_adata)
    adata.obsm[EMB_KEY] = new_adata.obsm["X_pca"]

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
