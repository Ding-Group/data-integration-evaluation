"""
Run CCA on a dataset

Usage:
    run_cca.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --seed=<int>           Random seed. [default: 0]
"""

from os.path import join
import os
os.environ["R_HOME"] = (
    "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/lib/R"
)

from docopt import docopt
import scanpy as sc
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

    k_weight = min(100, adata.obs[batch_key_final].value_counts().min())

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
        r('obj[["RNA"]] <- split(obj[["RNA"]], f = obj$batch_vec, layer = "counts")')
        r("obj <- NormalizeData(obj)")
        r("obj <- FindVariableFeatures(obj)")
        r("obj <- ScaleData(obj)")
        r("obj <- RunPCA(obj)")

        r(f'obj <- IntegrateLayers(object = obj, method = CCAIntegration, orig.reduction = "pca", new.reduction = "X_integration", k.weight={k_weight})')

        emb = r('Embeddings(obj, "X_integration")')

    adata.obsm[EMB_KEY] = emb

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
