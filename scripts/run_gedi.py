#!/usr/bin/env python

"""
Run GEDI on a dataset

Usage:
    run_gedi.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help               Show this screen.
    -o <path>               Path to output directory.
    -b <batch>              One or more batch keys.
    -c <key>                Cell type label key.
    --iterations=<int>      Number of iterations. [default: 50]
    --latent-dim=<int>      Number of latent dims. [default: 5]
    --mode=<str>            Mode to use. [default: Bsphere]
    --oi-shrinkage=<float>  Shrinkage multiplier for oi. [default: 0.001]
    --seed=<int>            Random seed. [default: 0]
"""

import os
from os.path import join

os.environ["R_HOME"] = (
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/CondaEnv/gedi-env/lib/R"
)

import anndata as ad
import pandas as pd
from docopt import docopt
from utils.adata_utils import EMB_KEY, add_umap, combine_batch, minimise_anndata


def main():
    args = docopt(__doc__)
    print(args)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    iterations = int(args["--iterations"])
    latent_dim = int(args["--latent-dim"])
    mode = args["--mode"]
    oi_shrinkage = float(args["--oi-shrinkage"])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    batch, batch_key_final = combine_batch(adata, batch_keys)
    batch = batch.cat.codes

    import anndata2ri
    import rpy2.robjects as ro
    from rpy2.robjects import conversion, default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr

    print("Running integration")
    converter = (
        default_converter
        + numpy2ri.converter
        + pandas2ri.converter
        + anndata2ri.converter
    )
    with conversion.localconverter(converter):
        r = ro.r

        importr("SingleCellExperiment")
        importr("scuttle")
        importr("anndataR")
        importr("GEDI")
        importr("scran")
        importr("scater")

        r.assign("batch_vec", batch)
        r(f'sce <- read_h5ad("{file}", as = "SingleCellExperiment")')
        r("sce$batch_vec <- batch_vec")

        r('sum_genes<- rowSums(assay(sce, "counts")>3)')
        r("genes_use<- sum_genes>3")
        r("genes_use<- names(which(genes_use))")
        r("sce<- sce[genes_use,]")
        r("meta<- data.frame(colData(sce))")
        r('raw_counts<- as(as.matrix(assay(sce, "counts")), "dgCMatrix")')
        r('model <- new("GEDI")')

        r(f"""
model$setup(Samples = sce$batch_vec, # Vector indicating which sample belongs to each cell
            colData=meta, # Metadata (optional)
            M = raw_counts, # Expression data
            K = {latent_dim}, # Number of latent variables to use
            mode = "{mode}", # Modes to use: Either Bsphere (hyperellipsoid) or Bl2 (hyperplane)
            oi_shrinkage = {oi_shrinkage} # Shrinkage multiplier for oi. In here we use 0.001, to better accommodated the mean abundance differences that exist between multiple scRNA-seq technologies.
            ) 
          """)

        r(f"model$initialize.LVs(randomSeed = {seed})")
        r(f"model$optimize(iterations={iterations})")
        r("meta<- meta[model$aux$cellIDs,]")
        r("svd_res <- svd.gedi( model )")
        r("embedding_res_svd<- svd_res$v %*% diag(svd_res$d)")
        embs = r("embedding_res_svd")
        rownames = r("rownames(embedding_res_svd)")
        embs = pd.DataFrame(data = embs, index=pd.Index(data=rownames))

        embs = embs.loc[adata.obs_names]
        adata.obsm[EMB_KEY] = embs.to_numpy()

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
