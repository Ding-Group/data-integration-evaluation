#!/usr/bin/env python

"""
Run scAlign on a dataset

Usage:
    run_scalign.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --ccs=<int>            Number of CCs to use. [Default: 15]
    --steps=<int>          Number of iterations. [Default: 15000]
    --batch-size=<int>     Batch size. [Default: 300]
    --perplexity=<int>     Perplexity. [Default: 30]
    --num-dim=<int>        Number of dimensions. [Default: 32]
    --seed=<int>           Random seed. [Default: 0]
"""

import os
from os.path import join

os.environ["R_HOME"] = (
    "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/CondaEnv/scalign-env/lib/R"
)

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
    ccs = int(args['--ccs'])
    steps = int(args['--steps'])
    batch_size = int(args['--batch-size'])
    perplexity = int(args['--perplexity'])
    num_dim = int(args['--num-dim'])
    seed = int(args["--seed"])

    print("Loading data")
    adata = sc.read(file)
    batch, batch_key_final = combine_batch(adata, batch_keys)
    batch = batch.cat.codes

    import anndata2ri
    import rpy2.robjects as ro
    from rpy2.robjects import conversion, default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr

    print("Running integration")
    converter = default_converter + numpy2ri.converter + pandas2ri.converter + anndata2ri.converter
    with conversion.localconverter(converter):
        r = ro.r
        
        importr("reticulate")
        r('use_python("/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/CondaEnv/scalign-env/bin/python")')

        importr("scAlign")
        importr("Seurat")
        importr("SingleCellExperiment")
        importr("anndataR")
        importr("tensorflow")
        
        r.assign("batch_vec", batch)
        r(f'obj <- read_h5ad("{file}", as = "Seurat")')
        r("obj$batch_vec <- batch_vec")
        r('obj.list <- SplitObject(obj, split.by = "batch_vec")')
        r('''
obj.list <- lapply(X = obj.list, FUN = function(x) {
    x <- NormalizeData(x)
    x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
    x <- ScaleData(x)
})
          ''')
        r("var.features.list <- lapply(obj.list, VariableFeatures)")
        r("genes.use <- Reduce(intersect, var.features.list)")

        r('''
obj.list <- lapply(X = obj.list, FUN = function(x) {
    sce <- SingleCellExperiment(
      assays = list(counts = x@assays$RNA$counts[genes.use,],
                    logcounts  = x@assays$RNA$data[genes.use,],
                    scale.data = x@assays$RNA$scale.data[genes.use,])
    )
    return(sce)
})
          ''')
        r(f'''
scAlignObj = scAlignCreateObject(sce.objects = obj.list,
                                 genes.use = genes.use,
                                 pca.reduce = FALSE,
                                 cca.reduce = TRUE,
                                 ccs.compute = {ccs},
                                 project.name="scAlign")
          ''')
        r(f'''
scAlignObj = scAlignMulti(scAlignObj,
                        options=scAlignOptions(steps={steps},
                                               log.every={steps // 3},
                                               batch.size={batch_size},
                                               perplexity={perplexity},
                                               norm=TRUE,
                                               batch.norm.layer=FALSE,
                                               architecture="large",  ## 3 layer neural network
                                               num.dim={num_dim},
                                               seed={seed}),            ## Number of latent dimensions
                        encoder.data="MultiCCA",
                        supervised='none',
                        run.encoder=TRUE,
                        run.decoder=FALSE,
                        log.results=FALSE,
                        device="GPU")
          ''')
        
        mat = r("reducedDim(scAlignObj, 'ALIGNED-MultiCCA')")
        obs_names = r("colnames(scAlignObj)")

        r('try({tensorflow::tf$keras$backend$clear_session()}, silent=TRUE)')
        r('try({tensorflow::tf$compat$v1$reset_default_graph()}, silent=TRUE)')
        r('gc()')

        import gc
        gc.collect()
        
        mat = pd.DataFrame(data = mat, index = obs_names)
        mat = mat.loc[adata.obs_names]

        adata.obsm[EMB_KEY] = mat.to_numpy()

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

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
