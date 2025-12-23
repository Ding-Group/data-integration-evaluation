"
Run scMerge from R

Usage:
    run_scmerge.R -o <path> (-b <batch>)... [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Output directory
    -b <batch>             One or more batch keys.
    --no-use-hvg           Do not use HVG selection. [default: FALSE]
    --ruvk=<int>           ruvK (number of unwanted variation factors removed). [default: 20]
    --k-pseudobulk=<int>   k_pseudoBulk. [default: 30]
    --k-celltype=<int>     k_celltype. [default: 10]
    --seed=<int>           Random seed. [default: 0]
" -> doc


library(anndataR)
library(SingleCellExperiment)
library(scMerge)
library(scater)
library(data.table)
library(docopt)


combine_batch <- function(sce, batch_keys) {
    # Ensure input is a vector of column names
    if (is.character(batch_keys) && length(batch_keys) == 1) {
        return(list(sce, colData(sce)[[batch_keys]], batch_keys))
    }

    keys <- as.character(batch_keys)

    # Check that all columns exist
    missing_keys <- setdiff(keys, colnames(colData(sce)))
    if (length(missing_keys) > 0) {
    stop(sprintf("Columns not found in colData(sce): %s", paste(missing_keys, collapse = ", ")))
    }

    # Generate a unique column name
    new_col <- make_unique_colname(colnames(colData(sce)))

    # Combine batch columns into a tuple-like string
    combined <- apply(as.data.frame(colData(sce)[, keys, drop = FALSE]), 1, function(x) {
        paste(x, collapse = "_")
    })

    colData(sce)[[new_col]] <- as.factor(combined)

    return(list(sce, colData(sce)[[new_col]], new_col))
}

# Helper function to generate a unique column name
make_unique_colname <- function(existing_cols, prefix = "combined") {
    new_col <- prefix
    i <- 1
    while (new_col %in% existing_cols) {
    new_col <- paste0(prefix, "_", i)
    i <- i + 1
    }
    return(new_col)
}


#' The main script function
main <- function() {
    args <- docopt::docopt(doc)
    print(args)

    out_dir <- args[['-o']]
    batch_keys <- args[['-b']]
    use_hvg <- !args[['--no-use-hvg']]
    ruvk <- args['--ruvk']
    k_pseudoBulk <- args['--k-pseudobulk']
    k_celltype <- args['--k-celltype']
    seed <- args[['--seed']]
    file <- args[['<file>']]
    
    print("Loading data")
    obj <- read_h5ad(file, as = 'SingleCellExperiment')
    logcounts(obj) <- assays(obj)$X
    result <- combine_batch(obj, batch_keys)
    obj <- result[[1]]
    batch_col <- result[[3]]

    print(batch_col)

    hvg <- rownames(obj)[rowData(obj)$highly_variable]
    if (!use_hvg) {
        hvg <- rownames(obj)
    }

    print("Running integration")
    scMerge2_res <- scMerge2(
        exprsMat = logcounts(obj),
        batch = obj[[batch_col]],
        chosen.hvg = hvg,
        verbose = TRUE
    )

    print("Writing outputs")
    assay(obj, "scMerge2") <- scMerge2_res$newY
    obj <- scater::runPCA(obj, exprs_values = 'scMerge2')
    fwrite(reducedDim(obj, "PCA"), file = paste0(out_dir, "/EMB.txt"), sep = ",")
    print("Done!")
}

if (sys.nframe() == 0) {
    main()
}