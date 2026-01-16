"""
Run scPRINT on a dataset

Usage:
    run_scprint.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --seed=<int>           Random seed. [Default: 0]
"""

from os.path import join

import anndata as ad
import lamindb as ln
import numpy as np
import torch
from docopt import docopt
from scdataloader import Preprocessor
from scdataloader.utils import load_genes
from scprint import scPrint
from scprint.tasks import Embedder
from utils.adata_utils import EMB_KEY, add_umap, minimise_anndata

torch.set_float32_matmul_precision("medium")


def run_scprint(adata, seed):
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    adata_copy = adata.copy()
    adata_copy.X = adata_copy.layers['counts'].copy()

    adata_copy.obs['organism_ontology_term_id'] = 'NCBITaxon:9606'
    adata_copy.obs['self_reported_ethnicity_ontology_term_id'] = 'unknown'
    adata_copy.obs['sex_ontology_term_id'] = 'unknown'
    adata_copy.obs['disease_ontology_term_id'] = 'unknown'
    adata_copy.obs['cell_type_ontology_term_id'] = 'unknown'
    adata_copy.obs['development_stage_ontology_term_id'] = 'unknown'
    adata_copy.obs['tissue_ontology_term_id'] = 'unknown'
    adata_copy.obs['assay_ontology_term_id'] = 'unknown'

    preprocessor = Preprocessor(do_postp=False)
    adata_copy = preprocessor(adata_copy)

    model_checkpoint_file = "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scPRINT/medium-v1.5.ckpt"
    # make sure that you check if you have a GPU with flashattention or not (see README)
    try:
        m = torch.load(model_checkpoint_file)
    # if not use this instead since the model weights are by default mapped to GPU types
    except RuntimeError:
        m = torch.load(model_checkpoint_file, map_location=torch.device("cpu"))

    # again here by default the model was trained with flash attention, so if you do not have a GPU you will need to replace the attention mechanism with regular attention
    transformer = "flash" if torch.cuda.is_available() else "normal"
    # both are for compatibility issues with different versions of the pretrained model, so we need to load it with the correct transformer
    if "prenorm" in m["hyper_parameters"]:
        m["hyper_parameters"].pop("prenorm")
        torch.save(m, model_checkpoint_file)
    if "label_counts" in m["hyper_parameters"]:
        # you need to set precpt_gene_emb=None otherwise the model will look for its precomputed gene embeddings files although they were already converted into model weights, so you don't need this file for a pretrained model
        model = scPrint.load_from_checkpoint(
            model_checkpoint_file,
            precpt_gene_emb=None,
            classes=m["hyper_parameters"]["label_counts"],
            transformer=transformer,
        )
    else:
        model = scPrint.load_from_checkpoint(
            model_checkpoint_file, precpt_gene_emb=None, transformer=transformer
        )

    # this might happen if you have a model that was trained with a different set of genes than the one you are using in the ontology (e.g. newer ontologies), While having genes in the onlogy not in the model is fine. the opposite is not, so we need to remove the genes that are in the model but not in the ontology
    missing = set(model.genes) - set(load_genes(model.organisms).index)
    if len(missing) > 0:
        print(
            "Warning: some genes missmatch exist between model and ontology: solving...",
        )
        model._rm_genes(missing)

    # again if not on GPU you need to convert the model to float64
    if not torch.cuda.is_available():
        model = model.to(torch.float32)

    # you can perform your inference on float16 if you have a GPU, otherwise use float64
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # the models are often loaded with some parts still displayed as "cuda" and some as "cpu", so we need to make sure that the model is fully on the right device
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Embedder(
        # can work on random genes or most variables etc..
        how="random expr",
        # number of genes to use
        max_len=4000,
        # the model is trained on a minibatch of 64 cells but you can choose whatever
        batch_size=16,
        # for the dataloading
        num_workers=8,
        # we will only use the cell type embedding here.
        pred_embedding=[],
        # , "disease_ontology_term_id"],
        # we will now
        save_every=40_000,
        doclass=False,
        doplot=False,
        dtype=dtype,
    )
    adata_final, _ = embedder(model, adata_copy, cache=False)
    adata.obsm[EMB_KEY] = adata_final.obsm['scprint_emb']
    return adata

def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_scprint(adata, seed)
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
