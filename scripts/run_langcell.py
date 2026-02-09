#!/usr/bin/env python

"""
Run LangCell on a dataset

Usage:
    run_geneformer.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --seed=<int>           Random seed. [Default: 0]
"""

import os
from os.path import join
import sys

import anndata as ad
import pandas as pd
from docopt import docopt
from datasets import load_from_disk
import torch.nn as nn
import torch.nn.functional as F
import torch
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata

sys.path.insert(0, '/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/LangCell/')
from transformers import BertTokenizer, BertModel
from data_preprocess.utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
from data_preprocess.utils import LangCellTranscriptomeTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader


class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output


def run_langcell(adata, cell_type_key):
    if 'ensembl_id' not in adata.var.columns:
        mart = pd.read_csv(
            "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/data/gene_name_to_ensembl.csv",
            index_col=0,
        )
        adata_copy = adata[:, mart.index.intersection(adata.var_names)].copy()
        adata_copy.var["ensembl_id"] = mart.loc[adata_copy.var_names]["Gene stable ID"]
    else:
        adata_copy = adata.copy()

    adata_copy.X = adata_copy.layers['counts'].copy()
    adata_copy.obs['n_counts'] = adata_copy.X.sum(axis=1)
    adata_copy.obs['barcode'] = adata.obs.index

    tk = LangCellTranscriptomeTokenizer(dict([(k, k) for k in adata_copy.obs.keys()]), nproc=4)
    tokenized_cells, cell_metadata = tk.tokenize_anndata(adata_copy)
    tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)

    tokenized_dataset.save_to_disk(os.path.join(os.environ['TMPDIR'], 'data.dataset'))

    model = BertModel.from_pretrained('/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/LangCell/ckpt/cell_bert')
    model.pooler = Pooler(model.config, pretrained_proj='/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/LangCell/ckpt/cell_proj.bin', proj_dim=256)
    proj = model.pooler.proj
    # model = model.module
    model = model.to("cuda")

    text_pretrained_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    tokenizer = BertTokenizer.from_pretrained(text_pretrained_model)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  

    def cell_encode(cell_input_ids, cell_atts):
        cell = model(cell_input_ids.to("cuda"), cell_atts.to("cuda"))
        cell_last_h = cell.last_hidden_state
        cell_pooler = cell.pooler_output
        return cell_last_h, cell_pooler


    dataset = load_from_disk(os.path.join(os.environ['TMPDIR'], 'data.dataset'))
    types = list(set(dataset[cell_type_key]))
    type2num = dict([(t, i) for i, t in enumerate(types)])

    def classes_to_ids(example):
        example["label"] = type2num[example[cell_type_key]]
        return example
    
    testdataset = dataset.map(classes_to_ids, num_proc=16)
    remove_columns = testdataset.column_names
    remove_columns.remove('input_ids')
    remove_columns.remove('label')
    testdataset = testdataset.remove_columns(remove_columns)
    batchsize = 32
    collator = DataCollatorForCellClassification()
    dataloader = DataLoader(testdataset, batch_size=batchsize, collate_fn=collator, shuffle=False)

    cell_embs = torch.zeros(len(testdataset), 256)
    model.eval()
    with torch.no_grad():
        for i, d in tqdm(enumerate(dataloader)):
            _, cellemb = cell_encode(d['input_ids'], d['attention_mask']) # batchsize * 256
            cell_embs[i * batchsize: (i + 1) * batchsize] = cellemb.cpu()

    df = pd.DataFrame(cell_embs.numpy(), index=dataset['barcode'])
    df.loc[adata.obs_names]
    adata.obsm[EMB_KEY] = cell_embs.numpy()

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
    adata = run_langcell(adata, cell_type_key)
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
