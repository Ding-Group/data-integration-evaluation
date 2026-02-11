#!/usr/bin/env python

"""
Run scMM on a dataset

Usage:
    run_scmm.py -o <path> (-b <batch>)...  -c <key> [options] <file1> <file2>

Options:
    -h --help              Show this screen.
    -o <path>              Path to output directory.
    -b <batch>             One or more batch keys.
    -c <key>               Cell type label key.
    --latent_dim=<int>     Latent dimension. [Default: 20]
    --epochs=<int>         Number of epochs. [Default: 50]
    --seed=<int>           Random seed. [Default: 0]
"""

from os.path import join

import sys
sys.path.insert(0, '/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scMM/src')

import os
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import mkdtemp
from collections import defaultdict

import scipy
import models
import objectives
import datetime
from utils import Timer, save_vars, EarlyStopping
from datasets import RNA_Dataset, ATAC_Dataset
import torch
from torch.utils.data import Subset
from torch import optim

from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata


def run_scmm(rna, atac, latent_dim, epochs, seed):
    rna_copy = rna[:, rna.var.highly_variable].copy()
    atac = atac[:, atac.var.highly_variable].copy()

    class Args():
        pass

    args = Args()
    args.seed = seed
    args.no_cuda = False
    args.experiment = "benchmark"
    args.model = 'rna_atac'
    args.dataset_path = Path(join(os.environ['TMPDIR'], 'data'))
    args.pre_trained = False

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    runId = datetime.datetime.now().isoformat()
    experiment_dir = Path(join(os.environ['TMPDIR'], 'experiment'))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))

    dataset_path = args.dataset_path
    if args.model == 'rna_atac':
        modal = 'ATAC-seq'
    elif args.model == 'rna_protein':
        modal = 'CITE-seq'
    rna_path = dataset_path / 'RNA-seq'
    modal_path = dataset_path / '{}'.format(modal)

    (dataset_path / 'RNA-seq').mkdir(parents=True)
    (dataset_path / 'ATAC-seq').mkdir(parents=True)

    scipy.io.mmwrite(dataset_path / 'RNA-seq/RNA_count.mtx', rna_copy.layers['counts'].T)
    with open(dataset_path / 'RNA-seq/barcode.tsv', 'w') as f:
        for b in rna_copy.obs_names:
            f.write(f'{b}\n')
    with open(dataset_path / 'RNA-seq/gene.tsv', 'w') as f:
        for b in rna_copy.var_names:
            f.write(f'{b}\n')
    scipy.io.mmwrite(dataset_path / 'ATAC-seq/ATAC_count.mtx', atac.layers['counts'].T)
    with open(dataset_path / 'ATAC-seq/barcode.tsv', 'w') as f:
        for b in atac.obs_names:
            f.write(f'{b}\n')
    with open(dataset_path / 'ATAC-seq/gene.tsv', 'w') as f:
        for b in atac.var_names:
            f.write(f'{b}\n')

    r_dataset = RNA_Dataset(str(rna_path)) 
    args.r_dim = r_dataset.data.shape[1]

    modal_dataset = ATAC_Dataset(str(modal_path)) if args.model == 'rna_atac' else RNA_Dataset(str(modal_path))
    args.p_dim = modal_dataset.data.shape[1]

    num_cell = r_dataset.data.shape[0]
    t_size = np.round(num_cell*0.80).astype('int')
    t_id = np.random.choice(a=num_cell, size=t_size, replace=False) 
    s_id = np.delete(range(num_cell),t_id)
    torch.save(t_id, runPath + '/t_id.rar')
    torch.save(s_id, runPath + '/s_id.rar')

    train_dataset = [Subset(r_dataset, t_id), Subset(modal_dataset, t_id)]
    test_dataset = [Subset(r_dataset, s_id), Subset(modal_dataset, s_id)]
    full_dataset = [r_dataset, modal_dataset]

    args.latent_dim = latent_dim
    args.num_hidden_layers = 2
    args.r_hidden_dim = 100
    args.p_hidden_dim = 20
    args.learn_prior = True
    args.llik_scaling = 1
    args.batch_size = 32
    args.lr = 0.0001
    args.obj = 'm_elbo_naive_warmup'
    args.deterministic_warmup = 25
    args.epochs = epochs
    args.print_freq = 0

    pretrained_path = ""
    modelC = getattr(models, 'VAE_{}'.format(args.model))
    print(args)

    model = modelC(args).to(device)
    # torch.save(args,runPath+'/args.rar')

    #Dataloader
    train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, device=device)
    test_loader = model.getDataLoaders(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)

    # preparation for training
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.lr, amsgrad=True)
    objective = getattr(objectives,args.obj) 
    s_objective = getattr(objectives,args.obj)
    full_loader = model.getDataLoaders(full_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)

    def train(epoch, agg, W):
        model.train()
        b_loss = 0
        for i, dataT in enumerate(train_loader):
            beta = (epoch - 1) / W  if epoch <= W else 1
            if dataT[0].size()[0] == 1:
                continue
            data = [d.to(device) for d in dataT] #multimodal
            optimizer.zero_grad()
            loss = -objective(model, data, beta)
            loss.backward()
            optimizer.step()
            b_loss += loss.item()
            if args.print_freq > 0 and i % args.print_freq == 0:
                print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
        agg['train_loss'].append(b_loss / len(train_loader.dataset))
        print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))
        return b_loss

    def test(epoch, agg, W):
        model.eval()
        b_loss = 0
        with torch.no_grad():
            for i, dataT in enumerate(test_loader):
                beta = (epoch - 1) / W  if epoch <= W else 1
                if dataT[0].size()[0] == 1:
                    continue
                data = [d.to(device) for d in dataT]
                loss = -s_objective(model, data, beta)
                b_loss += loss.item()
        agg['test_loss'].append(b_loss / len(test_loader.dataset))
        print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))
        
    def get_latent(dataloader, train_test, runPath):
        model.eval()
        with torch.no_grad():
            if args.model == 'rna_atac':
                modal = ['rna', 'atac'] 
            elif args.model == 'rna_protein':
                modal = ['rna', 'protein']
            pred = []
            for i, dataT in enumerate(dataloader):
                data = [d.to(device) for d in dataT]
                lats = model.latents(data, sampling=False)
                if i == 0:
                    pred = lats
                else:
                    for m,lat in enumerate(lats):
                        pred[m] = torch.cat([pred[m], lat], dim=0) 

            for m,lat in enumerate(pred):
                lat = lat.cpu().detach().numpy()
                lat = pd.DataFrame(lat)
                lat.to_csv('{}/lat_{}_{}.csv'.format(runPath, train_test, modal[m]))
            mean_lats = sum(pred)/len(pred)
            mean_lats = mean_lats.cpu().detach().numpy()
    #         mean_lats = pd.DataFrame(mean_lats)
    #         mean_lats.to_csv('{}/lat_{}_mean.csv'.format(runPath,train_test))
        return mean_lats
    
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=10, verbose=True) 
        W = args.deterministic_warmup
        start_early_stop = W
        for epoch in range(1, args.epochs + 1):
            b_loss = train(epoch, agg, W)
            if torch.isnan(torch.tensor([b_loss])):
                break          
            test(epoch, agg, W)
            save_vars(agg, runPath + '/losses.rar')
            if epoch > start_early_stop: 
                early_stopping(agg['test_loss'][-1], model, runPath)
            if early_stopping.early_stop:
                print('Early stopping')
                break
    latents = get_latent(full_loader, 'full', runPath)

    rna.obsm[EMB_KEY] = latents
    return rna


def main():
    args = docopt(__doc__)
    rna = args["<file1>"]
    atac = args["<file2>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    latent_dim = int(args['--latent_dim'])
    epochs = int(args['--epochs'])
    seed = int(args["--seed"])

    print("Loading data")
    rna_adata = sc.read(rna)
    atac_adata = sc.read(atac)
    print("Running integration")
    adata = run_scmm(rna_adata, atac_adata, latent_dim, epochs, seed)
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
