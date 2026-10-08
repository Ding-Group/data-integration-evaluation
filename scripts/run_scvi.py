"""
Run scVI on a dataset

Usage:
    run_scvi.py -o <path> (-b <batch>)...  -c <key> [options] <file>

Options:
    -h --help                 Show this screen.
    -o <path>                 Path to output directory.
    -b <batch>                One or more batch keys.
    -c <key>                  Cell type label key.
    --no-use-hvg              Do not use HVG selection. [default: False]
    --hvg-method=<method>     HVG selection method. Default uses the pre-computed HVGs.
    --n_latent=<int>          Number of latent dimensions for embedding. [default: 10]
    --n_hidden=<int>          Width of hidden layer. [default: 128]
    --n_layers=<int>          Number of NN layers. [default: 1]
    --dropout_rate=<float>    Dropout rate. [default: 0.1]
    --lr=<float>              Learning rate. [default: 0.001]
    --batch_size=<int>        Batch size. [default: 128]
    --seed=<int>              Random seed. [default: 0]
"""

from os.path import join

import anndata as ad
import scvi
from docopt import docopt
from project_utils.adata_utils import EMB_KEY, add_umap, minimise_anndata
from project_utils.hvg_utils import select_hvgs


def run_scvi(adata, batch_keys, use_hvg, hvg_method, n_latent, n_layers, n_hidden, dropout_rate, batch_size, lr, seed):
    scvi.settings.seed = seed

    adata_copy = select_hvgs(adata, use_hvg=use_hvg, hvg_method=hvg_method, subset=True)

    new_batch_keys= []
    for b in batch_keys:
        adata_copy.obs.rename({b: b.replace(".", "_")}, axis=1, inplace=True)
        new_batch_keys.append(b.replace(".", "_"))

    scvi.model.SCVI.setup_anndata(
        adata_copy,
        layer="counts",
        batch_key=new_batch_keys[0],
        categorical_covariate_keys=new_batch_keys[1:],
    )
    model = scvi.model.SCVI(
        adata_copy,
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_layers=n_layers,
        dropout_rate=dropout_rate
    )
    model.train(
        batch_size=batch_size,
        plan_kwargs={
            "lr": float(lr),
        }
    )

    adata.obsm[EMB_KEY] = model.get_latent_representation()
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    n_latent = int(args['--n_latent'])
    n_layers = int(args['--n_layers'])
    n_hidden = int(args['--n_hidden'])
    dropout_rate = float(args['--dropout_rate'])
    batch_size = int(args['--batch_size'])
    lr = float(args['--lr'])
    seed = int(args["--seed"])

    print("Loading data")
    adata = ad.read_h5ad(file)
    print("Running integration")
    adata = run_scvi(adata, batch_keys, use_hvg, hvg_method, n_latent, n_layers, n_hidden, dropout_rate, batch_size, lr, seed)
    print("Adding UMAP")
    add_umap(adata)
    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=batch_keys + [cell_type_key],
        obsm=[EMB_KEY, "X_umap"],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()
