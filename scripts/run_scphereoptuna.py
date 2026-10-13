"""
Run scPhere on a dataset using Optuna trial hyperparameters

Usage:
    run_scphereoptuna.py -o <path> (-b <batch>)... -c <key> --params-json=<file> [options] <file>

Options:
    -h --help                   Show this screen.
    -o <path>                   Path to output directory.
    -b <batch>                  One or more batch keys.
    -c <key>                    Cell type label key.
    --params-json=<file>        JSON file with hyperparameters
    --no-use-hvg                 Do not use HVG selection. [default: False]
    --hvg-method=<method>        HVG selection method. Default uses the pre-computed HVGs.
    --hvg-n=<int>                Number of HVGs if computing [default: 2000].
    --batch-size=<int>           Minibatch size [default: 128].
    --seed=<int>                 Random seed [default: 0].
"""
from datetime import datetime
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("START", current_time_str)

import sys
sys.path.append('/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/libraries/scPhere')

import json
import random
from os.path import join

import anndata as ad
import scanpy as sc
import numpy as np
import torch
from docopt import docopt
from scphere.model.vae import SCPHERE
from scphere.util.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
from project_utils.adata_utils import EMB_KEY, add_umap, dense_counts, minimise_anndata
from project_utils.hvg_utils import select_hvgs


ENCODER_LAYER_OPTIONS = {
    "128-64-32": [128, 64, 32],
    "256-128-64": [256, 128, 64],
    "128-64": [128, 64],
    "256-128-64-32": [256, 128, 64, 32],
    "128": [128],
    "64-32-16": [64, 32, 16],
}

DECODER_LAYER_OPTIONS = {
    "32-128": [32, 128],
    "64-128": [64, 128],
    "64-256": [64, 256],
    "32-64-128": [32, 64, 128],
    "16-64": [16, 64],
}


def parse_layer_arg(value):
    """Still supported for direct list-of-ints input, e.g. from a manual params.json
    or if someone wants to pass raw layer sizes without going through the name tables."""
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]

    text = str(value).strip()
    if text == "":
        return None

    return [int(item.strip()) for item in text.split(",") if item.strip()]


def resolve_layer(params, values_key, name_key, options_table):
    """Resolve encoder/decoder layer sizes from a params dict, accepting either
    an explicit list (values_key) or a named preset (name_key)."""
    if values_key in params and params[values_key] is not None:
        return parse_layer_arg(params[values_key])
    if name_key in params and params[name_key] is not None:
        try:
            return options_table[params[name_key]]
        except KeyError:
            raise ValueError(
                f"Unknown {name_key} '{params[name_key]}'. "
                f"Valid options: {list(options_table.keys())}"
            )
    raise ValueError(
        f"params-json must contain either '{values_key}' (a list of ints) "
        f"or '{name_key}' (one of {list(options_table.keys())})"
    )


def load_hyperparams(params_json_path):
    with open(params_json_path) as f:
        params = json.load(f)

    required_scalar_keys = ["learning_rate", "max_epoch", "z_dim"]
    missing = [k for k in required_scalar_keys if k not in params]
    if missing:
        raise ValueError(f"params-json missing required keys: {missing}")

    hyperparams = {
        "learning_rate": float(params["learning_rate"]),
        "max_epoch": int(params["max_epoch"]),
        "z_dim": int(params["z_dim"]),
        "encoder_layer": resolve_layer(
            params, "encoder_layer_values", "encoder_layer_name", ENCODER_LAYER_OPTIONS
        ) if "encoder_layer_name" in params or "encoder_layer_values" in params
          else parse_layer_arg(params.get("encoder_layer")),
        "decoder_layer": resolve_layer(
            params, "decoder_layer_values", "decoder_layer_name", DECODER_LAYER_OPTIONS
        ) if "decoder_layer_name" in params or "decoder_layer_values" in params
          else parse_layer_arg(params.get("decoder_layer")),
    }

    if hyperparams["encoder_layer"] is None or hyperparams["decoder_layer"] is None:
        raise ValueError(
            "params-json must specify encoder_layer/decoder_layer "
            "(as a list, or via encoder_layer_name/decoder_layer_name)"
        )

    return hyperparams, params.get("trial_number")


def run_scphere(
    adata: ad.AnnData,
    batch_keys: list[str],
    use_hvg: bool,
    hvg_method,
    hvg_n: int,
    epochs: int,
    batch_size: int,
    lr: float,
    emb_dim: int,
    encoder_layer,
    decoder_layer,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed: int = 0
) -> ad.AnnData:
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    adata = select_hvgs(
        adata, use_hvg=use_hvg, hvg_method=hvg_method, hvg_n=hvg_n, subset=True
    )

    X = dense_counts(adata)  # (n_cells, n_genes)

    if len(batch_keys) >= 1:
        batch_vec = np.vstack(
            [adata.obs[col].astype("category").cat.codes for col in batch_keys]
        ).T
        batch_vec = torch.Tensor(batch_vec).long()
        n_batch = [len(np.unique(adata.obs[col])) for col in batch_keys]
    else:
        batch_vec = torch.zeros((adata.n_obs, 1))
        n_batch = 0

    model = SCPHERE(
        n_gene=X.shape[1],
        n_batch=n_batch,  # 0 (no batch) or K
        z_dim=int(emb_dim),  # latent dim
        encoder_layer=encoder_layer,
        decoder_layer=decoder_layer
    )
    model.to(device)
    model.train()

    trainer = Trainer(
        model=model,
        x=X,
        batch_id=batch_vec,
        max_epoch=int(epochs),
        mb_size=int(batch_size),
        learning_rate=float(lr),
    )
    trainer.train()

    model.eval()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(batch_vec))
    loader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=False,
        drop_last=False
    )

    embs = []
    for _, (data, batch) in enumerate(loader):
        model.forward(data.to(device), batch.to(device))
        z_mu = model.z_mu.detach().cpu().numpy()
        embs.append(z_mu)
    z = np.concatenate(embs)

    adata.obsm[EMB_KEY] = z
    return adata


def main():
    args = docopt(__doc__)
    file = args["<file>"]
    out_dir = args["-o"]
    batch_keys = args["-b"]
    cell_type_key = args["-c"]
    seed = int(args["--seed"])
    use_hvg = not args["--no-use-hvg"]
    hvg_method = args["--hvg-method"]
    hvg_n = int(args["--hvg-n"])
    batch_size = int(args["--batch-size"])
    params_json_path = args["--params-json"]

    hyperparams, trial_number = load_hyperparams(params_json_path)

    print(torch.cuda.is_available())
    if trial_number is not None:
        print(f"Running trial {trial_number}")
    print(f"Hyperparameters: {hyperparams}")

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("TRAIN", current_time_str)
    print("Loading data")
    adata = ad.read_h5ad(file)

    print("Running scPhere")
    adata = run_scphere(
        adata=adata,
        batch_keys=batch_keys,
        use_hvg=use_hvg,
        hvg_method=hvg_method,
        hvg_n=hvg_n,
        epochs=hyperparams["max_epoch"],
        batch_size=batch_size,
        lr=hyperparams["learning_rate"],
        emb_dim=hyperparams["z_dim"],
        encoder_layer=hyperparams["encoder_layer"],
        decoder_layer=hyperparams["decoder_layer"],
        seed=seed
    )
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("ENDTRAIN", current_time_str)

    sc.pp.neighbors(adata, use_rep=EMB_KEY)

    print("Saving data")
    output = minimise_anndata(
        adata,
        obs=(list(batch_keys) if batch_keys else []) + [cell_type_key],
        obsm=[EMB_KEY],
        uns=["neighbors"],
        obsp=["connectivities", "distances"]
    )
    output.write(join(out_dir, "adata.h5ad"))
    print("Done!")


if __name__ == "__main__":
    main()