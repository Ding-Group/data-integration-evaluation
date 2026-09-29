import argparse
import json
import os

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

ENCODER_LAYER_OPTIONS = {
    "128-64-32": [128, 64, 32], "256-128-64": [256, 128, 64],
    "128-64": [128, 64], "256-128-64-32": [256, 128, 64, 32],
    "128": [128], "64-32-16": [64, 32, 16],
}
DECODER_LAYER_OPTIONS = {
    "32-128": [32, 128], "64-128": [64, 128], "64-256": [64, 256],
    "32-64-128": [32, 64, 128], "16-64": [16, 64],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.study_dir, exist_ok=True)

    journal_path = os.path.join(args.study_dir, "optuna_journal.log")
    storage = JournalStorage(JournalFileBackend(journal_path))

    study = optuna.create_study(
        study_name="optuna",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=TPESampler(constant_liar=True)
    )

    trial = study.ask()

    params = {
        "trial_number": trial.number,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "max_epoch": trial.suggest_int("max_epoch", 50, 150),
        "z_dim": trial.suggest_categorical("z_dim", [2, 5, 10, 20, 30]),
        "encoder_layer_name": trial.suggest_categorical("encoder_layer", list(ENCODER_LAYER_OPTIONS)),
        "decoder_layer_name": trial.suggest_categorical("decoder_layer", list(DECODER_LAYER_OPTIONS)),
    }
    params["encoder_layer"] = ENCODER_LAYER_OPTIONS[params["encoder_layer_name"]]
    params["decoder_layer"] = DECODER_LAYER_OPTIONS[params["decoder_layer_name"]]

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()