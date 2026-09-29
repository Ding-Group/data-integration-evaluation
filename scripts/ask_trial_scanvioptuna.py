import argparse
import json
import os

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


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
        'n_latent' : trial.suggest_int("n_latent", 5, 64),
        'n_hidden' : trial.suggest_categorical("n_hidden", [64, 128, 256]),
        'n_layers' : trial.suggest_int("n_layers", 1, 3),
        'dropout_rate': trial.suggest_float("dropout_rate", 0.01, 0.2),
        'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()