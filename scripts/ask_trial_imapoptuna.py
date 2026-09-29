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
        "n_epochs": trial.suggest_int("n_epochs", 50, 300, step=50),
        "latent_dim": trial.suggest_categorical("latent_dim", [32, 64, 128, 256]),
        "lambda_co": trial.suggest_float(
            "lambda_co",
            1.0,
            8.0,
        ),
        "lambda_rc": trial.suggest_float(
            "lambda_rc",
            0.5,
            3.0,
        )
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()