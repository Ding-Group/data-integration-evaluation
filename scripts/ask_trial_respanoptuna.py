import argparse
import json
import os

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

REDUCTION_OPTIONS = ["pca", "cca", "kpca"]
SUBSAMPLE_OPTIONS = [1000, 3000, 5000]
EPOCH_OPTIONS = [100, 200, 300, 500]


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
        "reduction": trial.suggest_categorical("reduction", REDUCTION_OPTIONS),
        "subsample": trial.suggest_categorical("subsample", SUBSAMPLE_OPTIONS),
        "epochs": trial.suggest_categorical("epoch", EPOCH_OPTIONS),
        'lambda_1': trial.suggest_float("lambda_1", 1, 20),
        'lr': trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()