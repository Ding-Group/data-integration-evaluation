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
        'resolution' : trial.suggest_float("resolution", 1.0, 5.0, step=0.5),
        'K_in' : trial.suggest_int("K_in", 3, 15, step=2),
        'K_bw' : trial.suggest_int("K_bw", 5, 30, step=5),
        'margin' : trial.suggest_float("margin", 0.05, 0.5, log=True),
        'num_epochs' : trial.suggest_int("num_epochs", 50, 150, step=25),
        'hidden_width' : trial.suggest_categorical("hidden_width", [64, 128, 256, 512])
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()