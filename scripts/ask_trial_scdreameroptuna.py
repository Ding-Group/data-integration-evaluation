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
        'lr_ae' : trial.suggest_float("lr_ae", 0.0001, 0.01, log=True),
        'lr_dis' : trial.suggest_float("lr_dis", 0.0001, 0.01, log=True),
        'lr_bc': trial.suggest_float("lr_bc", 0.0001, 0.01, log=True),
        'z_dim' : trial.suggest_int("z_dim", 5, 30),
        'epochs': trial.suggest_int("epochs", 100, 500, step=50)
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()