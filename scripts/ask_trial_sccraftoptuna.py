import argparse
import json
import os

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

Z_DIM_OPTIONS = [64, 128, 256, 384]


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
        'kl_coef' : trial.suggest_float("kl_coef", 1e-4, 5e-3, log=True),
        'd_coef' : trial.suggest_float("d_coef", 0.05, 0.5),
        'z_dim' : trial.suggest_categorical("z_dim", Z_DIM_OPTIONS),
        'resolution1': trial.suggest_float('resolution1', 0.2, 0.7),
        'resolution2': trial.suggest_float('resolution2', 4, 10),
        'method' : trial.suggest_categorical('method', ['Louvain', 'Leiden'])
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()