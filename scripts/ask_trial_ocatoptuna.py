import argparse
import json
import os

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

N_GHOSTS_MIN = 40
N_GHOSTS_MAX = 100
DIM_MIN = 40
DIM_MAX = 125
P_MIN = 0.1
P_MAX = 0.5


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
        'n_ghosts' : trial.suggest_int("n_ghosts", N_GHOSTS_MIN, N_GHOSTS_MAX),
        'dim' : trial.suggest_int("dim", DIM_MIN, DIM_MAX),
        'p' : trial.suggest_float("p", P_MIN, P_MAX)
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()