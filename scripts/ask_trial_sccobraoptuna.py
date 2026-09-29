import argparse
import json
import os

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

BATCH_SIZE_OPTIONS = [64, 128, 256, 512]
LR_MIN = 5e-5
LR_MAX = 1e-3
MAX_ITERATION_MIN = 10000
MAX_ITERATION_MAX = 50000



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
        'batch_size' : trial.suggest_categorical("batch_size", BATCH_SIZE_OPTIONS),
        'lr' : trial.suggest_float("lr", LR_MIN, LR_MAX, log=True),
        'max_iteration' : trial.suggest_int(
            "max_iteration",
            MAX_ITERATION_MIN,
            MAX_ITERATION_MAX,
        )
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()