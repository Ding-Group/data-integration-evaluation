import argparse
import json
import os

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

LAMBDACOS_MIN = 10.0
LAMBDACOS_MAX = 50.0
TRAINING_STEPS_MIN = 1000
TRAINING_STEPS_MAX = 3000


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
        'lambdacos' : trial.suggest_float("lambdacos", LAMBDACOS_MIN, LAMBDACOS_MAX),
        'training_steps' : trial.suggest_int(
            "training_steps",
            TRAINING_STEPS_MIN,
            TRAINING_STEPS_MAX,
        )
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()