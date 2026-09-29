import argparse
import json
import os

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

EMBEDDING_DIMS_OPTIONS = [2, 5, 10, 20, 30]
K_MIN = 20
K_MAX = 200
MARGIN_MIN = 0.2
MARGIN_MAX = 2.0
K_TO_M_RATIO_OPTIONS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]

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
        'embedding_dims' : trial.suggest_categorical(
            "embedding_dims",
            EMBEDDING_DIMS_OPTIONS,
        ),
        'k' : trial.suggest_int(
            "k",
            K_MIN,
            K_MAX,
        ),
        'margin' : trial.suggest_float(
            "margin",
            MARGIN_MIN,
            MARGIN_MAX,
        ),
        'k_to_m_ratio' : trial.suggest_categorical(
            "k_to_m_ratio",
            K_TO_M_RATIO_OPTIONS,
        )
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()