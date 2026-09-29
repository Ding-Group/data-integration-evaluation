import argparse
import json
import os

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

Z_DISTANCE_CYCLE_WEIGHT_MIN = 1.0
Z_DISTANCE_CYCLE_WEIGHT_MAX = 20.0
N_PRIOR_COMPONENTS_MIN = 2
N_PRIOR_COMPONENTS_MAX = 32
KL_WEIGHT_MIN = 0.2
KL_WEIGHT_MAX = 5
N_LATENT_MIN = 5
N_LATENT_MAX = 50
N_HIDDEN_OPTIONS = [64, 128, 256]
N_LAYERS_MIN = 1
N_LAYERS_MAX = 3
DROPOUT_RATE_MIN = 0.01
DROPOUT_RATE_MAX = 0.3


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
        'z_distance_cycle_weight' : trial.suggest_float(
            "z_distance_cycle_weight",
            Z_DISTANCE_CYCLE_WEIGHT_MIN,
            Z_DISTANCE_CYCLE_WEIGHT_MAX,
            log=True,
        ),
        'n_prior_components' : trial.suggest_int(
            "n_prior_components",
            N_PRIOR_COMPONENTS_MIN,
            N_PRIOR_COMPONENTS_MAX,
            log=True,
        ),
        'kl_weight' : trial.suggest_float(
            "kl_weight",
            KL_WEIGHT_MIN,
            KL_WEIGHT_MAX,
        ),
        'n_latent':trial.suggest_int(
            "n_latent",
            N_LATENT_MIN,
            N_LATENT_MAX,
        ),
        'n_hidden':trial.suggest_categorical(
            "n_hidden",
            N_HIDDEN_OPTIONS,
        ),
        'n_layers':trial.suggest_int(
            "n_layers",
            N_LAYERS_MIN,
            N_LAYERS_MAX,
        ),
        'dropout_rate':trial.suggest_float(
            "dropout_rate",
            DROPOUT_RATE_MIN,
            DROPOUT_RATE_MAX,
        ),
    }

    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()