"""
Best hyperparams for ResPAN-Optim

Usage:
    get_best_respanoptuna.py --study-dir=<dir> --out=<file> --best-metrics-out=<file>
"""
import json
import os
import sys

import optuna
from docopt import docopt
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def main():
    args = docopt(__doc__)
    study_dir = args["--study-dir"]
    out_path = args["--out"]
    best_metrics_path = args["--best-metrics-out"]

    journal_path = os.path.join(study_dir, "optuna_journal.log")
    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.create_study(
        study_name="optuna",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        sys.stderr.write(f"No completed trials found in study at {study_dir}\n")
        sys.exit(1)

    best = study.best_trial

    best_params = {
        "trial_number": best.number,
        "reduction": best.params["reduction"],
        "subsample": best.params["subsample"],
        "epochs": best.params["epoch"],
        "lambda_1": best.params["lambda_1"],
        "lr": best.params["lr"],
    }

    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)

    best_metrics = {
        "trial_number": best.number,
        "best_value": best.value,
        "n_completed_trials": len(completed),
        "n_total_trials": len(study.trials),
        "params": best.params,
    }
    with open(best_metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"Best trial: #{best.number}  joint_score={best.value:.4f}", flush=True)
    print(f"({len(completed)}/{len(study.trials)} trials completed)", flush=True)


if __name__ == "__main__":
    main()