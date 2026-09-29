"""
Best hyperparams for scCRAFT-Optim

Usage:
    get_best_sccraftoptuna.py --study-dir=<dir> --out=<file> --best-metrics-out=<file>
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

    # Reconstruct the same JSON shape ask_trial.py produces, so run_scphere.py's
    # load_hyperparams() can consume it unchanged.
    best_params = {
        "trial_number": best.number,
        "kl_coef": best.params["kl_coef"],
        "d_coef": best.params["d_coef"],
        "z_dim": best.params["z_dim"],
        "resolution1": best.params["resolution1"],
        "resolution2": best.params["resolution2"],
        "method": best.params["method"]
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