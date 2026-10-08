"""
Run a metric with metric parameters and report to an Optuna study

Usage:
    run_metrics_final_optuna.py (-b <key> | --batch-key=<key>)
                                (-c <key> | --cell-type-key=<key>)
                                (-d <str> | --dataset=<str>)
                                (-m <str> | --method=<str>)
                                (-p <str> | --params=<str>)
                                [options]
                                <unintegrated_data> <integrated_data> <outfile>

Options:
    -h --help                    Show this screen.
    -b --batch-key=<key>         Batch key.
    -c --cell-type-key=<key>     Cell type label key.
    -d --dataset=<str>           Dataset name.
    -m --method=<str>            Integration method name.
    -p --params=<str>            Params for metrics as a str
    --study-dir=<dir>            Optuna study directory (enables reporting back to study).
    --params-json=<file>        JSON file with trial_number, produced by ask_trial.py.
"""

import json
import os
import sys

sys.path.insert(
    0, "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scib/"
)
os.environ["R_HOME"] = (
    "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/lib/R"
)

import numpy as np
import scanpy as sc
from docopt import docopt
from project_utils.adata_utils import EMB_KEY
from scib.metrics import final_metrics

BIO_METRICS = ["BAMI", "CASW", "SCGRAPHM", "TRAJ", "ISOLISI"]
BATCH_METRICS = ["BRAS", "KBET", "CPCR"]


def joint_score_from_result(result):
    """result is the raw scib final_metrics Series (metric name -> value)."""
    bio = np.nanmean([result.loc[m] for m in BIO_METRICS if m in result.index])
    batch = np.nanmean([result.loc[m] for m in BATCH_METRICS if m in result.index])
    return bio * 0.6 + batch * 0.4


def report_to_optuna(study_dir, trial_params_path, joint_score, failed=False):
    """Best-effort report back to the shared Optuna study. Only runs if both
    --study-dir and --trial-params were provided; silently no-ops otherwise."""
    if not study_dir or not trial_params_path:
        return

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    with open(trial_params_path) as f:
        trial_info = json.load(f)
    trial_number = trial_info["trial_number"]

    journal_path = os.path.join(study_dir, "optuna_journal.log")
    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.create_study(
        study_name="optuna",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    if failed:
        study.tell(trial_number, state=optuna.trial.TrialState.FAIL)
        sys.stderr.write(f"[trial {trial_number}] reported FAIL to study\n")
    else:
        study.tell(trial_number, joint_score)
        print(f"[trial {trial_number}] joint_score={joint_score:.4f} reported to study", flush=True)


def main():
    args = docopt(__doc__, options_first=True)
    file = args["<unintegrated_data>"]
    file_int = args["<integrated_data>"]
    outfile = args["<outfile>"]
    batch_key = args["--batch-key"]
    label_key = args["--cell-type-key"]
    dataset = args["--dataset"]
    method = args["--method"]
    params = args["--params"]
    params = json.loads(params)
    study_dir = args["--study-dir"]
    trial_params_path = args["--params-json"]

    metric_kwargs = {}
    if params is not None:
        for p in params:
            for metric_name, metric_params in p.items():
                metric_kwargs[f'{metric_name.lower()}_kwargs'] = metric_params
    # print(metric_kwargs)

    try:
        adata = sc.read(file)
    except FileNotFoundError as e:
        sys.stderr.write(
            f"Metric computation failed on dataset {dataset} and integration {method}\n"
        )
        report_to_optuna(study_dir, trial_params_path, None, failed=True)
        raise e

    try:
        adata_int = sc.read(file_int)

        # This is to post-hoc add in the new label keys for the granularity experiments
        if label_key not in adata_int.obs.columns:
            adata_int.obs[label_key] = adata.obs[label_key]

        if "." in batch_key:
            new_batch_key = batch_key.replace(".", "_")
            adata.obs[new_batch_key] = adata.obs[batch_key]
            adata_int.obs[new_batch_key] = adata_int.obs[batch_key]
            batch_key = new_batch_key
        if "." in label_key:
            new_label_key = label_key.replace(".", "_")
            adata.obs[new_label_key] = adata.obs[label_key]
            adata_int.obs[new_label_key] = adata_int.obs[label_key]
            label_key = new_label_key

        result = final_metrics(
            adata,
            adata_int,
            batch_key,
            label_key,
            EMB_KEY,
            metric_kwargs=metric_kwargs
        )
    except Exception:
        sys.stderr.write(
            f"Metric computation failed on dataset {dataset} and integration {method}\n"
        )
        report_to_optuna(study_dir, trial_params_path, None, failed=True)
        raise

    joint_score = joint_score_from_result(result)

    results = result.reset_index()
    results = results.rename({0: "Value", "index": "Metric"}, axis=1)
    results["Dataset"] = dataset
    results["Integration"] = method
    results["BatchKey"] = batch_key
    results["LabelKey"] = label_key
    results = results[
        ["Dataset", "Integration", "Metric", "BatchKey", "LabelKey", "Value"]
    ].dropna()

    if metric_kwargs:
        results.insert(len(results.columns) - 1, "Params", str(metric_kwargs))
    results.to_csv(outfile, index=False)
    report_to_optuna(study_dir, trial_params_path, joint_score, failed=False)


if __name__ == "__main__":
    main()
