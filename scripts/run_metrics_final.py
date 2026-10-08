"""
Run a metric with metric parameters

Usage:
    run_metrics_final.py (-b <key> | --batch-key=<key>)
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
"""

import sys
import os
import json

sys.path.insert(
    0, "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scib/"
)
os.environ["R_HOME"] = (
    "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/lib/R"
)

from docopt import docopt
import scanpy as sc
from scib.metrics import final_metrics
from project_utils.adata_utils import EMB_KEY


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
        raise e

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


if __name__ == "__main__":
    main()
