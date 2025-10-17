"""
Run a metric.

Usage:
    run_metrics.py (-b <key> | --batch-key=<key>)
                   (-c <key> | --cell-type-key=<key>)
                   (-d <str> | --dataset=<str>)
                   (-m <str> | --method=<str>)
                   [options]
                   <metric> <unintegrated_data> <integrated_data> <outfile> [<args>...]

Options:
    -h --help                    Show this screen.
    -b --batch-key=<key>         Batch key.
    -c --cell-type-key=<key>     Cell type label key.
    -d --dataset=<str>           Dataset name.
    -m --method=<str>            Integration method name.
"""

import sys
import os

sys.path.insert(
    0, "/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scib/"
)
os.environ["R_HOME"] = (
    "/ubc/cs/research/beaver/projects/yinian/miniconda3/envs/scmetrics/lib/R"
)

from docopt import docopt
import scanpy as sc
import pandas as pd
from scib.metrics import metrics
from utils.adata_utils import EMB_KEY


def main():
    args = docopt(__doc__, options_first=True)
    metric = args["<metric>"].lower()
    file = args["<unintegrated_data>"]
    file_int = args["<integrated_data>"]
    outfile = args["<outfile>"]
    batch_key = args["--batch-key"]
    label_key = args["--cell-type-key"]
    dataset = args["--dataset"]
    method = args["--method"]
    params = args["<args>"]

    metric_kwargs = {}
    for token in params:
        if token.startswith("--") and "=" in token:
            k, v = token[2:].split("=")
            metric_kwargs[k] = v

    adata = sc.read(file)
    adata_int = sc.read(file_int)

    kwargs = {metric + '_': True}

    result = metrics(
        adata,
        adata_int,
        batch_key,
        label_key,
        EMB_KEY,
        metric_kwargs=metric_kwargs,
        **kwargs,
    )

    results = pd.DataFrame(
        [
            {
                "Dataset": dataset,
                "Integration": method,
                "Metric": metric,
                "Value": result.loc[metric.upper()].to_numpy()[0],
            }
        ]
    )
    results.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
