"""
Run a metric.

Usage:
    run_metrics.py (-d <str> | --dataset=<str>)
                   (-m <str> | --method=<str>)
                   [options] <metric> <outfile>

Options:
    -h --help                    Show this screen.
    -d --dataset=<str>           Dataset name.
    -m --method=<str>            Integration method name.
"""

import sys

import numpy as np
import pandas as pd
from docopt import docopt


def main():
    args = docopt(__doc__, options_first=True)
    metric = args["<metric>"].lower()
    outfile = args["<outfile>"]
    dataset = args["--dataset"]
    method = args["--method"]

    sys.stderr.write(f"Error in {metric}, {method}, {dataset}\n")

    results = pd.DataFrame(
        [
            {
                "Dataset": dataset,
                "Integration": method,
                "Metric": metric,
                "Value": np.nan,
            }
        ]
    )
    results.to_csv(outfile, index=False)
    raise ValueError('Metric failure')


if __name__ == "__main__":
    main()
