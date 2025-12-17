"""
Combine a set of metrics output files with the AnnDatas and the metric values.

Usage:
    combine-metrics.py --out-file=<path> [options] <file>...

Options:
    -h --help                Show this screen.
    --out-file=<path>        Path to combined metrics output file.
"""

import re
from os.path import splitext

import scanpy as sc
import pandas as pd

def main():
    """The main script function"""
    from docopt import docopt

    args = docopt(__doc__)

    files = args["<file>"]
    out_file = args["--out-file"]

    # Split the input stream into the CSVs and AnnDatas
    adatas = {}
    csvs = {}
    for f in files:
        _, ext = splitext(f)
        if ext == ".h5ad":
            pattern = r"(\w+)\.h5ad$"
            match = re.search(pattern, f)
            hex_string = match.group(1)
            adatas[hex_string] = f
        else:
            pattern = r"(\w+)\.csv$"
            match = re.search(pattern, f)
            hex_string = match.group(1)
            csvs[hex_string] = f

    all_dfs = []
    for key in adatas.keys():
        adata = sc.read(adatas[key])
        df = pd.read_csv(csvs[key])

        # Add in the information from the adata
        neighbor_params = adata.uns['neighbors_params']
        for k, v in neighbor_params.items():
            df[f'neighbor_{k}'] = v
        clustering_params = adata.uns['clustering_params']
        for k, v in clustering_params.items():
            df[f'clustering_{k}'] = v
        df['hex'] = key
        all_dfs.append(df)
    
    final_df = pd.concat(all_dfs)
    final_df.to_csv(out_file, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
