# Comprehensive evaluation of single-cell benchmarking

This repository contains the code for evaluating metrics, datasets, and hyperparameter optimization.
The repository follows a similar structure to the [feature selection benchmark](https://github.com/theislab/atlas-feature-selection-benchmark)
by Zappia et al., for which we are grateful for their work.

This repository is regrettably not super clean, but we explain the structure below.

## Directory structure

* `notebooks/`: Notebooks for analysis and presentation of the results. Please refer to the README in the directory for the specific relevant analyses.
* `scripts/`: Python scripts used for running integration methods and metrics.
* `conf/`: Configuration files for Nextflow experiments run.
* `data/`: AnnData objects of analysis data. The data files that are not easily producible from public sources are available in [FigShare](10.6084/m9.figshare.34442223). Other data files can be reproduced/downloaded from Supplementary Table S4.
* `CondaEnv/`: Directory containing the Conda environments. 
  A few of the Conda environments we used were not built in this directory.
  Instead of the full environments, we supply YAML files containing their configuration.
* `results`: Results from the NextFlow pipelines.
  All results except for source data spreadsheets are excluded from this repository.
  These spreadsheets contain metric values to produce the results.
  The set of integrations used to analyze metrics ("Single-cell integration zoo") is available in [Zenodo](doi.org/10.5281/zenodo.21983919
)
* `work` and `ablation_work`: Intermediate work from the Nextflow piplines.
  This is excluded from this repository.


## Example metric computation pipeline

An example Nextflow pipeline run using just a few integration methods can be found in `conf/minimal.yml`.
The `scmetrics` environment in `CondaEnv/scmetrics.yaml` contains NextFlow and [sc-complete-metrics](https://github.com/Ding-Group/sc-complete-metrics) 
and will be used to run the pipeline.

```bash
cd CondaEnv/
conda env create -f basic_env.yaml -n <name_of_basic_env>
conda env create -f scmetrics.yaml -n <name_of_scmetrics_env>
```

In `conf/minimal.yml`, you will need to edit `conda_location` to the path of your `conda.sh` and replace `<path/to/basic_env>` and
`<path/to/scmetrics_env>` with the corresponding values from above. You will also need to update the SLURM configuration, if you
are using SLURM, to match your cluster details (our config file is `conf/ubc_slurm.config`).

```bash
# Download the cellranger_versions dataset and place it into the data/ directory
# https://figshare.com/account/articles/34442223
gzip -d data/cellranger_versions.h5ad.gz

conda activate <name_of_scmetrics_env>
export JAVA_CMD=$(which java)

nextflow run main.nf -entry WF_ALL -profile ubc_slurm -params-file conf/minimal.yml -w work/minimal
```

The results will be located in `results/minimal_example/metrics/all-metrics.csv`.