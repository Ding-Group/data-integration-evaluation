# Comprehensive evaluation of single-cell benchmarking

This repository contains the code for evaluating metrics, datasets, and hyperparameter optimization.
The repository follows a similar structure to the [feature selection benchmark](https://github.com/theislab/atlas-feature-selection-benchmark)
by Zappia et al., for which we are grateful for their work.

This repository is regrettably not super clean, but we explain the structure below.

## Directory structure

* `notebooks/`: Notebooks for analysis and presentation of the results. Please refer to the README in the directory for the specific relevant analyses.
* `scripts/`: Python scripts used for running integration methods and metrics.
* `conf/`: Configuration files for Nextflow experiments run.
* `data/`: AnnData objects of analysis data. The data files can be downloaded from **TBD**.
* `CondaEnv/`: Directory containing the Conda environments. 
  A few of the Conda environments we used were not built in this directory.
  Instead of the full environments, we supply YAML files containing their configuration.
* `scib/`: Directory containing the complete set of [metric implementations](https://github.com/Ding-Group/sc-complete-metrics).
* `results`: Results from the NextFlow pipelines.
  This is excluded from this repository.
* `work` and `ablation_work`: Intermediate work from the Nextflow piplines.
  This is excluded from this repository.
