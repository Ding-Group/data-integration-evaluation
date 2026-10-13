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
* `scib/`: Directory containing the complete set of [metric implementations](https://github.com/Ding-Group/sc-complete-metrics).
* `results`: Results from the NextFlow pipelines.
  All results except for select spreadsheets are excluded from this repository.
  The set of integrations used to analyze metrics ("Single-cell integration zoo") is available in [Zenodo](doi.org/10.5281/zenodo.21983919
)
* `work` and `ablation_work`: Intermediate work from the Nextflow piplines.
  This is excluded from this repository.
