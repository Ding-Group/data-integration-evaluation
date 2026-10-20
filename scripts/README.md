# Tool running scripts

This directory contains the Python scripts used to run all the integration methods.
There are the following script purposes:

* `run_{method}.py`: These scripts run the given `method` on a particular dataset. This script is used for generating integrations for the metric analysis, data characteristics analysis, and the final benchmark. All methods are run under default hyperparameters.
* `run_{method}optuna.py`: These scripts run the given `method` in concert with Optuna. Some scripts (CCA, Harmony, RPCA, scVI, MNN, Scanorama, LIGER) run Optuna and the integration method in the same script, following our initial hyperparameter experiments. The remaining methods' scripts function similarly to the `run_{method}.py` scripts, taking in a hyperparameter JSON.
* `ask_trial_{method}optuna.py`: These scripts produce a set of hyperparameters for an Optuna trial, which will be fed into `run_{method}optuna.py` for methods that are optimized following this procedure.
* `get_best_{method}optuna.py`: These scripts take the completed runs and produce the best set of hyperparameters.
* `run_metrics.py` runs the given metric on a given integration. This script is used for running individual metrics for our metric analysis.
* `run_metrics_final.py` runs the eight final core metrics on a given integration. This script is used for getting benchmarking scores for our dataset characteristics analysis and final benchmarking.
* `run_metrics_final_optuna.py` operates like `run_metrics_final.py` except it reports back to Optuna the benchmarking score for hyperparameter optimization.

In `survey/`, the two scripts `ask_gemini.py` and `ask_gpt.py` ask the two respective LLMs the survey questions about each data integration benchmark.

In `metric_ablation/`, there are scripts to compute alternative clustering algorithms (K-means, Leiden, Louvain, GMM) and alternative nearest neighbor graph construction algorithms (FAISS, Annoy, Default, Exact, FLANN, hnswlib).

In `optuna_multi_metrics/`, there are scripts for our first analysis of hyperparameter optimization involving five methods (LIGER, CCA, Scanorama, Harmony, scVI) and nine datasets.

In `project_utils`, there are some helper scripts for the notebooks and other scripts.
