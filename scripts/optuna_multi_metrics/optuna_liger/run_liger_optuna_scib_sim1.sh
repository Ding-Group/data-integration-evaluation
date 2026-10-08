#!/bin/bash

#SBATCH --time=4-00:00:00
#SBATCH --job-name=liger_joint_scib_sim1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=beaver
#SBATCH --mail-user=syao08@student.ubc.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_liger/log_scib_sim1/liger_joint_scib_sim1_%j.out
#SBATCH --error=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_liger/log_scib_sim1/liger_joint_scib_sim1_%j.err

set -euo pipefail

PROJECT_ROOT="/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

SCRIPT_DIR="${PROJECT_ROOT}/scripts/optuna_multi_metrics/optuna_liger"
LOG_DIR="${SCRIPT_DIR}/log_scib_sim1"
RESULT_DIR="${PROJECT_ROOT}/results/optuna_multi_metrics/liger/scib_sim1/expanded_metrics_scgraphm_v1"
DATA_PATH="${PROJECT_ROOT}/data/scib_sim1.h5ad"

PY_SCRIPT="optimize_liger_multi_metrics.py"
STUDY_NAME="liger_scib_sim1_joint_expanded_metrics_scgraphm_v1"

echo "============================================================"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
echo "Script dir: ${SCRIPT_DIR}"
echo "Python script: ${PY_SCRIPT}"
echo "Data path: ${DATA_PATH}"
echo "Log dir: ${LOG_DIR}"
echo "Result dir: ${RESULT_DIR}"
echo "Study name: ${STUDY_NAME}"
echo "============================================================"

cd "${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"
mkdir -p "${RESULT_DIR}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda not found / cannot init conda" >&2
  exit 1
fi

set +u
conda activate "${PROJECT_ROOT}/CondaEnv/pyliger_env"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/scib:${PYTHONPATH:-}"
export R_HOME="${PROJECT_ROOT}/CondaEnv/pyliger_env/lib/R"

echo "============================================================"
echo "Environment check"
echo "============================================================"
which python
python --version
echo "R_HOME=${R_HOME}"
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"

python -c "import optuna; print('optuna ok')"
python -c "import scanpy as sc; print('scanpy ok')"
python -c "import anndata as ad; print('anndata ok')"
python -c "import pyliger; print('pyliger ok')"
python -c "from docopt import docopt; print('docopt ok')"
python -c "from scib.metrics.metrics import metrics; print('repo metrics ok')"

echo "============================================================"
echo "Running formal joint-metric LIGER Optuna on scib_sim1"
echo "============================================================"

cd "${SCRIPT_DIR}"

python "${PY_SCRIPT}" \
  -o "${RESULT_DIR}" \
  -b Batch \
  -c Group \
  --dataset-name=scib_sim1 \
  --study-name="${STUDY_NAME}" \
  --seed=1 \
  --mode=core \
  --layer=counts \
  --bami \
  --casw \
  --scgraphm \
  --isolisi \
  --bras \
  --kbet \
  --cpcr \
  "${DATA_PATH}"

echo "============================================================"
echo "Job finished"
echo "============================================================"
