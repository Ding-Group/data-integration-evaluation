#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH --job-name=scanorama_joint_mouse_regulatory_T
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=beaver
#SBATCH --mail-user=syao08@student.ubc.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_scanorama/log_mouse_regulatory_T/scanorama_joint_mouse_regulatory_T_%j.out
#SBATCH --error=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_scanorama/log_mouse_regulatory_T/scanorama_joint_mouse_regulatory_T_%j.err

set -euo pipefail

PROJECT_ROOT="/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

SCRIPT_DIR="${PROJECT_ROOT}/scripts/optuna_multi_metrics/optuna_scanorama"
LOG_DIR="${SCRIPT_DIR}/log_mouse_regulatory_T"
RESULT_DIR="${PROJECT_ROOT}/results/optuna_multi_metrics/scanorama/mouse_regulatory_T/expanded_metrics_v2"
DATA_PATH="${PROJECT_ROOT}/data/mouse_regulatory_T.h5ad"

PY_SCRIPT="optimize_scanorama_multi_metrics.py"
STUDY_NAME="scanorama_mouse_regulatory_T_joint_expanded_metrics_v2"

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

conda activate "${PROJECT_ROOT}/CondaEnv/harmony_optuna"

export PYTHONNOUSERSITE=1
export R_HOME="${PROJECT_ROOT}/CondaEnv/harmony_optuna/lib/R"

echo "============================================================"
echo "Environment check"
echo "============================================================"
which python
python --version
echo "R_HOME=${R_HOME}"
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"

python -c "import optuna; print('optuna ok')"
python -c "import scanpy as sc; print('scanpy ok')"
python -c "import scanorama; print('scanorama ok')"
python -c "from docopt import docopt; print('docopt ok')"
python -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/scib/'); from scib.metrics.metrics import metrics; print('repo metrics ok')"

echo "============================================================"
echo "Running formal joint-metric Scanorama Optuna on mouse_regulatory_T"
echo "============================================================"

cd "${SCRIPT_DIR}"

python "${PY_SCRIPT}" \
  -o "${RESULT_DIR}" \
  -b method \
  -c cell.type \
  --dataset-name=mouse_regulatory_T \
  --study-name="${STUDY_NAME}" \
  --seed=0 \
  --bami \
  --casw \
  --scgraphm \
  --bras \
  --kbet \
  --cpcr \
  "${DATA_PATH}"

echo "============================================================"
echo "Job finished"
echo "============================================================"
