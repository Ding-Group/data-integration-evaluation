#!/bin/bash

#SBATCH --time=21-00:00:00
#SBATCH --job-name=cca_joint_bmmc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --partition=beaver
#SBATCH --exclude=beaver01
#SBATCH --mail-user=syao08@student.ubc.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_cca/log_bmmc/cca_joint_bmmc_%j.out
#SBATCH --error=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_cca/log_bmmc/cca_joint_bmmc_%j.err

set -eo pipefail

PROJECT_ROOT="/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

SCRIPT_DIR="${PROJECT_ROOT}/scripts/optuna_multi_metrics/optuna_cca"
LOG_DIR="${PROJECT_ROOT}/scripts/optuna_multi_metrics/optuna_cca/log_bmmc"
RESULT_DIR="${PROJECT_ROOT}/results/optuna_multi_metrics/cca/neurips_bmmc/expanded_metrics_scgraphm_v1"
DATA_PATH="${PROJECT_ROOT}/data/neurips_bmmc.h5ad"

ENV_PREFIX="${PROJECT_ROOT}/CondaEnv/cca_optuna"

PY_SCRIPT="optimize_cca_multi_metrics.py"
STUDY_NAME="cca_neurips_bmmc_joint_expanded_metrics_scgraphm_v1"

echo "============================================================"
echo "Job started"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
echo "Script dir: ${SCRIPT_DIR}"
echo "Python script: ${PY_SCRIPT}"
echo "Data path: ${DATA_PATH}"
echo "Log dir: ${LOG_DIR}"
echo "Result dir: ${RESULT_DIR}"
echo "Conda env: ${ENV_PREFIX}"
echo "Study name: ${STUDY_NAME}"
echo "============================================================"

cd "${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"
mkdir -p "${RESULT_DIR}"

if [ ! -d "${ENV_PREFIX}" ]; then
  echo "ERROR: Conda environment directory not found: ${ENV_PREFIX}" >&2
  exit 1
fi

if [ ! -f "${ENV_PREFIX}/bin/python" ]; then
  echo "ERROR: Python not found in conda environment: ${ENV_PREFIX}/bin/python" >&2
  exit 1
fi

# Conda init and activation may reference unset variables internally.
set +u

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

conda activate "${ENV_PREFIX}"

set -u

export PYTHONNOUSERSITE=1
export R_HOME="${ENV_PREFIX}/lib/R"
export R_LIBS_USER="${ENV_PREFIX}/lib/R/library"
export R_LIBS="${ENV_PREFIX}/lib/R/library"
export PYTHONPATH="${PROJECT_ROOT}/scib:${PYTHONPATH:-}"

echo "============================================================"
echo "Environment check"
echo "============================================================"

which python
python --version

echo "CONDA_PREFIX=${CONDA_PREFIX:-NA}"
echo "R_HOME=${R_HOME}"
echo "R_LIBS_USER=${R_LIBS_USER}"
echo "R_LIBS=${R_LIBS}"
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"
echo "PYTHONPATH=${PYTHONPATH}"

which Rscript || true
"${ENV_PREFIX}/bin/Rscript" --version

python -c "import optuna; print('optuna ok')"
python -c "import scanpy as sc; print('scanpy ok')"
python -c "import anndata as ad; print('anndata ok')"
python -c "import scipy; print('scipy ok')"
python -c "from docopt import docopt; print('docopt ok')"
python -c "from scib.metrics.metrics import metrics; print('scib metrics ok')"

"${ENV_PREFIX}/bin/Rscript" --vanilla -e ".libPaths('${ENV_PREFIX}/lib/R/library'); library(Seurat); library(SeuratObject); library(Matrix); cat('Seurat R packages ok\n')"

echo "============================================================"
echo "Running joint-metric Seurat CCA Optuna"
echo "============================================================"

cd "${SCRIPT_DIR}"

python "${PY_SCRIPT}" \
  -o "${RESULT_DIR}" \
  -b batch \
  -c cell_type \
  --dataset-name=neurips_bmmc \
  --study-name="${STUDY_NAME}" \
  --seed=0 \
  --layer=counts \
  --nfeatures-min=1000 \
  --nfeatures-max=2500 \
  --nfeatures-step=500 \
  --npcs-min=30 \
  --npcs-max=40 \
  --npcs-step=5 \
  --bami \
  --casw \
  --scgraphm \
  --traj \
  --isolisi \
  --bras \
  --kbet \
  --cpcr \
  "${DATA_PATH}"

echo "============================================================"
echo "Job finished"
echo "============================================================"
