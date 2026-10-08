#!/bin/bash
#SBATCH --time=21-00:00:00
#SBATCH --job-name=scvi_joint_bmmc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=beaver
#SBATCH --gres=gpu:1
#SBATCH --mail-user=syao08@student.ubc.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_scvi/log_bmmc/scvi_joint_bmmc_%j.out
#SBATCH --error=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_scvi/log_bmmc/scvi_joint_bmmc_%j.err

set -euo pipefail

PROJECT_ROOT="/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"

SCRIPT_DIR="${PROJECT_ROOT}/scripts/optuna_multi_metrics/optuna_scvi"
LOG_DIR="${PROJECT_ROOT}/scripts/optuna_multi_metrics/optuna_scvi/log_bmmc"
RESULT_DIR="${PROJECT_ROOT}/results/optuna_multi_metrics/scvi/bmmc"
DATA_PATH="${PROJECT_ROOT}/data/neurips_bmmc.h5ad"

PY_SCRIPT="optimize_scvi_multi_metrics.py"
STUDY_NAME="scvi_bmmc_joint_metrics_v4_clean_repo_metrics"

echo "============================================================"
echo "Job started"
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

conda activate "${PROJECT_ROOT}/CondaEnv/scvi_py310"


export PYTHONNOUSERSITE=1

export PYTHONPATH="${PROJECT_ROOT}/scib:${PROJECT_ROOT}:${PYTHONPATH:-}"

export R_HOME="${CONDA_PREFIX}/lib/R"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export TMPDIR="${PROJECT_ROOT}/tmp"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"

mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

echo "============================================================"
echo "Environment check"
echo "============================================================"
which python
python --version
echo "R_HOME=${R_HOME}"
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"

python -c "import optuna; print('optuna ok')"
python -c "import scanpy as sc; print('scanpy ok')"
python -c "import scvi; print('scvi ok')"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
python -c "from docopt import docopt; print('docopt ok')"
python -c "import anndata2ri; print('anndata2ri ok')"
python -c "from scib_metrics.metrics import bras; print('scib_metrics bras ok')"
python -c "from scib.metrics.metrics import metrics; print('scib metrics ok')"
python -m pip check || echo "[WARN] pip check found dependency conflicts, ignored."

echo "============================================================"
echo "Running joint-metric scVI Optuna"
echo "============================================================"

cd "${SCRIPT_DIR}"

/opt/slurm/bin/srun python "${PY_SCRIPT}" \
  -o "${RESULT_DIR}" \
  -b batch \
  -c cell_type \
  --study-name="${STUDY_NAME}" \
  --seed=0 \
  --bami \
  --casw \
  --scgraphm \
  --bras \
  --kbet \
  --traj \
  "${DATA_PATH}"

echo "============================================================"
echo "Job finished"
echo "============================================================"
