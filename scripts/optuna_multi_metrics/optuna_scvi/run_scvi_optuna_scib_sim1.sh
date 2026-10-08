#!/bin/bash
#SBATCH --time=21-00:00:00
#SBATCH --job-name=scvi_joint_scib_sim1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=beaver
#SBATCH --gres=gpu:1
#SBATCH --mail-user=syao08@student.ubc.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_scvi/log_scib_sim1/scvi_joint_scib_sim1_%j.out
#SBATCH --error=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_scvi/log_scib_sim1/scvi_joint_scib_sim1_%j.err

set -euo pipefail

PROJECT_ROOT="/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/optuna_multi_metrics/optuna_scvi"
LOG_DIR="${SCRIPT_DIR}/log_scib_sim1"
RESULT_DIR="${PROJECT_ROOT}/results/optuna_multi_metrics/scvi/scib_sim1/expanded_metrics_v1"
DATA_PATH="${PROJECT_ROOT}/data/scib_sim1.h5ad"
PY_SCRIPT="optimize_scvi_multi_metrics.py"
STUDY_NAME="scvi_scib_sim1_joint_expanded_metrics_v1"

echo "============================================================"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Data path: ${DATA_PATH}"
echo "Result dir: ${RESULT_DIR}"
echo "Study name: ${STUDY_NAME}"
echo "============================================================"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

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
export R_LIBS_SITE="${PROJECT_ROOT}/CondaEnv/harmony_optuna/lib/R/library"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${PROJECT_ROOT}/tmp"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

which python
python --version
python -c "import optuna, scanpy, scvi, torch; print('core imports ok; cuda available:', torch.cuda.is_available())"
python -c "import anndata2ri; from docopt import docopt; print('bridge imports ok')"
python -c "from scib_metrics.metrics import bras; from balanced_clustering import balanced_adjusted_mutual_info; print('metric dependencies ok')"
python -c "import inspect; from scib.metrics.metrics import metrics; p=inspect.signature(metrics).parameters; required=('bami_', 'casw_', 'scgraphm_', 'bras_', 'kbet_', 'cpcr_', 'traj_', 'isolisi_'); missing=[x for x in required if x not in p]; assert not missing, missing; print('metric signature ok:', required)"
python -c "from rpy2.robjects.packages import importr; importr('kBET'); print('R kBET ok')"

cd "${SCRIPT_DIR}"
/opt/slurm/bin/srun python "${PY_SCRIPT}" \
  -o "${RESULT_DIR}" \
  -b Batch \
  -c Group \
  --dataset-name=scib_sim1 \
  --study-name="${STUDY_NAME}" \
  --seed=0 \
  --bami \
  --casw \
  --scgraphm \
  --isolisi \
  --bras \
  --kbet \
  --cpcr \
  "${DATA_PATH}"

echo "Job finished"
