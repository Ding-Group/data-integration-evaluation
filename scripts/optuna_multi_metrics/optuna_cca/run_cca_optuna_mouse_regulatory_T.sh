#!/bin/bash

#SBATCH --time=10-00:00:00
#SBATCH --job-name=cca_joint_mouse_regulatory_T
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=beaver
#SBATCH --exclude=beaver01
#SBATCH --mail-user=syao08@student.ubc.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_cca/log_mouse_regulatory_T/cca_joint_mouse_regulatory_T_%j.out
#SBATCH --error=/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/scripts/optuna_multi_metrics/optuna_cca/log_mouse_regulatory_T/cca_joint_mouse_regulatory_T_%j.err

set -euo pipefail

PROJECT_ROOT="/ubc/cs/research/beaver/projects/Single-cell-integration-metrics"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/optuna_multi_metrics/optuna_cca"
LOG_DIR="${SCRIPT_DIR}/log_mouse_regulatory_T"
RESULT_DIR="${PROJECT_ROOT}/results/optuna_multi_metrics/cca/mouse_regulatory_T/expanded_metrics_scgraphm_v2"
DATA_PATH="${PROJECT_ROOT}/data/mouse_regulatory_T.h5ad"
ENV_PREFIX="${PROJECT_ROOT}/CondaEnv/cca_optuna"
PY_SCRIPT="${SCRIPT_DIR}/optimize_cca_multi_metrics.py"
STUDY_NAME="cca_mouse_regulatory_T_joint_expanded_metrics_scgraphm_v2"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  echo "ERROR: Python not found: ${ENV_PREFIX}/bin/python" >&2
  exit 1
fi
if [ ! -x "${ENV_PREFIX}/bin/Rscript" ]; then
  echo "ERROR: Rscript not found: ${ENV_PREFIX}/bin/Rscript" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export R_HOME="${ENV_PREFIX}/lib/R"
export R_LIBS_USER="${ENV_PREFIX}/lib/R/library"
export R_LIBS="${ENV_PREFIX}/lib/R/library"
export PYTHONPATH="${PROJECT_ROOT}/scib:${PYTHONPATH:-}"

echo "Dataset: mouse_regulatory_T"
echo "Data path: ${DATA_PATH}"
echo "Result dir: ${RESULT_DIR}"
echo "Study name: ${STUDY_NAME}"

"${ENV_PREFIX}/bin/python" -c "import optuna, scanpy, anndata, scipy; from docopt import docopt; from scib.metrics.metrics import metrics; print('CCA Python environment ok')"
"${ENV_PREFIX}/bin/Rscript" --vanilla -e ".libPaths('${ENV_PREFIX}/lib/R/library'); library(Seurat); library(SeuratObject); library(Matrix); cat('CCA R environment ok\n')"

"${ENV_PREFIX}/bin/python" "${PY_SCRIPT}" \
  -o "${RESULT_DIR}" \
  -b method \
  -c cell.type \
  --dataset-name=mouse_regulatory_T \
  --study-name="${STUDY_NAME}" \
  --seed=0 \
  --layer=counts \
  --nfeatures-min=1000 \
  --nfeatures-max=5000 \
  --nfeatures-step=500 \
  --npcs-min=30 \
  --npcs-max=60 \
  --npcs-step=5 \
  --bami \
  --casw \
  --scgraphm \
  --bras \
  --kbet \
  --cpcr \
  "${DATA_PATH}"
