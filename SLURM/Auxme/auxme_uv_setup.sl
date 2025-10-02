#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-uv-setup
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=auxme-uv-setup-%j.out

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR}
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-/scratch/${USER}}
ENV_DIR=${UV_ENV_DIR:-${SCRATCH_ROOT}/robot_sf_uv}
UV_INSTALL_PREFIX=${UV_INSTALL_PREFIX:-${HOME}/.local}
PY_VERSION=${UV_PYTHON_VERSION:-3.11}
MODULE_LIST=${AUXME_MODULES:-"gcc/13.2.0"}
CUDA_MODULE=${AUXME_CUDA_MODULE:-cuda/12.1}
ENABLE_CUDA=${AUXME_ENABLE_CUDA:-0}

if [[ ! -d ${PROJECT_ROOT}/.git ]]; then
  echo "[auxme] Run this script from within the repository." >&2
  exit 1
fi

module purge
for mod in ${MODULE_LIST}; do
  [[ -z "${mod}" ]] && continue
  echo "[auxme] Loading module ${mod}"
  module load "${mod}"
done

if [[ "${ENABLE_CUDA}" == "1" ]]; then
  echo "[auxme] Loading CUDA module ${CUDA_MODULE}"
  module load "${CUDA_MODULE}"
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[auxme] curl is required; load a module providing it." >&2
  exit 1
fi


echo "[auxme] Found uv at ${UV_BIN}; updating if necessary"
"${UV_BIN}" self update || true


export PATH="${UV_INSTALL_PREFIX}/bin:${PATH}"
mkdir -p "$(dirname "${ENV_DIR}")"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"

echo "[auxme] Ensuring git submodules are synchronised"
cd "${PROJECT_ROOT}"
git submodule sync --recursive
git submodule update --init --recursive

echo "[auxme] Installing Python ${PY_VERSION} via uv"
uv python install --if-missing "${PY_VERSION}"

echo "[auxme] Synchronising dependencies"
uv sync --python "${PY_VERSION}" --extra dev

echo "[auxme] Installing pre-commit hooks"
uv run pre-commit install

echo "[auxme] Environment ready at ${ENV_DIR}"
