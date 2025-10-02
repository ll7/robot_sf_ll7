#!/usr/bin/env bash
#SBATCH --job-name=robot-sf-uv-setup
#SBATCH --account=mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=auxme-uv-setup-%j.out

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR}
SCRATCH_ROOT=${AUXME_SCRATCH_ROOT:-/scratch/${USER}}
ENV_DIR=${UV_ENV_DIR:-${SCRATCH_ROOT}/robot_sf_uv}
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

if ! command -v uv >/dev/null 2>&1; then
  echo "[auxme] uv is not available in PATH. Load the appropriate module or contact the cluster admins." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENV_DIR}")"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"

cd "${PROJECT_ROOT}"

echo "[auxme] Synchronising git submodules"
git submodule sync --recursive
git submodule update --init --recursive

echo "[auxme] Installing Python ${PY_VERSION} via uv"
uv python install --if-missing "${PY_VERSION}"

echo "[auxme] Synchronising project dependencies"
uv sync --python "${PY_VERSION}" --extra dev

echo "[auxme] Installing pre-commit hooks"
uv run pre-commit install

echo "[auxme] Environment ready at ${ENV_DIR}"
cat <<MSG
Activate with:
  module purge
  for mod in ${MODULE_LIST}; do module load \$mod; done
  [[ "${ENABLE_CUDA}" == "1" ]] && module load ${CUDA_MODULE}
  source "${ENV_DIR}/bin/activate"

To rerun dependency sync:
  uv sync --extra dev
MSG
