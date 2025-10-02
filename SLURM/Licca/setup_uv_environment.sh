#!/usr/bin/env bash
# Set up a uv-managed Python environment for Robot SF on LiCCA.
# This script installs uv (if needed), loads cluster modules, and
# synchronises project dependencies using uv.
# Usage: bash setup_uv_environment.sh [environment_path]
# Environment variables:
#   LICCA_UV_MODULES     - whitespace-separated module list (default: "miniforge gcc/13.2.0")
#   LICCA_UV_ENABLE_CUDA - if set to 1, load CUDA module defined by LICCA_UV_CUDA_MODULE
#   LICCA_UV_CUDA_MODULE - CUDA module to load when LICCA_UV_ENABLE_CUDA=1 (default: cuda/12.1.1)
#   UV_INSTALL_PREFIX    - install prefix for uv (default: "$HOME/.local")
#   UV_PYTHON_VERSION    - Python version to install via uv (default: 3.11)
#   UV_ENV_DIR           - environment path override (takes precedence over positional arg)

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  echo "[licca] Lmod 'module' command not available. Run from a LiCCA login node." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "[licca] Could not locate project root (missing pyproject.toml)." >&2
  exit 1
fi

DEFAULT_ENV_DIR="/hpc/gpfs2/scratch/u/${USER}/venvs/robot-sf-uv"
ENV_DIR="${1:-${UV_ENV_DIR:-${DEFAULT_ENV_DIR}}}"
UV_INSTALL_PREFIX="${UV_INSTALL_PREFIX:-${HOME}/.local}"
PY_VERSION="${UV_PYTHON_VERSION:-3.11}"
CUDA_MODULE="${LICCA_UV_CUDA_MODULE:-cuda/12.1.1}"

module purge
MODULE_LIST="${LICCA_UV_MODULES:-miniforge gcc/13.2.0}"
for module_name in ${MODULE_LIST}; do
  [[ -z "${module_name}" ]] && continue
  echo "[licca] Loading module ${module_name}"
  module load "${module_name}"
done

if [[ "${LICCA_UV_ENABLE_CUDA:-0}" == "1" ]]; then
  echo "[licca] Loading CUDA module ${CUDA_MODULE}"
  module load "${CUDA_MODULE}"
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[licca] curl is required to download uv. Load a module providing curl." >&2
  exit 1
fi

mkdir -p "${UV_INSTALL_PREFIX}/bin"
UV_BIN="${UV_INSTALL_PREFIX}/bin/uv"
if [[ ! -x "${UV_BIN}" ]]; then
  echo "[licca] Installing uv into ${UV_INSTALL_PREFIX}/bin"
  UV_INSTALL_DIR="${UV_INSTALL_PREFIX}" curl -LsSf https://astral.sh/uv/install.sh | sh
else
  echo "[licca] Found uv at ${UV_BIN}; updating if necessary"
  "${UV_BIN}" self update || true
fi

export PATH="${UV_INSTALL_PREFIX}/bin:${PATH}"

mkdir -p "$(dirname "${ENV_DIR}")"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"

echo "[licca] Installing Python ${PY_VERSION} via uv"
uv python install --if-missing "${PY_VERSION}"

echo "[licca] Synchronising project dependencies (dev extras)"
cd "${REPO_ROOT}"
uv sync --python "${PY_VERSION}" --extra dev

echo "[licca] Installing pre-commit hooks"
uv run pre-commit install

if [[ "${LICCA_UV_ENABLE_CUDA:-0}" == "1" ]]; then
  CUDA_ACTIVATION_LINE="  module load ${CUDA_MODULE}"
else
  CUDA_ACTIVATION_LINE="  # module load ${CUDA_MODULE} (enable if your job needs CUDA)"
fi

echo "[licca] uv environment ready at ${ENV_DIR}"
cat <<MSG
Activate it with:
  module purge
  for mod in ${MODULE_LIST}; do module load \$mod; done
${CUDA_ACTIVATION_LINE}
  source "${ENV_DIR}/bin/activate"

All commands (pytest, scripts) can be run with 'uv run <command>' from the repo root.
MSG
