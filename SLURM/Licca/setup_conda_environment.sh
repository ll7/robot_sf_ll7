#!/usr/bin/env bash
# Create/update a Robot SF conda or micromamba environment on LiCCA.
# Run this on a LiCCA login node from the project root.
# Usage: bash setup_conda_environment.sh [env_name]
# Environment variables:
#   LICCA_ENV_MANAGER=miniforge|micromamba (default: miniforge)
#   PY_VERSION (default: 3.11)
#   EXTRA_PIP_FLAGS - appended to pip install commands

set -euo pipefail

ENV_NAME=${1:-robot-sf}
ENV_MANAGER=${LICCA_ENV_MANAGER:-miniforge}
PY_VERSION=${PY_VERSION:-3.11}

# Guard variables referenced by conda's module wrappers when set -u is active.
: "${_CE_M:=}"
: "${_CE_CONDA:=}"

echo "[licca] Preparing ${ENV_MANAGER} environment '${ENV_NAME}'"

module purge
case "${ENV_MANAGER}" in
  miniforge)
    module load miniforge
    CREATE_CMD=(conda create -y -n "${ENV_NAME}" -c conda-forge "python=${PY_VERSION}" pip)
    RUN_CMD=(conda run -n "${ENV_NAME}")
    ACTIVATE_BIN="conda"
    PACKAGE_CMD=(conda install -y -n "${ENV_NAME}" -c conda-forge)
    ;;
  micromamba)
    module load micromamba
    CREATE_CMD=(mm create -y -n "${ENV_NAME}" -c conda-forge "python=${PY_VERSION}" pip)
    RUN_CMD=(mm run -n "${ENV_NAME}")
    ACTIVATE_BIN="micromamba"
    PACKAGE_CMD=(mm install -y -n "${ENV_NAME}" -c conda-forge)
    ;;
  *)
    echo "Unsupported LICCA_ENV_MANAGER='${ENV_MANAGER}' (use miniforge or micromamba)" >&2
    exit 1
    ;;
esac

if ! "${RUN_CMD[@]}" python -c "import sys" >/dev/null 2>&1; then
  echo "[licca] Creating environment '${ENV_NAME}'"
  "${CREATE_CMD[@]}"
else
  echo "[licca] Environment '${ENV_NAME}' already exists"
fi

# Ensure OpenCV can load without a system OpenGL stack.
OPENGL_RUNTIME_PACKAGES=(libglvnd mesa-libgl-cos7-x86_64)
echo "[licca] Installing OpenGL runtime packages: ${OPENGL_RUNTIME_PACKAGES[*]}"
"${PACKAGE_CMD[@]}" "${OPENGL_RUNTIME_PACKAGES[@]}"

# Upgrade pip and install Robot SF in editable mode with dev extras.
"${RUN_CMD[@]}" python -m pip install --upgrade pip ${EXTRA_PIP_FLAGS:-}
"${RUN_CMD[@]}" python -m pip install --editable ".[dev]" ${EXTRA_PIP_FLAGS:-}

# Force the headless OpenCV build to avoid libGL loader issues in CI/HPC.
echo "[licca] Replacing OpenCV with headless build"
"${RUN_CMD[@]}" python -m pip install --upgrade opencv-python-headless ${EXTRA_PIP_FLAGS:-}

cat <<MSG
[licca] Environment '${ENV_NAME}' ready.
Activate it with:
  module purge
  module load ${ENV_MANAGER}
  ${ACTIVATE_BIN} activate ${ENV_NAME}
MSG
