#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./private_ops.sh
source "${SCRIPT_DIR}/private_ops.sh"

PRIVATE_SCRIPT="$(private_ops_require_file auxme/scripts/dev/sbatch_oracle_imitation_traces_issue1470.sh)"
exec "${PRIVATE_SCRIPT}" "$@"
