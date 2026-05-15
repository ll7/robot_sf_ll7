#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_setup.sh
source "$SCRIPT_DIR/common_setup.sh"

BASE_REF="${BASE_REF:-origin/main}"

uv run --active python scripts/validation/check_docs_proof_consistency.py --base "$BASE_REF"
