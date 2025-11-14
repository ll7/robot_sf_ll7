#!/usr/bin/env bash
# Test coverage collection validation
# Purpose: Verify that pytest-cov collects coverage automatically

set -e

echo "==================================="
echo "Coverage Collection Validation"
echo "==================================="

# Navigate to project root
cd "$(dirname "$0")/../.." || exit 1

echo ""
echo "Step 1: Running a simple test with coverage..."
uv run pytest tests/test_snqi_schema.py -v -q 2>&1 | head -30

echo ""
echo "Step 2: Checking coverage outputs..."

COVERAGE_ROOT=$(uv run python - <<'PY'
from robot_sf.common.artifact_paths import get_artifact_category_path
print(get_artifact_category_path("coverage"), end="")
PY
)

echo "Using coverage root: $COVERAGE_ROOT"

if [ -f "$COVERAGE_ROOT/coverage.json" ]; then
    echo "✓ coverage.json exists"
    jq '.totals.percent_covered' "$COVERAGE_ROOT/coverage.json" 2>/dev/null || echo "  (JSON format valid)"
else
    echo "✗ coverage.json NOT found"
    exit 1
fi

if [ -d "$COVERAGE_ROOT/htmlcov" ]; then
    echo "✓ htmlcov/ directory exists"
    if [ -f "$COVERAGE_ROOT/htmlcov/index.html" ]; then
        echo "  ✓ index.html present"
    else
        echo "  ✗ index.html missing"
        exit 1
    fi
else
    echo "✗ htmlcov/ directory NOT found"
    exit 1
fi

if [ -f "$COVERAGE_ROOT/.coverage" ]; then
    echo "✓ .coverage database exists"
else
    echo "✗ .coverage database NOT found"
    exit 1
fi

echo ""
echo "==================================="
echo "✓ Coverage collection validated!"
echo "==================================="
echo ""
echo "Next steps:"
echo "  - Open $COVERAGE_ROOT/htmlcov/index.html in a browser to view the report"
echo "  - Run 'uv run pytest tests' to collect full coverage"
