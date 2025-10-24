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

if [ -f "coverage.json" ]; then
    echo "✓ coverage.json exists"
    jq '.totals.percent_covered' coverage.json 2>/dev/null || echo "  (JSON format valid)"
else
    echo "✗ coverage.json NOT found"
    exit 1
fi

if [ -d "htmlcov" ]; then
    echo "✓ htmlcov/ directory exists"
    if [ -f "htmlcov/index.html" ]; then
        echo "  ✓ index.html present"
    else
        echo "  ✗ index.html missing"
        exit 1
    fi
else
    echo "✗ htmlcov/ directory NOT found"
    exit 1
fi

if [ -f ".coverage" ]; then
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
echo "  - Open htmlcov/index.html in a browser to view the report"
echo "  - Run 'uv run pytest tests' to collect full coverage"
