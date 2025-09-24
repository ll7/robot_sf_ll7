#!/usr/bin/env python3
"""
CI Performance Dashboard

Simple dashboard to display CI performance metrics and trends.
Shows package installation times, performance improvements, and CI health.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Main dashboard function."""
    # Default metrics file location (now in tmp folder)
    metrics_file = Path("tmp/ci_performance_metrics.json")

    if not metrics_file.exists():
        print("❌ No CI performance metrics found.")
        print(f"Expected metrics file: {metrics_file}")
        print("\nRun CI pipeline first to generate metrics, or specify custom path:")
        print("python scripts/ci_dashboard.py --metrics-file /path/to/metrics.json")
        return 1

    print("📊 CI Performance Dashboard")
    print("=" * 50)

    # Run the performance metrics script to generate report
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/ci-tests/performance_metrics.py",
                "--metrics-file",
                str(metrics_file),
                "--report",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating performance report: {e}")
        print(f"stderr: {e.stderr}")
        return 1

    # Add dashboard-specific summary
    print("\n" + "=" * 50)
    print("🎯 Performance Targets:")
    print("• Package installation: < 73 seconds (50% reduction)")
    print("• Overall CI job: Within acceptable time limits")
    print("\n💡 Tips:")
    print("• Green checkmarks indicate targets met")
    print("• Monitor trends across multiple CI runs")
    print("• Use 'act' tool for local CI testing")

    return 0


if __name__ == "__main__":
    sys.exit(main())
