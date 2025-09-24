#!/usr/bin/env python3
"""
Performance Metrics Collection Script

Collects and analyzes CI performance metrics to track optimization effectiveness.
Generates reports on package installation times and overall CI performance.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class PerformanceMetrics:
    """Handles collection and analysis of CI performance metrics."""

    def __init__(self, metrics_file: Path):
        """Initialize metrics handler.

        Args:
            metrics_file: Path to JSON file containing metrics data
        """
        self.metrics_file = metrics_file
        self.data = self._load_metrics()

    def _load_metrics(self) -> Dict[str, Any]:
        """Load metrics data from file."""
        if not self.metrics_file.exists():
            return {"runs": []}

        try:
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not load metrics from {self.metrics_file}")
            return {"runs": []}

    def save_metrics(self) -> None:
        """Save metrics data to file."""
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_run(self, run_data: Dict[str, Any]) -> None:
        """Add a new CI run to the metrics.

        Args:
            run_data: Dictionary containing run metrics
        """
        if "runs" not in self.data:
            self.data["runs"] = []

        # Add timestamp if not provided
        if "timestamp" not in run_data:
            run_data["timestamp"] = datetime.now().isoformat()

        self.data["runs"].append(run_data)
        self.save_metrics()

    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent CI runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of recent run data
        """
        runs = self.data.get("runs", [])
        return sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

    def analyze_package_installation_times(self) -> Dict[str, Any]:
        """Analyze package installation performance across runs.

        Returns:
            Dictionary with analysis results
        """
        runs = self.data.get("runs", [])
        if not runs:
            return {"error": "No runs available for analysis"}

        install_times = []
        for run in runs:
            steps = run.get("steps", {})
            if "install_system_packages" in steps:
                duration = steps["install_system_packages"].get("duration_seconds")
                if duration is not None:
                    install_times.append(duration)

        if not install_times:
            return {"error": "No package installation timing data available"}

        install_times.sort()

        return {
            "count": len(install_times),
            "min": min(install_times),
            "max": max(install_times),
            "mean": sum(install_times) / len(install_times),
            "median": install_times[len(install_times) // 2],
            "p95": install_times[int(len(install_times) * 0.95)] if install_times else None,
            "target_achieved": min(install_times) < 73.0,  # Target: under 1min 13sec
            "improvement_ratio": max(install_times) / min(install_times)
            if min(install_times) > 0
            else None,
        }

    def generate_report(self) -> str:
        """Generate a human-readable performance report.

        Returns:
            Formatted report string
        """
        analysis = self.analyze_package_installation_times()
        recent_runs = self.get_recent_runs(5)

        report_lines = [
            "# CI Performance Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Package Installation Analysis",
        ]

        if "error" in analysis:
            report_lines.append(f"Error: {analysis['error']}")
        else:
            report_lines.extend(
                [
                    f"- Total runs analyzed: {analysis['count']}",
                    f"- Best time: {analysis['min']:.1f}s",
                    f"- Worst time: {analysis['max']:.1f}s",
                    f"- Average time: {analysis['mean']:.1f}s",
                    f"- Median time: {analysis['median']:.1f}s",
                    f"- 95th percentile: {analysis.get('p95', 'N/A')}",
                    f"- Target achieved (under 73s): {'✓' if analysis['target_achieved'] else '✗'}",
                ]
            )

            if analysis.get("improvement_ratio"):
                ratio = analysis["improvement_ratio"]
                report_lines.append(f"- Performance improvement: {ratio:.1f}x")

        report_lines.extend(["", "## Recent Runs"])

        for i, run in enumerate(recent_runs, 1):
            timestamp = run.get("timestamp", "Unknown")
            status = run.get("status", "Unknown")
            total_time = run.get("total_duration_seconds", "N/A")

            install_step = run.get("steps", {}).get("install_system_packages", {})
            install_time = install_step.get("duration_seconds", "N/A")

            report_lines.append(
                f"{i}. {timestamp} - Status: {status}, Total: {total_time}s, Install: {install_time}s"
            )

        return "\n".join(report_lines)


def main() -> int:
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="CI Performance Metrics Analysis")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("results/ci_metrics.json"),
        help="Path to metrics JSON file",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate and print performance report"
    )
    parser.add_argument("--add-run", type=Path, help="Add a new run from JSON file")

    args = parser.parse_args()

    metrics = PerformanceMetrics(args.metrics_file)

    if args.add_run:
        try:
            with open(args.add_run, "r", encoding="utf-8") as f:
                run_data = json.load(f)
            metrics.add_run(run_data)
            print(f"Added run data from {args.add_run}")
        except Exception as e:
            print(f"Error adding run: {e}")
            return 1

    if args.report:
        report = metrics.generate_report()
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
