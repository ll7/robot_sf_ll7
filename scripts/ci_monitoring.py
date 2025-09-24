#!/usr/bin/env python3
"""
CI Performance Monitoring Script

This script monitors and collects performance metrics for CI jobs,
specifically tracking the system package installation step duration.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class CIPerformanceMonitor:
    """Monitor CI job performance metrics."""

    def __init__(self, output_dir: str = "tmp/ci_metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_job(self, job_id: str, branch: str = "main") -> str:
        """Start monitoring a CI job.

        Args:
            job_id: Unique identifier for the CI job
            branch: Git branch being tested

        Returns:
            Monitoring session ID
        """
        session_id = f"{job_id}_{int(time.time())}"
        session_data = {
            "session_id": session_id,
            "job_id": job_id,
            "branch": branch,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "steps": {},
        }

        session_file = self.output_dir / f"{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        return session_id

    def start_step(self, session_id: str, step_name: str) -> None:
        """Start monitoring a specific CI step.

        Args:
            session_id: Monitoring session ID
            step_name: Name of the CI step
        """
        session_file = self.output_dir / f"{session_id}.json"
        if not session_file.exists():
            return

        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["steps"][step_name] = {"start_time": datetime.now().isoformat(), "status": "running"}

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def end_step(self, session_id: str, step_name: str, success: bool = True) -> None:
        """End monitoring a specific CI step.

        Args:
            session_id: Monitoring session ID
            step_name: Name of the CI step
            success: Whether the step succeeded
        """
        session_file = self.output_dir / f"{session_id}.json"
        if not session_file.exists():
            return

        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if step_name in data["steps"]:
            start_time = datetime.fromisoformat(data["steps"][step_name]["start_time"])
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            data["steps"][step_name].update(
                {
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "status": "completed" if success else "failed",
                }
            )

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def end_job(self, session_id: str, success: bool = True) -> None:
        """End monitoring a CI job.

        Args:
            session_id: Monitoring session ID
            success: Whether the job succeeded
        """
        session_file = self.output_dir / f"{session_id}.json"
        if not session_file.exists():
            return

        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.update(
            {"end_time": datetime.now().isoformat(), "status": "completed" if success else "failed"}
        )

        # Calculate total duration
        if "start_time" in data and "end_time" in data:
            start = datetime.fromisoformat(data["start_time"])
            end = datetime.fromisoformat(data["end_time"])
            data["total_duration_seconds"] = (end - start).total_seconds()

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a completed CI job.

        Args:
            session_id: Monitoring session ID

        Returns:
            Metrics dictionary or None if not found
        """
        session_file = self.output_dir / f"{session_id}.json"
        if not session_file.exists():
            return None

        with open(session_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_package_installation_time(self, session_id: str) -> Optional[float]:
        """Get the package installation step duration.

        Args:
            session_id: Monitoring session ID

        Returns:
            Duration in seconds or None if not available
        """
        metrics = self.get_metrics(session_id)
        if not metrics:
            return None

        step = metrics.get("steps", {}).get("System packages for headless")
        if step and "duration_seconds" in step:
            return step["duration_seconds"]

        return None


def main():
    """Command-line interface for CI monitoring."""
    parser = argparse.ArgumentParser(description="CI Performance Monitor")
    parser.add_argument(
        "action", choices=["start-job", "start-step", "end-step", "end-job", "get-metrics"]
    )
    parser.add_argument("--session-id", help="Monitoring session ID")
    parser.add_argument("--job-id", help="CI job ID")
    parser.add_argument("--step-name", help="CI step name")
    parser.add_argument("--branch", default="main", help="Git branch")
    parser.add_argument(
        "--success", action="store_true", default=True, help="Step/job success status"
    )

    args = parser.parse_args()
    monitor = CIPerformanceMonitor()

    # Action handlers to reduce complexity
    handlers = {
        "start-job": lambda: handle_start_job(args, monitor),
        "start-step": lambda: handle_start_step(args, monitor),
        "end-step": lambda: handle_end_step(args, monitor),
        "end-job": lambda: handle_end_job(args, monitor),
        "get-metrics": lambda: handle_get_metrics(args, monitor),
    }

    try:
        handlers[args.action]()
    except ValueError as e:
        parser.error(str(e))


def handle_start_job(args, monitor):
    """Handle start-job action."""
    if not args.job_id:
        raise ValueError("--job-id required for start-job")
    session_id = monitor.start_job(args.job_id, args.branch)
    print(f"Started monitoring session: {session_id}")


def handle_start_step(args, monitor):
    """Handle start-step action."""
    if not args.session_id or not args.step_name:
        raise ValueError("--session-id and --step-name required for start-step")
    monitor.start_step(args.session_id, args.step_name)
    print(f"Started monitoring step: {args.step_name}")


def handle_end_step(args, monitor):
    """Handle end-step action."""
    if not args.session_id or not args.step_name:
        raise ValueError("--session-id and --step-name required for end-step")
    monitor.end_step(args.session_id, args.step_name, args.success)
    print(f"Ended monitoring step: {args.step_name}")


def handle_end_job(args, monitor):
    """Handle end-job action."""
    if not args.session_id:
        raise ValueError("--session-id required for end-job")
    monitor.end_job(args.session_id, args.success)
    print("Ended monitoring job")


def handle_get_metrics(args, monitor):
    """Handle get-metrics action."""
    if not args.session_id:
        raise ValueError("--session-id required for get-metrics")
    metrics = monitor.get_metrics(args.session_id)
    if metrics:
        print(json.dumps(metrics, indent=2))
        package_time = monitor.get_package_installation_time(args.session_id)
        if package_time:
            print(f"\nPackage installation time: {package_time:.2f} seconds")
    else:
        print("Metrics not found")


if __name__ == "__main__":
    main()
