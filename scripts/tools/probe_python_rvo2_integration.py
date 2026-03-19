#!/usr/bin/env python3
"""Validate vendored Python-RVO2 provenance and Robot SF ORCA adapter behavior."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.planner.socnav import ORCAPlannerAdapter, SocNavPlannerConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rvo2-root",
        type=Path,
        default=Path("third_party/python-rvo2"),
        help="Path to the vendored Python-RVO2 checkout.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional Markdown output path.",
    )
    return parser


def _make_obs() -> dict[str, Any]:
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([0.0], dtype=np.float32),
            "speed": np.array([0.0], dtype=np.float32),
            "radius": np.array([0.5], dtype=np.float32),
        },
        "goal": {
            "current": np.array([5.0, 0.0], dtype=np.float32),
            "next": np.array([0.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": np.array([[2.0, 0.0]], dtype=np.float32),
            "velocities": np.zeros((1, 2), dtype=np.float32),
            "radius": np.array([0.4], dtype=np.float32),
            "count": np.array([1.0], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
    }


def _validate_required_files(rvo2_root: Path) -> dict[str, str]:
    required = {
        "upstream_note": "UPSTREAM.md",
        "readme": "README.md",
        "license": "LICENSE",
        "example": "example.py",
    }
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for key, rel_path in required.items():
        candidate = rvo2_root / rel_path
        if not candidate.exists():
            missing.append(rel_path)
        else:
            resolved[key] = str(candidate)
    if missing:
        raise FileNotFoundError(
            f"Vendored Python-RVO2 checkout is missing required files: {', '.join(missing)}"
        )
    return resolved


def _run_upstream_example(rvo2_root: Path) -> dict[str, Any]:
    example_path = rvo2_root / "example.py"
    proc = subprocess.run(
        [sys.executable, str(example_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return {
        "returncode": proc.returncode,
        "stdout_preview": stdout_lines[:6],
        "stderr_preview": [line for line in proc.stderr.splitlines() if line.strip()][:6],
    }


def _probe_adapter() -> dict[str, Any]:
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig(), allow_fallback=False)
    v, w = adapter.plan(_make_obs())
    metadata = enrich_algorithm_metadata(
        algo="orca",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner_kinematics = dict(metadata.get("planner_kinematics") or {})
    return {
        "linear_velocity": float(v),
        "angular_velocity": float(w),
        "upstream_reference": dict(metadata.get("upstream_reference") or {}),
        "planner_kinematics": planner_kinematics,
    }


def build_report(rvo2_root: Path) -> dict[str, Any]:
    """Collect upstream-example and adapter-contract evidence for Python-RVO2."""
    files = _validate_required_files(rvo2_root)
    example_result = _run_upstream_example(rvo2_root)
    if example_result["returncode"] != 0:
        verdict = "not yet viable"
    else:
        verdict = "viable benchmark prototype"
    adapter_result = _probe_adapter()
    return {
        "verdict": verdict,
        "vendored_root": str(rvo2_root),
        "required_files": files,
        "upstream_example": example_result,
        "adapter_probe": adapter_result,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for the integration probe report."""
    example = report["upstream_example"]
    adapter = report["adapter_probe"]
    kin = adapter["planner_kinematics"]
    upstream = adapter["upstream_reference"]
    lines = [
        "# Python-RVO2 Integration Probe",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Upstream validation",
        "",
        f"- vendored root: `{report['vendored_root']}`",
        f"- upstream example return code: `{example['returncode']}`",
        f"- upstream repo: `{upstream.get('repo_url', 'unknown')}`",
        f"- upstream commit: `{upstream.get('commit', 'unknown')}`",
        "",
        "## Adapter contract",
        "",
        f"- adapter boundary: {upstream.get('adapter_boundary', 'unknown')}",
        f"- upstream command space: `{kin.get('upstream_command_space', 'unknown')}`",
        f"- benchmark command space: `{kin.get('benchmark_command_space', 'unknown')}`",
        f"- projection policy: `{kin.get('projection_policy', 'unknown')}`",
        f"- adapter probe action: `v={adapter['linear_velocity']:.4f}`, `w={adapter['angular_velocity']:.4f}`",
    ]
    if example["stdout_preview"]:
        lines.extend(["", "## Upstream example preview", ""])
        lines.extend([f"- `{line}`" for line in example["stdout_preview"]])
    if example["stderr_preview"]:
        lines.extend(["", "## Upstream example stderr preview", ""])
        lines.extend([f"- `{line}`" for line in example["stderr_preview"]])
    return "\n".join(lines) + "\n"


def _write_optional(path: Path | None, content: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    """Run the probe from CLI arguments and emit JSON plus optional report files."""
    args = _build_parser().parse_args()
    report = build_report(args.rvo2_root)
    payload = json.dumps(report, indent=2)
    markdown = render_markdown(report)
    _write_optional(args.output_json, payload + "\n")
    _write_optional(args.output_md, markdown)
    print(payload)
    return 0 if report["verdict"] == "viable benchmark prototype" else 1


if __name__ == "__main__":
    raise SystemExit(main())
