#!/usr/bin/env python3
"""Replay original wall-affected diagnostic campaigns with explicit seed admission.

Run inside a Slurm allocation. Outputs are diagnostic; historical evidence and
Stage B preregistration remain unchanged. No Stage B confirmation is submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from robot_sf.evidence.writers import write_json, write_sha256sums

# The original schedules were read from the committed evidence manifests/summaries.
CAMPAIGNS = (
    ("demo", "build_issue_5149_emergent_phenomena_demo.py", (5149,), ()),
    (
        "face_validity",
        "build_issue_5149_emergent_phenomena_campaign.py",
        tuple(range(5149, 5159)),
        (),
    ),
    (
        "sensitivity",
        "run_issue_6962_lane_formation_sensitivity.py",
        tuple(range(5149, 5159)),
        (
            "--lengths-m",
            "16,24",
            "--half-widths-m",
            "1.75,2.5",
            "--pedestrian-counts",
            "16,24",
            "--steps",
            "200,400",
            "--calibrations",
            "released_default,literature_typical",
            "--lane-segregation-thresholds",
            "0.15,0.3,0.5",
            "--lane-purity-thresholds",
            "0.4,0.6,0.8",
        ),
    ),
    (
        "reference",
        "run_issue_6969_lane_formation_reference.py",
        (5149, 5150, 5151),
        (
            "--conditions",
            "mixed_sustained_flow,separated_lane_control",
            "--calibrations",
            "released_default,literature_typical",
        ),
    ),
    (
        "stage_a",
        "run_issue_6969_parameter_screen.py",
        (5149, 5150, 5151),
        ("--profiles", "8", "--profile-seed", "6969"),
    ),
)
REFERENCE_PROTOCOL = (
    "--sampling-strides",
    "1,2,4",
    "--length-m",
    "24",
    "--half-width-m",
    "2.5",
    "--pedestrian-count",
    "24",
    "--warmup-steps",
    "100",
    "--observation-steps",
    "200",
    "--recycle-margin-m",
    "0.2",
    "--lane-offset-m",
    "0.85",
    "--entry-y-span-m",
    "1.2",
)
SEALED = frozenset(
    (
        50036,
        50140,
        50331,
        50403,
        50813,
        51339,
        51709,
        51767,
        52094,
        52175,
        52257,
        52671,
        52850,
        52971,
        53020,
        53198,
        53239,
        53636,
        53671,
        53779,
        55022,
        55379,
        55568,
        56170,
        56966,
        57077,
        57113,
        57494,
        57943,
        59019,
    )
)


def check_seeds(name: str, seeds: tuple[int, ...]) -> None:
    """Print the resolved schedule and reject both forbidden evaluation bands."""
    print(f"RESOLVED {name} seeds: {list(seeds)}", flush=True)
    if any(111 <= seed <= 140 or seed in SEALED for seed in seeds):
        raise ValueError(f"Forbidden held-out seed in {name}")


def main() -> int:
    """Run the unreduced original matrices and preserve commands and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Campaign execution requires a Slurm allocation")
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for name, script, seeds, flags in CAMPAIGNS:
        check_seeds(name, seeds)
        if name == "stage_a":
            check_seeds("profile sampling (no environment steps)", (6969,))
        command = [
            sys.executable,
            str(root / "scripts/validation" / script),
            "--output-dir",
            str(args.output_dir / name),
        ]
        if name != "demo":
            command += ["--seeds", ",".join(map(str, seeds))]
        command += list(flags)
        if name in {"reference", "stage_a"}:
            command += list(REFERENCE_PROTOCOL)
        commands.append({"campaign": name, "seeds": list(seeds), "command": command})
        write_json(
            args.output_dir / "receipt.json",
            {
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "job_id": os.environ["SLURM_JOB_ID"],
                "runtime": platform.platform(),
                "python": platform.python_version(),
                "commands": commands,
                "reduction": "none",
                "status": "running",
                "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
            },
        )
        with (args.output_dir / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"COMPLETE {name}", flush=True)
    receipt = json.loads((args.output_dir / "receipt.json").read_text())
    receipt["status"] = "complete"
    write_json(args.output_dir / "receipt.json", receipt)
    write_sha256sums(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
