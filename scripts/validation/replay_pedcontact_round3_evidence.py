"""Extract SHA-verified PEDCONTACT analysis drivers and optionally render evidence.

Raw metadata remains in the owned lane. This entry point never acquires episodes,
submits cluster jobs or restores trajectories. Rendering requires the full extras
environment and the preserved Round 3 metadata corpus.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ANALYSIS_DRIVERS = (
    "verify_producer_round3_final.py",
    "robot_failure_geometry_round3_final.py",
    "robot_spawn_audit_round3_final.py",
    "v6_negative_control_round3_final.py",
    "fit_runtime_round3_final.py",
    "audit_results_round3_final.py",
    "fit_interpretation_round3_final.py",
    "prepare_purity_round3_final.py",
)


def extract(bundle, lane):
    """Verify source bytes and write only missing files inside the explicit lane.

    Returns:
        Number of verified files and newly extracted files.
    """
    created = 0
    for name, entry in bundle["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe driver path: {name}")
        destination = lane / relative
        if not destination.resolve().is_relative_to(lane):
            raise ValueError(f"Driver path escapes lane: {name}")
        content = entry["content"].encode("utf-8")
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Driver digest mismatch: {name}")
        if destination.exists():
            if destination.read_bytes() != content:
                raise ValueError(f"Existing file differs; preserve it: {name}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            created += 1
    return len(bundle["files"]), created


def render(lane, bundle):
    """Run the explicit analysis-only generators against preserved metadata."""
    names = (*ANALYSIS_DRIVERS, "publish_round3_final.py", "publication_round3_final.py")
    if any(name not in bundle["files"] for name in names):
        raise ValueError("Historical bundle lacks the current analysis drivers")
    repo = lane / "repo"
    (lane / "tmp").mkdir(exist_ok=True)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        XDG_CACHE_HOME=str(lane / "cache"),
        UV_CACHE_DIR=str(lane / "cache/uv"),
        NUMBA_CACHE_DIR=str(lane / "cache/numba"),
        MPLCONFIGDIR=str(lane / "cache/mpl"),
        TMPDIR=str(lane / "tmp"),
    )
    for name in ANALYSIS_DRIVERS:
        subprocess.run([sys.executable, str(lane / name)], cwd=repo, env=env, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validation.render_pedcontact_round3_evidence",
            "--evidence-root",
            str(lane / "evidence/round3-final"),
            "--output-dir",
            str(repo / "docs"),
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    for name in ("publish_round3_final.py", "publication_round3_final.py"):
        subprocess.run([sys.executable, str(lane / name)], cwd=repo, env=env, check=True)


def main():
    """Read the committed bundle; extraction is the default, rendering explicit."""
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=repo / "docs/pedcontact_10101_round3_drivers.json"
    )
    parser.add_argument("--lane-root", type=Path, default=repo.parent)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    lane = args.lane_root.resolve()
    bundle = json.loads(args.bundle.read_text())
    count, created = extract(bundle, lane)
    print(f"Verified {count} driver/input files; extracted {created}")
    if args.render:
        render(lane, bundle)


if __name__ == "__main__":
    main()
