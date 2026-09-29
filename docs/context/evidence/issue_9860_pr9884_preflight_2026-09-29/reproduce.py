#!/usr/bin/env python3
"""Re-run the three setup-only #9860 preflights from this PR checkout."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pair_reports import (
    BASE_CHECKER_COMMIT,
    FIXED_CHECKER_COMMIT,
    pair_reports,
    read_report,
    write_pair,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CHECKER = Path("robot_sf/benchmark/spawn_preflight.py")
PREFLIGHT = Path("scripts/benchmark/preflight_spawn_clearance.py")
CORRECTED_MANIFEST = Path("configs/benchmarks/releases/issue_9860_corrected_diagnostic_v1.yaml")
GUARD_MANIFEST = Path("configs/benchmarks/releases/issue_9860_unsafe_probe_diagnostic_v1.yaml")
GOAL_SAMPLING_OVERLAY = HERE / "goal_clearance_runtime.patch.gz"


def digest(path: Path) -> str:
    """Hash a file exactly as recorded in the input closure."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    """Run Git from the PR checkout."""
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_input_closure(root: Path, hashes: dict[str, str]) -> None:
    """Fail on any absent or changed manifest dependency."""
    for rel, expected in hashes.items():
        path = root / rel
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"input closure mismatch: {rel}")


def apply_diagnostic_overlay(checkout: Path) -> None:
    """Apply the committed goal-sampling dependency in an isolated checkout."""
    patch_bytes = gzip.decompress(GOAL_SAMPLING_OVERLAY.read_bytes())
    subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=checkout,
        input=patch_bytes,
        check=True,
    )
    subprocess.run(["git", "apply", "-"], cwd=checkout, input=patch_bytes, check=True)


def run_preflight(
    checkout: Path, manifest: Path, output_dir: Path, name: str, expected_rc: int
) -> tuple[Path, Path]:
    """Run one matrix preflight with the original settings."""
    json_path = output_dir / f"{name}.json"
    markdown_path = output_dir / f"{name}.md"
    cmd = [
        sys.executable,
        str(PREFLIGHT),
        "--manifest",
        str(manifest),
        "--workers",
        "4",
        "--json-output",
        str(json_path),
        "--markdown-output",
        str(markdown_path),
    ]
    result = subprocess.run(cmd, cwd=checkout, text=True, capture_output=True, check=False)
    if result.returncode != expected_rc:
        raise RuntimeError(
            f"{name} preflight exited {result.returncode}, expected {expected_rc}: "
            f"{result.stderr[-1500:]}"
        )
    return json_path, markdown_path


def compare_report(name: str, fresh: Path, markdown: Path) -> dict[str, object]:
    """Verify every field except the observed wall time, then check Markdown bytes."""
    old, old_raw, old_stable = read_report(HERE / f"{name}.json.gz")
    new, new_raw, new_stable = read_report(fresh)
    if old_stable != new_stable:
        differing = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
        raise RuntimeError(f"{name} report differs beyond runtime_s: {differing}")
    if (HERE / f"{name}.md").read_bytes() != markdown.read_bytes():
        raise RuntimeError(f"{name} Markdown report differs")
    return {
        "stored_json_sha256": old_raw,
        "rerun_json_sha256": new_raw,
        "stable_sha256": old_stable,
        "stored_runtime_s": old["runtime_s"],
        "rerun_runtime_s": new["runtime_s"],
        "raw_hash_match": old_raw == new_raw,
        "only_difference_if_raw_hash_changed": "runtime_s",
    }


def main() -> None:
    """Reconstruct baseline inputs, run three preflights, and compare receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if git("rev-parse", "--show-toplevel") != str(ROOT):
        raise RuntimeError("run from the PR checkout containing this evidence")
    if git("rev-parse", f"{BASE_CHECKER_COMMIT}^{{commit}}") != BASE_CHECKER_COMMIT:
        raise RuntimeError("baseline checker commit is missing")
    if git("rev-parse", f"{FIXED_CHECKER_COMMIT}^{{commit}}") != FIXED_CHECKER_COMMIT:
        raise RuntimeError("fixed checker commit is missing")
    fixed_bytes = subprocess.check_output(
        ["git", "show", f"{FIXED_CHECKER_COMMIT}:{CHECKER}"], cwd=ROOT
    )
    if (ROOT / CHECKER).read_bytes() != fixed_bytes:
        raise RuntimeError("current checker differs from the fixed checker commit")

    closure = json.loads((HERE / "input_closure.json").read_text(encoding="utf-8"))
    hashes = closure["sha256_files"]
    assert len(hashes) == closure["file_count"] == 79
    verify_input_closure(ROOT, hashes)
    if digest(HERE / "seed_sets_v1.yaml") != hashes["configs/benchmarks/seed_sets_v1.yaml"]:
        raise RuntimeError("adjacent seed-set copy differs from the manifest input")

    with tempfile.TemporaryDirectory(prefix="pr9884-base-") as tmp:
        base = Path(tmp) / "base"
        git("worktree", "add", "--detach", "--quiet", str(base), BASE_CHECKER_COMMIT)
        try:
            for rel, expected in hashes.items():
                target = base / rel
                if not target.is_file() or digest(target) != expected:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / rel, target)
            verify_input_closure(base, hashes)
            apply_diagnostic_overlay(base)
            before_json, before_md = run_preflight(
                base, CORRECTED_MANIFEST, output_dir, "before", 2
            )
        finally:
            git("worktree", "remove", "--force", str(base))

    with tempfile.TemporaryDirectory(prefix="pr9884-fixed-") as tmp:
        fixed = Path(tmp) / "fixed"
        git("worktree", "add", "--detach", "--quiet", str(fixed), "HEAD")
        try:
            verify_input_closure(fixed, hashes)
            apply_diagnostic_overlay(fixed)
            after_json, after_md = run_preflight(fixed, CORRECTED_MANIFEST, output_dir, "after", 0)
            guard_json, guard_md = run_preflight(
                fixed, GUARD_MANIFEST, output_dir, "unsafe_probe", 2
            )
        finally:
            git("worktree", "remove", "--force", str(fixed))
    comparisons = {
        "before": compare_report("before", before_json, before_md),
        "after": compare_report("after", after_json, after_md),
        "unsafe_probe": compare_report("unsafe_probe", guard_json, guard_md),
    }
    fresh_pair = pair_reports(before_json, after_json, guard_json)
    stored_pair = json.loads((HERE / "paired.json").read_text(encoding="utf-8"))
    for key in ("before_report_sha256", "after_report_sha256", "unsafe_probe_report_sha256"):
        fresh_pair.pop(key)
        stored_pair.pop(key)
    if fresh_pair != stored_pair:
        raise RuntimeError("rerun pairing differs from the committed receipt")
    write_pair(pair_reports(before_json, after_json, guard_json), output_dir)
    receipt = {
        "schema_version": "issue_9860_pr9884_fresh_clone_reproduction.v1",
        "source_head": git("rev-parse", "HEAD"),
        "input_closure_sha256": digest(HERE / "input_closure.json"),
        "diagnostic_overlay_sha256": digest(GOAL_SAMPLING_OVERLAY),
        "reports": comparisons,
        "transition_counts": stored_pair["transitions"],
        "guard_unsafe_nominal_blocked": stored_pair["unsafe_nominal_goals_blocked"],
        "guard_doorway_blocked": stored_pair["historical_doorway_probe_cells_blocked"],
        "planner_actions": 0,
    }
    (output_dir / "reproduction_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
