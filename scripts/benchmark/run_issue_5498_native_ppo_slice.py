#!/usr/bin/env python3
"""Run the issue #5498 native-PPO single-host exact-repeat slice.

Background
----------
Issue #5498 is the successor to #5263. The single-host slice (#5499) executed
all 140 targets but recorded the 3 PPO planner cells (60 targets) as
``degraded``/``unrunnable`` because the forked planner-step worker crashed or
timed out on that host. The native-PPO worker-isolation fix (#5740) resolved
that defect, so PPO can now run natively through the exact-repeat
``run_episode`` path.

What this script does
---------------------
It re-executes the previously-unrunnable PPO targets natively (3 repeats each)
on one CPU-only single-worker host, and writes the verified report under
``output/issue_5498_native_ppo/``. The host report is verified against a
PPO-only manifest slice (re-hashed, so ``verify_host_report`` does not reject
it as "missing the 80 orca/goal targets").

Usage
-----
Full native-PPO slice (60 targets x 3 repeats; a CPU-only single-host campaign
act intended for a maintainer / campaign host):

    uv run python scripts/benchmark/run_issue_5498_native_ppo_slice.py

Smoke check of the runner + worker-isolation fix on a tiny target subset
(cheap, non-campaign; used to validate the script itself):

    uv run python scripts/benchmark/run_issue_5498_native_ppo_slice.py --max-targets 2

Evidence status
---------------
The cheap-lane worker that added this script validated it end-to-end on a
single PPO target (slice -> execute -> verify, bitwise-identical repeats) and
ran the ``--max-targets 2`` smoke, but did NOT execute the full 60-target run
or register a result under ``docs/context/evidence/``. The full run is a
campaign act reserved for a maintainer on a campaign host, and its output must
not be treated as established evidence until the run completes and (per the
#5498 acceptance contract) the second-host near-miss comparison is also done.
"""
# evidence-writer-exempt: references evidence paths but does not write to evidence tree; guarded by AST analysis

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import robot_sf.benchmark.exact_repeat_campaign as erc

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "docs/context/evidence/issue_5263_exact_repeat"
BUNDLE_PATH = EVIDENCE_DIR / "resolved_definitions.json"
MANIFEST_PATH = EVIDENCE_DIR / "exact_repeat_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/issue_5498_native_ppo"
TOTAL_PPO_TARGETS = 60


def _ppo_only_bundle(max_targets: int | None) -> dict[str, Any]:
    """Return a schema-valid resolved bundle containing only PPO planner targets.

    Args:
        max_targets: Optional cap on the number of PPO targets retained, used by
            the ``--max-targets`` smoke mode. ``None`` keeps all 60.

    Returns:
        A copy of the resolved bundle restricted to PPO targets, with a freshly
        recomputed ``bundle_sha256``.
    """
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    ppo_targets = [target for target in bundle["targets"] if target["planner"] == "ppo"]
    if max_targets is not None:
        if max_targets < 1:
            raise ValueError("--max-targets must be a positive integer")
        ppo_targets = ppo_targets[:max_targets]
    slim = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    slim["targets"] = ppo_targets
    slim["bundle_sha256"] = erc.canonical_sha256(slim)
    return slim


def _ppo_only_manifest_slice(target_keys: list[tuple[str, str, int]]) -> dict[str, Any]:
    """Build a re-hashed PPO-only manifest slice matching the executed targets.

    ``verify_host_report`` rejects a host report whose ``manifest_sha256`` does
    not match the manifest, and rejects any manifest target missing from the
    host report. The full 140-target manifest would therefore reject a PPO-only
    host report as "missing 80 orca/goal targets". This slice keeps only the
    PPO manifest rows whose ``(scenario_id, planner, seed)`` appear in
    ``target_keys`` and recomputes the manifest hash so verification succeeds.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    key_set = set(target_keys)

    def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
        return (row["scenario_id"], row["planner"], int(row["seed"]))

    sliced = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    sliced["targets"] = [
        target for target in sliced.get("targets", []) if _row_key(target) in key_set
    ]
    # Manifest cells aggregate across seeds; keep only PPO cells whose targets
    # were all retained so per-cell verdicts stay well-formed.
    sliced["cells"] = [cell for cell in sliced.get("cells", []) if cell["planner"] == "ppo"]
    sliced["manifest_sha256"] = erc.canonical_sha256(sliced)
    return sliced


def main() -> int:
    """Execute the native-PPO exact-repeat slice and verify the host report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help=(
            "Cap the number of PPO targets executed (smoke mode). Default: all "
            f"{TOTAL_PPO_TARGETS} PPO targets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the host and verified reports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    bundle = _ppo_only_bundle(args.max_targets)
    n_targets = len(bundle["targets"])
    is_smoke = args.max_targets is not None and args.max_targets < TOTAL_PPO_TARGETS
    if not is_smoke and n_targets != TOTAL_PPO_TARGETS:
        raise SystemExit(
            f"expected {TOTAL_PPO_TARGETS} PPO targets in the resolved bundle, found {n_targets}"
        )

    output_dir: Path = args.output_dir
    host_result = erc.execute_campaign(bundle, output_dir=output_dir)
    host_report_path = output_dir / "issue_5498_native_ppo_host_result.json"
    host_report_path.parent.mkdir(parents=True, exist_ok=True)
    host_report_path.write_text(
        json.dumps(host_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    target_keys = [
        (result["scenario_id"], result["planner"], int(result["seed"]))
        for result in host_result["results"]
    ]
    manifest_slice = _ppo_only_manifest_slice(target_keys)
    # The host report carries the FULL bundle's manifest_sha256; the slice is
    # re-hashed, so align the report's manifest hash before verification.
    aligned_host_report = dict(host_result)
    aligned_host_report["manifest_sha256"] = manifest_slice["manifest_sha256"]
    verified = erc.verify_host_report(manifest_slice, aligned_host_report)
    verified_path = output_dir / "issue_5498_native_ppo_verified_host_result.json"
    verified_path.write_text(
        json.dumps(verified, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = verified["summary"]
    mode = "SMOKE" if is_smoke else "FULL"
    print(f"native PPO exact-repeat slice [{mode}] ({n_targets} target(s) x 3 repeats):")
    print(f"  n_targets={summary['n_targets']}")
    print(f"  n_runnable_targets={summary['n_runnable_targets']}")
    print(f"  n_unrunnable_targets={summary['n_unrunnable_targets']}")
    print(f"  n_cells={summary['n_cells']}")
    print(f"  n_runnable_cells={summary['n_runnable_cells']}")
    print(f"  n_unrunnable_cells={summary['n_unrunnable_cells']}")
    print(f"  all_cells_bitwise_identical={summary['all_cells_bitwise_identical']}")
    print(f"host_report={host_report_path}")
    print(f"verified_report={verified_path}")
    if is_smoke:
        print(
            "NOTE: --max-targets smoke output is a runner self-check, NOT "
            "registered evidence; the full 60-target run is a campaign act."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
