#!/usr/bin/env python3
"""Verify preserved development acquisition and emit portable review evidence.

Run from the frozen producer checkout with that checkout on PYTHONPATH. This reads existing rows;
it never creates an environment, executes a planner, scores or ranks planners.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from build_snqi_v2_determinism_receipt import build_receipt, load_rows
from scipy.stats import spearmanr

from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.fallback_policy import campaign_status_axes_payload, resolve_execution_mode
from robot_sf.benchmark.robot_force_contract import validate_robot_force_provenance
from robot_sf.benchmark.snqi.v2_binding import bind_acquired_anchors
from robot_sf.benchmark.snqi.v2_calibration import freeze_campaign_anchors
from robot_sf.benchmark.snqi.v2_spec import PP_EQUIV_FORCE, SIMULATED_FORCE
from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.evidence.writers import write_review_sidecar


def digest(path: Path) -> str:
    """Return a streamed SHA-256 of actual bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_member(root: Path, relative: str) -> Path:
    """Resolve a regular custody member, refusing traversal and symlinks."""
    member = Path(relative)
    candidate = root / member
    if member.is_absolute() or ".." in member.parts or not candidate.is_file():
        raise ValueError(f"unsafe or missing custody member: {relative}")
    if candidate.is_symlink() or any(p.is_symlink() for p in candidate.parents):
        raise ValueError(f"symlinked custody member: {relative}")
    return candidate


def check_producer(root: Path, head: str) -> tuple[dict, dict]:
    """Verify complete source-bound producer custody before numeric analysis."""
    signed = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        sha, relative = line.split("  ", 1)
        if relative in signed or digest(safe_member(root, relative)) != sha:
            raise ValueError("duplicate or mismatched producer SHA256SUMS member")
        signed[relative] = sha
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "SHA256SUMS"
    }
    if not signed or set(signed) != actual:
        raise ValueError("producer manifest does not cover the complete tree")
    producer = json.loads((root / "producer_provenance.json").read_text())
    completion = json.loads((root / "producer_exit.json").read_text())
    if (
        producer["source_commit"] != head
        or completion["output_status"] != "complete"
        or completion["campaign_exit_code"] != 0
        or completion["sync_exit_code"] != 0
        or completion["finalization_errors"]
    ):
        raise ValueError("incomplete or wrong-source producer")
    return signed, producer


def check_cold_snapshot(args: argparse.Namespace) -> tuple[Path, dict, dict]:
    """Compare every remote downloaded object and decoded byte with the snapshot."""
    manifest_path = args.cold_root / "campaign_preservation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(args.preservation_receipt.read_text())
    canonical = json.dumps(
        {k: v for k, v in manifest.items() if k != "manifest_digest"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    if manifest["manifest_digest"] != "sha256:" + hashlib.sha256(canonical.encode()).hexdigest():
        raise ValueError("cold preservation manifest digest mismatch")
    expected = {entry["path"] for entry in manifest["files"]}
    actual = {
        str(p.relative_to(args.snapshot_root)) for p in args.snapshot_root.rglob("*") if p.is_file()
    }
    if not expected or len(expected) != len(manifest["files"]) or expected != actual:
        raise ValueError("preservation manifest must cover the complete independent snapshot")
    if (
        receipt["status"] != "verified"
        or receipt["manifest_digest"] != manifest["manifest_digest"]
        or not receipt["two_copy_policy"]["satisfied"]
    ):
        raise ValueError("preservation receipt did not verify the two-copy contract")
    for entry in manifest["files"]:
        cold = safe_member(args.cold_root, entry["stored_path"])
        snapshot = safe_member(args.snapshot_root, entry["path"])
        with gzip.open(cold, "rb") as decoded:
            decoded_sha = hashlib.file_digest(decoded, "sha256").hexdigest()
        if (
            digest(cold) != entry["stored_sha256"]
            or decoded_sha != entry["sha256"]
            or digest(snapshot) != entry["sha256"]
        ):
            raise ValueError("cold/snapshot byte mismatch")
    return manifest_path, manifest, receipt


def parse_args() -> argparse.Namespace:
    """Require explicit preserved input and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "producer-root",
        "anchors",
        "snapshot-root",
        "cold-root",
        "preservation-receipt",
        "rehearsal",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--repeat-producer-root", type=Path)
    parser.add_argument("--rehearsal-root", type=Path)
    parser.add_argument("--private-determinism-receipt", type=Path)
    parser.add_argument("--repeat-snapshot-root", type=Path)
    parser.add_argument("--repeat-cold-root", type=Path)
    parser.add_argument("--repeat-preservation-receipt", type=Path)
    return parser.parse_args()


def check_execution(execution: dict) -> None:
    """Reject actual arm-level failures and availability substitutions."""
    counts = execution["row_status_summary"]
    if (
        execution["evidence_status"] != "valid"
        or counts["successful_evidence_rows"] != 14
        or any(
            counts[k]
            for k in (
                "accepted_unavailable_rows",
                "unexpected_failed_rows",
                "fallback_or_degraded_rows",
            )
        )
    ):
        raise ValueError("incomplete or degraded planner-arm execution")


def check_candidate_attachment(source: Path, campaign: Path, anchors_path: Path, head: str) -> None:
    """Attach actual acquired custody and artifact-only bytes without executing episodes."""
    candidate = load_campaign_config(
        source
        / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml",
        repository_root=source,
    )
    for custody_root in (campaign, None):
        bound = bind_acquired_anchors(
            candidate,
            calibration_root=custody_root,
            anchors_path=anchors_path,
            source_commit=head,
            diagnostic=True,
        )
        if bound.snqi_v2_spec.hashes["anchors"] != digest(anchors_path):
            raise ValueError("candidate attachment anchor digest differs")


def check_force_samples(row: dict) -> None:
    """Check finite recorded samples, including explicit authored no-pedestrian rows."""
    metrics = row["metrics"]
    samples = metrics["force_sample_stats"]
    if (
        samples["invalid_samples"]
        or samples["raw_samples"] != samples["finite_samples"]
        or samples["zero_force_samples"] + samples["nonzero_force_samples"]
        != samples["finite_samples"]
    ):
        raise ValueError("nonfinite recorded per-pedestrian force samples")
    if samples["raw_samples"] > 0 and samples["status"] in {"ok", "all-zero"}:
        return
    if (
        samples["raw_samples"] == 0
        and samples["status"] == "no-pedestrians"
        and row["scenario_params"]["metadata"].get("density_advisory")
        == "zero_baseline_route_spawn"
        and row["interaction_exposure"]["first_clearance_reason"] == "no_pedestrians"
        and metrics[SIMULATED_FORCE] == metrics[PP_EQUIV_FORCE] == 0
    ):
        return
    raise ValueError("missing force samples without the authored zero-pedestrian baseline")


def write_scalars(output_dir: Path, scalars: list) -> None:
    """Emit sortable, independently readable calibration scalar columns."""
    csv_path = output_dir / "calibration-scalars.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "arm",
                "scenario_id",
                "seed",
                "simulated_force",
                "pp_equivalent_force",
                "close_clearance_fraction",
                "jerk_mean",
                "curvature_mean",
            )
        )
        writer.writerows(sorted(scalars))
    write_review_sidecar(csv_path, repo_root=Path(__file__).resolve().parents[2])


def write_review_outputs(args: argparse.Namespace, proof: dict) -> None:
    """Retain exact official anchors and render their portable review and replay notes."""
    root = Path(__file__).resolve().parents[2]
    anchor_output = args.output_dir / "anchors.v2.0.acquired.json"
    anchor_output.write_bytes(args.anchors.read_bytes())
    write_review_sidecar(anchor_output, repo_root=root)
    metadata = args.output_dir / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "distance_convention": "surface_clearance",
                "scope": "Underlying near-miss predicate; close_clearance_fraction is dimensionless count/steps",
                "source_commit": proof["source_commit"],
                "scalar_file": "calibration-scalars.csv",
            },
            indent=2,
        )
        + "\n"
    )
    write_review_sidecar(metadata, repo_root=root)
    values = proof["upper_anchors_recomputed"]
    delta = proof["rehearsal_delta"]
    repeat_notes = ""
    if "determinism_receipt" in proof:
        repeat_notes = render_determinism_notes(args.output_dir, proof)
    readme = args.output_dir / "README.md"
    readme.write_text(f"""# Frozen-source SNQI v2 development acquisition

AI-GENERATED NEEDS-REVIEW. Scientific review remains pending.

Job {proof["scheduler_job_id"]} acquired 14 arms × 48 authored cells × dev seeds 1001/1002
at `{proof["source_commit"]}`. The 1,344 unique rows and complete producer
hashes/sidecars passed the repository's strict freezer. All 14 arms completed;
no fallback, degraded, failed or unavailable execution was accepted. Navigation
collisions and timeouts remain outcomes, separate from planner execution failures.

[Acquired anchors](anchors.v2.0.acquired.json), [recomputed proof](acquisition-proof.json)
and [scalar rows](calibration-scalars.csv) retain the actual acquisition. Modes:
goal and PPO native (96 each), guarded PPO mixed (96), eleven adapter arms
(96 each). All rows declare recorded robot/pedestrian social force sampled
pre-integration: model acceleration, not measured human discomfort. Force sample
statistics are finite; 28 authored `classic_bottleneck_low` zero-pedestrian cells
are explicitly counted separately from 1,316 sampled cells. No missing force
value is zero-imputed. The compact rows retain recorded scalars and provenance;
raw force vectors are not claimed reconstructed from the scalar CSV.
[Scalar metadata](metadata.json) declares `distance_convention=surface_clearance`
for the underlying near-miss predicate; the close-clearance fraction is dimensionless.

## Anchors and rehearsal comparison

| Term | Acquired p95 | Change from d56092ed rehearsal |
| --- | ---: | ---: |
| F | {values["F"]!r} | {delta["F"]!r} |
| J | {values["J"]!r} | {delta["J"]!r} |
| K | {values["K"]!r} | {delta["K"]!r} |

T=3 and N=0.25 remain normative. The selected F field remains
`robot_force_impulse_total`, because the recomputed absolute Spearman rho
{proof["force_decision_rho_recomputed"]!r} is below the preregistered 0.9 boundary.
J rose by approximately 0.93634%. Its zero-based p95 index is 1275.85; both
bracketing acquired values are {values["J"]!r}, from the seed-1001
`classic_doorway_high` rows of `hybrid_rule_v4_fast_progress_static_escape` and
`scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4`. The proof records those
actual order statistics. This is a changed empirical upper anchor, not rounding
of the rehearsal's 1.9396757022730264.

The acquisition configuration is byte-identical across rehearsal d56092ed and
freeze 3e73b04b; metric definitions, force kernel and dependency lock are unchanged.
The producer source and acquired trajectories are fresh. `changed_source_paths`
records source differences; rr10126 independently found them behaviourally inert
for this grid (bicycle-only changes, default-off flags and diagnostic metadata).
These two acquisitions do not isolate a causal effect
of any single source change or runtime nondeterminism; no such attribution is made.
The independent reviewer must assess this empirical difference against raw custody.
The new anchor SHA also binds the new source, run ID, manifest and row/sidecar hashes.

Guarded PPO arbitration counters are aggregated directly from all 96 raw rows in
`guard_arbitration_counts`, next to `command_mode_counts`. Guard-selected safe
controller actions are arbitration interventions, distinct from degraded planner
execution. `pedestrian_model.development_model=unknown` remains a provenance
limitation in these immutable rows; its writer repair is deferred to 0.0.9 in
[issue #10127](https://github.com/ll7/robot_sf_ll7/issues/10127).

{repeat_notes}

## Preservation and authority boundary

W&B `{proof["preservation"]["qualified_name"]}` is COMMITTED and checksum-verified.
Manifest `{proof["preservation"]["manifest_digest"]}` covers the full producer,
startup/source/config/checkpoint/runtime and scheduler/watcher receipts and anchors.
All {proof["preservation"]["cold_files_byte_verified"]} source members passed stored,
gzip-decoded and independent snapshot SHA-256 comparison using a fresh API and empty
cold cache. The operator retains those inputs and the receipt outside public Git.
Raw-custody and artifact-only attachment to the unchanged candidate passed without
any reset or step. The source pending-anchor file and acquisition YAML remain intact.
The placeholder pin test is inherited from #10125; stale comment work remains #10124.
All six runbook scientific-review boxes and private Git-blob trust pins remain pending.
No smoke, final mint, DOI change, sealed execution or publication was performed.

## Regeneration

Obtain the preserved complete producer and its independent snapshot, cold download
and original preservation receipt. Operator-supplied roots below contain no fixed
machine locator. `PRODUCER_SOURCE` must be clean at the named freeze with rebuilt
all-extras dependencies and staged checkpoints; `REVIEW_REPO` contains this script.
Run from the producer checkout because learned-checkpoint registry lookup is relative
to it. These commands analyze existing data only:

```bash
cd "$PRODUCER_SOURCE"
export PYTHONPATH="$PRODUCER_SOURCE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/tools/analyze_snqi_contract.py \
  --campaign-root "$PRODUCER_ROOT/benchmarks/{proof["run_id"]}" \
  --freeze-v2-anchors "$ANCHORS_OUTPUT"
.venv/bin/python "$REVIEW_REPO/scripts/dev/build_snqi_v2_acquisition_evidence.py" \
  --producer-root "$PRODUCER_ROOT" --anchors "$ANCHORS_OUTPUT" \
  --snapshot-root "$SNAPSHOT_ROOT" --cold-root "$COLD_ROOT" \
  --preservation-receipt "$PRESERVATION_RECEIPT" \
  --rehearsal docs/context/evidence/2026-10-03_issue10112_mintorder/calibration-d56092ed-grid-proof.json \
  --output-dir "$REVIEW_REPO/docs/context/evidence/2026-10-04_freeze008_calibration" \
  --source-commit {proof["source_commit"]} \
  --runtime-commit {proof["producer_runtime_commit"]} \
  --launcher-sha256 {proof["launcher_sha256"]} \
  --repeat-producer-root "$REPEAT_PRODUCER_ROOT" --rehearsal-root "$REHEARSAL_ROOT" \
  --repeat-snapshot-root "$REPEAT_SNAPSHOT_ROOT" --repeat-cold-root "$REPEAT_COLD_ROOT" \
  --repeat-preservation-receipt "$REPEAT_PRESERVATION_RECEIPT"
```
""")
    write_review_sidecar(readme, repo_root=root)


def compare_rehearsal(
    args: argparse.Namespace, producer: dict, head: str, scalars: list
) -> tuple[dict, dict]:
    """Compare actual source/config bytes and identify the acquired J order statistics."""
    if head != "3e73b04b43aa99b9fbe4a6ab34b89a5a9f1933b6":
        raise ValueError("inert-source audit applies only to the reviewed freeze")
    rehearsal = json.loads(args.rehearsal.read_text())["anchors"]
    old = rehearsal["anchors"]
    previous = rehearsal["calibration"]["source_commit"]
    if previous != "d56092ed9d4b442f9dfb99e31010a9f5a8666547":
        raise ValueError("inert-source audit applies only to the reviewed rehearsal")
    config_path = "configs/benchmarks/snqi_v2/calibration.dev1001_1002_scheduled_acquisition.yaml"
    previous_config = subprocess.check_output(["git", "show", f"{previous}:{config_path}"])
    if hashlib.sha256(previous_config).hexdigest() != producer["config_sha256"]:
        raise ValueError("rehearsal and acquired configuration bytes differ")
    changes = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            previous,
            head,
            "--",
            "robot_sf",
            "fast-pysf",
            "configs",
            "uv.lock",
        ],
        text=True,
    ).splitlines()
    jerk_order = sorted((row[6], row[0], row[1], row[2]) for row in scalars)
    jerk_index = (len(jerk_order) - 1) * 0.95
    return old, {
        "source_commit": previous,
        "acquisition_config_byte_identical": True,
        "changed_source_paths": changes,
        "changed_source_paths_behaviourally_inert_for_this_grid": True,
        "changed_source_paths_behaviourally_inert_reason": "rr10126 independently reviewed the complete d56092ed-to-3e73b04b diff: behavioural changes are bicycle-only or gated by default-off flags on this differential-drive grid; other changes are diagnostic metadata.",
        "causal_attribution": "not isolated by these two acquisitions",
        "jerk_p95_zero_based_index": jerk_index,
        "jerk_p95_bracketing_rows": jerk_order[int(jerk_index) : int(jerk_index) + 2],
    }


def aggregate_guard_counters(arm: str, row: dict, totals: dict) -> None:
    """Retain nonnegative raw arbitration counters for the guarded PPO arm."""
    if arm != "guarded_ppo":
        return
    counters = row["algorithm_metadata"]["guard_stats"]
    if not counters or any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError("missing or invalid guarded PPO arbitration counters")
    totals.setdefault(arm, Counter()).update(counters)


def verify_repeat_census(campaign: Path, force_source: str, execution: dict) -> dict:
    """Retain the repeat's checked force and guard census beside its metric-row receipt."""
    rows, _ = load_rows(campaign)
    sources, timings, sample_statuses, force_samples, guards = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        {},
    )
    for key, row in rows.items():
        check_force_samples(row)
        provenance = validate_robot_force_provenance(row["metrics"], force_source)
        sources[provenance["source"]] += 1
        timings[provenance["sample_timing"]] += 1
        samples = row["metrics"]["force_sample_stats"]
        sample_statuses[samples["status"]] += 1
        force_samples.update(
            {
                field: samples[field]
                for field in (
                    "raw_samples",
                    "finite_samples",
                    "invalid_samples",
                    "zero_force_samples",
                    "nonzero_force_samples",
                )
            }
        )
        aggregate_guard_counters(key[0], row, guards)
    return {
        "rows": len(rows),
        "arm_execution": execution,
        "force_sources": sources,
        "force_sample_timing": timings,
        "recorded_force_sample_totals": force_samples,
        "force_sample_status_counts": sample_statuses,
        "guard_arbitration_counts": guards,
    }


def render_determinism_notes(output_dir: Path, proof: dict) -> str:
    """Render the measured result while keeping the six scientific decisions independent."""
    receipt = json.loads((output_dir / "determinism-receipt.json").read_text())
    compared = receipt["original_vs_repeat"]
    previous = receipt["rehearsal_vs_original"]
    context = receipt["execution_contexts"]
    verdict = receipt["classification"]
    context_status = "match" if receipt["same_recorded_environment"] else "do not match"
    message = (
        "J is reproducible for this fixed recorded environment; keep the point anchor."
        if verdict == "a"
        else "The repeat does not establish fixed-environment reproducibility; release remains blocked."
    )
    custody_notes = ""
    if "repeat_preservation" in proof:
        custody = proof["repeat_preservation"]
        custody_notes = f"Repeat W&B `{custody['qualified_name']}` is COMMITTED; all {custody['cold_files_byte_verified']} source members pass stored/decoded cold and independent snapshot hashes. Manifest `{custody['manifest_digest']}` includes the complete repeat, both determinism receipts and recovered rehearsal raw inputs."
    return f"""## Fixed-environment repeat and environment sensitivity

[Determinism receipt](determinism-receipt.json) compares all 1,344 metric-column,
steps and status hashes between jobs {receipt["scheduler_job_ids"]["original"]} and
{receipt["scheduler_job_ids"]["repeat"]}: {compared["identical_rows"]} identical and
{compared["different_rows"]} different rows, classification `{verdict}`.
{message} The recorded node identity, CPU/software/thread context {context_status};
16 workers, subprocess arm isolation and all three thread limits of one remain fixed.
Public custody uses hashed node identities; private scheduler receipts retain actual names.

The rehearsal raw rows were recovered and all 14 file hashes match its committed
d56092ed anchor proof. Compared with job 21331, {previous["different_rows"]} rows differ:
{json.dumps(previous["different_rows_by_arm"], sort_keys=True)}; {previous["step_differences"]}
step-count and {previous["status_differences"]} navigation-status differences.
Every differing row and both steps/status values are retained in the receipt.
Rehearsal CPU: {context["rehearsal"]["cpu_model"]}; acquisition CPU:
{context["original"]["cpu_model"]}. Node, kernel and glibc differ; recorded
Python/NumPy/Numba versions and thread limits match. F and K delta are zero;
J delta is {receipt["J_delta_percent"]!r}% ({receipt["rehearsal_to_original_delta"]["J"]!r}).
This is the anchors' measured environment sensitivity, consistent with the
[documented machine/compiler-conditional dynamics sensitivity](../../../benchmark_release_reproducibility.md).
The experiment does not isolate a pedestrian fast-math or PPO arithmetic mechanism.

The rehearsal p95 is a linear interpolation between 1.930676903661017
(`guarded_ppo`, `francis2023_leave_group`, 1002) and 1.9412637255574998
(`socnav_sampling`, `classic_bottleneck_high`, 1002), so it need not appear in any
raw jerk sample. The unchanged hybrid three-way tie moves into the p95 bracket
as the PPO-arm distribution changes. Exact brackets and all paired metric hashes
are recorded; the acquired anchor bytes remain unchanged.

{custody_notes}
"""


def write_repeat_preservation(args: argparse.Namespace, proof: dict) -> None:
    """Recheck independently downloaded repeat custody without creating circular receipts."""
    repeat_cold = (
        args.repeat_snapshot_root,
        args.repeat_cold_root,
        args.repeat_preservation_receipt,
    )
    if any(repeat_cold):
        if not all(repeat_cold) or not args.repeat_producer_root:
            raise ValueError("repeat cold custody requires all three roots and repeat producer")
        namespace = argparse.Namespace(
            snapshot_root=args.repeat_snapshot_root,
            cold_root=args.repeat_cold_root,
            preservation_receipt=args.repeat_preservation_receipt,
        )
        manifest_path, manifest, receipt = check_cold_snapshot(namespace)
        proof["repeat_preservation"] = {
            "qualified_name": receipt["artifact"]["qualified_name"],
            "manifest_digest": manifest["manifest_digest"],
            "receipt_sha256": digest(args.repeat_preservation_receipt),
            "cold_files_byte_verified": len(manifest["files"]),
            "cold_manifest_sha256": digest(manifest_path),
        }


def write_determinism_evidence(
    args: argparse.Namespace, proof: dict, campaign: Path, root: Path, producer: dict, head: str
) -> None:
    """Bind separately preserved repeat and rehearsal rows to this original acquisition."""
    if bool(args.repeat_producer_root) != bool(args.rehearsal_root):
        raise ValueError("repeat producer and rehearsal raw roots must be provided together")
    if args.repeat_producer_root:
        _, repeated_producer = check_producer(args.repeat_producer_root, head)
        for field in ("private_ops_commit", "launcher_sha256", "config_sha256"):
            if repeated_producer[field] != producer[field]:
                raise ValueError(f"repeat producer changed {field}")
        repeated_startup = json.loads((args.repeat_producer_root / "startup.json").read_text())[
            "identities"
        ]
        repeated_roots = list((args.repeat_producer_root / "benchmarks").iterdir())
        if len(repeated_roots) != 1 or repeated_startup["public_commit"] != head:
            raise ValueError("repeat campaign or source binding differs")
        repeat_campaign = repeated_roots[0]
        repeat_execution = campaign_status_axes_payload(
            json.loads((repeat_campaign / "reports/campaign_summary.json").read_text()),
            expected_total_runs=14,
        )
        check_execution(repeat_execution)
        with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
            repeat_anchor_path = Path(temporary) / "repeat-anchors.json"
            freeze_campaign_anchors(repeat_campaign, repeat_anchor_path)
            repeated_anchors = json.loads(repeat_anchor_path.read_text())
        proof["repeat_census"] = verify_repeat_census(
            repeat_campaign, repeated_anchors["force_decision"]["source"], repeat_execution
        )
        determinism = build_receipt(campaign, repeat_campaign, args.rehearsal_root)
        if determinism["upper_anchors"]["repeat"] != {
            term: repeated_anchors["anchors"][term]["upper"] for term in ("F", "J", "K")
        }:
            raise ValueError("repeat strict freezer and scalar recomputation differ")
        determinism["repeat_command_mode_counts"] = repeated_anchors["calibration"][
            "command_mode_counts"
        ]
        determinism["source_commit"] = head
        determinism["producer_runtime_commit"] = producer["private_ops_commit"]
        determinism["launcher_sha256"] = producer["launcher_sha256"]
        determinism["config_sha256"] = producer["config_sha256"]
        determinism["workers"] = 16
        determinism["arm_isolation"] = "subprocess"
        if (
            determinism["episode_files_sha256"]["rehearsal"]
            != json.loads(args.rehearsal.read_text())["anchors"]["calibration"][
                "episode_files_sha256"
            ]
        ):
            raise ValueError("rehearsal raw hashes differ from preserved public proof")
        determinism["scheduler_job_ids"] = {
            "original": json.loads((root / "startup.json").read_text())["identities"]["job_id"],
            "repeat": repeated_startup["job_id"],
            "rehearsal": "20299",
        }
        determinism["producer_manifest_sha256"] = {
            "original": digest(root / "SHA256SUMS"),
            "repeat": digest(args.repeat_producer_root / "SHA256SUMS"),
        }
        if args.private_determinism_receipt:
            private = dict(determinism)
            private["execution_contexts"] = build_receipt(
                campaign, repeat_campaign, args.rehearsal_root, portable=False
            )["execution_contexts"]
            args.private_determinism_receipt.write_text(
                json.dumps(private, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        receipt_path = args.output_dir / "determinism-receipt.json"
        receipt_path.write_text(
            json.dumps(determinism, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        write_review_sidecar(receipt_path, repo_root=Path(__file__).resolve().parents[2])
        proof["determinism_receipt"] = {
            "path": str(receipt_path.relative_to(Path(__file__).resolve().parents[2])),
            "sha256": digest(receipt_path),
            "classification": determinism["classification"],
            "same_recorded_environment": determinism["same_recorded_environment"],
            "identical_rows": determinism["original_vs_repeat"]["identical_rows"],
            "different_rows": determinism["original_vs_repeat"]["different_rows"],
        }
    write_repeat_preservation(args, proof)


def main() -> None:
    """Validate producer and cold bytes, rederive anchors, then emit review inputs."""
    args = parse_args()
    source = get_repository_root()
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != args.source_commit or Path.cwd().resolve() != source.resolve():
        raise ValueError(
            "analysis imports and working directory must use the exact producer checkout"
        )
    root = args.producer_root.resolve()
    signed, producer = check_producer(root, head)
    startup = json.loads((root / "startup.json").read_text())["identities"]
    if (
        startup["public_commit"] != head
        or producer["private_ops_commit"] != args.runtime_commit
        or producer["launcher_sha256"] != args.launcher_sha256
        or producer["config_sha256"] != digest(root / "config/benchmark_config.yaml")
        or producer["config_sha256"]
        != digest(
            source
            / "configs/benchmarks/snqi_v2/calibration.dev1001_1002_scheduled_acquisition.yaml"
        )
    ):
        raise ValueError("startup or retained acquisition config binding mismatch")
    anchors = json.loads(args.anchors.read_text())
    campaign = root / "benchmarks" / anchors["calibration"]["run_id"]
    summary = json.loads((campaign / "reports/campaign_summary.json").read_text())
    execution = campaign_status_axes_payload(summary, expected_total_runs=14)
    check_execution(execution)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
        rebuilt = Path(temporary) / "anchors.json"
        freeze_campaign_anchors(campaign, rebuilt)
        if rebuilt.read_bytes() != args.anchors.read_bytes():
            raise ValueError("raw-custody anchor rederivation differs")
    check_candidate_attachment(source, campaign, args.anchors, head)
    scalars, modes, outcomes, sources, timings = [], {}, Counter(), Counter(), Counter()
    force_samples, sample_statuses = Counter(), Counter()
    guard_counters = {}
    for relative, sha in sorted(anchors["calibration"]["episode_files_sha256"].items()):
        path = safe_member(campaign, relative)
        if digest(path) != sha:
            raise ValueError("anchor episode hash mismatch")
        arm = path.parent.name.removesuffix("__differential_drive")
        modes[arm] = Counter()
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                metrics = row["metrics"]
                samples = metrics["force_sample_stats"]
                sample_statuses[samples["status"]] += 1
                check_force_samples(row)
                force_samples.update(
                    {
                        k: samples[k]
                        for k in (
                            "raw_samples",
                            "finite_samples",
                            "invalid_samples",
                            "zero_force_samples",
                            "nonzero_force_samples",
                        )
                    }
                )
                provenance = validate_robot_force_provenance(
                    metrics, anchors["force_decision"]["source"]
                )
                sources[provenance["source"]] += 1
                timings[provenance["sample_timing"]] += 1
                outcomes[row["status"]] += 1
                mode = resolve_execution_mode(row["algorithm_metadata"])
                modes[arm][mode] += 1
                aggregate_guard_counters(arm, row, guard_counters)
                scalars.append(
                    (
                        arm,
                        row["scenario_id"],
                        row["seed"],
                        metrics[SIMULATED_FORCE],
                        metrics[PP_EQUIV_FORCE],
                        metrics["near_misses"] / row["steps"],
                        metrics["jerk_mean"],
                        metrics["curvature_mean"],
                    )
                )
    rho = float(
        spearmanr([r[3] for r in scalars], np.clip([r[5] / 0.25 for r in scalars], 0, 1)).statistic
    )
    selected = PP_EQUIV_FORCE if abs(rho) >= 0.9 else SIMULATED_FORCE
    values = {
        term: float(np.percentile([r[column] for r in scalars], 95, method="linear"))
        for term, column in (("F", 4 if selected == PP_EQUIV_FORCE else 3), ("J", 6), ("K", 7))
    }
    if (
        modes != anchors["calibration"]["command_mode_counts"]
        or selected != anchors["force_decision"]["source"]
        or any(values[k] != anchors["anchors"][k]["upper"] for k in values)
    ):
        raise ValueError("independent scalar recomputation differs")
    manifest_path, manifest, receipt = check_cold_snapshot(args)
    old, comparison = compare_rehearsal(args, producer, head, scalars)
    proof = {
        "schema": "snqi-v2-acquisition-review-input.v1",
        "scientific_review": "pending_independent_review",
        "source_commit": head,
        "run_id": anchors["calibration"]["run_id"],
        "rehearsal_comparison": comparison,
        "scheduler_job_id": startup["job_id"],
        "submission_id": startup["submission_id"],
        "producer_runtime_commit": producer["private_ops_commit"],
        "launcher_sha256": producer["launcher_sha256"],
        "calibration_rows": len(scalars),
        "seeds": anchors["calibration"]["seeds"],
        "producer_signed_files": len(signed),
        "producer_manifest_sha256": digest(root / "SHA256SUMS"),
        "producer_provenance_sha256": digest(root / "producer_provenance.json"),
        "config_sha256": producer["config_sha256"],
        "anchors_sha256": digest(args.anchors),
        "candidate_attachment": {
            "strict_raw_custody": "passed",
            "artifact_only": "passed",
            "episodes_executed": 0,
            "scientific_admission": False,
        },
        "command_mode_counts": modes,
        "guard_arbitration_counts": guard_counters,
        "arm_execution": execution,
        "navigation_outcomes": outcomes,
        "force_sources": sources,
        "force_sample_timing": timings,
        "recorded_force_sample_totals": force_samples,
        "force_sample_status_counts": sample_statuses,
        "force_decision_rho_recomputed": rho,
        "upper_anchors_recomputed": values,
        "rehearsal_delta": {k: values[k] - old[k]["upper"] for k in values},
        "preservation": {
            "qualified_name": receipt["artifact"]["qualified_name"],
            "manifest_digest": manifest["manifest_digest"],
            "receipt_sha256": digest(args.preservation_receipt),
            "cold_files_byte_verified": len(manifest["files"]),
            "independent_snapshot_files_byte_verified": len(manifest["files"]),
            "cold_manifest_sha256": digest(manifest_path),
        },
    }
    write_determinism_evidence(args, proof, campaign, root, producer, head)
    output = args.output_dir / "acquisition-proof.json"
    output.write_text(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_review_sidecar(output, repo_root=Path(__file__).resolve().parents[2])
    write_scalars(args.output_dir, scalars)
    write_review_outputs(args, proof)


if __name__ == "__main__":
    main()
