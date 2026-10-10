#!/usr/bin/env python3
"""Produce behaviour-receipt payloads and compact PR-body headers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci import behaviour_receipt  # noqa: E402

DEV_SEED_MIN = 1001
DEV_SEED_MAX = 1200
GATE_SEED_MAX = 1030
DEFAULT_SEEDS = tuple(range(DEV_SEED_MIN, GATE_SEED_MAX + 1))
PLACEHOLDER_REVIEW_URI = "https://example.org/replace-with-independent-exact-head-refute-review"
PLACEHOLDER_EVIDENCE_URI = "https://example.org/replace-with-durable-behaviour-evidence"


@dataclass(frozen=True)
class SweepRun:
    """Recorded sweep execution or submission metadata used by the receipt header."""

    source_sha: str
    output_dir: Path
    artifact_uri: str
    artifact_sha256: str
    job_id: str


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"


def _canonical_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def _full_sha(repo_root: Path, ref: str) -> str:
    return _git(repo_root, "rev-parse", "--verify", f"{ref}^{{commit}}")


def _check_seed_range(seeds: list[int]) -> list[int]:
    if not seeds:
        raise ValueError("at least one development seed is required")
    bad = [seed for seed in seeds if seed < DEV_SEED_MIN or seed > DEV_SEED_MAX]
    if bad:
        raise ValueError(f"seeds outside development range {DEV_SEED_MIN}-{DEV_SEED_MAX}: {bad}")
    unsupported = [seed for seed in seeds if seed > GATE_SEED_MAX]
    if unsupported:
        raise ValueError(
            "the current behaviour gate admits only seeds "
            f"{DEV_SEED_MIN}-{GATE_SEED_MAX}; unsupported seeds: {unsupported}"
        )
    return seeds


def _release_source(repo_root: Path, baseline: str) -> str:
    try:
        return _full_sha(repo_root, f"refs/tags/{baseline}")
    except subprocess.CalledProcessError:
        subprocess.run(
            ["git", "fetch", "origin", f"refs/tags/{baseline}:refs/tags/{baseline}"],
            cwd=repo_root,
            check=True,
        )
        return _full_sha(repo_root, f"refs/tags/{baseline}")


def _sweep_command(
    source_sha: str,
    output_dir: Path,
    seeds: list[int],
    *,
    arms: list[str] | None,
    scenarios: list[str] | None,
    workers: int,
    check_only: bool,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/validation/run_empty_world_sweep.py",
        "--head-sha",
        source_sha,
        "--suite",
        "both",
        "--seeds",
        *[str(seed) for seed in seeds],
        "--workers",
        str(workers),
        "--output-dir",
        str(output_dir),
    ]
    if arms:
        cmd.extend(["--arms", *arms])
    if scenarios:
        cmd.extend(["--scenarios", *scenarios])
    if check_only:
        cmd.append("--check-only")
    return cmd


def _ensure_worktree(repo_root: Path, work_dir: Path, label: str, source_sha: str) -> Path:
    checkout = work_dir / "worktrees" / label
    if checkout.exists():
        return checkout
    checkout.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(checkout), source_sha],
        cwd=repo_root,
        check=True,
    )
    return checkout


def _run_local_sweep(  # noqa: PLR0913
    repo_root: Path,
    work_dir: Path,
    label: str,
    source_sha: str,
    seeds: list[int],
    *,
    arms: list[str] | None,
    scenarios: list[str] | None,
    workers: int,
    check_only: bool,
) -> SweepRun:
    checkout = _ensure_worktree(repo_root, work_dir, label, source_sha)
    output_dir = work_dir / "sweeps" / label
    cmd = _sweep_command(
        source_sha,
        output_dir,
        seeds,
        arms=arms,
        scenarios=scenarios,
        workers=workers,
        check_only=check_only,
    )
    subprocess.run(cmd, cwd=checkout, check=True)
    manifest = output_dir / "README.md"
    artifact_sha = _sha256(manifest) if manifest.is_file() else _canonical_digest(cmd)
    return SweepRun(
        source_sha=source_sha,
        output_dir=output_dir,
        artifact_uri=f"{PLACEHOLDER_EVIDENCE_URI}/{label}",
        artifact_sha256=artifact_sha,
        job_id="1",
    )


def _submit_sbatch_sweep(
    repo_root: Path,
    work_dir: Path,
    label: str,
    source_sha: str,
    seeds: list[int],
    *,
    arms: list[str] | None,
    scenarios: list[str] | None,
    workers: int,
) -> SweepRun:
    checkout = _ensure_worktree(repo_root, work_dir, label, source_sha)
    output_dir = work_dir / "sweeps" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _sweep_command(
        source_sha,
        output_dir,
        seeds,
        arms=arms,
        scenarios=scenarios,
        workers=workers,
        check_only=False,
    )
    script = work_dir / f"behaviour_receipt_{label}.sbatch"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"#SBATCH --job-name=behaviour-{label}",
                f"#SBATCH --output={output_dir / 'slurm-%j.out'}",
                "#SBATCH --time=24:00:00",
                "set -euo pipefail",
                f"cd {checkout}",
                shlex.join(cmd),
                "",
            ]
        ),
        encoding="utf-8",
    )
    job_id = subprocess.check_output(["sbatch", "--parsable", str(script)], text=True).strip()
    submission = {
        "schema_version": "behaviour-sweep-submission.v1",
        "label": label,
        "source_sha": source_sha,
        "job_id": job_id,
        "command": cmd,
        "script": str(script),
    }
    manifest = output_dir / "submission.json"
    manifest.write_bytes(_json_bytes(submission))
    return SweepRun(
        source_sha=source_sha,
        output_dir=output_dir,
        artifact_uri=f"{PLACEHOLDER_EVIDENCE_URI}/{label}",
        artifact_sha256=_sha256(manifest),
        job_id=job_id,
    )


def _existing_sweep(
    source_sha: str,
    output_dir: Path,
    *,
    artifact_uri: str,
    artifact_sha256: str | None,
    job_id: str,
) -> SweepRun:
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    _verify_sweep_metadata(output_dir, source_sha)
    digest = _canonical_digest(
        {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        }
    )
    if artifact_sha256 is not None and artifact_sha256 != digest:
        raise ValueError("supplied sweep digest differs from actual artifact bytes")
    return SweepRun(source_sha, output_dir, artifact_uri, digest, job_id)


def _verify_sweep_metadata(output_dir: Path, source_sha: str) -> None:
    summaries = sorted(output_dir.glob("episodes_*.jsonl"))
    if not summaries:
        raise ValueError("sweep has no episode suites")
    for summary in summaries:
        suite = summary.stem.removeprefix("episodes_")
        metadata_path = output_dir / f"execution_{suite}.json"
        if not metadata_path.is_file():
            raise ValueError(f"missing authoritative execution metadata for {suite}")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta.get("head_sha") != source_sha:
            raise ValueError(f"{suite} recorded SHA differs from claimed source SHA")
        if meta.get("suite") != suite:
            raise ValueError(f"{suite} execution suite identity differs")
        complete = (
            meta.get("complete") is True
            and meta.get("trace_requested") is True
            and meta.get("campaign_execution_status") == "completed"
            and meta.get("exit_code") == 0
            and meta.get("unexpected_failed_runs") == 0
            and all(
                meta.get(key) == []
                for key in (
                    "failed_slots",
                    "missing_slots",
                    "duplicate_slots",
                    "unexpected_slots",
                    "incomplete_trace_slots",
                )
            )
        )
        if not complete:
            raise ValueError(f"{suite} sweep execution is incomplete")
        records = [json.loads(line) for line in summary.read_text().splitlines() if line.strip()]
        counts = [meta.get(key) for key in ("expected_slots", "written_rows", "total_episodes")]
        if not records or any(count != len(records) for count in counts):
            raise ValueError(f"{suite} execution row counts differ")
        seeds = meta.get("seeds")
        if not isinstance(seeds, list) or sorted(_check_seed_range(seeds)) != sorted(
            {row["seed"] for row in records}
        ):
            raise ValueError(f"{suite} execution seed identity differs")
        if any(row.get("execution_status") != "written" for row in records):
            raise ValueError(f"{suite} has unsuccessful row execution")


def _load_sweep_rows(sweep_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(sweep_dir.glob("episodes_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["arm"]), str(row["scenario"]), int(row["seed"]))
            suite = path.stem.removeprefix("episodes_")
            campaign = sweep_dir / "campaigns" / f"empty_world_{suite}"
            source = row.get("source_file")
            if not isinstance(source, str):
                raise ValueError(f"missing raw controller trace source for {key}")
            source_path = (campaign / source).resolve()
            if not source_path.is_relative_to(campaign.resolve()):
                raise ValueError("raw controller trace path escapes the campaign")
            raw_rows = [
                json.loads(item) for item in source_path.read_text().splitlines() if item.strip()
            ]
            matches = [
                raw for raw in raw_rows if (raw.get("scenario_id"), raw.get("seed")) == key[1:]
            ]
            if len(matches) != 1 or source_path.parent.name.split("__")[0] != key[0]:
                raise ValueError(f"raw episode identity differs for {key}")
            raw = matches[0]
            if _row_success(raw) != _row_success(row) or _row_collisions(raw) != _row_collisions(
                row
            ):
                raise ValueError(f"raw episode outcome differs from summary for {key}")
            if key in rows:
                raise ValueError(f"duplicate sweep row {key}")
            rows[key] = {**raw, "execution_status": row["execution_status"]}
    if not rows:
        raise ValueError(f"no sweep rows found in {sweep_dir}")
    return rows


def _row_success(row: dict[str, Any]) -> bool:
    if "success" in row:
        return bool(row["success"])
    metrics = row.get("metrics") or {}
    outcome = row.get("outcome") or {}
    return bool(metrics.get("success") or outcome.get("route_complete"))


def _row_collisions(row: dict[str, Any]) -> int:
    if row.get("collisions") is not None:
        return int(row["collisions"])
    metrics = row.get("metrics") or {}
    return int(metrics.get("total_collision_count", metrics.get("collisions", 0)) or 0)


def _execution_evidence(row: dict[str, Any]) -> dict[str, Any]:
    from robot_sf.benchmark.fallback_policy import (
        resolve_execution_mode,
        runtime_fallback_or_degraded_marker,
    )
    from scripts.validation.run_empty_world_sweep import _trace_complete

    metadata = row.get("algorithm_metadata") or {}
    algorithm = row.get("algo") or row.get("algorithm")
    mode = resolve_execution_mode(metadata)
    marker = runtime_fallback_or_degraded_marker(
        row,
        expected_algorithm=algorithm,
        algorithm_metadata=metadata,
    )
    trace = metadata.get("simulation_step_trace") or {}
    steps = trace.get("steps") or []
    executed = (
        row.get("execution_status") == "written"
        and metadata.get("status") == "ok"
        and _trace_complete(row)
        and all(
            isinstance((step.get("planner") or {}).get("selected_action"), dict)
            and step["planner"]["selected_action"]
            for step in steps
        )
        and row.get("controller_executed", True) is True
    )
    return {
        "algorithm": algorithm or "unknown",
        "execution_mode": mode if row.get("execution_mode", mode) == mode else "unknown",
        "controller_executed": bool(executed),
        "fallback": row.get("fallback") is True
        or (marker is not None and "fallback" in str(marker)),
        "degraded": row.get("degraded") is True or marker is not None,
    }


def build_rows_and_classifications(
    head_rows: dict[tuple[str, str, int], dict[str, Any]],
    baseline_rows: dict[tuple[str, str, int], dict[str, Any]],
    scope: dict[str, Any],
    *,
    classification_class: str,
    evidence_base_uri: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Join head/baseline sweep rows into gate payload rows and classifications."""
    rows: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    for arm in scope["arms"]:
        for map_id in scope["maps"]:
            for seed in DEFAULT_SEEDS:
                key = (arm, map_id, seed)
                if key not in head_rows:
                    raise ValueError(f"head sweep omitted {key}")
                if key not in baseline_rows:
                    raise ValueError(f"baseline sweep omitted {key}")
                head = head_rows[key]
                baseline = baseline_rows[key]
                success = _row_success(head)
                collisions = _row_collisions(head)
                baseline_success = _row_success(baseline)
                baseline_collisions = _row_collisions(baseline)
                rows.append(
                    {
                        "arm": arm,
                        "map": map_id,
                        "seed": seed,
                        "success": success,
                        "collisions": collisions,
                        "baseline_success": baseline_success,
                        "baseline_collisions": baseline_collisions,
                        **_execution_evidence(head),
                    }
                )
                evidence = f"{evidence_base_uri.rstrip('/')}/{arm}/{map_id}/{seed}"
                if baseline_success and not success:
                    classifications.append(
                        {
                            "arm": arm,
                            "map": map_id,
                            "seed": seed,
                            "kind": "success_to_failure",
                            "class": classification_class,
                            "evidence": evidence,
                            "count": 1,
                        }
                    )
                if collisions > baseline_collisions:
                    classifications.append(
                        {
                            "arm": arm,
                            "map": map_id,
                            "seed": seed,
                            "kind": "new_collision",
                            "class": classification_class,
                            "evidence": evidence,
                            "count": collisions - baseline_collisions,
                        }
                    )
    totals = {
        "episodes": len(rows),
        "new_failures": sum(row["kind"] == "success_to_failure" for row in classifications),
        "new_collisions": sum(
            row["count"] for row in classifications if row["kind"] == "new_collision"
        ),
    }
    return rows, classifications, totals


def _audit_status(rows: list[dict[str, Any]]) -> str:
    if not rows or any(
        row.get("controller_executed") is not True
        or row.get("execution_mode") not in {"native", "adapter"}
        or not row.get("algorithm")
        or row.get("algorithm") == "unknown"
        or type(row.get("fallback")) is not bool
        or type(row.get("degraded")) is not bool
        for row in rows
    ):
        return "fail"
    return "degraded" if any(row["fallback"] or row["degraded"] for row in rows) else "pass"


def write_real_row_audit(
    path: Path,
    *,
    source_sha: str,
    rows: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
) -> str:
    """Write the real-row audit artifact and return its byte digest."""
    audit = {
        "schema_version": "behaviour-real-row-audit.v1",
        "source_sha": source_sha,
        "status": _audit_status(rows),
        "rows": len(rows),
        "controller_executed_rows": sum(row.get("controller_executed") is True for row in rows),
        "fallback_rows": sum(row.get("fallback") is True for row in rows),
        "degraded_rows": sum(row.get("degraded") is True for row in rows),
        "classification_rows": len(classifications),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(audit))
    return _sha256(path)


def write_receipt_and_header(  # noqa: PLR0913
    *,
    repo_root: Path,
    receipt_id: str,
    head_sha: str,
    scheduler: SweepRun,
    baseline: SweepRun,
    baseline_release: str,
    baseline_body_id: str,
    baseline_config_sha256: str,
    baseline_differences: list[str],
    scope: dict[str, Any],
    rows: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    totals: dict[str, int],
    audit_uri: str,
    audit_sha256: str,
    audit_source_sha: str,
    refute_review_uri: str,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Write the committed rows payload and return the compact PR header."""
    payload = {
        "schema_version": "behaviour-change-rows.v1",
        "rows": rows,
        "classifications": classifications,
    }
    receipt_path = output_dir / f"{receipt_id}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(payload)
    receipt_path.write_bytes(raw)
    relative = receipt_path.relative_to(repo_root).as_posix()
    vehicle_body = {
        "vehicle_id": scope["vehicle_id"],
        "exceptions": scope["exceptions"],
    }
    job_id = scheduler.job_id
    if scheduler.job_id != baseline.job_id:
        job_id = f"{scheduler.job_id}_{baseline.job_id}"
    header = {
        "schema_version": "behaviour-change-receipt-header.v2",
        "head_sha": head_sha,
        "scheduler": {
            "kind": "slurm",
            "job_id": job_id,
            "source_sha": scheduler.source_sha,
        },
        "artifact": {"uri": scheduler.artifact_uri, "sha256": scheduler.artifact_sha256},
        "baseline": {
            "release": baseline_release,
            "source_sha": baseline.source_sha,
            "artifact_uri": baseline.artifact_uri,
            "artifact_sha256": baseline.artifact_sha256,
            "comparison": "development_reconstruction",
            "body_id": baseline_body_id,
            "config_sha256": baseline_config_sha256,
            "differences": baseline_differences,
        },
        "vehicle": {
            "id": scope["vehicle_id"],
            "body_sha256": _canonical_digest(vehicle_body),
        },
        "classifications": {
            "count": len(classifications),
            "sha256": _canonical_digest(classifications),
        },
        "exceptions": scope["exceptions"],
        "totals": totals,
        "interaction_audit": {
            "uri": audit_uri,
            "sha256": audit_sha256,
            "source_sha": audit_source_sha,
        },
        "refute_review": {
            "head_sha": head_sha,
            "verdict": "pending_independent_review",
            "uri": refute_review_uri,
        },
        "scope_sha256": _canonical_digest(scope),
        "rows_artifact": {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()},
    }
    return receipt_path, header


def _load_scope(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument(
        "--head-source-sha", help="Recorded execution source before receipt-only commits"
    )
    parser.add_argument("--baseline", default="latest")
    parser.add_argument("--repo", default="ll7/robot_sf_ll7")
    parser.add_argument("--receipt-id", required=True)
    parser.add_argument("--mode", choices=("existing", "local", "sbatch"), default="existing")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--head-sweep-dir", type=Path)
    parser.add_argument("--baseline-sweep-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("output/behaviour_receipt"))
    parser.add_argument("--output-dir", type=Path, default=Path("receipts/behaviour"))
    parser.add_argument("--scope-path", type=Path, default=behaviour_receipt.SCOPE_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--scenarios", nargs="*")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--head-artifact-uri", default=f"{PLACEHOLDER_EVIDENCE_URI}/head")
    parser.add_argument("--head-artifact-sha256")
    parser.add_argument("--baseline-artifact-uri", default=f"{PLACEHOLDER_EVIDENCE_URI}/baseline")
    parser.add_argument("--baseline-artifact-sha256")
    parser.add_argument("--job-id", default="1")
    parser.add_argument("--baseline-job-id", default="1")
    parser.add_argument("--audit-uri", default=f"{PLACEHOLDER_EVIDENCE_URI}/real-row-audit.json")
    parser.add_argument("--audit-source-sha")
    parser.add_argument("--refute-review-uri", default=PLACEHOLDER_REVIEW_URI)
    parser.add_argument(
        "--classification-class",
        choices=(
            "defect",
            "known_limitation",
            "vehicle_specific_infeasible",
            "measurement_inconsistency",
        ),
        default="known_limitation",
    )
    parser.add_argument("--classification-evidence-base-uri", default=PLACEHOLDER_EVIDENCE_URI)
    parser.add_argument("--baseline-body-id", default="differential_drive_r1m")
    parser.add_argument("--baseline-config-sha256")
    parser.add_argument(
        "--baseline-difference",
        action="append",
        dest="baseline_differences",
        default=[],
    )
    return parser.parse_args(argv)


def _prepare_sweeps(
    args: argparse.Namespace, repo_root: Path, baseline_source: str
) -> tuple[SweepRun, SweepRun]:
    seeds = _check_seed_range(list(args.seeds))
    work_dir = (repo_root / args.work_dir).resolve()
    if args.mode == "existing":
        if args.head_sweep_dir is None or args.baseline_sweep_dir is None:
            raise ValueError("--mode existing requires --head-sweep-dir and --baseline-sweep-dir")
        return (
            _existing_sweep(
                args.head_source_sha or args.head_sha,
                args.head_sweep_dir.resolve(),
                artifact_uri=args.head_artifact_uri,
                artifact_sha256=args.head_artifact_sha256,
                job_id=args.job_id,
            ),
            _existing_sweep(
                baseline_source,
                args.baseline_sweep_dir.resolve(),
                artifact_uri=args.baseline_artifact_uri,
                artifact_sha256=args.baseline_artifact_sha256,
                job_id=args.baseline_job_id,
            ),
        )
    if args.mode == "local":
        return (
            _run_local_sweep(
                repo_root,
                work_dir,
                "head",
                args.head_sha,
                seeds,
                arms=args.arms,
                scenarios=args.scenarios,
                workers=args.workers,
                check_only=args.check_only,
            ),
            _run_local_sweep(
                repo_root,
                work_dir,
                "baseline",
                baseline_source,
                seeds,
                arms=args.arms,
                scenarios=args.scenarios,
                workers=args.workers,
                check_only=args.check_only,
            ),
        )
    return (
        _submit_sbatch_sweep(
            repo_root,
            work_dir,
            "head",
            args.head_sha,
            seeds,
            arms=args.arms,
            scenarios=args.scenarios,
            workers=args.workers,
        ),
        _submit_sbatch_sweep(
            repo_root,
            work_dir,
            "baseline",
            baseline_source,
            seeds,
            arms=args.arms,
            scenarios=args.scenarios,
            workers=args.workers,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the behaviour receipt producer CLI."""
    args = _parse_args(argv)
    repo_root = ROOT
    head_sha = _full_sha(repo_root, args.head_sha)
    args.head_sha = head_sha
    if args.head_source_sha:
        args.head_source_sha = _full_sha(repo_root, args.head_source_sha)
        behaviour_receipt.require_receipt_only_ancestor(args.head_source_sha, head_sha, "job")
    baseline_release = (
        behaviour_receipt.latest_release(args.repo) if args.baseline == "latest" else args.baseline
    )
    baseline_source = _release_source(repo_root, baseline_release)
    scheduler, baseline = _prepare_sweeps(args, repo_root, baseline_source)
    if args.submit_only:
        print(json.dumps({"head_job_id": scheduler.job_id, "baseline_job_id": baseline.job_id}))
        return 0
    scope = _load_scope(args.scope_path)
    head_rows = _load_sweep_rows(scheduler.output_dir)
    baseline_rows = _load_sweep_rows(baseline.output_dir)
    rows, classifications, totals = build_rows_and_classifications(
        head_rows,
        baseline_rows,
        scope,
        classification_class=args.classification_class,
        evidence_base_uri=args.classification_evidence_base_uri,
    )
    audit_path = (repo_root / args.work_dir / "real_row_audit.json").resolve()
    audit_source_sha = args.audit_source_sha or scheduler.source_sha
    audit_sha = write_real_row_audit(
        audit_path,
        source_sha=audit_source_sha,
        rows=[*rows, *[_execution_evidence(row) for row in baseline_rows.values()]],
        classifications=classifications,
    )
    if json.loads(audit_path.read_text())["status"] != "pass":
        raise ValueError(f"real-row audit did not pass; inspect {audit_path}")
    baseline_config_sha = args.baseline_config_sha256 or _canonical_digest(
        {
            "release": baseline_release,
            "source_sha": baseline_source,
            "scope_sha256": _canonical_digest(scope),
        }
    )
    differences = args.baseline_differences or [
        "development reconstruction against the published baseline; reviewer must confirm body/config/source provenance"
    ]
    receipt_path, header = write_receipt_and_header(
        repo_root=repo_root,
        receipt_id=args.receipt_id,
        head_sha=head_sha,
        scheduler=scheduler,
        baseline=baseline,
        baseline_release=baseline_release,
        baseline_body_id=args.baseline_body_id,
        baseline_config_sha256=baseline_config_sha,
        baseline_differences=differences,
        scope=scope,
        rows=rows,
        classifications=classifications,
        totals=totals,
        audit_uri=args.audit_uri,
        audit_sha256=audit_sha,
        audit_source_sha=audit_source_sha,
        refute_review_uri=args.refute_review_uri,
        output_dir=(repo_root / args.output_dir).resolve(),
    )
    print(f"wrote {receipt_path.relative_to(repo_root)}", file=sys.stderr)
    print("<!-- behaviour-change-receipt:v2")
    print(json.dumps(header, indent=2, sort_keys=True))
    print("-->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
