#!/usr/bin/env python3
"""Build the issue #5592 cross-matrix agreement/disagreement table.

This is the artifact-first generator for the pre-registered second scenario
matrix (issue #5592): it compares the structural-class ordering independently on
the reference ``classic_interactions`` matrix and on the candidate atomic-topology
matrix, then emits a side-by-side agreement/disagreement table. It does NOT merge
the two matrices into a single ranking.

The structural ranking for each matrix is supplied as an input artifact (the
output of the future paired campaign run). When a matrix's ranking is absent the
builder fails closed and emits ``blocked_missing_matrix`` rows rather than
inventing a ranking; this keeps the generator honest under the cheap-lane
constraint that no campaign is run here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = REPO_ROOT / "configs/benchmarks/issue_5592_cross_matrix_preregistration.yaml"
SCHEMA_VERSION = "issue_5592_cross_matrix_preregistration.v1"
REPORT_SCHEMA_VERSION = "issue_5592_cross_matrix_agreement.v1"

# Structural classes, in the canonical cross-cut order. The reference ranking
# input lists one rank per structural class; the candidate ranking is compared
# against it.
STRUCTURAL_CLASS_ORDER = [
    "constraint_first_hybrid",
    "learned_policy",
    "predictive",
    "baseline_reactive",
]
ALLOWED_AGREEMENT_STATUSES = {
    "agreement",
    "disagreement",
    "blocked_missing_matrix",
    "blocked_incomparable_roster",
}
PRIMARY_OUTPUT = "cross_matrix_agreement.csv"
REQUIRED_COLUMNS = [
    "matrix",
    "structural_class",
    "rank",
    "reference_rank",
    "rank_delta",
    "agreement_status",
    "caveat",
]
EXPECTED_RANKS = frozenset(range(1, len(STRUCTURAL_CLASS_ORDER) + 1))
ROSTER_SIGNATURE_COLUMN = "roster_signature"
RANKING_INPUT_COLUMNS = frozenset({"structural_class", "rank", ROSTER_SIGNATURE_COLUMN})


class BuildError(ValueError):
    """Raised when issue #5592 inputs or the pre-registration are malformed."""


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BuildError(f"{path} must contain a YAML mapping")
    return payload


def _public_path(path: Path) -> str:
    """Return a repo-public path without local home/worktree prefixes."""
    resolved = path.resolve()
    for anchor in ("docs", "configs", "scripts", "tests", "output"):
        if anchor in resolved.parts:
            index = resolved.parts.index(anchor)
            return str(Path(*resolved.parts[index:]))
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _roster_signature(packet: Mapping[str, Any]) -> str:
    """Return the deterministic signature of the preregistered planner roster."""
    roster = packet.get("planner_roster")
    if not isinstance(roster, dict):
        raise BuildError("packet.planner_roster must be a mapping")
    structural_classes = roster.get("structural_classes")
    if not isinstance(structural_classes, dict):
        raise BuildError("packet.planner_roster.structural_classes must be a mapping")
    if set(structural_classes) != set(STRUCTURAL_CLASS_ORDER):
        raise BuildError("packet planner roster structural classes mismatch")

    canonical: dict[str, list[str]] = {}
    planners: list[str] = []
    for structural_class in STRUCTURAL_CLASS_ORDER:
        class_planners = structural_classes.get(structural_class)
        if not isinstance(class_planners, list) or not class_planners:
            raise BuildError(f"planner roster for {structural_class!r} must be a non-empty list")
        normalized = [str(planner).strip() for planner in class_planners]
        if any(not planner for planner in normalized):
            raise BuildError(f"planner roster for {structural_class!r} contains an empty planner")
        canonical[structural_class] = normalized
        planners.extend(normalized)
    if len(planners) != len(set(planners)):
        raise BuildError("packet planner roster contains duplicate planner keys")
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_ranking_csv(path: Path, *, expected_roster_signature: str) -> dict[str, int]:
    """Load a single-matrix structural-class ranking from a CSV.

    The CSV must carry ``structural_class``, ``rank``, and the roster signature
    generated from the pre-registration. Returns a complete map from structural
    class to a unique integer rank in ``1..4``.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = RANKING_INPUT_COLUMNS - fieldnames
        if missing_columns:
            raise BuildError(
                f"ranking input {path} missing required columns: {sorted(missing_columns)}"
            )
        rows = list(reader)
    ranking: dict[str, int] = {}
    for row_number, row in enumerate(rows, start=2):
        roster_signature = (row.get(ROSTER_SIGNATURE_COLUMN) or "").strip()
        if roster_signature != expected_roster_signature:
            raise BuildError(
                f"ranking input {path} row {row_number} has an incompatible planner roster"
            )
        klass = (row.get("structural_class") or row.get("class") or "").strip()
        rank_raw = row.get("rank")
        if not klass or rank_raw is None or str(rank_raw).strip() == "":
            raise BuildError(f"ranking input {path} row {row_number} is missing class or rank")
        if klass in ranking:
            raise BuildError(f"duplicate structural class {klass!r} in {path}")
        try:
            rank = int(str(rank_raw).strip())
        except (TypeError, ValueError) as exc:
            raise BuildError(f"invalid integer rank for {klass!r} in {path}: {rank_raw!r}") from exc
        if rank <= 0:
            raise BuildError(f"rank for {klass!r} in {path} must be positive")
        ranking[klass] = rank
    expected_classes = set(STRUCTURAL_CLASS_ORDER)
    if set(ranking) != expected_classes:
        missing = sorted(expected_classes - set(ranking))
        extra = sorted(set(ranking) - expected_classes)
        raise BuildError(
            f"ranking input {path} must contain exactly the four structural classes; "
            f"missing={missing}, extra={extra}"
        )
    if set(ranking.values()) != EXPECTED_RANKS:
        raise BuildError(f"ranking input {path} must contain a unique rank permutation 1..4")
    return ranking


def _load_packet(packet_path: Path) -> dict[str, Any]:
    packet = _load_yaml(packet_path)
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise BuildError("packet schema_version mismatch")
    if packet.get("issue") != 5592:
        raise BuildError("packet.issue must be 5592")
    if packet.get("status") != "pre_registered":
        raise BuildError("packet.status must be pre_registered")
    return packet


def _validate_comparison_contract(packet: Mapping[str, Any]) -> None:
    comparison = packet.get("comparison_contract")
    if not isinstance(comparison, dict):
        raise BuildError("comparison_contract must be a mapping")
    if comparison.get("primary_output") != PRIMARY_OUTPUT:
        raise BuildError("comparison primary_output must be cross_matrix_agreement.csv")
    required = comparison.get("required_columns")
    if not isinstance(required, list) or set(REQUIRED_COLUMNS) - set(required):
        raise BuildError("comparison required_columns mismatch")
    if comparison.get("rank_unit") != "structural_class":
        raise BuildError("comparison rank_unit must be structural_class")
    if comparison.get("metric") != "constraints_first_structural_rank":
        raise BuildError("comparison metric mismatch")
    allowed_statuses = comparison.get("allowed_agreement_statuses")
    if (
        not isinstance(allowed_statuses, list)
        or set(allowed_statuses) != ALLOWED_AGREEMENT_STATUSES
    ):
        raise BuildError("comparison allowed_agreement_statuses mismatch")
    if comparison.get("must_emit_disagreement_rows") is not True:
        raise BuildError("comparison must_emit_disagreement_rows must be true")


def _git_head() -> str | None:
    """Return the current full Git commit, or ``None`` outside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _source_commit(output_dir: Path) -> str:
    """Resolve required source provenance from an explicit file or Git HEAD."""
    commit_path = output_dir / "SOURCE_COMMIT"
    try:
        source_commit = (
            commit_path.read_text(encoding="utf-8").strip() if commit_path.exists() else ""
        )
    except OSError as exc:
        raise BuildError(f"cannot read source provenance file {commit_path}") from exc
    source_commit = source_commit or (_git_head() or "")
    if not source_commit or source_commit == "unknown":
        raise BuildError("source commit provenance is required and could not be recorded")
    return source_commit


def _classify_agreement(
    structural_class: str,
    reference_rank: int | None,
    candidate_rank: int | None,
) -> tuple[str, str]:
    """Return (agreement_status, caveat) for one structural class comparison."""
    if reference_rank is None or candidate_rank is None:
        status = "blocked_missing_matrix"
        caveat = "one or both matrices lack a structural-class rank; no generalization conclusion"
    elif reference_rank == candidate_rank:
        status = "agreement"
        caveat = "structural-class rank holds on both matrices"
    else:
        status = "disagreement"
        caveat = "structural ranking flips on the atomic-topology matrix"
    _assert_allowed_status(status, structural_class)
    return status, caveat


def _assert_allowed_status(status: str, structural_class: str) -> None:
    if status not in ALLOWED_AGREEMENT_STATUSES:
        raise BuildError(f"agreement_status {status!r} for {structural_class!r} not in allowed set")


def _build_rows(
    reference_ranking: Mapping[str, int] | None,
    candidate_ranking: Mapping[str, int] | None,
) -> list[dict[str, Any]]:
    """Build the cross-matrix agreement rows from two structural-class rankings."""
    rows: list[dict[str, Any]] = []
    for klass in STRUCTURAL_CLASS_ORDER:
        reference_rank = reference_ranking.get(klass) if reference_ranking else None
        candidate_rank = candidate_ranking.get(klass) if candidate_ranking else None
        status, caveat = _classify_agreement(klass, reference_rank, candidate_rank)
        rank_delta = ""
        if candidate_rank is not None and reference_rank is not None:
            rank_delta = candidate_rank - reference_rank
        # `rank` carries the candidate (atomic-topology) rank; reference_rank is
        # the classic_interactions rank used for the delta.
        rows.append(
            {
                "matrix": "atomic_topology",
                "structural_class": klass,
                "rank": candidate_rank if candidate_rank is not None else "",
                "reference_rank": reference_rank if reference_rank is not None else "",
                "rank_delta": rank_delta,
                "agreement_status": status,
                "caveat": caveat,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sha256sums(output_dir: Path) -> None:
    lines: list[str] = []
    for path in sorted(item for item in output_dir.iterdir() if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {_public_path(path)}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(output_dir: Path, status: str, *, next_action: str | None = None) -> None:
    next_action_line = f"\nCurrent blocker next action: {next_action}\n" if next_action else ""
    readme = f"""# Issue #5592 cross-matrix structural-ranking agreement

status: {status}
{next_action_line}
This is the cross-matrix generalization check for the structural planner ranking: the
structural-class ordering is compared independently on the reference `classic_interactions`
matrix and on the candidate atomic-topology matrix, then reported side by side. The two
matrices are never merged into one ranking.

Each supplied ranking CSV must include `structural_class`, `rank`, and a
`roster_signature` matching the preregistered 12-planner roster. Ranks must be a complete
`1..4` permutation; malformed or incomparable inputs fail closed.

`cross_matrix_agreement.csv` is the primary output. Each row carries the candidate
(atomic-topology) rank, the reference (classic_interactions) rank, the rank delta, and an
explicit agreement_status (`agreement`, `disagreement`, or a `blocked_*` status). Disagreement
rows are always emitted when both matrices are present; they are not hidden in a merge.

Claim boundary: this is evidence about transfer to one additional geometry distribution, not a
general-purpose generalization guarantee. No campaign, Slurm/GPU submission, or
paper/dissertation claim is produced here.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def build_packet(
    *,
    packet_path: Path,
    reference_ranking_path: Path | None,
    candidate_ranking_path: Path | None,
    output_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Build the issue #5592 cross-matrix agreement evidence packet."""
    packet = _load_packet(packet_path)
    _validate_comparison_contract(packet)
    roster_signature = _roster_signature(packet)

    for ranking_path in (reference_ranking_path, candidate_ranking_path):
        if ranking_path is not None and ranking_path.exists() and not ranking_path.is_file():
            raise BuildError(f"ranking input is not a file: {ranking_path}")

    reference_present = reference_ranking_path is not None and reference_ranking_path.is_file()
    candidate_present = candidate_ranking_path is not None and candidate_ranking_path.is_file()

    reference_ranking = (
        _read_ranking_csv(reference_ranking_path, expected_roster_signature=roster_signature)
        if reference_present
        else None
    )
    candidate_ranking = (
        _read_ranking_csv(candidate_ranking_path, expected_roster_signature=roster_signature)
        if candidate_present
        else None
    )

    rows = _build_rows(reference_ranking, candidate_ranking)

    missing = not (reference_present and candidate_present)
    status = "blocked_missing_matrix" if missing else "ready"
    next_action = None
    if missing:
        missing_matrices = []
        if not reference_present:
            missing_matrices.append("reference (classic_interactions)")
        if not candidate_present:
            missing_matrices.append("candidate (atomic_topology)")
        next_action = (
            "Run the pre-registered paired campaign (5 seeds, h600, 12-planner roster) for the "
            f"missing matrix ranking(s): {', '.join(missing_matrices)}. Then re-run this builder "
            "with the resulting structural-class ranking CSVs. No ranking is fabricated here."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_commit = _source_commit(output_dir)
    _write_csv(output_dir / PRIMARY_OUTPUT, rows, REQUIRED_COLUMNS)

    agreement_statuses = {str(row["agreement_status"]) for row in rows}
    if not (agreement_statuses <= ALLOWED_AGREEMENT_STATUSES):
        raise BuildError(f"unallowed agreement_status in output: {agreement_statuses}")

    disagreement_rows = [row for row in rows if row["agreement_status"] == "disagreement"]
    if not missing and not disagreement_rows:
        # When both matrices are present, the contract requires disagreement rows to be
        # representable. They exist only when a rank flip occurs; if the ranking is identical
        # across matrices there are no flips, which is a valid (full-agreement) result. We still
        # assert the generator is capable of emitting them by construction above.
        pass

    metadata = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "issue": 5592,
        "generated_at": generated_at,
        "preregistration": _public_path(packet_path),
        "reference_ranking_present": reference_present,
        "candidate_ranking_present": candidate_present,
        "structural_classes": list(STRUCTURAL_CLASS_ORDER),
        "agreement_statuses": sorted(agreement_statuses),
        "disagreement_row_count": len(disagreement_rows),
        "source_commit": source_commit,
        "roster_signature": roster_signature,
        "claim_boundary": "Cross-matrix transfer evidence for one additional geometry "
        "distribution only; not a general-purpose generalization guarantee.",
        "next_action": next_action,
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_readme(output_dir, status, next_action=next_action)
    _write_sha256sums(output_dir)
    return metadata


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--reference-ranking", type=Path, default=None)
    parser.add_argument("--candidate-ranking", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-at", default="now")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the issue #5592 cross-matrix agreement builder CLI."""
    args = _parse_args(argv or sys.argv[1:])
    generated_at = (
        datetime.now(UTC).isoformat(timespec="seconds")
        if args.generated_at == "now"
        else str(args.generated_at)
    )
    try:
        summary = build_packet(
            packet_path=args.packet,
            reference_ranking_path=args.reference_ranking,
            candidate_ranking_path=args.candidate_ranking,
            output_dir=args.output_dir,
            generated_at=generated_at,
        )
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"status: {summary['status']}")
        print(f"output_dir: {args.output_dir}")
        if summary.get("next_action"):
            print(f"next_action: {summary['next_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
