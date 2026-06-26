"""Tests for the performance-PR evidence contract check.

The contract fires only for `perf`-typed conventional-commit changes (the #3611
failure mode: a `perf(planner): ...` PR whose claimed speed-up was never
substantiated and had to be reverted in #3613). It requires the PR body to carry
concrete before/after evidence, a representative command, a rollback criterion,
and — when caching is claimed — a cache-hit/reuse counter.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.dev.check_perf_evidence import (
    analyze_perf_evidence,
    claims_caching,
    perf_commit_subjects,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "check_perf_evidence.py"


def _evidence_section(
    *,
    baseline: str = "42.0 s on the S20 planner slice (3 seeds)",
    changed: str = "30.1 s on the same slice + seeds",
    command: str = "scripts/dev/run_tests_parallel.sh --lane optional",
    cache_counter: str = "NA - no caching claimed",
    rollback: str = "revert if the slice regresses > 5% vs baseline",
) -> str:
    return f"""## Performance Evidence
- Baseline runtime: {baseline}
- Changed runtime: {changed}
- Representative command: {command}
- Hot-path call count or profile anchor: cProfile on compute_shortest_path_length
- Cache-hit or reuse counter: {cache_counter}
- Rollback or failure criterion: {rollback}
"""


def _body(section: str = "") -> str:
    return f"""## Summary
Speed up the planner hot path.

{section}
"""


# --- trigger detection -------------------------------------------------------


def test_perf_commit_subjects_detects_conventional_perf_types() -> None:
    """Conventional-commit perf subjects trigger the contract; others do not."""
    subjects = (
        "perf(planner): cache motion-planning grid to unblock the #1554 S20 matrix",
        "perf!: drop redundant recompute",
        "feat(benchmark): add evaluator",
        "fix: guard NaN",
        "refactor(planner): rename helper",
    )

    triggered = perf_commit_subjects(subjects)

    assert triggered == (
        "perf(planner): cache motion-planning grid to unblock the #1554 S20 matrix",
        "perf!: drop redundant recompute",
    )


def test_claims_caching_reads_perf_subjects_only() -> None:
    """Caching is inferred from the perf commit subjects, not the body."""
    assert claims_caching(("perf(planner): cache motion-planning grid",))
    assert claims_caching(("perf: memoize feasibility lookups",))
    assert not claims_caching(("perf(planner): pin CPU and raise worker count",))


# --- analyze_perf_evidence ---------------------------------------------------


def test_non_perf_change_is_skipped_regardless_of_body() -> None:
    """A change with no perf commits is not subject to the contract."""
    report = analyze_perf_evidence(_body(), perf_subjects=(), source="fixture")

    assert report.status == "skipped"


def test_full_evidence_section_passes() -> None:
    """A perf change with a complete evidence section passes."""
    report = analyze_perf_evidence(
        _body(_evidence_section()),
        perf_subjects=("perf(planner): pin CPU and raise worker count",),
        source="fixture",
    )

    assert report.status == "ok", report.message


def test_missing_section_fails() -> None:
    """A perf change with no Performance Evidence section fails closed."""
    report = analyze_perf_evidence(
        _body(),
        perf_subjects=("perf(planner): pin CPU and raise worker count",),
        source="fixture",
    )

    assert report.status == "missing_perf_evidence"


def test_placeholder_core_field_is_incomplete() -> None:
    """A placeholder/NA baseline runtime is reported as a missing field."""
    report = analyze_perf_evidence(
        _body(_evidence_section(baseline="NA")),
        perf_subjects=("perf(planner): pin CPU and raise worker count",),
        source="fixture",
    )

    assert report.status == "incomplete_perf_evidence"
    assert "Baseline runtime" in report.missing_fields


def test_caching_claim_requires_cache_counter() -> None:
    """When the perf subject claims caching, an NA cache counter fails."""
    report = analyze_perf_evidence(
        _body(_evidence_section(cache_counter="NA - no caching claimed")),
        perf_subjects=("perf(planner): cache motion-planning grid",),
        source="fixture",
    )

    assert report.status == "missing_cache_counter"
    assert "Cache-hit or reuse counter" in report.missing_fields


def test_caching_claim_passes_with_real_counter() -> None:
    """A caching perf change passes when the cache counter is concrete."""
    report = analyze_perf_evidence(
        _body(_evidence_section(cache_counter="hits=18234 misses=512 (97.3% reuse)")),
        perf_subjects=("perf(planner): cache motion-planning grid",),
        source="fixture",
    )

    assert report.status == "ok", report.message


def test_non_caching_change_does_not_require_cache_counter() -> None:
    """A non-caching perf change with an NA cache counter still passes."""
    report = analyze_perf_evidence(
        _body(_evidence_section(cache_counter="NA")),
        perf_subjects=("perf(planner): pin CPU and raise worker count",),
        source="fixture",
    )

    assert report.status == "ok", report.message


# --- CLI ---------------------------------------------------------------------


def _run_cli(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    repo_root = SCRIPT.resolve().parents[2]
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_passes_for_complete_perf_evidence(tmp_path: Path) -> None:
    """A perf change with a complete evidence section exits 0 via the CLI."""
    body = _write(tmp_path, "body.md", _body(_evidence_section()))
    commits = _write(tmp_path, "commits.txt", "perf(planner): pin CPU and raise worker count\n")

    result = _run_cli(["--body-file", str(body), "--commits-file", str(commits)], tmp_path)

    assert result.returncode == 0, result.stderr


def test_cli_fails_for_missing_perf_evidence(tmp_path: Path) -> None:
    """A perf change with no evidence section exits 2 via the CLI."""
    body = _write(tmp_path, "body.md", _body())
    commits = _write(tmp_path, "commits.txt", "perf(planner): cache grid\n")

    result = _run_cli(["--body-file", str(body), "--commits-file", str(commits)], tmp_path)

    assert result.returncode == 2
    assert "missing_perf_evidence" in (result.stderr + result.stdout)


def test_cli_advisory_downgrades_failure(tmp_path: Path) -> None:
    """Advisory mode emits a warning annotation and exits 0 on a violation."""
    body = _write(tmp_path, "body.md", _body())
    commits = _write(tmp_path, "commits.txt", "perf(planner): cache grid\n")

    result = _run_cli(
        ["--body-file", str(body), "--commits-file", str(commits), "--advisory"], tmp_path
    )

    assert result.returncode == 0
    assert "::warning" in result.stdout


def test_cli_skips_non_perf_change(tmp_path: Path) -> None:
    """A change with no perf commits exits 0 regardless of the body."""
    body = _write(tmp_path, "body.md", _body())
    commits = _write(tmp_path, "commits.txt", "feat: add thing\nfix: guard\n")

    result = _run_cli(["--body-file", str(body), "--commits-file", str(commits)], tmp_path)

    assert result.returncode == 0


def test_cli_requires_body_when_perf_change_and_no_body(tmp_path: Path) -> None:
    """--require-body fails closed for a perf change with no body source."""
    commits = _write(tmp_path, "commits.txt", "perf: cache grid\n")

    result = _run_cli(["--commits-file", str(commits), "--require-body"], tmp_path)

    assert result.returncode == 2
    assert "missing_body" in (result.stderr + result.stdout)


def test_cli_missing_body_without_require_is_lenient(tmp_path: Path) -> None:
    """Without --require-body, a perf change with no body source exits 0."""
    commits = _write(tmp_path, "commits.txt", "perf: cache grid\n")

    result = _run_cli(["--commits-file", str(commits)], tmp_path)

    assert result.returncode == 0
