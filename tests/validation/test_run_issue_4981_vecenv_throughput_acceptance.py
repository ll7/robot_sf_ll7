"""Contract tests for the issue #4981 VecEnv throughput acceptance runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

from scripts.validation.run_issue_4981_vecenv_throughput_acceptance import (  # noqa: E402
    DEFAULT_PROFILE,
    REPORT_SCHEMA,
    SOURCE_CLAIM_BOUNDARY,
    SOURCE_SCHEMA,
    adjudicate,
    build_comparator_command,
    expected_source_provenance,
    load_profile,
    main,
)


def _repetitions(mode: str, count: int, tps: float) -> list[dict[str, object]]:
    return [
        {
            "mode": mode,
            "transitions_per_second": tps,
            "speedup_vs_baseline": None,
            "status": "ok",
            "error": None,
            "repetition": index,
        }
        for index in range(count)
    ]


def _evidence(
    *,
    current_commit: str,
    speedups: dict[str, float] | None = None,
) -> dict[str, object]:
    profile = load_profile(DEFAULT_PROFILE)
    provenance = expected_source_provenance(profile)
    baseline_tps = 100.0
    resolved_speedups = {
        "dummy": 1.05,
        "subproc": 2.8,
        "threaded": 3.1,
        "threaded_lidar_batch": 2.9,
        **(speedups or {}),
    }
    results = []
    for mode in profile.modes:
        speedup = resolved_speedups[mode]
        tps = round(baseline_tps * speedup, 2)
        results.append(
            {
                "mode": mode,
                "transitions_per_second": tps,
                "speedup_vs_baseline": round(speedup, 3),
                "status": "ok",
                "error": None,
                "repetition_results": _repetitions(mode, profile.repetitions, tps),
            }
        )
    return {
        "schema": SOURCE_SCHEMA,
        "status": "ok",
        **provenance,
        "scenario_selection": {"strategy": "first", "index": 0},
        "commit": current_commit,
        "host": "test-host",
        "python": "3.13.0",
        "platform": "test-platform",
        "num_envs": profile.num_envs,
        "repetitions": profile.repetitions,
        "base_seed": profile.base_seed,
        "warmup_steps": profile.warmup_steps,
        "measure_steps": profile.measure_steps,
        "modes": list(profile.modes),
        "baseline_mode": "dummy",
        "baseline_num_envs": 1,
        "baseline": {
            "mode": "dummy",
            "transitions_per_second": baseline_tps,
            "speedup_vs_baseline": 1.0,
            "status": "ok",
            "error": None,
            "repetition_results": _repetitions("dummy", profile.repetitions, baseline_tps),
        },
        "results": results,
        "failures": [],
        "claim_boundary": SOURCE_CLAIM_BOUNDARY,
    }


def _adjudicate(evidence: dict[str, object], current_commit: str) -> dict[str, object]:
    profile = load_profile(DEFAULT_PROFILE)
    return adjudicate(
        profile,
        evidence,
        evidence_path=Path("output/comparison.json"),
        current_commit=current_commit,
        preflight={"status": "passed", "pytest_nodes": list(profile.required_pytest_nodes)},
    )


def test_profile_freezes_the_standard_workload_and_strict_threshold() -> None:
    """The tracked profile makes the formerly vague acceptance workload reviewable."""
    profile = load_profile(DEFAULT_PROFILE)

    assert profile.num_envs == 4
    assert profile.modes == ("dummy", "subproc", "threaded", "threaded_lidar_batch")
    assert profile.repetitions == 5
    assert profile.warmup_steps == 1000
    assert profile.measure_steps == 10000
    assert profile.candidate_modes == ("threaded", "threaded_lidar_batch")
    assert profile.minimum_speedup_exclusive == pytest.approx(3.0)
    assert "paper/dissertation" in profile.claim_boundary


def test_comparator_command_is_derived_only_from_the_profile(tmp_path: Path) -> None:
    """The acceptance measurement cannot silently drift through ad-hoc CLI overrides."""
    profile = load_profile(DEFAULT_PROFILE)

    command = build_comparator_command(
        profile,
        tmp_path / "comparison.json",
        python_executable="python-under-test",
    )

    assert command[:2] == [
        "python-under-test",
        str(_REPO_ROOT / "scripts/validation/run_vecenv_worker_mode_throughput.py"),
    ]
    assert command[command.index("--config") + 1] == str(_REPO_ROOT / profile.config_path)
    assert command[command.index("--num-envs") + 1] == "4"
    assert command[command.index("--repetitions") + 1] == "5"
    assert command[command.index("--warmup-steps") + 1] == "1000"
    assert command[command.index("--measure-steps") + 1] == "10000"
    modes_index = command.index("--modes")
    assert command[modes_index + 1 : modes_index + 5] == list(profile.modes)


def test_adjudicator_marks_strictly_greater_than_three_speedup_met() -> None:
    """A complete current-head artifact with a 3.1x candidate meets the engineering gate."""
    current_commit = "a" * 40

    report = _adjudicate(_evidence(current_commit=current_commit), current_commit)

    assert report["schema"] == REPORT_SCHEMA
    assert report["status"] == "met"
    assert report["acceptance_met"] is True
    assert report["best_mode"] == "threaded"
    assert report["best_speedup_vs_baseline"] == pytest.approx(3.1)
    assert report["blockers"] == []
    assert "paper/dissertation" in report["claim_boundary"]


def test_exactly_three_speedup_does_not_satisfy_greater_than_gate() -> None:
    """The issue says greater than 3x, so equality must remain a valid not-met result."""
    current_commit = "b" * 40
    evidence = _evidence(
        current_commit=current_commit,
        speedups={"subproc": 2.8, "threaded": 3.0, "threaded_lidar_batch": 2.9},
    )

    report = _adjudicate(evidence, current_commit)

    assert report["status"] == "not_met"
    assert report["acceptance_met"] is False
    assert report["best_speedup_vs_baseline"] == pytest.approx(3.0)
    assert report["blockers"] == []


def test_subproc_speedup_is_comparison_only_not_acceptance() -> None:
    """The existing process-per-environment mode cannot satisfy the in-process issue goal."""
    current_commit = "c" * 40
    evidence = _evidence(
        current_commit=current_commit,
        speedups={"subproc": 3.5, "threaded": 2.9, "threaded_lidar_batch": 2.8},
    )

    report = _adjudicate(evidence, current_commit)

    assert report["status"] == "not_met"
    assert report["acceptance_met"] is False
    assert {row["mode"] for row in report["mode_speedups"]} == {
        "threaded",
        "threaded_lidar_batch",
    }


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    [
        (lambda data: data.__setitem__("measure_steps", 9999), "measure_steps"),
        (lambda data: data.__setitem__("commit", "c" * 40), "current HEAD"),
        (lambda data: data.__setitem__("config_sha256", "0" * 64), "config_sha256"),
        (lambda data: data.__setitem__("status", "failed"), "source status"),
        (lambda data: data.__setitem__("failures", [{"scope": "mode"}]), "failures"),
    ],
)
def test_incomplete_or_stale_evidence_blocks_claim_promotion(mutation, expected_blocker) -> None:
    """Short, stale, failed, or provenance-mismatched artifacts stay blocked."""
    current_commit = "d" * 40
    evidence = _evidence(current_commit=current_commit)
    mutation(evidence)

    report = _adjudicate(evidence, current_commit)

    assert report["status"] == "blocked"
    assert report["acceptance_met"] is None
    assert any(expected_blocker in blocker for blocker in report["blockers"])


def test_aggregate_throughput_must_match_retained_repetitions() -> None:
    """A hand-edited headline speedup cannot override the retained sample records."""
    current_commit = "e" * 40
    evidence = _evidence(current_commit=current_commit)
    threaded = next(row for row in evidence["results"] if row["mode"] == "threaded")
    for repetition in threaded["repetition_results"]:
        repetition["transitions_per_second"] = 999.0

    report = _adjudicate(evidence, current_commit)

    assert report["status"] == "blocked"
    assert any("median repetition throughput" in blocker for blocker in report["blockers"])


def test_dry_run_writes_a_non_evidence_preflight_without_running_commands(tmp_path: Path) -> None:
    """Reviewers can validate the frozen command without launching the long CPU measurement."""
    with patch(
        "scripts.validation.run_issue_4981_vecenv_throughput_acceptance.subprocess.run"
    ) as run:
        exit_code = main(["--dry-run", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    run.assert_not_called()
    report = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert report["status"] == "dry_run_ready"
    assert report["acceptance_met"] is None
    assert report["claim_boundary"].startswith("Preflight only")
    assert report["comparator_command"]
