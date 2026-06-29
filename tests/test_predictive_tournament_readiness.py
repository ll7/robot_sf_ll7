"""Fixture tests for the predictive hard-case tournament readiness helper (#3215).

These tests exercise the presence-only classifier against synthetic repository roots so the
ready / blocked / missing-path logic is covered without depending on the real checkout layout
or any SLURM/GPU execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.tools.predictive_tournament_readiness import (
    ARMS,
    RUN_GATES,
    SHARED_PROTOCOL_PATHS,
    evaluate_arm,
    evaluate_readiness,
    main,
    render_text,
)

if TYPE_CHECKING:
    from pathlib import Path


def _touch(repo_root: Path, rel: Path) -> None:
    """Create an empty file (and parents) at ``rel`` under ``repo_root``."""
    target = repo_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")


def _stage_all_prerequisites(repo_root: Path) -> None:
    """Materialize every shared-protocol and per-arm prerequisite path."""
    for rel in SHARED_PROTOCOL_PATHS:
        _touch(repo_root, rel)
    for arm in ARMS:
        for rel in arm.required_paths:
            # The authority arm requires a directory; create a dir for any path without a suffix.
            if rel.suffix:
                _touch(repo_root, rel)
            else:
                (repo_root / rel).mkdir(parents=True, exist_ok=True)


def test_all_present_reports_ready(tmp_path: Path) -> None:
    """When every prerequisite exists, all components classify as ready."""
    _stage_all_prerequisites(tmp_path)
    report = evaluate_readiness(tmp_path)

    assert report["prerequisites_status"] == "ready"
    assert report["shared_protocol"]["status"] == "ready"
    assert all(arm["status"] == "ready" for arm in report["arms"])
    assert all(not arm["missing_paths"] for arm in report["arms"])


def test_run_is_never_authorized_even_when_ready(tmp_path: Path) -> None:
    """The helper is presence-only: prerequisites ready must not imply run authorization."""
    _stage_all_prerequisites(tmp_path)
    report = evaluate_readiness(tmp_path)

    assert report["run_authorized"] is False
    assert report["run_gates"] == list(RUN_GATES)
    assert report["run_gates"], "standing run gates must be reported"


def test_missing_arm_path_reports_blocked_with_blockers(tmp_path: Path) -> None:
    """A missing arm prerequisite yields a blocked arm that names the missing path."""
    _stage_all_prerequisites(tmp_path)
    model_arm = next(arm for arm in ARMS if arm.arm_id == "model")
    missing = model_arm.required_paths[0]
    (tmp_path / missing).unlink()

    report = evaluate_readiness(tmp_path)
    model = next(arm for arm in report["arms"] if arm["id"] == "model")

    assert report["prerequisites_status"] == "blocked"
    assert model["status"] == "blocked"
    assert missing.as_posix() in model["missing_paths"]
    assert model["blockers"] == [
        {"path": missing.as_posix(), "reason": "required prerequisite path is missing"}
    ]


def test_report_surfaces_expected_configs_and_output_paths(tmp_path: Path) -> None:
    """Machine-readable report names per-arm configs and expected output paths."""
    _stage_all_prerequisites(tmp_path)

    report = evaluate_readiness(tmp_path)
    shared = report["shared_protocol"]
    selection = next(arm for arm in report["arms"] if arm["id"] == "selection")
    model = next(arm for arm in report["arms"] if arm["id"] == "model")

    assert "configs/benchmarks/predictive_sweep_planner_grid_v1.yaml" in shared["expected_configs"]
    assert "configs/research/predictive_checkpoint_proxy_v1.yaml" in selection["expected_configs"]
    assert (
        "configs/training/predictive/predictive_retraining_readiness_issue_3214.yaml"
        in model["expected_configs"]
    )
    assert selection["expected_output_paths"] == [selection["expected_output_path"]]


def test_missing_shared_protocol_blocks_overall(tmp_path: Path) -> None:
    """A missing shared-protocol file blocks the portfolio even if all arms are ready."""
    _stage_all_prerequisites(tmp_path)
    (tmp_path / SHARED_PROTOCOL_PATHS[0]).unlink()

    report = evaluate_readiness(tmp_path)

    assert report["shared_protocol"]["status"] == "blocked"
    assert report["prerequisites_status"] == "blocked"
    assert SHARED_PROTOCOL_PATHS[0].as_posix() in report["shared_protocol"]["missing_paths"]


def test_empty_repo_reports_all_paths_missing(tmp_path: Path) -> None:
    """Against an empty root, every expected path is recorded as missing (not crashing)."""
    report = evaluate_readiness(tmp_path)

    assert report["prerequisites_status"] == "blocked"
    for arm in report["arms"]:
        assert arm["status"] == "blocked"
        assert arm["missing_paths"] == [p["path"] for p in arm["paths"]]
        assert all(p["exists"] is False for p in arm["paths"])


def test_authority_directory_prerequisite_is_satisfied_by_directory(tmp_path: Path) -> None:
    """The authority algo-config directory counts as present when the directory exists."""
    authority = next(arm for arm in ARMS if arm.arm_id == "authority")
    dir_prereq = next(p for p in authority.required_paths if not p.suffix)
    (tmp_path / dir_prereq).mkdir(parents=True, exist_ok=True)

    component = evaluate_arm(tmp_path, authority)
    dir_status = next(p for p in component.paths if p.path == dir_prereq.as_posix())

    assert dir_status.exists is True


def test_arms_cover_the_three_named_bets() -> None:
    """The portfolio must expose exactly the selection / authority / model bets."""
    assert {arm.arm_id for arm in ARMS} == {"selection", "authority", "model"}
    assert {arm.child_issue for arm in ARMS} == {3204, 3213, 3214}


def test_model_arm_uses_frozen_weighted_training_config() -> None:
    """Model arm follows #3215's #3214 -> #3254 frozen config handoff."""
    model = next(arm for arm in ARMS if arm.arm_id == "model")

    assert [path.as_posix() for path in model.required_paths] == [
        "configs/training/predictive/predictive_retraining_readiness_issue_3214.yaml",
        "configs/training/predictive/predictive_crossing_conflict_weighted_issue_3254.yaml",
    ]
    assert model.child_issue == 3214
    assert "#3254" in model.notes


def test_run_gates_preserve_read_only_boundary_without_stale_budget_blocker() -> None:
    """Live issue says budget was decided; helper still cannot submit compute."""
    gate_text = "\n".join(RUN_GATES)

    assert "pre-authorize" not in gate_text
    assert "outside this read-only helper" in gate_text
    assert "Autonomous Usage Stop Guard" in gate_text


def test_render_text_runs_and_mentions_status(tmp_path: Path) -> None:
    """The text renderer produces output and surfaces the prerequisite status."""
    _stage_all_prerequisites(tmp_path)
    text = render_text(evaluate_readiness(tmp_path))

    assert "tournament readiness" in text.lower()
    assert "READY" in text


def test_main_exit_code_reflects_prerequisite_status(tmp_path: Path, capsys) -> None:
    """main() exits 0 when ready and 1 when blocked, in both text and JSON modes."""
    _stage_all_prerequisites(tmp_path)
    assert main(["--repo-root", str(tmp_path), "--json"]) == 0
    capsys.readouterr()

    (tmp_path / SHARED_PROTOCOL_PATHS[0]).unlink()
    assert main(["--repo-root", str(tmp_path)]) == 1


@pytest.mark.parametrize("flag", [[], ["--json"]])
def test_main_against_real_checkout_does_not_crash(flag: list[str], capsys) -> None:
    """Running against the real repository root produces a report without raising."""
    rc = main(flag)
    out = capsys.readouterr().out
    assert rc in (0, 1)
    assert out.strip()
