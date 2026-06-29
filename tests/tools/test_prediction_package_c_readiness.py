"""Tests for the issue #3080 Package C prediction readiness preflight.

These tests build synthetic repository trees so they exercise the fail-closed
ready / blocked / missing status logic without depending on (or executing) any
real benchmark campaign.  One test runs against the real repository root to keep
the preflight wired to the actual configs and entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.tools.prediction_package_c_readiness import (
    ARMS,
    DEFAULT_CLOSED_LOOP_OUTPUT_ROOT,
    DEFAULT_OBSERVATION_REPLAY_OUTPUT_ROOT,
    REQUIRED_CODE,
    REQUIRED_CONFIGS,
    assess_package_c_readiness,
    render_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_wired_repo(root: Path, *, with_coupling_store: bool = False) -> Path | None:
    """Materialize every required Package C input under ``root``.

    Writes the two coordination configs (with a declared seed plan and output
    root) plus every required code entry point, including all arm baselines.
    Returns the coupling result-store path when ``with_coupling_store`` is set.
    """
    for rel in REQUIRED_CONFIGS + REQUIRED_CODE:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")

    # Open-loop config (#2915): declare seeds + output root.
    (root / REQUIRED_CONFIGS[0]).write_text(
        "seeds:\n  - 111\n  - 2868\noutput:\n  evidence_dir: docs/context/evidence/issue_2915\n",
        encoding="utf-8",
    )
    # Closed-loop config (#2916): declare the fixture seed.
    (root / REQUIRED_CONFIGS[1]).write_text(
        "fixture:\n  seed: 111\n",
        encoding="utf-8",
    )
    # Register every arm baseline in the forecast module stub.
    baseline_ids = [arm.baseline_id for arm in ARMS if arm.baseline_id is not None]
    (root / "robot_sf/benchmark/pedestrian_forecast.py").write_text(
        "\n".join(f"def {bid}():\n    return None\n" for bid in baseline_ids),
        encoding="utf-8",
    )

    if with_coupling_store:
        store = root / "store"
        store.mkdir(parents=True, exist_ok=True)
        (store / "summary.json").write_text("{}", encoding="utf-8")
        return store
    return None


def test_all_arms_blocked_when_inputs_present_but_no_coupling_store(tmp_path: Path) -> None:
    """Fully wired inputs without a durable #2916 store fail closed to blocked."""
    _write_wired_repo(tmp_path)
    report = assess_package_c_readiness(tmp_path)

    assert report["overall_status"] == "blocked"
    assert [arm["status"] for arm in report["arms"]] == ["blocked"] * len(ARMS)
    assert report["seed_plan"] == [111, 2868]
    assert report["output_roots"] == [
        "docs/context/evidence/issue_2915",
        DEFAULT_OBSERVATION_REPLAY_OUTPUT_ROOT,
        DEFAULT_CLOSED_LOOP_OUTPUT_ROOT,
    ]
    # The named blocker references the #2916 coupling gate, not a vague reason.
    assert all("#2916" in arm["blockers"][0] for arm in report["arms"])


def test_all_arms_ready_when_coupling_store_present(tmp_path: Path) -> None:
    """A durable coupling store clears the blocker and yields ready arms."""
    store = _write_wired_repo(tmp_path, with_coupling_store=True)
    report = assess_package_c_readiness(tmp_path, coupling_result_store=store)

    assert report["overall_status"] == "ready"
    assert [arm["status"] for arm in report["arms"]] == ["ready"] * len(ARMS)
    assert report["coupling_result_store_available"] is True
    assert all(arm["blockers"] == [] for arm in report["arms"])


def test_missing_config_marks_arms_missing(tmp_path: Path) -> None:
    """A removed required config fails closed to missing for every arm."""
    _write_wired_repo(tmp_path)
    (tmp_path / REQUIRED_CONFIGS[1]).unlink()

    report = assess_package_c_readiness(tmp_path)

    assert report["overall_status"] == "missing"
    assert all(arm["status"] == "missing" for arm in report["arms"])
    assert all(REQUIRED_CONFIGS[1] in arm["missing_inputs"] for arm in report["arms"])


def test_missing_baseline_only_marks_that_arm_missing(tmp_path: Path) -> None:
    """A baseline absent from the forecast module marks only its own arm missing.

    The no-forecast control arm has no baseline and stays blocked (not missing)
    when every shared input is present.
    """
    _write_wired_repo(tmp_path)
    # Drop the semantic baseline registration; keep the others.
    kept = [
        arm.baseline_id
        for arm in ARMS
        if arm.baseline_id is not None and arm.baseline_id != "semantic_cv_baseline"
    ]
    (tmp_path / "robot_sf/benchmark/pedestrian_forecast.py").write_text(
        "\n".join(f"def {bid}():\n    return None\n" for bid in kept),
        encoding="utf-8",
    )

    report = assess_package_c_readiness(tmp_path)
    by_arm = {arm["arm"]: arm for arm in report["arms"]}

    assert by_arm["semantic_cv"]["status"] == "missing"
    assert any("semantic_cv_baseline" in m for m in by_arm["semantic_cv"]["missing_inputs"])
    # Other arms remain blocked (inputs wired, coupling store absent).
    assert by_arm["no_forecast"]["status"] == "blocked"
    assert by_arm["cv"]["status"] == "blocked"
    assert by_arm["interaction_aware"]["status"] == "blocked"
    # Overall is missing because at least one arm is missing.
    assert report["overall_status"] == "missing"


def test_markdown_renders_status_and_blockers(tmp_path: Path) -> None:
    """The Markdown view surfaces the overall status and the #2916 blocker."""
    _write_wired_repo(tmp_path)
    report = assess_package_c_readiness(tmp_path)

    markdown = render_markdown(report)

    assert "Prediction Package C Readiness Preflight" in markdown
    assert "`blocked`" in markdown
    assert "#2916" in markdown


def test_real_repo_preflight_is_wired_and_fails_closed() -> None:
    """Against the real repo, every required input resolves; default is blocked.

    No coupling result store is supplied, so the preflight must fail closed to
    blocked rather than missing: this proves the declared configs and entry
    points exist in the live tree.
    """
    report = assess_package_c_readiness()

    assert report["overall_status"] == "blocked"
    assert all(arm["status"] == "blocked" for arm in report["arms"])
    # No arm should report a missing input against the real repository.
    assert all(arm["missing_inputs"] == [] for arm in report["arms"])
    # The real seed plan is the same-seed union declared by #2915 and #2916.
    assert report["seed_plan"] == [111, 2868]
    assert report["output_roots"] == [
        "docs/context/evidence/issue_2915_forecast_baselines_2026-06-20",
        DEFAULT_OBSERVATION_REPLAY_OUTPUT_ROOT,
        DEFAULT_CLOSED_LOOP_OUTPUT_ROOT,
    ]
