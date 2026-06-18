"""Tests for issue #3064 behavior-variant preflight reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.tools import report_behavior_variant_preflight as report

if TYPE_CHECKING:
    from pathlib import Path


def _touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_json(root: Path, relative: str, *, pairs: int, deltas: int) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "{\n"
            '  "schema_version": "test",\n'
            '  "result": {\n'
            f'    "candidate_pairs_compared": {pairs},\n'
            f'    "non_identical_pairs_found": {deltas},\n'
            '    "max_per_frame_abs_delta": 0.0,\n'
            '    "decision": "test"\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )


def _minimal_repo(root: Path) -> None:
    for relative in (
        "robot_sf/baselines/social_force.py",
        "robot_sf/planner/socnav.py",
        "robot_sf/planner/social_navigation_pyenvs_orca.py",
        "robot_sf/planner/social_navigation_pyenvs_force_model.py",
        "robot_sf/planner/social_navigation_pyenvs_hsfm.py",
        "configs/baselines/social_force_default.yaml",
        "configs/baselines/social_force_ammv_aware.yaml",
    ):
        _touch(root, relative)
    for evidence in report.AMMV_EVIDENCE:
        _write_json(root, evidence.as_posix(), pairs=3, deltas=0)


def _row(payload: dict, variant_id: str) -> dict:
    return next(row for row in payload["rows"] if row["variant_id"] == variant_id)


def test_report_classifies_missing_external_adapters_fail_closed(tmp_path: Path) -> None:
    """Missing Social-Navigation-PyEnvs runtime inputs should not become benchmark evidence."""
    _minimal_repo(tmp_path)

    payload = report.build_report(repo_root=tmp_path)

    assert _row(payload, "social_force")["benchmark_validity"] == "benchmark_valid_candidate"
    assert _row(payload, "ammv_social_force")["benchmark_validity"] == "diagnostic_only"
    assert _row(payload, "social_navigation_pyenvs_orca")["benchmark_validity"] == "not_available"
    assert payload["status_counts"] == {
        "benchmark_valid_candidate": 1,
        "diagnostic_only": 1,
        "not_available": 4,
    }


def test_report_allows_external_adapters_when_checkout_exists(tmp_path: Path, monkeypatch) -> None:
    """Non-SocialForce external adapters can become candidates when their checkout exists."""
    _minimal_repo(tmp_path)
    external_root = tmp_path / "output/repos/Social-Navigation-PyEnvs"
    external_root.mkdir(parents=True)
    monkeypatch.setattr(report.importlib.util, "find_spec", lambda package: None)

    payload = report.build_report(repo_root=tmp_path, social_navigation_pyenvs_root=external_root)

    assert (
        _row(payload, "social_navigation_pyenvs_orca")["benchmark_validity"]
        == "benchmark_valid_candidate"
    )
    assert (
        _row(payload, "social_navigation_pyenvs_sfm_helbing")["benchmark_validity"]
        == "benchmark_valid_candidate"
    )
    assert (
        _row(payload, "social_navigation_pyenvs_socialforce")["benchmark_validity"]
        == "not_available"
    )


def test_render_markdown_separates_diagnostic_and_unavailable_rows(tmp_path: Path) -> None:
    """The Markdown report should make caveated rows visible."""
    _minimal_repo(tmp_path)

    text = report.render_markdown(report.build_report(repo_root=tmp_path))

    assert "`ammv_social_force`" in text
    assert "`diagnostic_only`" in text
    assert "`not_available`" in text
    assert "must not be counted as benchmark-success evidence" in text
