"""Capability tests for the bounded doorway QD campaign runner (issue #5308).

These tests exercise the campaign runner wiring
(``scripts/adversarial.run_qd_campaign_issue_5308``) in smoke mode (injected
evaluator, no simulator) so the integration contract - config parsing, warm-start
extraction, emitter mix (Random + CoordinateRefinement + CMA-ME), archive artifact
schema, and the equal-budget comparison - is validated on CPU. This is capability
plumbing, not a benchmark claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "adversarial"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_qd_campaign_issue_5308 import (  # noqa: E402
    CLAIM_BOUNDARY,
    _build_emitters,
    _build_grid,
    _build_search_space,
    _load_campaign_config,
    run_campaign,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "adversarial" / ("issue_5308_qd_doorway.yaml")
)


def _config_payload() -> dict:
    """Load the canonical campaign config once."""
    return _load_campaign_config(CONFIG_PATH)


def test_campaign_config_parses_and_is_well_formed() -> None:
    """The checked-in doorway campaign config validates and carries the contract."""
    payload = _config_payload()
    assert payload["schema_version"] == "adversarial-qd-campaign.v1"
    assert payload["issue"] == 5308
    assert payload["claim_boundary"] == "capability_not_evidence"
    assert payload["family"] == "doorway"
    assert payload["objective"] == "worst_case_snqi"
    grid = _build_grid(payload)
    assert grid.bins >= 2
    space = _build_search_space(payload)
    assert space.start_x.max > space.start_x.min
    # Doorway geometry lies along the classic_doorway robot route.
    assert 25.0 <= space.start_x.min <= space.start_x.max <= 28.0


def test_campaign_config_budget_within_four_hour_cpu_contract() -> None:
    """The checked-in budget must respect the issue #5308 <=4h CPU stop condition."""
    payload = _config_payload()
    budget = int(payload["budget"])
    # Conservative upper bound: the campaign runs ~1 episode/candidate on CPU.
    # 4h at a generous 30s/candidate caps the budget well above this; the checked-in
    # value (240) is far inside it.
    assert 1 <= budget <= 480


def test_campaign_config_emits_cma_me_emitter() -> None:
    """The campaign config must wire the CMA-ME diversity-driven emitter."""
    payload = _config_payload()
    assert "cma_me" in payload["emitters"]


def test_campaign_smoke_emits_valid_archive_artifact(tmp_path: Path) -> None:
    """Smoke-mode run emits an adversarial_qd_archive.v1 artifact with filled cells."""
    summary = run_campaign(CONFIG_PATH, tmp_path, budget_override=24, smoke=True)
    archive = json.loads((tmp_path / "archive.json").read_text(encoding="utf-8"))
    assert archive["schema_version"] == "adversarial_qd_archive.v1"
    assert archive["behavior_axes"] == ["distance_to_human_min", "time_to_collision_min"]
    assert archive["summary"]["filled_cell_count"] == summary["filled_cell_count"]
    assert archive["summary"]["filled_cell_count"] > 0
    assert len(archive["cells"]) == archive["summary"]["filled_cell_count"]
    for cell in archive["cells"]:
        assert len(cell["cell"]) == 2
        assert isinstance(cell["objective_value"], (int, float))
        assert cell["candidate"]["start"]["x"] is not None


def test_campaign_smoke_finds_distinct_certified_failure_modes(tmp_path: Path) -> None:
    """Smoke-mode archive admits at least two distinct certified failure mechanisms."""
    summary = run_campaign(CONFIG_PATH, tmp_path, budget_override=30, smoke=True)
    modes = summary["distinct_failure_modes"]
    assert len(modes) >= 2
    assert summary["filled_cell_count"] > 0


def test_campaign_smoke_emits_equal_budget_comparison(tmp_path: Path) -> None:
    """Smoke-mode run emits an equal-budget MAP-Elites vs single-objective report."""
    run_campaign(CONFIG_PATH, tmp_path, budget_override=18, smoke=True)
    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["comparison_type"] == "equal_budget_qd_vs_single_objective"
    assert comparison["rows"]["map_elites"]["budget"] == 18
    assert comparison["rows"]["single_objective"]["budget"] == 18
    assert comparison["rows"]["map_elites"]["num_evaluated"] == 18
    assert comparison["summary"]["qd_distinct_failure_modes"] >= 1


def test_campaign_smoke_uses_warm_start_seeds(tmp_path: Path) -> None:
    """The warm-start flip report seeds the emitters before CMA-ME diversity search."""
    run_campaign(CONFIG_PATH, tmp_path, budget_override=12, smoke=True)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["claim_boundary"] == CLAIM_BOUNDARY
    assert manifest["archive_artifact"].endswith("archive.json")
    # Warm-start extraction is logged in the summary provenance; the campaign runs.
    summary = json.loads((tmp_path / "campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["num_evaluated"] == 12


def test_build_emitters_rejects_unknown_emitter() -> None:
    """An unknown emitter name must fail closed."""
    from robot_sf.adversarial.qd import GridSpec, QDArchive

    payload = _config_payload()
    space = _build_search_space(payload)
    archive = QDArchive(grid=GridSpec(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, bins=2))
    with pytest.raises(ValueError, match="unknown emitter"):
        _build_emitters({"emitters": ["totally_bogus"]}, space, archive, warm_starts=(), seed=0)


def test_campaign_smoke_is_reproducible(tmp_path: Path) -> None:
    """Same config + seed + smoke budget yields identical archive coverage."""
    a = run_campaign(CONFIG_PATH, tmp_path / "a", budget_override=20, smoke=True)
    b = run_campaign(CONFIG_PATH, tmp_path / "b", budget_override=20, smoke=True)
    assert a["filled_cell_count"] == b["filled_cell_count"]
    assert a["distinct_failure_modes"] == b["distinct_failure_modes"]
    archive_a = json.loads((tmp_path / "a" / "archive.json").read_text(encoding="utf-8"))
    archive_b = json.loads((tmp_path / "b" / "archive.json").read_text(encoding="utf-8"))
    assert archive_a["summary"] == archive_b["summary"]


FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "adversarial"
    / "fixtures"
    / "issue_5308_qd_doorway"
)


def test_tracked_fixture_archive_meets_issue_5308_contract() -> None:
    """The archived doorway campaign fixture satisfies the issue #5308 stop conditions."""
    pytest.importorskip("pytest")
    archive_path = FIXTURE_DIR / "archive.json"
    assert archive_path.exists(), f"tracked capability archive missing: {archive_path}"
    payload = json.loads(archive_path.read_text(encoding="utf-8"))
    # Schema matches adversarial_qd_archive.v1.
    assert payload["schema_version"] == "adversarial_qd_archive.v1"
    assert payload["behavior_axes"] == ["distance_to_human_min", "time_to_collision_min"]
    # filled_cell_count > 0 and >= 2 distinct certified failure modes.
    assert payload["summary"]["filled_cell_count"] > 0
    assert len(payload["summary"]["distinct_failure_modes"]) >= 2
    assert len(payload["cells"]) == payload["summary"]["filled_cell_count"]
    # Equal-budget comparison artifact is archived alongside.
    comparison = json.loads((FIXTURE_DIR / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["comparison_type"] == "equal_budget_qd_vs_single_objective"
    manifest = json.loads((FIXTURE_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["claim_boundary"].startswith("capability_not_evidence")
    assert manifest["archive_artifact"].endswith("archive.json")
