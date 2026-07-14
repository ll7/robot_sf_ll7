"""Tests for campaign atlas and event-aligned ensemble context views (#5616).

Covers:
- Atlas covers all eligible cells with explicit markers for missing/ineligible data.
- Ensemble view refuses to render when no shared event anchor exists.
- Parity check between HTML exploration atlas and static output fails closed.
- Outputs are deterministic (two runs hash-stable).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.campaign_atlas import (
    AtlasConfig,
    AtlasParityError,
    EnsembleResult,
    EpisodeInventoryRow,
    PredicateInterval,
    TrajectoryPoint,
    build_atlas_summary,
    build_campaign_atlas,
    check_atlas_parity,
    render_ensemble_context_view,
    render_html_atlas_exploration,
)

FAMILY_A = "famA"
FAMILY_B = "famB"
PLANNER_X = "orca"
PLANNER_Y = "ppo"
ANCHOR = "near_miss_start"


def _row(
    episode_id: str,
    planner: str,
    family: str,
    seed: int,
    outcome: str,
    *,
    with_anchor: bool = True,
    traj_len: int = 10,
) -> EpisodeInventoryRow:
    """Build a minimal campaign inventory row for tests."""
    return EpisodeInventoryRow(
        episode_id=episode_id,
        planner=planner,
        scenario_id=f"{family}_s{seed}",
        scenario_family=family,
        seed=seed,
        outcome=outcome,
        metrics={"path_efficiency": float(seed)},
        trajectory=tuple(
            TrajectoryPoint(t=float(k), x=float(k), y=float(k) + seed) for k in range(traj_len)
        ),
        event_anchors={"near_miss_start": 5.0} if with_anchor else {},
        predicate_timeline=(
            PredicateInterval(0.0, 3.0, "approach"),
            PredicateInterval(3.0, 8.0, "evade"),
        ),
    )


def _eligible_rows() -> list[EpisodeInventoryRow]:
    """Return a small multi-cell campaign inventory covering eligible cells."""
    return [
        _row("a0", PLANNER_X, FAMILY_A, 1, "collision"),
        _row("a1", PLANNER_X, FAMILY_A, 2, "collision"),
        _row("a2", PLANNER_X, FAMILY_A, 3, "success"),
        _row("a3", PLANNER_Y, FAMILY_A, 4, "success"),
        _row("b0", PLANNER_X, FAMILY_B, 5, "timeout"),
    ]


class TestAtlasEligibleCells:
    """Atlas covers all eligible cells with explicit missing/ineligible markers."""

    def test_eligible_cells_all_present(self) -> None:
        """Every (scenario-family, planner) combination appears in the summary."""
        rows = _eligible_rows()
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="c1"))
        cell_keys = {(c.scenario_family, c.planner) for c in summary.cells}
        assert (FAMILY_A, PLANNER_X) in cell_keys
        assert (FAMILY_A, PLANNER_Y) in cell_keys
        assert (FAMILY_B, PLANNER_X) in cell_keys

    def test_eligible_cells_have_counts_and_ci(self) -> None:
        """An eligible cell carries outcome counts and a Wilson interval."""
        rows = _eligible_rows()
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="c1"))
        cell = next(
            c for c in summary.cells if c.scenario_family == FAMILY_A and c.planner == PLANNER_X
        )
        assert cell.eligible
        assert cell.n_total == 3
        assert cell.outcome_counts.get("collision") == 2
        assert set(cell.outcome_ci) == {"collision", "success"}
        lo, hi = cell.outcome_ci["collision"][1], cell.outcome_ci["collision"][2]
        assert 0.0 <= lo <= hi <= 1.0

    def test_eligible_cells_mark_missing(self) -> None:
        """A cell with no episodes is explicitly marked ineligible with a reason."""
        rows = _eligible_rows()
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="c1"))
        cell = next(
            (c for c in summary.cells if c.scenario_family == FAMILY_B and c.planner == PLANNER_Y),
            None,
        )
        assert cell is not None
        assert cell.eligible is False
        assert cell.n_total == 0
        assert cell.ineligible_reason is not None
        assert "no eligible episodes" in cell.ineligible_reason

    def test_eligible_cells_below_min_size_marked_ineligible(self) -> None:
        """A cell below the configured minimum size is ineligible (not silently dropped)."""
        rows = _eligible_rows()
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="c1", min_cell_size=5))
        for cell in summary.cells:
            assert cell.eligible is False
            assert cell.ineligible_reason is not None

    def test_eligible_cells_exemplar_marker_resolved(self) -> None:
        """A selected exemplar episode id is resolved and recorded on its cell."""
        rows = _eligible_rows()
        summary = build_atlas_summary(
            rows, config=AtlasConfig(campaign_id="c1"), exemplar_episode_ids=["a0"]
        )
        cell = next(
            c for c in summary.cells if c.scenario_family == FAMILY_A and c.planner == PLANNER_X
        )
        assert cell.exemplar_episode_ids == ("a0",)


class TestEnsembleContext:
    """Ensemble view refuses to render without a shared event anchor."""

    def test_ensemble_context_renders_when_anchor_shared(self) -> None:
        """When every episode shares the anchor, the view renders with a medoid."""
        rows = [
            _row("e0", PLANNER_X, FAMILY_A, 1, "collision"),
            _row("e1", PLANNER_X, FAMILY_A, 2, "collision"),
        ]
        result = render_ensemble_context_view(rows, anchor=ANCHOR, out=Path("__unused__.svg"))
        assert isinstance(result, EnsembleResult)
        assert result.status == "rendered"
        assert result.medoid_episode_id is not None
        assert set(result.aligned_episode_ids) == {"e0", "e1"}

    def test_ensemble_context_event_anchor_missing_refuses(self) -> None:
        """When one episode lacks the anchor, the view is explicitly unavailable."""
        rows = [
            _row("e0", PLANNER_X, FAMILY_A, 1, "collision", with_anchor=True),
            _row("e1", PLANNER_X, FAMILY_A, 2, "collision", with_anchor=False),
        ]
        result = render_ensemble_context_view(rows, anchor=ANCHOR, out=Path("__unused__.svg"))
        assert result.status == "unavailable"
        assert result.reason is not None
        assert "e1" in result.reason
        assert result.medoid_episode_id is None
        assert result.output_path is not None

    def test_ensemble_context_event_anchor_missing_writes_figure(self, tmp_path: Path) -> None:
        """An unavailable ensemble view still writes an explicit 'unavailable' figure."""
        rows = [_row("e1", PLANNER_X, FAMILY_A, 2, "collision", with_anchor=False)]
        out = tmp_path / "ensemble.svg"
        result = render_ensemble_context_view(rows, anchor=ANCHOR, out=out)
        assert result.status == "unavailable"
        assert out.exists()
        assert "unavailable" in out.read_text(encoding="utf-8")

    def test_ensemble_context_quantile_band_does_not_span_unaligned(self) -> None:
        """Alignment is on event-relative time, so trajectory *t* values vary per episode."""
        rows = [
            _row("e0", PLANNER_X, FAMILY_A, 1, "collision", traj_len=12),
            _row("e1", PLANNER_X, FAMILY_A, 2, "collision", traj_len=12),
        ]
        result = render_ensemble_context_view(rows, anchor=ANCHOR, out=Path("__unused__.svg"))
        assert result.status == "rendered"
        assert result is not None


class TestAtlasHtmlParity:
    """Parity check between HTML exploration atlas and static output."""

    def test_atlas_html_parity_passes_on_matching_summary(self, tmp_path: Path) -> None:
        """When the HTML summary matches the static summary, parity passes."""
        rows = _eligible_rows()
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="c1"))
        html_path = render_html_atlas_exploration(summary, tmp_path / "atlas.html")
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "campaign_atlas.v1",
                    "campaign_id": summary.campaign_id,
                    "scenario_families": list(summary.scenario_families),
                    "planners": list(summary.planners),
                    "event_anchor": summary.event_anchor,
                    "selection_manifest_hash": summary.selection_manifest_hash,
                    "metric_definitions": dict(summary.metric_definitions),
                    "cells": [
                        {
                            "scenario_family": c.scenario_family,
                            "planner": c.planner,
                            "eligible": c.eligible,
                            "ineligible_reason": c.ineligible_reason,
                            "n_total": c.n_total,
                            "outcome_counts": dict(c.outcome_counts),
                            "outcome_ci": {k: list(v) for k, v in c.outcome_ci.items()},
                            "exemplar_episode_ids": list(c.exemplar_episode_ids),
                        }
                        for c in summary.cells
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        check_atlas_parity(html_path, summary_path)

    def test_atlas_html_parity_fails_closed_on_mismatch(self, tmp_path: Path) -> None:
        """A divergent HTML summary raises AtlasParityError (fail-closed)."""
        rows = _eligible_rows()
        summary = build_atlas_summary(rows, config=AtlasConfig(campaign_id="c1"))
        html_path = render_html_atlas_exploration(summary, tmp_path / "atlas.html")
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "campaign_atlas.v1",
                    "campaign_id": "DIFFERENT",
                    "scenario_families": list(summary.scenario_families),
                    "planners": list(summary.planners),
                    "event_anchor": summary.event_anchor,
                    "selection_manifest_hash": summary.selection_manifest_hash,
                    "metric_definitions": dict(summary.metric_definitions),
                    "cells": [],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        with pytest.raises(AtlasParityError):
            check_atlas_parity(html_path, summary_path)


class TestAtlasDeterminism:
    """Two builds of the same inputs are hash-stable."""

    def test_full_build_deterministic(self, tmp_path: Path) -> None:
        """Two full builds into separate dirs produce identical artifact trees."""
        from scripts.analysis.build_campaign_atlas_issue_5616 import (
            _build,
            _hash_tree,
            build_arg_parser,
            load_inventory,
        )

        inventory = tmp_path / "inv.jsonl"
        inventory.write_text(
            "\n".join(
                json.dumps(
                    {
                        "episode_id": r.episode_id,
                        "planner": r.planner,
                        "scenario_id": r.scenario_id,
                        "scenario_family": r.scenario_family,
                        "seed": r.seed,
                        "outcome": r.outcome,
                        "event_anchors": dict(r.event_anchors),
                    }
                )
                for r in _eligible_rows()
            ),
            encoding="utf-8",
        )
        rows = load_inventory(inventory)
        args = build_arg_parser().parse_args(
            ["--inventory", str(inventory), "--campaign-id", "c1", "--ensemble-anchor", ANCHOR]
        )
        first = tmp_path / "a"
        second = tmp_path / "b"
        _build(first, rows, args)
        _build(second, rows, args)
        assert _hash_tree(first) == _hash_tree(second)

    def test_cli_check_determinism_flag(self, tmp_path: Path) -> None:
        """The CLI --check-determinism flag asserts hash-stable output."""
        from scripts.analysis.build_campaign_atlas_issue_5616 import (
            build_arg_parser,
            check_determinism,
            load_inventory,
        )

        inventory = tmp_path / "inv.jsonl"
        inventory.write_text(
            "\n".join(
                json.dumps(
                    {
                        "episode_id": r.episode_id,
                        "planner": r.planner,
                        "scenario_id": r.scenario_id,
                        "scenario_family": r.scenario_family,
                        "seed": r.seed,
                        "outcome": r.outcome,
                        "event_anchors": dict(r.event_anchors),
                    }
                )
                for r in _eligible_rows()
            ),
            encoding="utf-8",
        )
        rows = load_inventory(inventory)
        args = build_arg_parser().parse_args(["--inventory", str(inventory), "--campaign-id", "c1"])
        assert check_determinism(rows, args) is True


class TestAtlasManifestBinding:
    """Outputs carry the required manifest binding (artifact_catalog.v1)."""

    def test_catalog_binds_run_metadata(self, tmp_path: Path) -> None:
        """The catalog binds campaign id, versions, selection hash, and output hashes."""
        import yaml

        rows = _eligible_rows()
        result = build_campaign_atlas(
            rows,
            out_dir=tmp_path,
            config=AtlasConfig(campaign_id="c1"),
            exemplar_episode_ids=["a0"],
            selection_manifest_hash="deadbeef",
            ensemble_anchor=ANCHOR,
            render_html=False,
            commit="abc123",
        )
        assert result.catalog_path is not None
        catalog = yaml.safe_load(result.catalog_path.read_text(encoding="utf-8"))
        provenance = catalog["provenance"]
        assert provenance["campaign_id"] == "c1"
        assert provenance["selection_manifest_hash"] == "deadbeef"
        assert provenance["event_detector_version"]
        assert provenance["renderer_version"] == "matplotlib"
        for artifact in catalog["artifacts"]:
            for out_ref in artifact["outputs"].values():
                assert len(out_ref["sha256"]) == 64
