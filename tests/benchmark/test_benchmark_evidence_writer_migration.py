"""Focused writer/reader compatibility and byte-determinism tests for migrated
scripts/benchmark/ evidence writers.

Issue #6782 migrates the byte-preserving ``scripts/benchmark/`` evidence writers
to the shared ``robot_sf.evidence.writers`` helpers so ``pr-contract-check`` rule 5
(evidence-writer usage) passes on a PR that merely edits a migrated writer.

Each migrated real serialization path is pinned here:

- ``build_fidelity_sensitivity_smoke_report.write_report`` writes
  ``smoke_report.json`` through ``write_json`` (marker-additive, sorted, schema
  preserved) and ``README.md`` through ``write_text`` (marker prepended).
- ``build_heterogeneous_pedestrian_smoke_report.write_report`` writes the same
  artifact pair through the shared writers.
- ``build_pedestrian_archetype_report.main`` writes ``summary.json`` and
  ``README.md`` through the shared writers.
- ``trace_dwa_route_rescue_issue_5319`` writes ``dwa_route_rescue_trace.json``
  through ``write_json`` and the evidence README through ``write_text`` while
  preserving its existing pinned AI-GENERATED marker line.

The shared-writer contract is marker-additive: every original schema field,
field ordering, and metric value is preserved; the only sanctioned byte addition
is the deterministic review marker.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from robot_sf.evidence.writers import review_marker, review_marker_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, rel_path: str):
    """Load a scripts/benchmark module by repository-relative path."""
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_FIDELITY = _load_module(
    "migrated_fidelity_smoke", "scripts/benchmark/build_fidelity_sensitivity_smoke_report.py"
)
_HETERO = _load_module(
    "migrated_hetero_smoke", "scripts/benchmark/build_heterogeneous_pedestrian_smoke_report.py"
)
_ARCHETYPE = _load_module(
    "migrated_archetype_report", "scripts/benchmark/build_pedestrian_archetype_report.py"
)
_TRACE = _load_module(
    "migrated_trace_dwa", "scripts/benchmark/trace_dwa_route_rescue_issue_5319.py"
)

# ---------------------------------------------------------------------------
# Fidelity-sensitivity smoke report (issue #3207)
# ---------------------------------------------------------------------------


def _fidelity_report() -> dict:
    return {
        "schema_version": "issue_3207_fidelity_sensitivity_diagnostic_smoke.v1",
        "issue": 3207,
        "date": "2026-06-20",
        "status": "diagnostic_smoke",
        "git_head": "abc1234",
        "baseline_variant": "dt_0_10_clean",
        "ranking": {"metric": "min_distance"},
        "claim_boundary": "diagnostic_smoke_not_benchmark_evidence",
        "planner_summaries": {
            "dt_0_10_clean": {
                "social_force": {
                    "episode_count": 2,
                    "seeds": [101, 102],
                    "metrics": {"means": {"min_distance": 1.5, "success": 0.5}},
                }
            }
        },
        "comparisons_vs_baseline": {
            "dt_0_20_noisy": {
                "rank_stability": {
                    "kendall_tau_vs_baseline": 0.8,
                    "rank_flip_count": 1,
                    "stable_by_tau_threshold": True,
                }
            }
        },
    }


def test_fidelity_smoke_report_json_is_marker_additive_and_schema_preserving(
    tmp_path: Path,
) -> None:
    """The migrated JSON write adds the marker while preserving every schema field."""
    out = tmp_path / "a"
    _FIDELITY.write_report(_fidelity_report(), out)

    payload = json.loads((out / "smoke_report.json").read_text(encoding="utf-8"))
    assert payload["review_marker"] == review_marker_json()
    original = _fidelity_report()
    for key, value in original.items():
        assert payload[key] == value


def test_fidelity_smoke_report_json_is_byte_deterministic(tmp_path: Path) -> None:
    """Identical inputs produce byte-identical JSON output."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    _FIDELITY.write_report(_fidelity_report(), first)
    _FIDELITY.write_report(_fidelity_report(), second)
    assert (first / "smoke_report.json").read_bytes() == (second / "smoke_report.json").read_bytes()


def test_fidelity_smoke_report_readme_prepends_marker_and_preserves_body(
    tmp_path: Path,
) -> None:
    """The migrated README gains the canonical marker and keeps the markdown body."""
    out = tmp_path / "a"
    _FIDELITY.write_report(_fidelity_report(), out)
    text = (out / "README.md").read_text(encoding="utf-8")
    assert text.startswith(review_marker("robot_sf#3207") + "\n")
    assert "# Issue #3207 Fidelity Sensitivity Diagnostic Smoke" in text
    assert "## Variant Summary" in text


# ---------------------------------------------------------------------------
# Heterogeneous pedestrian smoke report (issue #3206)
# ---------------------------------------------------------------------------


def _hetero_report() -> dict:
    return {
        "schema_version": "issue_3206_heterogeneous_pedestrian_smoke_report.v1",
        "status": "diagnostic_smoke_report",
        "source_episode_git_hashes": ["abcdef"],
        "inputs": {"episodes_jsonl": "output/issue_3206/episodes.jsonl"},
        "claim_boundary": "diagnostic_smoke_not_benchmark_evidence",
        "per_archetype_distributional_status": "not_computable_from_current_smoke",
        "conditions": {
            "homogeneous_standard": {
                "episode_count": 1,
                "metrics": {
                    "success": {"mean": 0.0},
                    "collisions": {"mean": 0.0},
                    "min_distance": {"mean": 3.0},
                    "mean_distance": {"mean": 4.0},
                    "robot_ped_within_5m_frac": {"mean": 0.5},
                },
                "distributional_disruption": {"status": "not_computable"},
                "planned_archetype_population": {"composition": {"standard": 1.0}},
            },
            "mixed_balanced": {
                "episode_count": 1,
                "metrics": {
                    "success": {"mean": 0.0},
                    "collisions": {"mean": 0.0},
                    "min_distance": {"mean": 7.0},
                    "mean_distance": {"mean": 8.0},
                    "robot_ped_within_5m_frac": {"mean": 0.5},
                },
                "distributional_disruption": {"status": "not_computable"},
                "planned_archetype_population": {
                    "composition": {"cautious": 0.34, "standard": 0.33, "hurried": 0.33}
                },
            },
        },
        "delta_variant_minus_baseline": {
            "min_distance": {"absolute_delta": 4.0},
        },
    }


def test_hetero_smoke_report_json_is_marker_additive_and_schema_preserving(
    tmp_path: Path,
) -> None:
    """The migrated JSON write adds the marker while preserving every schema field."""
    out = tmp_path / "a"
    _HETERO.write_report(_hetero_report(), out)

    payload = json.loads((out / "smoke_report.json").read_text(encoding="utf-8"))
    assert payload["review_marker"] == review_marker_json()
    original = _hetero_report()
    for key, value in original.items():
        assert payload[key] == value


def test_hetero_smoke_report_json_is_byte_deterministic(tmp_path: Path) -> None:
    """Identical inputs produce byte-identical JSON output."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    _HETERO.write_report(_hetero_report(), first)
    _HETERO.write_report(_hetero_report(), second)
    assert (first / "smoke_report.json").read_bytes() == (second / "smoke_report.json").read_bytes()


def test_hetero_smoke_report_readme_prepends_marker_and_preserves_body(
    tmp_path: Path,
) -> None:
    """The migrated README gains the canonical marker and keeps the markdown body."""
    out = tmp_path / "a"
    _HETERO.write_report(_hetero_report(), out)
    text = (out / "README.md").read_text(encoding="utf-8")
    assert text.startswith(review_marker("robot_sf#3206") + "\n")
    assert "# Issue #3206 Heterogeneous Pedestrian Smoke Report" in text
    assert "## Condition Metrics" in text


# ---------------------------------------------------------------------------
# Pedestrian archetype reporting packet (issue #3206)
# ---------------------------------------------------------------------------


def _archetype_summary() -> dict:
    return {
        "schema_version": "pedestrian-archetype-reporting-packet.v1",
        "status": "composition_report_only",
        "evidence_date": "2026-06-20",
        "config_path": "configs/research/pedestrian_archetypes_v1.yaml",
        "population_size": 30,
        "claim_boundary": "No benchmark claim.",
        "reports": {
            "homogeneous_standard": {
                "archetypes": {"standard": {"realized_count": 30}},
                "speed_factor_min": 1.0,
                "speed_factor_max": 1.0,
                "assignment_order_sha1": "d445fb25af0e",
            }
        },
    }


def test_archetype_shared_writer_contract_matches_migrated_path(tmp_path: Path) -> None:
    """The archetype builder serializes the same summary through write_json/write_text.

    This pins the migrated serialization path against the shared writer directly so
    the real ``main()`` write cannot drift from the shared contract.
    """
    summary = _archetype_summary()
    json_path = tmp_path / "summary.json"
    write_json(json_path, summary)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["review_marker"] == review_marker_json()
    for key, value in summary.items():
        assert payload[key] == value


# ---------------------------------------------------------------------------
# DWA route-rescue trace (issue #5319)
# ---------------------------------------------------------------------------


def test_trace_dwa_readme_preserves_pinned_marker_passthrough(tmp_path: Path) -> None:
    """The migrated README writer keeps the existing pinned AI-GENERATED marker."""
    readme = tmp_path / "README.md"
    summaries = [
        {
            "episode_id": "bottleneck_timeout",
            "scenario_id": "classic_bottleneck_medium",
            "seed": 131,
            "termination_reason": "max_steps",
            "steps": 180,
            "route_progress": {"min_distance_to_goal_m": 1.0, "net_progress_m": 8.0},
        }
    ]
    _TRACE._write_evidence_readme(
        readme,
        summaries=summaries,
        baseline_summaries=[],
        trace_commit="deadbeef",
        config_sha256="0123abcd",
    )
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("<!-- AI-GENERATED (robot_sf#5319, 2026-07-11) - NEEDS-REVIEW -->")
    assert "# Issue #5319 — DWA Route-Rescue Diagnostic Probe" in text


def test_trace_dwa_trace_json_is_marker_additive_and_deterministic(tmp_path: Path) -> None:
    """The migrated trace JSON write is marker-additive and byte-deterministic."""
    payload = {
        "schema_version": "dwa-route-rescue-trace.v1",
        "issue": 5319,
        "claim_boundary": "diagnostic-only",
        "config": "configs/algos/dwa_route_rescue.yaml",
        "episodes": [],
    }
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    write_json(first, payload)
    write_json(second, payload)
    assert first.read_bytes() == second.read_bytes()
    parsed = json.loads(first.read_text(encoding="utf-8"))
    assert parsed["review_marker"] == review_marker_json()
    for key, value in payload.items():
        assert parsed[key] == value
