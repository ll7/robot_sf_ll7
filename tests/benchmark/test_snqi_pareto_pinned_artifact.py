"""Drift guard tying the pinned SNQI Pareto SVG to the live renderer.

Issue #5418 arose because a renderer change (#5403 label-layout fix) landed in
``format_pareto_svg`` while the tracked evidence-packet SVG at
``docs/context/evidence/issue_3653_snqi_decision_disagreement_job_13175`` was not
regenerated, leaving the durable diagnostic artifact stale relative to the code.

The existing coverage does not catch that class of drift:

- ``tests/benchmark/test_snqi_pareto_numeric_axes.py`` exercises the renderer on a
  *synthetic* report, so it never touches the pinned artifact on disk.
- ``tests/validation/test_check_issue_3653_snqi_decision_disagreement_packet.py``
  only checks that the recorded checksums are internally self-consistent with the
  file bytes, so a stale-but-self-consistent artifact still passes.

This test closes the gap: it re-renders the SVG from the *pinned report JSON* and
asserts it reproduces the tracked SVG byte-for-byte, and that the render's hash
matches the checksum the packet checker enforces. If the renderer changes without
regenerating the durable artifact (and updating the checker constant), this fails.

Scope: diagnostic-only artifact provenance guard; no benchmark, metric, schema,
or paper/dissertation claim is asserted.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.snqi_scalarization_sensitivity import format_pareto_svg

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "docs/context/evidence/issue_3653_snqi_decision_disagreement_job_13175"
REPORT_PATH = EVIDENCE_DIR / "snqi_scalarization_sensitivity.json"
SVG_PATH = EVIDENCE_DIR / "snqi_scalarization_sensitivity_pareto.svg"
CHECKER_PATH = (
    REPO_ROOT / "scripts/validation/check_issue_3653_snqi_decision_disagreement_packet.py"
)

# The two declared Pareto-front planners for job 13175 (see provenance.json).
PARETO_FRONT_PLANNERS = frozenset({"ppo", "hybrid_rule_v3_fast_progress_static_escape"})


def _load_checker():
    spec = importlib.util.spec_from_file_location("_issue_3653_snqi_packet_check", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.base_sensitive
def test_pinned_svg_regenerates_identically_from_pinned_report() -> None:
    """The tracked SVG is exactly what the current renderer produces from its report.

    This is the drift guard: it fails if ``format_pareto_svg`` changes without the
    pinned issue-3653 artifact being regenerated to match (the #5418 failure mode).
    """
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    rendered = format_pareto_svg(report)

    assert rendered == SVG_PATH.read_text(encoding="utf-8"), (
        "Pinned SNQI Pareto SVG is stale relative to format_pareto_svg(). "
        "Regenerate the artifact and update the recorded checksums "
        "(validation script, YAML packet, artifact_inventory.json, packet.json)."
    )


@pytest.mark.base_sensitive
def test_pinned_svg_hash_matches_packet_checker_constant() -> None:
    """The pinned SVG bytes match the checksum the packet checker enforces.

    Ties renderer output -> tracked file -> gate constant in one chain, so a
    regeneration that forgets to update the checker constant is caught here.
    """
    checker = _load_checker()
    expected = checker.EXPECTED_EXPORT_ARTIFACT_HASHES[SVG_PATH.name]

    assert sha256_file(SVG_PATH) == expected


@pytest.mark.base_sensitive
def test_pinned_svg_keeps_both_pareto_front_points_identifiable() -> None:
    """Acceptance item 4: the two Pareto-front points stay identifiable.

    Under the numbered-marker rendering the front points carry an explicit
    ``pareto-point front`` class and a connecting front polyline.
    """
    svg = SVG_PATH.read_text(encoding="utf-8")
    front_planners = {
        line.split('data-planner="', 1)[1].split('"', 1)[0]
        for line in svg.splitlines()
        if 'class="pareto-point front"' in line
    }

    assert front_planners == PARETO_FRONT_PLANNERS
    assert '<polyline points="' in svg
