"""Contract tests for the Issue #6720 Flint analysis-only foundation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from robot_sf.benchmark.flint_chart import (
    ATLAS_SCHEMA_VERSION,
    SURFACE_SCHEMA_VERSION,
    FlintChartContractError,
    build_atlas_manifest,
    build_surface,
    load_json,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/flint_chart"
INPUT_SCHEMA = json.loads(
    (REPO_ROOT / "tools/flint-chart/surface-input-schema.v1.json").read_text(encoding="utf-8")
)
SURFACE_SCHEMA = json.loads(
    (REPO_ROOT / "tools/flint-chart/surface-schema.v1.json").read_text(encoding="utf-8")
)
ATLAS_SCHEMA = json.loads(
    (REPO_ROOT / "tools/flint-chart/atlas-manifest-schema.v1.json").read_text(encoding="utf-8")
)


def _payload(name: str = "figure_7_6_release_input.json") -> dict:
    """Load one synthetic/public-safe surface input."""
    return dict(load_json(FIXTURE_ROOT / name))


def test_versioned_schemas_are_valid_and_fixture_surface_is_schema_valid() -> None:
    """The versioned schemas compile and accept a built candidate surface."""
    jsonschema.Draft202012Validator.check_schema(INPUT_SCHEMA)
    jsonschema.Draft202012Validator.check_schema(SURFACE_SCHEMA)
    jsonschema.Draft202012Validator.check_schema(ATLAS_SCHEMA)

    input_payload = _payload()
    assert list(jsonschema.Draft202012Validator(INPUT_SCHEMA).iter_errors(input_payload)) == []
    surface = build_surface(input_payload)

    assert surface["schema_version"] == SURFACE_SCHEMA_VERSION
    assert surface["render_status"] == "not_run"
    assert list(jsonschema.Draft202012Validator(SURFACE_SCHEMA).iter_errors(surface)) == []


def test_surface_is_deterministic_and_preserves_exact_ties_without_rank() -> None:
    """Repeated builds are byte-equivalent and tie cells carry no rank."""
    first = build_surface(_payload())
    second = build_surface(_payload())

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["parity"]["status"] == "passed"
    assert first["coverage"]["expected_cells"] == 4
    tie_groups = first["tie_groups"]
    assert len(tie_groups) == 1
    assert tie_groups[0]["members"] == [["ppo", "doorway"], ["sacadrl", "doorway"]]
    tied = [cell for cell in first["cells"] if cell["tie_group"]]
    assert len(tied) == 2
    assert all(cell["rank"] is None for cell in tied)
    assert all(cell["rank"] is None for cell in first["cells"])


def test_atlas_keeps_release_and_replay_contexts_separate(tmp_path: Path) -> None:
    """The atlas accepts the same surface id only when contexts remain separate."""
    release = tmp_path / "release.surface.json"
    replay = tmp_path / "replay.surface.json"
    write_json(release, build_surface(_payload()))
    write_json(replay, build_surface(_payload("figure_7_6_replay_input.json")))

    manifest = build_atlas_manifest([release, replay], atlas_id="fixture-atlas")

    assert manifest["schema_version"] == ATLAS_SCHEMA_VERSION
    assert manifest["contexts"] == {
        "release": ["figure_7_6_success_by_family"],
        "replay": ["figure_7_6_success_by_family"],
    }
    assert manifest["coverage"]["surface_count"] == 2
    assert list(jsonschema.Draft202012Validator(ATLAS_SCHEMA).iter_errors(manifest)) == []


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda payload: payload["candidate_cells"][0].update({"value": 0.7}), "parity drift"),
        (
            lambda payload: payload["candidate_cells"].pop(),
            "candidate cells do not match display population",
        ),
        (
            lambda payload: payload["candidate_cells"][0].update({"catalog_rank": 1}),
            "unsupported fields",
        ),
    ],
)
def test_surface_fails_closed_on_drift_population_or_catalog_rank(mutation, expected: str) -> None:
    """Candidate changes cannot silently become a promotion-ready surface."""
    payload = _payload()
    mutation(payload)

    with pytest.raises(FlintChartContractError, match=expected):
        build_surface(payload)


def test_figure_7_1_requires_uncertainty_and_direct_label_policy() -> None:
    """The uncertainty/direct-label gate is checked before any rendering path."""
    payload = _payload()
    payload["figure_id"] = "figure_7_1"
    payload["renderer_policy"] = {
        "canonical_renderer": "matplotlib/pgf/tikz",
        "tie_policy": "exact_ties_no_catalog_rank",
        "source_context_separation": True,
        "requires_uncertainty": True,
        "requires_direct_labels": True,
    }

    surface = build_surface(payload)

    assert surface["render_status"] == "not_run"
    assert surface["renderer_policy"]["requires_direct_labels"] is True


def test_atlas_rejects_duplicate_surface_context(tmp_path: Path) -> None:
    """A repeated release surface cannot create ambiguous atlas coverage."""
    surface = tmp_path / "surface.json"
    write_json(surface, build_surface(_payload()))

    with pytest.raises(FlintChartContractError, match="duplicate surface context"):
        build_atlas_manifest([surface, surface], atlas_id="duplicate")


def test_output_surface_mutation_cannot_claim_complete_population(tmp_path: Path) -> None:
    """The atlas rechecks cell/population counts instead of trusting coverage flags."""
    surface = build_surface(_payload())
    surface["cells"] = surface["cells"][:-1]
    path = tmp_path / "mutated.surface.json"
    write_json(path, surface)

    with pytest.raises(FlintChartContractError, match="cells do not match display population"):
        build_atlas_manifest([path], atlas_id="mutated")


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (
            lambda surface: surface["renderer_policy"].update(
                {"canonical_renderer": "unapproved/renderer"}
            ),
            "renderer_policy.canonical_renderer",
        ),
        (lambda surface: surface["metric"].pop("id"), "metric.id"),
        (
            lambda surface: surface["cells"][0].update({"tie_group": None}),
            "cell tie references",
        ),
        (lambda surface: surface.update({"claim_boundary": "promoted"}), "analysis-only"),
    ],
)
def test_atlas_rejects_mutated_surface_contract(tmp_path: Path, mutation, expected: str) -> None:
    """The atlas must not accept hand-edited surface metadata."""
    surface = build_surface(_payload())
    mutation(surface)
    path = tmp_path / "mutated-contract.surface.json"
    write_json(path, surface)

    with pytest.raises(FlintChartContractError, match=expected):
        build_atlas_manifest([path], atlas_id="mutated-contract")


def test_payload_copy_is_independent() -> None:
    """The fixture helper does not expose mutable state across builds."""
    original = _payload()
    cloned = copy.deepcopy(original)
    cloned["metric"]["unit"] = "fraction"
    assert original == _payload()
