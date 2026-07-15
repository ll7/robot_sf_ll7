"""Focused contract tests for the #5756 request-to-figure pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import robot_sf.benchmark.candidate_trace_resolution as resolution_module
import robot_sf.benchmark.trace_scene_figure as trace_figure_module
from robot_sf.benchmark.candidate_trace_resolution import (
    ISSUE_5756_MAPPING_SCHEMA_VERSION,
    ISSUE_5756_PINNED_PROVENANCE,
    CandidateTraceResolutionError,
    load_episode_mapping,
    load_episode_requests,
    resolve_episode_requests,
    validate_candidate_trace_resolution,
)
from robot_sf.benchmark.trace_scene_figure import TraceSchemaError, load_episode_from_trace_export
from scripts.analysis import render_worked_example_trace_figures_issue_5756 as render_cli
from scripts.analysis import resolve_candidate_traces_issue_5615 as resolver_cli

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json"
)


def _request_manifest(**request_overrides: object) -> dict[str, object]:
    request = {
        "scenario_id": "classic_bottleneck_medium",
        "planner": "hybrid_rule_v0_minimal",
        "seed": "111",
        "episode_id": "fixture_episode_001",
        **request_overrides,
    }
    return {
        "schema_version": "issue_5446_trace_reexport_list.v1",
        "n_tuples": 1,
        "tuples": [request],
    }


def _mapping_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario_id": "classic_bottleneck_medium",
        "planner": "hybrid_rule_v0_minimal",
        "seed": 111,
        "episode_id": "fixture_episode_001",
        "release_episode_id": "fixture_episode_001",
        "expected_release_outcome": "success",
        "rerun_outcome": "success",
        "trace_artifact_uri": str(TRACE),
    }
    row.update(overrides)
    if "trace_sha256" not in overrides:
        trace_path = Path(str(row["trace_artifact_uri"]))
        row["trace_sha256"] = (
            hashlib.sha256(trace_path.read_bytes()).hexdigest()
            if trace_path.is_file()
            else "0" * 64
        )
    return row


def _load_test_requests(path: Path, payload: dict[str, object]) -> Any:
    _write_json(path, payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return load_episode_requests(
        path,
        expected_count=int(payload["n_tuples"]),
        expected_sha256=digest,
    )


def _mapping_receipt(
    rows: list[dict[str, object]],
    *,
    request_sha256: str,
    provenance_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    provenance = {**ISSUE_5756_PINNED_PROVENANCE, "request_manifest_sha256": request_sha256}
    provenance.update(provenance_overrides or {})
    return {
        "schema_version": ISSUE_5756_MAPPING_SCHEMA_VERSION,
        "n_rows": len(rows),
        "provenance": provenance,
        "rows": rows,
    }


def _load_test_mapping(
    path: Path,
    rows: list[dict[str, object]],
    *,
    request_sha256: str,
) -> Any:
    expected_provenance = {
        **ISSUE_5756_PINNED_PROVENANCE,
        "request_manifest_sha256": request_sha256,
    }
    return load_episode_mapping(
        _write_json(
            path,
            _mapping_receipt(rows, request_sha256=request_sha256),
        ),
        expected_count=len(rows),
        expected_provenance=expected_provenance,
    )


def test_request_resolution_validates_identity_outcome_and_trace(tmp_path: Path) -> None:
    """A concrete request resolves only after all provenance gates pass."""
    request_path = tmp_path / "requests.json"
    mapping_path = tmp_path / "mapping.json"
    request_manifest = _load_test_requests(request_path, _request_manifest())
    assert request_manifest.rows[0]["seed"] == 111
    mapping = _load_test_mapping(
        mapping_path,
        [_mapping_row()],
        request_sha256=request_manifest.content_sha256,
    )
    result = resolve_episode_requests(request_manifest, mapping)

    assert result["summary"] == {
        "n_candidates": 1,
        "n_resolved": 1,
        "n_trace_missing": 0,
        "n_schema_mismatch": 0,
        "n_provenance_incomplete": 0,
    }
    assert result["rows"][0]["reason_code"] == "trace_schema_and_provenance_valid:outcome=success"
    assert result["rows"][0]["exact_repeat_determinism"] == "pinned_artifact"
    assert validate_candidate_trace_resolution(result)["ok"]


@pytest.mark.parametrize(
    ("request_overrides", "mapping_overrides", "status", "reason"),
    [
        (
            {"seed": 112, "episode_id": "missing"},
            {},
            "provenance-incomplete",
            "missing_episode_mapping",
        ),
        (
            {},
            {"scenario_id": "classic_doorway_medium"},
            "provenance-incomplete",
            "mapping_identity_mismatch:scenario_id",
        ),
        ({}, {"rerun_outcome": "collision_event"}, "provenance-incomplete", "outcome_mismatch"),
        (
            {},
            {"trace_artifact_uri": "/not/a/trace.json"},
            "trace-missing",
            "trace_artifact_not_found",
        ),
    ],
)
def test_request_resolution_fails_closed(
    tmp_path: Path,
    request_overrides: dict[str, object],
    mapping_overrides: dict[str, object],
    status: str,
    reason: str,
) -> None:
    """Missing mapping, identity drift, outcome drift, and trace absence are explicit."""
    request_manifest = _load_test_requests(
        tmp_path / "requests.json", _request_manifest(**request_overrides)
    )
    mapping = _load_test_mapping(
        tmp_path / "mapping.json",
        [_mapping_row(**mapping_overrides)],
        request_sha256=request_manifest.content_sha256,
    )
    result = resolve_episode_requests(request_manifest, mapping)
    row = result["rows"][0]
    assert row["resolution_status"] == status
    assert row["reason_code"].startswith(reason)


def test_duplicate_request_tuple_is_rejected(tmp_path: Path) -> None:
    """The 90-request contract cannot silently resolve a duplicate tuple twice."""
    payload = _request_manifest()
    payload["n_tuples"] = 2
    payload["tuples"] = [payload["tuples"][0], dict(payload["tuples"][0])]  # type: ignore[index]
    path = _write_json(tmp_path / "requests.json", payload)
    with pytest.raises(CandidateTraceResolutionError, match="duplicate episode request tuple"):
        load_episode_requests(path, expected_count=2, expected_sha256=None)


def test_trace_export_adapter_provides_renderer_episode() -> None:
    """The typed trace export becomes the renderer's existing derived-series contract."""
    episode = load_episode_from_trace_export(TRACE, outcome="success")
    assert episode.metadata["episode_id"] == "fixture_episode_001"
    assert episode.metadata["episode_status"] == "success"
    assert episode.steps == (0, 1)
    assert episode.nearest_pedestrian_id == ("ped_1", "ped_1")
    assert episode.min_robot_ped_distance_m[0] > episode.min_robot_ped_distance_m[1]


def test_release_episode_id_alias_joins_to_rerun_episode(tmp_path: Path) -> None:
    """A rerun may assign a new id while retaining the release request id."""
    request_manifest = _load_test_requests(tmp_path / "requests.json", _request_manifest())
    rerun_trace = json.loads(TRACE.read_text(encoding="utf-8"))
    rerun_trace["source"]["episode_id"] = "rerun_episode_001"
    rerun_trace_path = _write_json(tmp_path / "rerun_trace.json", rerun_trace)
    mapping = _load_test_mapping(
        tmp_path / "mapping.json",
        [
            _mapping_row(
                episode_id="rerun_episode_001",
                release_episode_id="fixture_episode_001",
                trace_artifact_uri=str(rerun_trace_path),
            )
        ],
        request_sha256=request_manifest.content_sha256,
    )
    result = resolve_episode_requests(request_manifest, mapping)
    assert result["summary"]["n_resolved"] == 1
    assert result["rows"][0]["episode_id"] == "rerun_episode_001"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": "wrong", "tuples": [], "n_tuples": 0},
        {"schema_version": "issue_5446_trace_reexport_list.v1", "tuples": [], "n_tuples": 0},
        {
            "schema_version": "issue_5446_trace_reexport_list.v1",
            "tuples": [_request_manifest()["tuples"][0]],
            "n_tuples": 2,
        },
        {
            "schema_version": "issue_5446_trace_reexport_list.v1",
            "tuples": ["not-an-object"],
            "n_tuples": 1,
        },
        {
            "schema_version": "issue_5446_trace_reexport_list.v1",
            "tuples": [{"scenario_id": "s", "planner": "p"}],
            "n_tuples": 1,
        },
        {
            "schema_version": "issue_5446_trace_reexport_list.v1",
            "tuples": [{"scenario_id": "s", "planner": "p", "seed": 1, "expected_outcome": "bad"}],
            "n_tuples": 1,
        },
    ],
)
def test_request_loader_rejects_malformed_contract(tmp_path: Path, payload: object) -> None:
    """Malformed request inputs fail before any trace lookup occurs."""
    with pytest.raises(CandidateTraceResolutionError):
        load_episode_requests(_write_json(tmp_path / "requests.json", payload))


def test_request_loader_rejects_unreadable_json(tmp_path: Path) -> None:
    """An unreadable request path is a machine-readable resolver failure."""
    with pytest.raises(CandidateTraceResolutionError, match="unreadable"):
        load_episode_requests(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(CandidateTraceResolutionError, match="unreadable"):
        load_episode_requests(invalid)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"rows": []},
        {"rows": ["not-an-object"]},
        {"rows": [{"episode_id": "only-id"}]},
    ],
)
def test_episode_mapping_loader_rejects_malformed_contract(tmp_path: Path, payload: object) -> None:
    """A rerun mapping cannot omit identity or silently contain empty rows."""
    with pytest.raises(CandidateTraceResolutionError):
        load_episode_mapping(_write_json(tmp_path / "mapping.json", payload))


def test_episode_mapping_loader_rejects_duplicate_identity(tmp_path: Path) -> None:
    """Episode and tuple identity are unique joins, including release aliases."""
    duplicate_episode = [_mapping_row(), _mapping_row(seed=112)]
    with pytest.raises(CandidateTraceResolutionError, match="duplicate mapped episode_id"):
        load_episode_mapping(
            _write_json(
                tmp_path / "episode-duplicate.json",
                _mapping_receipt(
                    duplicate_episode,
                    request_sha256=ISSUE_5756_PINNED_PROVENANCE["request_manifest_sha256"],
                ),
            ),
            expected_count=2,
        )

    duplicate_tuple = [_mapping_row(), _mapping_row(episode_id="other")]
    with pytest.raises(CandidateTraceResolutionError, match="duplicate mapped episode tuple"):
        load_episode_mapping(
            _write_json(
                tmp_path / "tuple-duplicate.json",
                _mapping_receipt(
                    duplicate_tuple,
                    request_sha256=ISSUE_5756_PINNED_PROVENANCE["request_manifest_sha256"],
                ),
            ),
            expected_count=2,
        )

    duplicate_release = [
        _mapping_row(episode_id="rerun-1", release_episode_id="release-1"),
        _mapping_row(seed=112, episode_id="rerun-2", release_episode_id="release-1"),
    ]
    with pytest.raises(CandidateTraceResolutionError, match="duplicate mapped release"):
        load_episode_mapping(
            _write_json(
                tmp_path / "release-duplicate.json",
                _mapping_receipt(
                    duplicate_release,
                    request_sha256=ISSUE_5756_PINNED_PROVENANCE["request_manifest_sha256"],
                ),
            ),
            expected_count=2,
        )


def test_episode_mapping_loader_rejects_unversioned_direct_mapping(tmp_path: Path) -> None:
    """Direct episode-id dictionaries cannot bypass the versioned receipt contract."""
    payload = {"fixture_episode_001": _mapping_row()}
    with pytest.raises(CandidateTraceResolutionError, match="schema_version"):
        load_episode_mapping(_write_json(tmp_path / "mapping.json", payload))


@pytest.mark.parametrize(
    "request_manifest",
    [{}, {"tuples": ["bad"]}, {"tuples": [{"scenario_id": "only"}]}],
)
def test_request_resolution_rejects_malformed_runtime_payload(
    request_manifest: dict[str, object],
) -> None:
    """The public resolver also guards callers that bypass the file loader."""
    with pytest.raises(CandidateTraceResolutionError):
        resolve_episode_requests(request_manifest, {})


def test_request_resolution_rejects_invalid_outcome_and_trace_source(tmp_path: Path) -> None:
    """Outcome and embedded source disagreement never becomes a rendered row."""
    request = _load_test_requests(tmp_path / "requests.json", _request_manifest())
    with pytest.raises(CandidateTraceResolutionError, match="canonical rerun outcome"):
        _load_test_mapping(
            tmp_path / "invalid-outcome.json",
            [_mapping_row(rerun_outcome="other")],
            request_sha256=request.content_sha256,
        )

    mismatched_trace = json.loads(TRACE.read_text(encoding="utf-8"))
    mismatched_trace["source"]["scenario_id"] = "classic_doorway_medium"
    mismatched_path = _write_json(tmp_path / "mismatched-trace.json", mismatched_trace)
    mismatch_mapping = _load_test_mapping(
        tmp_path / "mismatch-mapping.json",
        [_mapping_row(trace_artifact_uri=str(mismatched_path))],
        request_sha256=request.content_sha256,
    )
    mismatch_result = resolve_episode_requests(request, mismatch_mapping)
    assert mismatch_result["rows"][0]["reason_code"] == "trace_source_mismatch:scenario_id"


def test_outcome_normalization_accepts_release_boolean_shapes() -> None:
    """Release rows may expose typed events as booleans or canonical strings."""
    assert resolution_module._normalize_outcome({"route_complete": True}) == "route_complete"
    assert resolution_module._normalize_outcome("goal_reached") == "route_complete"
    assert resolution_module._mapping_outcome({"collision_event": True}) == "collision_event"
    assert resolution_module._mapping_outcome({"outcome": "timeout_event"}) == "timeout_event"


def test_request_loader_requires_exact_pinned_manifest(tmp_path: Path) -> None:
    """The production loader must reject a valid-looking subset of the pinned 90 requests."""
    with pytest.raises(CandidateTraceResolutionError, match="90|SHA-256|sha256"):
        load_episode_requests(_write_json(tmp_path / "requests.json", _request_manifest()))


def test_mapping_requires_versioned_pinned_receipt_and_trace_digest(tmp_path: Path) -> None:
    """An unversioned identity/outcome/URI row is not sufficient provenance."""
    with pytest.raises(CandidateTraceResolutionError, match="schema|provenance|digest|SHA-256"):
        load_episode_mapping(_write_json(tmp_path / "mapping.json", {"rows": [_mapping_row()]}))


def test_mapping_rejects_wrong_pin_and_missing_expected_release_outcome(tmp_path: Path) -> None:
    """Receipt pins and per-row release outcomes are mandatory, not advisory metadata."""
    request_sha256 = ISSUE_5756_PINNED_PROVENANCE["request_manifest_sha256"]
    wrong_pin = _mapping_receipt(
        [_mapping_row()],
        request_sha256=request_sha256,
        provenance_overrides={"execution_commit": "f" * 40},
    )
    with pytest.raises(CandidateTraceResolutionError, match="execution_commit"):
        load_episode_mapping(
            _write_json(tmp_path / "wrong-pin.json", wrong_pin),
            expected_count=1,
        )

    wrong_bundle = _mapping_receipt(
        [_mapping_row()],
        request_sha256=request_sha256,
        provenance_overrides={"release_bundle_sha256": "0" * 64},
    )
    with pytest.raises(CandidateTraceResolutionError, match="release_bundle_sha256"):
        load_episode_mapping(
            _write_json(tmp_path / "wrong-bundle.json", wrong_bundle),
            expected_count=1,
        )

    missing_outcome_row = _mapping_row()
    del missing_outcome_row["expected_release_outcome"]
    with pytest.raises(CandidateTraceResolutionError, match="expected release outcome"):
        load_episode_mapping(
            _write_json(
                tmp_path / "missing-outcome.json",
                _mapping_receipt(
                    [missing_outcome_row],
                    request_sha256=request_sha256,
                ),
            ),
            expected_count=1,
        )


def test_resolution_rejects_trace_bytes_that_do_not_match_receipt_digest(
    tmp_path: Path,
) -> None:
    """A schema-valid trace is unresolved when its bytes differ from the receipt digest."""
    request = _load_test_requests(tmp_path / "requests.json", _request_manifest())
    mapping = _load_test_mapping(
        tmp_path / "mapping.json",
        [_mapping_row(trace_sha256="f" * 64)],
        request_sha256=request.content_sha256,
    )
    result = resolve_episode_requests(request, mapping)
    row = result["rows"][0]
    assert row["resolution_status"] == "provenance-incomplete"
    assert row["reason_code"].startswith("trace_sha256_mismatch")
    assert row["exact_repeat_determinism"] is None


def test_trace_search_rejects_ambiguous_matches(tmp_path: Path) -> None:
    """Legacy candidate lookup also fails closed instead of choosing lexicographically."""
    roots = [tmp_path / "one", tmp_path / "two"]
    for index, root in enumerate(roots):
        root.mkdir()
        _write_json(
            root / f"trace-fixture_episode_001-{index}.json",
            json.loads(TRACE.read_text(encoding="utf-8")),
        )
    with pytest.raises(CandidateTraceResolutionError, match="ambiguous trace artifact search"):
        resolution_module.resolve_trace_artifact(
            {"episode_id": "fixture_episode_001"},
            trace_search_roots=roots,
        )


def test_unreadable_trace_becomes_schema_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trace read failure is a structured fail-closed row, not an uncaught exception."""
    original_read_bytes = Path.read_bytes

    def _read_bytes(path: Path) -> bytes:
        if path == TRACE:
            raise OSError("fixture read blocked")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _read_bytes)
    result = resolution_module._validate_trace_file(TRACE)
    assert result["resolution_status"] == "schema-mismatch"
    assert result["reason_code"].startswith("trace_read_failed")


def test_renderer_rejects_noncanonical_outcome() -> None:
    """Figure annotations accept only the four canonical release outcome labels."""
    with pytest.raises(TraceSchemaError, match="outcome"):
        load_episode_from_trace_export(TRACE, outcome="goal_reached")


def test_renderer_rejects_boolean_vectors_and_missing_pedestrian_ids(tmp_path: Path) -> None:
    """Boolean coordinates and missing pedestrian ids fail with typed schema errors."""
    assert not trace_figure_module._is_xy_vector([True, False])
    payload = json.loads(TRACE.read_text(encoding="utf-8"))
    del payload["frames"][0]["pedestrians"][0]["id"]
    missing_id = _write_json(tmp_path / "missing-id.json", payload)
    with pytest.raises(TraceSchemaError, match="missing.*id"):
        load_episode_from_trace_export(missing_id, outcome="success")


def test_episode_mode_rejects_candidate_only_campaign_store_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Episode mode must reject flags it cannot honor instead of silently ignoring them."""
    request_path = _write_json(tmp_path / "requests.json", _request_manifest())
    mapping_path = _write_json(tmp_path / "mapping.json", {"rows": [_mapping_row()]})
    exit_code = resolver_cli.main(
        [
            "--episode-requests",
            str(request_path),
            "--episode-mapping",
            str(mapping_path),
            "--campaign-store",
            str(tmp_path / "ignored-store"),
        ]
    )
    assert exit_code == 2
    assert "--campaign-store" in capsys.readouterr().err


def test_render_stops_when_four_exemplars_resolve_but_one_other_request_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four renderable exemplars cannot bypass the complete 90/90 acceptance gate."""
    exemplar_specs = [
        ("classic_doorway_medium", "ppo", 113, "success"),
        ("classic_doorway_medium", "ppo", 114, "collision_event"),
        ("classic_realworld_double_bottleneck_high", "goal", 118, "success"),
        ("classic_realworld_double_bottleneck_high", "ppo", 118, "collision_event"),
    ]
    rows: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for index, (scenario_id, planner, seed, outcome) in enumerate(exemplar_specs):
        episode_id = f"episode-{index}"
        rows.append(
            {
                "scenario_id": scenario_id,
                "planner_id": planner,
                "seed": seed,
                "episode_id": episode_id,
                "trace_artifact_uri": str(TRACE),
                "resolution_status": "resolved",
                "reason_code": "ok",
            }
        )
        mapping[episode_id] = {"outcome": outcome}
    rows.append(
        {
            "scenario_id": "classic_doorway_medium",
            "planner_id": "ppo",
            "seed": 115,
            "episode_id": None,
            "trace_artifact_uri": None,
            "resolution_status": "trace-missing",
            "reason_code": "trace_artifact_not_found",
        }
    )
    incomplete = {
        "summary": {
            "n_candidates": 90,
            "n_resolved": 89,
            "n_trace_missing": 1,
            "n_schema_mismatch": 0,
            "n_provenance_incomplete": 0,
        },
        "rows": rows,
    }
    monkeypatch.setattr(render_cli, "load_episode_requests", lambda _path: ({}, []))
    monkeypatch.setattr(render_cli, "load_episode_mapping", lambda _path: mapping)
    monkeypatch.setattr(
        render_cli, "resolve_episode_requests", lambda *_args, **_kwargs: incomplete
    )
    monkeypatch.setattr(
        render_cli,
        "validate_candidate_trace_resolution",
        lambda _manifest: {"ok": True, "errors": []},
    )
    rendered: list[object] = []
    monkeypatch.setattr(render_cli, "_render_pair", lambda *_args, **_kwargs: rendered.append(1))

    exit_code = render_cli.main(
        [
            "--episode-requests",
            str(tmp_path / "requests.json"),
            "--episode-mapping",
            str(tmp_path / "mapping.json"),
            "--out-dir",
            str(tmp_path / "figures"),
        ]
    )

    assert exit_code == 2
    assert rendered == []


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
