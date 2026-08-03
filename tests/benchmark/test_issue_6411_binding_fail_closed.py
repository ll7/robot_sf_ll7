"""Fail-closed branch coverage for issue #6411 real-arm provenance binding.

The merged #6411 contract tests prove the happy path and a few drift failures
through ``bind_real_reexport_arms``. The readiness changed-line coverage gate
additionally requires the individual manifest/row identity guards to be proven
directly, so this module exercises each fail-closed branch of the binding
helpers with synthetic manifests and rows. No real re-export artifact is read
or written, and no benchmark claim is made.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from robot_sf.benchmark.trace_reexport_packaging import (
    EXECUTION_COMMIT,
    REAL_REEXPORT_ARMS,
    RealReexportArm,
    RealReexportBindingError,
    _compact_source_evidence,
    _discover_real_arm_inputs,
    _manifest_config_path,
    _manifest_job_id,
    _manifest_planner,
    _manifest_seed_list,
    _read_real_rows_with_raw_bytes,
    _real_outcome_index,
    _required_manifest_text,
    _source_job_id,
    _verify_real_algorithm_config,
    _verify_real_arm_manifest,
    _verify_real_rerun_row,
    _verify_real_row_config_hash,
)
from robot_sf.benchmark.utils import _config_hash

_ARM = REAL_REEXPORT_ARMS[0]
_KEY = (_ARM.planner, _ARM.scenario_id, 111)
_SCENARIO_MATRIX = "configs/scenarios/classic_interactions_francis2023.yaml"


def _valid_manifest(arm: RealReexportArm) -> dict[str, Any]:
    """Return a campaign manifest that satisfies every identity guard."""

    return {
        "campaign_id": f"campaign-{arm.key}",
        "job_id": arm.job_id,
        "name": arm.config_name,
        "config_path": arm.config_path,
        "config_hash": "campaign-config-hash",
        "scenario_matrix": _SCENARIO_MATRIX,
        "scenario_candidates": [arm.scenario_id],
        "seed_policy": {"resolved_seeds": list(arm.seeds)},
        "git": {"commit": EXECUTION_COMMIT},
        "planners": [{"key": arm.planner}],
    }


def _outcome_fields(success: bool = True) -> dict[str, bool]:
    """Return the four canonical release outcome booleans."""

    return {
        "success": success,
        "route_complete": success,
        "collision_event": not success,
        "timeout_event": False,
    }


def _flat_outcome_row(
    *, planner: str = "ppo", scenario_id: str = "classic_doorway_medium", seed: int = 111
) -> dict[str, Any]:
    """Return one request-list outcome row carrying flat outcome fields."""

    row: dict[str, Any] = {
        "planner": planner,
        "scenario_id": scenario_id,
        "seed": seed,
    }
    row.update(_outcome_fields())
    return row


def _valid_row() -> dict[str, Any]:
    """Return a rerun row that satisfies every row-level identity guard."""

    scenario_params = {"algo": _ARM.planner, "id": _ARM.scenario_id}
    return {
        "episode_id": "rerun-row",
        "git_hash": EXECUTION_COMMIT,
        "config_hash": _config_hash(scenario_params),
        "scenario_params": scenario_params,
        "metrics": {"success": True},
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
        "algorithm_metadata": {
            "planner_kinematics": {"robot_kinematics": "differential_drive"},
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "steps": [{"step": 0}],
            },
        },
    }


def test_required_manifest_text_rejects_non_text_values() -> None:
    """Required identity text must be a non-empty string or integer."""

    assert _required_manifest_text({"value": 42}, "identity", ("value",)) == "42"
    for bad in (True, None, "   ", [1]):
        with pytest.raises(RealReexportBindingError, match="lacks identity"):
            _required_manifest_text({"value": bad}, "identity", ("value",))


@pytest.mark.parametrize(
    ("manifest", "match"),
    [
        ({}, "lacks a list of resolved seeds"),
        ({"seeds": [1, True]}, "lacks a list of resolved seeds"),
        ({"seeds": ["not-an-int"]}, "resolved seeds are not integers"),
        ({"seeds": [111, 111]}, "resolved seeds contain duplicates"),
    ],
)
def test_manifest_seed_list_fails_closed(manifest: dict[str, Any], match: str) -> None:
    """Seed lists must be unique integer lists resolved from the manifest."""

    with pytest.raises(RealReexportBindingError, match=match):
        _manifest_seed_list(manifest)


def test_manifest_seed_list_preserves_manifest_order() -> None:
    """Alternate seed-policy paths remain readable in manifest order."""

    assert _manifest_seed_list({"seed_policy": {"seeds": [3, 1, 2]}}) == [3, 1, 2]


@pytest.mark.parametrize(
    ("manifest", "match"),
    [
        ({"planners": []}, "exactly one planner"),
        ({"planners": [{"key": "a"}, {"key": "b"}]}, "exactly one planner"),
        ({"planners": ["not-a-mapping"]}, "exactly one planner"),
        ({"planners": [{}]}, "lacks a planner key"),
        ({"planner": "   "}, "lacks a planner key"),
        ({}, "lacks a planner key"),
    ],
)
def test_manifest_planner_fails_closed(manifest: dict[str, Any], match: str) -> None:
    """Exactly one named planner must be declared by the campaign manifest."""

    with pytest.raises(RealReexportBindingError, match=match):
        _manifest_planner(manifest)


def test_manifest_planner_accepts_alternate_fields() -> None:
    """The planner key may come from the entry planner field or algorithm."""

    assert _manifest_planner({"planners": [{"planner": "goal"}]}) == "goal"
    assert _manifest_planner({"algorithm": "  goal "}) == "goal"


def test_manifest_config_path_rejects_invalid_shell_commands() -> None:
    """A recorded invocation that cannot be parsed must fail closed."""

    manifest = {"invoked_command": "python -m runner --config 'unterminated"}
    with pytest.raises(RealReexportBindingError, match="not a valid shell command"):
        _manifest_config_path(manifest)


def test_manifest_config_path_reads_nested_and_invocation_forms() -> None:
    """Config paths resolve from nested fields and recorded --config tokens."""

    assert _manifest_config_path({"config": {"path": " x.yaml "}}) == "x.yaml"
    assert _manifest_config_path({"command": "python run.py --config configs/a.yaml"}) == (
        "configs/a.yaml"
    )
    assert _manifest_config_path({"command": "python run.py --output out"}) is None
    assert _manifest_config_path({}) is None


def test_source_job_id_requires_a_unique_recovery() -> None:
    """Job ids are recovered only when the source identity is unambiguous."""

    assert _source_job_id({}, Path("/cluster/job-777")) == "777"
    assert _source_job_id({"results_root": "/cluster/job-2"}, Path("/cluster/job-1")) is None
    assert _source_job_id({}, Path("/cluster/run")) is None
    assert _source_job_id({"results_root": "/external/job-555"}, Path("/run")) == "555"


def test_compact_source_evidence_rejects_duplicate_preflight(tmp_path: Path) -> None:
    """More than one preflight record makes source evidence ambiguous."""

    manifest_path = tmp_path / "campaign_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    for subdir in ("a", "b"):
        record = tmp_path / subdir / "preflight" / "validate_config.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("{}", encoding="utf-8")

    with pytest.raises(RealReexportBindingError, match="at most one preflight"):
        _compact_source_evidence(tmp_path, manifest_path)


def test_compact_source_evidence_hashes_available_records(tmp_path: Path) -> None:
    """Manifest, run summary, and single preflight records are digest-bound."""

    manifest_path = tmp_path / "campaign_manifest.json"
    manifest_path.write_text('{"campaign": true}', encoding="utf-8")
    summary = tmp_path / "run_summary.yaml"
    summary.write_text("job: 13483\n", encoding="utf-8")
    preflight = tmp_path / "preflight" / "validate_config.json"
    preflight.parent.mkdir(parents=True, exist_ok=True)
    preflight.write_text('{"ok": true}', encoding="utf-8")

    evidence = _compact_source_evidence(tmp_path, manifest_path)

    assert [item["kind"] for item in evidence] == [
        "campaign_manifest",
        "run_summary.yaml",
        "validate_config.json",
    ]
    for item in evidence:
        digest = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        assert item["sha256"] == digest


def test_manifest_job_id_fails_closed(tmp_path: Path) -> None:
    """Declared job ids must be recoverable text, never booleans or blanks."""

    with pytest.raises(RealReexportBindingError, match="lacks job_id"):
        _manifest_job_id({"job_id": True}, tmp_path)
    with pytest.raises(RealReexportBindingError, match="lacks job_id"):
        _manifest_job_id({"job_id": None}, tmp_path)
    assert _manifest_job_id({"job_id": None, "results_root": "/x/job-555"}, tmp_path) == "555"
    assert _manifest_job_id({"job_id": 42}, tmp_path) == "42"
    assert _manifest_job_id({"job_id": " 13483 "}, tmp_path) == "13483"


def test_discover_real_arm_inputs_fails_closed(tmp_path: Path) -> None:
    """Exactly one manifest and one episode file must exist per arm root."""

    with pytest.raises(RealReexportBindingError, match="source root is unavailable"):
        _discover_real_arm_inputs(tmp_path / "missing", _ARM)

    root = tmp_path / "arm"
    root.mkdir()
    with pytest.raises(RealReexportBindingError, match="exactly one campaign_manifest"):
        _discover_real_arm_inputs(root, _ARM)

    for subdir in ("a", "b"):
        manifest = root / subdir / "campaign_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(RealReexportBindingError, match="exactly one campaign_manifest"):
        _discover_real_arm_inputs(root, _ARM)

    (root / "b" / "campaign_manifest.json").unlink()
    with pytest.raises(RealReexportBindingError, match="exactly one runs"):
        _discover_real_arm_inputs(root, _ARM)

    for run in ("run-1", "run-2"):
        episodes = root / "runs" / run / "episodes.jsonl"
        episodes.parent.mkdir(parents=True, exist_ok=True)
        episodes.write_text("", encoding="utf-8")
    with pytest.raises(RealReexportBindingError, match="exactly one runs"):
        _discover_real_arm_inputs(root, _ARM)


def test_discover_real_arm_inputs_returns_unique_paths(tmp_path: Path) -> None:
    """A well-formed arm root yields the manifest and episode paths."""

    root = tmp_path / "arm"
    manifest_path = root / "campaign_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"job_id": "13483"}', encoding="utf-8")
    episodes_path = root / "runs" / "ppo__differential_drive" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.write_text("", encoding="utf-8")

    found_manifest, found_episodes, manifest = _discover_real_arm_inputs(root, _ARM)

    assert found_manifest == manifest_path
    assert found_episodes == episodes_path
    assert manifest == {"job_id": "13483"}


def test_read_real_rows_with_raw_bytes_fails_closed() -> None:
    """JSONL parsing retains raw bytes and rejects malformed rows."""

    with pytest.raises(RealReexportBindingError, match="label:1: invalid JSON"):
        _read_real_rows_with_raw_bytes(b"{not-json\n", "label")
    with pytest.raises(RealReexportBindingError, match="label:1: expected a JSON object"):
        _read_real_rows_with_raw_bytes(b"[1, 2]\n", "label")
    with pytest.raises(RealReexportBindingError, match="label: contains no rows"):
        _read_real_rows_with_raw_bytes(b"\n   \n", "label")

    rows = _read_real_rows_with_raw_bytes(b'{"a": 1}\n\n{"b": 2}\n', "label")
    assert [(row, line_number) for _, (row, _, line_number) in enumerate(rows)] == [
        ({"a": 1}, 1),
        ({"b": 2}, 3),
    ]
    assert rows[0][1] == b'{"a": 1}\n'


_MUTATIONS: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
    ("wrong_commit", lambda m: m["git"].update({"commit": "deadbeef"}), "commit mismatch"),
    ("wrong_config_name", lambda m: m.update({"name": "other-config"}), "config name mismatch"),
    (
        "wrong_config_path",
        lambda m: m.update({"config_path": "other.yaml"}),
        "config path mismatch",
    ),
    (
        "wrong_scenarios",
        lambda m: m.update({"scenario_candidates": ["other_scenario"]}),
        "scenario candidate mismatch",
    ),
    (
        "wrong_seeds",
        lambda m: m.update({"seed_policy": {"resolved_seeds": [1]}}),
        "resolved seed set mismatch",
    ),
    (
        "wrong_planner",
        lambda m: m.update({"planners": [{"key": "other"}]}),
        "planner mismatch",
    ),
    (
        "wrong_scenario_matrix",
        lambda m: m.update({"scenario_matrix": "other.yaml"}),
        "scenario matrix mismatch",
    ),
    ("missing_config_hash", lambda m: m.pop("config_hash"), "lacks config_hash"),
    ("wrong_job", lambda m: m.update({"job_id": "99999"}), "job mismatch"),
)


@pytest.mark.parametrize(
    ("mutation_id", "mutate", "match"),
    _MUTATIONS,
    ids=[item[0] for item in _MUTATIONS],
)
def test_verify_real_arm_manifest_fails_closed(
    tmp_path: Path,
    mutation_id: str,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    """Every identity field must fail closed when the manifest drifts."""

    manifest = _valid_manifest(_ARM)
    mutate(manifest)
    with pytest.raises(RealReexportBindingError, match=match):
        _verify_real_arm_manifest(manifest, arm=_ARM, source_root=tmp_path)


def test_verify_real_arm_manifest_returns_normalized_identity(tmp_path: Path) -> None:
    """A valid manifest yields the normalized identity used by row checks."""

    identity = _verify_real_arm_manifest(_valid_manifest(_ARM), arm=_ARM, source_root=tmp_path)

    assert identity == {
        "job_id": _ARM.job_id,
        "execution_commit": EXECUTION_COMMIT,
        "config_name": _ARM.config_name,
        "config_hash": "campaign-config-hash",
        "campaign": f"campaign-{_ARM.key}",
    }


def test_real_outcome_index_rejects_invalid_shapes() -> None:
    """Outcome evidence must be keyed by exact tuples with full booleans."""

    with pytest.raises(RealReexportBindingError, match="tuple identities"):
        _real_outcome_index({"rows": {"not-a-tuple": _outcome_fields()}})
    with pytest.raises(RealReexportBindingError, match="lacks outcome fields"):
        _real_outcome_index({"rows": {_KEY: {"success": True}}})
    with pytest.raises(RealReexportBindingError, match="rows list"):
        _real_outcome_index({"rows": "not-a-list"})
    with pytest.raises(RealReexportBindingError, match="must be an object"):
        _real_outcome_index({"rows": ["not-an-object"]})
    duplicate = _flat_outcome_row()
    with pytest.raises(RealReexportBindingError, match="duplicate expected outcome tuple"):
        _real_outcome_index({"rows": [duplicate, dict(duplicate)]})
    with pytest.raises(RealReexportBindingError, match="lacks outcome fields"):
        _real_outcome_index({"rows": [{"planner": "ppo", "scenario_id": "s", "seed": 1}]})


def test_real_outcome_index_accepts_mapping_and_row_forms() -> None:
    """Both tuple-keyed mappings and request-list rows index canonically."""

    indexed = _real_outcome_index({"rows": {_KEY: _outcome_fields(success=False)}})
    assert indexed[_KEY] == _outcome_fields(success=False)

    row = _flat_outcome_row()
    indexed = _real_outcome_index({"rows": [row]})
    assert indexed[("ppo", "classic_doorway_medium", 111)] == _outcome_fields()

    metrics_row = {
        "planner": "ppo",
        "scenario_id": "classic_doorway_medium",
        "seed": 112,
        "metrics": {"success": True},
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
    }
    indexed = _real_outcome_index({"rows": [metrics_row]})
    assert indexed[("ppo", "classic_doorway_medium", 112)] == _outcome_fields()


def test_verify_real_row_config_hash_fails_closed() -> None:
    """Row hashes must bind the exact scenario parameters."""

    params = {"algo": _ARM.planner, "id": _ARM.scenario_id}
    with pytest.raises(RealReexportBindingError, match="lacks scenario config hash"):
        _verify_real_row_config_hash({}, key=_KEY, params=params)
    with pytest.raises(RealReexportBindingError, match="scenario/config hash mismatch"):
        _verify_real_row_config_hash({"config_hash": "wrong"}, key=_KEY, params=params)
    _verify_real_row_config_hash({"config_hash": _config_hash(params)}, key=_KEY, params=params)


def test_verify_real_algorithm_config_fails_closed() -> None:
    """Optional algorithm config provenance must hash-match when present."""

    _verify_real_algorithm_config({}, key=_KEY)
    _verify_real_algorithm_config({"config_hash": None}, key=_KEY)
    with pytest.raises(RealReexportBindingError, match="algorithm config hash is invalid"):
        _verify_real_algorithm_config({"config_hash": "   "}, key=_KEY)
    with pytest.raises(RealReexportBindingError, match="algorithm config hash mismatch"):
        _verify_real_algorithm_config({"config_hash": "wrong", "config": {"lr": 0.1}}, key=_KEY)
    config = {"lr": 0.1}
    _verify_real_algorithm_config({"config_hash": _config_hash(config), "config": config}, key=_KEY)


def test_verify_real_rerun_row_fails_closed() -> None:
    """Row-level identity, kinematics, and trace guards fail closed."""

    def _mutated(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        row = _valid_row()
        mutate(row)
        return row

    with pytest.raises(RealReexportBindingError, match="execution commit mismatch"):
        _verify_real_rerun_row(_mutated(lambda r: r.update({"git_hash": "deadbeef"})), key=_KEY)
    with pytest.raises(RealReexportBindingError, match="planner/config mismatch"):
        _verify_real_rerun_row(
            _mutated(lambda r: r["scenario_params"].update({"algo": "other"})), key=_KEY
        )
    with pytest.raises(RealReexportBindingError, match="lacks algorithm_metadata"):
        _verify_real_rerun_row(_mutated(lambda r: r.pop("algorithm_metadata")), key=_KEY)
    with pytest.raises(RealReexportBindingError, match="differential-drive"):
        _verify_real_rerun_row(
            _mutated(
                lambda r: r["algorithm_metadata"]["planner_kinematics"].update(
                    {"robot_kinematics": "unicycle"}
                )
            ),
            key=_KEY,
        )
    with pytest.raises(RealReexportBindingError, match="non-empty simulation trace"):
        _verify_real_rerun_row(
            _mutated(
                lambda r: r["algorithm_metadata"]["simulation_step_trace"].update({"steps": []})
            ),
            key=_KEY,
        )
    _verify_real_rerun_row(_valid_row(), key=_KEY)
