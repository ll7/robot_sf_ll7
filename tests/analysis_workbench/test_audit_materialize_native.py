"""A real canonical episode can supply retained state for a missing video."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.analysis_workbench.audit_materialize import (
    DERIVED_RENDER,
    FIDELITY_UNVERIFIABLE,
    materialize_episode,
)
from robot_sf.benchmark.analysis_trace import trace_artifact_sha256
from robot_sf.benchmark.runner import run_episode

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def native_episode() -> dict:
    """Retain one true canonical runner record for positive and negative checks."""

    original = run_episode(
        {"id": "ba05_native_probe", "density": "low", "flow": "uni", "obstacle": "open"},
        42,
        horizon=8,
        dt=0.1,
        robot_start=(-4.0, 0.0),
        robot_goal=(4.0, 0.0),
        record_forces=False,
        algo="simple_policy",
        telemetry={"analysis_trace": "all"},
        provenance={"test": "ba05-native-materialization"},
    )
    assert original["algorithm_metadata"]["planner_kinematics"]["execution_mode"] == "native"
    return original


def _selected_episode(original: dict) -> dict:
    return {
        "episode_id": original["episode_id"],
        "seed": original["seed"],
        "scenario_id": "ba05_native_probe",
        "algo": "simple_policy",
        "source_commit": original["git_hash"],
        "config_identity": original["algorithm_metadata"]["analysis_trace"]["config_digest"],
        "retained_trace": original["algorithm_metadata"]["analysis_trace"],
    }


def test_native_missing_video_renders_retained_trace_without_simulation(
    tmp_path: Path, native_episode: dict
) -> None:
    """Keep a newly rendered view distinct from the native source execution."""

    result = materialize_episode(
        _selected_episode(native_episode),
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
        output_directory="native-retained-render",
    )

    assert result.status == "complete", result.diagnostics
    assert result.materialization_kind == DERIVED_RENDER
    assert result.fidelity == FIDELITY_UNVERIFIABLE
    assert result.simulation_executed is False
    assert result.provenance["simulation_advanced"] is False
    assert "native_retained_trace_projected" in result.diagnostics
    trace_digest = native_episode["algorithm_metadata"]["analysis_trace"]["artifact_sha256"]
    assert result.source_digest == trace_digest
    assert result.provenance["native_trace_digest"] == trace_digest
    assert result.provenance["retained_state_digest"] != trace_digest
    manifest = json.loads(
        (
            tmp_path / "derived" / "native-retained-render" / "audit-materialization.v1.json"
        ).read_text()
    )
    assert manifest["source_digest"] == trace_digest
    assert (tmp_path / "derived" / "native-retained-render" / "trajectory.png").is_file()


@pytest.mark.parametrize(
    "mutation",
    ["digest", "scenario", "config", "planner", "commit", "episode", "seed", "execution"],
)
def test_native_trace_identity_or_digest_change_fails_closed(
    tmp_path: Path, native_episode: dict, mutation: str
) -> None:
    """A stale selected identity cannot produce a derived artifact."""

    selected = copy.deepcopy(_selected_episode(native_episode))
    if mutation == "digest":
        selected["retained_trace"]["steps"][0]["robot"]["position"][0] += 1.0
    elif mutation == "scenario":
        selected["scenario_id"] = "different-scenario"
    elif mutation == "config":
        selected["config_identity"] = "0" * 64
    elif mutation == "planner":
        selected["algo"] = "different_planner"
    elif mutation == "episode":
        selected["episode_id"] = "ba05_native_probe--43"
    elif mutation == "seed":
        selected["seed"] = 43
    elif mutation == "execution":
        selected["execution_id"] = "different-execution"
    else:
        selected["source_commit"] = "0" * 40

    result = materialize_episode(
        selected, source_root=tmp_path / "source", output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.reason == "native_retained_trace_unavailable"
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("missing", ["episode_id", "seed"])
def test_older_native_trace_without_episode_binding_fails_closed(
    tmp_path: Path, native_episode: dict, missing: str
) -> None:
    """Digest-valid older envelopes cannot claim an unrelated selected run."""

    selected = copy.deepcopy(_selected_episode(native_episode))
    trace = selected["retained_trace"]
    trace.pop(missing)
    trace["artifact_sha256"] = trace_artifact_sha256(trace)

    result = materialize_episode(
        selected, source_root=tmp_path / "source", output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.reason == "native_retained_trace_unavailable"
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize(
    "claimed_identity",
    [
        "campaign_id",
        "source_digest",
        "checkpoint_digest",
        "environment_digest",
        "initial_state_digest",
    ],
)
def test_native_trace_does_not_ignore_claimed_selected_identity(
    tmp_path: Path, native_episode: dict, claimed_identity: str
) -> None:
    """Missing optional trace identities cannot be silently treated as matches."""

    selected = _selected_episode(native_episode)
    selected[claimed_identity] = "different-identity"

    result = materialize_episode(
        selected, source_root=tmp_path / "source", output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.reason == "native_retained_trace_unavailable"
    assert not (tmp_path / "derived").exists()


def test_path_retained_native_trace_has_file_and_trace_digests(
    tmp_path: Path, native_episode: dict
) -> None:
    """A path-backed native trace remains a derived view, not original media."""

    selected = _selected_episode(native_episode)
    trace = selected.pop("retained_trace")
    source_root = tmp_path / "source"
    source_root.mkdir()
    content = json.dumps(trace, sort_keys=True).encode()
    (source_root / "trace.json").write_bytes(content)
    file_digest = hashlib.sha256(content).hexdigest()
    selected["retained_trace_source"] = {"uri": "trace.json", "sha256": file_digest}

    result = materialize_episode(
        selected, source_root=source_root, output_root=tmp_path / "derived"
    )

    assert result.status == "complete", result.diagnostics
    assert result.materialization_kind == DERIVED_RENDER
    assert result.source_digest == trace["artifact_sha256"]
    assert result.provenance["native_trace_digest"] == trace["artifact_sha256"]
    assert result.provenance["retained_source_file_digest"] == file_digest


def test_native_trace_matching_execution_id_is_bound(tmp_path: Path, native_episode: dict) -> None:
    """A claimed execution ID must be present in the digest-bound trace."""

    selected = copy.deepcopy(_selected_episode(native_episode))
    selected["execution_id"] = "probe-execution"
    trace = selected["retained_trace"]
    trace["execution_id"] = "probe-execution"
    trace["artifact_sha256"] = trace_artifact_sha256(trace)

    result = materialize_episode(
        selected, source_root=tmp_path / "source", output_root=tmp_path / "derived"
    )

    assert result.status == "complete", result.diagnostics
    assert result.materialization_kind == DERIVED_RENDER


def test_path_retained_native_trace_wrong_selected_seed_fails_closed(
    tmp_path: Path, native_episode: dict
) -> None:
    """A verified file digest is not sufficient to change the selected episode."""

    selected = _selected_episode(native_episode)
    trace = selected.pop("retained_trace")
    selected["episode_id"] = "ba05_native_probe--43"
    selected["seed"] = 43
    source_root = tmp_path / "source"
    source_root.mkdir()
    content = json.dumps(trace).encode()
    (source_root / "trace.json").write_bytes(content)
    selected["retained_trace_source"] = {
        "uri": "trace.json",
        "sha256": hashlib.sha256(content).hexdigest(),
    }

    result = materialize_episode(
        selected, source_root=source_root, output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.materialization_kind != DERIVED_RENDER
    assert not (tmp_path / "derived").exists()


def test_misdeclared_native_json_cannot_be_historical_original(
    tmp_path: Path, native_episode: dict
) -> None:
    """A digest-verified analysis trace is never an original recording."""

    selected = _selected_episode(native_episode)
    selected.pop("retained_trace")
    source_root = tmp_path / "source"
    source_root.mkdir()
    content = json.dumps(native_episode["algorithm_metadata"]["analysis_trace"]).encode()
    (source_root / "trace.json").write_bytes(content)
    selected["recording"] = {
        "uri": "trace.json",
        "sha256": hashlib.sha256(content).hexdigest(),
        "scenario_id": selected["scenario_id"],
        "source_commit": selected["source_commit"],
        "config_identity": selected["config_identity"],
    }

    result = materialize_episode(
        selected, source_root=source_root, output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.materialization_kind != "historical_original"
    assert "original_recording_format_unsupported" in result.diagnostics


@pytest.mark.parametrize(
    ("algo", "planner_id"),
    [("simple_policy", "foreign"), ("foreign", "simple_policy"), (None, "foreign")],
)
def test_conflicting_native_planner_aliases_fail_closed(
    tmp_path: Path, native_episode: dict, algo: str | None, planner_id: str
) -> None:
    """Every selected planner claim must match the retained native trace."""

    selected = _selected_episode(native_episode)
    selected["algo"] = algo
    selected["planner_id"] = planner_id

    result = materialize_episode(
        selected, source_root=tmp_path / "source", output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert "native_retained_trace_unavailable" == result.reason
    assert not (tmp_path / "derived").exists()


def test_misdeclared_runner_record_json_cannot_be_historical_original(
    tmp_path: Path, native_episode: dict
) -> None:
    """A runner JSON wrapper around a native trace is not historical video."""

    selected = _selected_episode(native_episode)
    selected.pop("retained_trace")
    source_root = tmp_path / "source"
    source_root.mkdir()
    content = json.dumps(native_episode).encode()
    (source_root / "record.json").write_bytes(content)
    selected["recording"] = {
        "uri": "record.json",
        "format": "video/mp4",
        "sha256": hashlib.sha256(content).hexdigest(),
        "scenario_id": selected["scenario_id"],
        "source_commit": selected["source_commit"],
        "config_identity": selected["config_identity"],
    }

    result = materialize_episode(
        selected, source_root=source_root, output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.materialization_kind != "historical_original"
    assert "original_recording_media_type_mismatch" in result.diagnostics


@pytest.mark.parametrize(
    "shape",
    [
        "duplicate_direct",
        "duplicate_wrapper",
        "deep_wrapper",
        "bom_direct",
        "bom_wrapper",
        "truncated_utf16le",
        "truncated_utf16be",
        "truncated_utf32le",
        "truncated_utf32be",
        "truncated_utf16le_nobom",
        "truncated_utf16be_nobom",
        "truncated_utf32le_nobom",
        "truncated_utf32be_nobom",
    ],
)
def test_uninspectable_native_json_cannot_be_historical_original(
    tmp_path: Path, native_episode: dict, shape: str
) -> None:
    """Duplicate or deep JSON cannot hide a native trace behind a media label."""

    selected = _selected_episode(native_episode)
    selected.pop("retained_trace")
    source_root = tmp_path / "source"
    source_root.mkdir()
    if shape == "duplicate_direct":
        content = json.dumps(native_episode["algorithm_metadata"]["analysis_trace"]).encode()
        content = content[:-1] + b',"schema_version":"opaque.v1"}'
    elif shape == "duplicate_wrapper":
        content = json.dumps(native_episode).encode()
        content = content[:-1] + b',"algorithm_metadata":{}}'
    elif shape == "bom_direct":
        content = (
            b"\xef\xbb\xbf"
            + json.dumps(native_episode["algorithm_metadata"]["analysis_trace"]).encode()
        )
    elif shape == "bom_wrapper":
        content = b"\xef\xbb\xbf" + json.dumps(native_episode).encode()
    elif shape.startswith("truncated_"):
        encoding = shape.removeprefix("truncated_")
        with_bom = not encoding.endswith("_nobom")
        encoding = encoding.removesuffix("_nobom")
        marker = {
            "utf16le": b"\xff\xfe",
            "utf16be": b"\xfe\xff",
            "utf32le": b"\xff\xfe\x00\x00",
            "utf32be": b"\x00\x00\xfe\xff",
        }[encoding]
        codec = {
            "utf16le": "utf-16-le",
            "utf16be": "utf-16-be",
            "utf32le": "utf-32-le",
            "utf32be": "utf-32-be",
        }[encoding]
        content = (marker if with_bom else b"") + json.dumps(
            native_episode["algorithm_metadata"]["analysis_trace"]
        )[:-1].encode(codec)
    else:
        nested = b"[" * 10_000 + b"0" + b"]" * 10_000
        metadata = json.dumps(native_episode["algorithm_metadata"]).encode()
        content = b'{"padding":' + nested + b',"algorithm_metadata":' + metadata + b"}"
    (source_root / "record.json").write_bytes(content)
    selected["recording"] = {
        "uri": "record.json",
        "format": "video/mp4",
        "sha256": hashlib.sha256(content).hexdigest(),
        "scenario_id": selected["scenario_id"],
        "source_commit": selected["source_commit"],
        "config_identity": selected["config_identity"],
    }

    result = materialize_episode(
        selected, source_root=source_root, output_root=tmp_path / "derived"
    )

    assert result.status == "unavailable"
    assert result.materialization_kind != "historical_original"
    assert "original_recording_media_type_mismatch" in result.diagnostics
