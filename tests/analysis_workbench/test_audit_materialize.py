"""Focused source-first tests for the BA-05 materialization leaf."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

import robot_sf.analysis_workbench.audit_materialize as materialize
from robot_sf.analysis_workbench.audit_contracts import EpisodeRef, record_to_dict
from robot_sf.analysis_workbench.audit_materialize import (
    DERIVED_RENDER,
    FIDELITY_UNAVAILABLE,
    FIDELITY_UNVERIFIABLE,
    FIDELITY_VERIFIED,
    HISTORICAL_ORIGINAL,
    UNAVAILABLE,
    build_materialization_cache_key,
    materialize_episode,
)
from robot_sf.analysis_workbench.review_contracts import ComponentResult, SourceRef
from robot_sf.benchmark.episode_replay_figure import FigureArtifact

TRACE_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "minimal_trace.json"
)
MP4_BYTES = base64.b64decode(
    (Path(__file__).resolve().parent / "fixtures" / "audit_tiny_mp4.b64").read_text(
        encoding="ascii"
    )
)


def _trace_payload() -> dict[str, Any]:
    """Load the retained canonical trace fixture."""
    return json.loads(TRACE_FIXTURE.read_text(encoding="utf-8"))


def test_verified_original_is_referenced_without_copy_or_render(tmp_path: Path) -> None:
    """A verified original wins and does not create derived output."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "episode.bin"
    original_bytes = MP4_BYTES
    source_path.write_bytes(original_bytes)
    digest = hashlib.sha256(original_bytes).hexdigest()
    output_root = tmp_path / "derived"
    episode = {
        "episode_id": "episode-001",
        "recording": {
            "uri": "episode.bin",
            "format": "video/mp4",
            "sha256": digest,
        },
    }

    result = materialize_episode(
        episode,
        source_root=source_root,
        output_root=output_root,
    )

    assert result.status == "complete"
    assert result.materialization_kind == HISTORICAL_ORIGINAL
    assert result.fidelity == FIDELITY_VERIFIED
    assert result.source_digest == digest
    assert result.simulation_executed is False
    assert result.output_directory is None
    assert result.artifacts[0]["uri"] == "episode.bin"
    assert result.provenance["renderer_invoked"] is False
    assert result.provenance["source_bytes_preserved"] is False
    assert result.provenance["source_bytes_verified"] is True
    assert result.provenance["source_reference_mutable"] is True
    assert source_path.read_bytes() == original_bytes
    assert not output_root.exists()


def test_original_receipt_uses_normalized_uri(tmp_path: Path) -> None:
    """Whitespace around an original URI is not retained in its receipt."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "episode.bin"
    source_bytes = MP4_BYTES
    source_path.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()

    result = materialize_episode(
        {
            "episode_id": "episode-normalized-uri",
            "recording": {"uri": " episode.bin ", "format": "video/mp4", "sha256": digest},
        },
        source_root=source_root,
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == HISTORICAL_ORIGINAL
    assert result.artifacts[0]["uri"] == "episode.bin"
    assert result.provenance["source_uri"] == "episode.bin"


def test_mp4_header_without_a_decodable_video_frame_is_not_original(tmp_path: Path) -> None:
    """A matching container prefix alone cannot certify historical video."""

    source_root = tmp_path / "source"
    source_root.mkdir()
    content = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2"
    (source_root / "episode.mp4").write_bytes(content)
    result = materialize_episode(
        {
            "episode_id": "spoofed-mp4",
            "recording": {
                "uri": "episode.mp4",
                "format": "video/mp4",
                "sha256": hashlib.sha256(content).hexdigest(),
            },
        },
        source_root=source_root,
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert "original_recording_media_type_mismatch" in result.diagnostics


def test_missing_media_decoder_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid MP4 is not admitted when the decoder cannot be launched."""

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "episode.mp4").write_bytes(MP4_BYTES)

    def unavailable_decoder() -> str:
        raise RuntimeError("decoder unavailable")

    monkeypatch.setattr(materialize.imageio_ffmpeg, "get_ffmpeg_exe", unavailable_decoder)
    result = materialize_episode(
        {
            "episode_id": "decoder-missing",
            "recording": {
                "uri": "episode.mp4",
                "format": "video/mp4",
                "sha256": hashlib.sha256(MP4_BYTES).hexdigest(),
            },
        },
        source_root=source_root,
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert "original_recording_media_decoder_unavailable" in result.diagnostics


def test_original_source_swap_after_initial_read_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source replaced after the first protected read is not admitted."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "episode.bin"
    source_path.write_bytes(b"original-source")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside-source")
    digest = hashlib.sha256(b"original-source").hexdigest()
    real_read = materialize._open_read_bounded
    swapped = False

    def read_then_swap(root: Path, uri: str) -> Any:
        nonlocal swapped
        result = real_read(root, uri)
        if not swapped:
            source_path.unlink()
            source_path.symlink_to(outside)
            swapped = True
        return result

    monkeypatch.setattr(materialize, "_open_read_bounded", read_then_swap)
    result = materialize_episode(
        {
            "episode_id": "episode-source-swap",
            "recording": {"uri": "episode.bin", "sha256": digest},
        },
        source_root=source_root,
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert result.reason == "retained_state_missing"
    assert "original_recording_source_changed" in result.diagnostics
    assert source_path.is_symlink()


def test_digest_mismatch_falls_back_to_retained_replay(tmp_path: Path) -> None:
    """A stale original declaration must never be presented as historical."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "episode.bin").write_bytes(b"new-bytes")
    output_root = tmp_path / "derived"
    episode = {
        "episode_id": "episode-002",
        "recording": {
            "uri": "episode.bin",
            "format": "video/mp4",
            "sha256": "0" * 64,
        },
        "retained_states": [
            {"time_s": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
            {"time_s": 1.0, "x": 1.0, "y": 0.0, "heading": 0.0},
        ],
    }

    result = materialize_episode(
        episode,
        source_root=source_root,
        output_root=output_root,
        output_directory="episode-002",
    )

    assert result.status == "complete"
    assert result.materialization_kind == DERIVED_RENDER
    assert result.fidelity == FIDELITY_UNVERIFIABLE
    assert result.simulation_executed is False
    assert result.diagnostics[0] == "original_recording_digest_mismatch"
    output = output_root / "episode-002"
    assert (output / "trajectory.png").is_file()
    manifest = json.loads((output / "audit-materialization.v1.json").read_text())
    assert manifest["materialization_kind"] == DERIVED_RENDER
    assert manifest["provenance"]["simulation_advanced"] is False


def test_retained_trace_uses_canonical_renderer_and_relative_artifacts(tmp_path: Path) -> None:
    """Retained trace state is rendered by SREV and remains diagnostic-only."""
    episode = {
        "episode_id": "fixture_episode_001",
        "scenario_id": "classic_bottleneck_medium",
        "retained_trace": _trace_payload(),
    }

    result = materialize_episode(
        episode,
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
        output_directory="trace-render",
    )

    assert result.status == "complete", result.diagnostics
    assert result.materialization_kind == DERIVED_RENDER
    assert result.fidelity == FIDELITY_UNVERIFIABLE
    assert result.simulation_executed is False
    output = tmp_path / "derived" / "trace-render"
    assert (output / "frame-source-map.json").is_file()
    assert (output / "component-descriptor.json").is_file()
    for artifact in result.artifacts:
        assert not artifact["uri"].startswith("trace-render/")
        assert (output / artifact["uri"]).is_file()
    manifest = json.loads((output / "audit-materialization.v1.json").read_text())
    assert manifest["provenance"]["renderer"] == "robot_sf.render.review_scene.run"
    assert manifest["provenance"]["simulation_advanced"] is False
    assert manifest["diagnostics"] == ["original_recording_not_declared"]


def test_retained_trace_source_requires_digest_and_never_guesses(tmp_path: Path) -> None:
    """A retained path without a receipt is unavailable rather than guessed."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "trace.json").write_bytes(TRACE_FIXTURE.read_bytes())
    episode = {
        "episode_id": "fixture_episode_001",
        "retained_trace_source": {"uri": "trace.json"},
        "video_path": "trace.json",
    }

    result = materialize_episode(
        episode,
        source_root=source_root,
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert result.fidelity == FIDELITY_UNAVAILABLE
    assert result.reason == "retained_state_missing"
    assert any("retained_trace_source_unavailable" in item for item in result.diagnostics)
    assert result.simulation_executed is False


def test_verified_retained_trace_source_keeps_source_digest(tmp_path: Path) -> None:
    """A retained trace path is rendered only after its byte receipt verifies."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "trace.json"
    source_bytes = TRACE_FIXTURE.read_bytes()
    source_path.write_bytes(source_bytes)
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    episode = {
        "episode_id": "fixture_episode_001",
        "retained_trace_source": {"uri": "trace.json", "sha256": source_digest},
    }

    result = materialize_episode(
        episode,
        source_root=source_root,
        output_root=tmp_path / "derived",
        output_directory="trace-source-render",
    )

    assert result.status == "complete", result.diagnostics
    assert result.materialization_kind == DERIVED_RENDER
    assert result.source_digest == source_digest
    assert "retained_trace_source_verified" in result.diagnostics
    manifest = json.loads(
        (tmp_path / "derived" / "trace-source-render" / "audit-materialization.v1.json").read_text()
    )
    assert manifest["source_digest"] == source_digest


def test_retained_trace_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    """A retained trace from another scenario cannot be rebound to this row."""
    episode = {
        "episode_id": "fixture_episode_001",
        "scenario_id": "different-scenario",
        "retained_trace": _trace_payload(),
    }

    result = materialize_episode(
        episode,
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert result.reason == "retained_trace_unavailable"
    assert any("scenario identity" in item for item in result.diagnostics)
    assert not (tmp_path / "derived" / "audit-materialization.v1.json").exists()


def test_exact_input_execution_is_explicitly_deferred(tmp_path: Path) -> None:
    """Exact-input rows do not silently fall back to a different policy."""
    episode = {
        "episode_id": "episode-exact-input",
        "config_identity": "config-sha",
        "source_commit": "commit-sha",
        "planner_config": {"planner": "intended"},
    }

    result = materialize_episode(
        episode,
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert result.fidelity == FIDELITY_UNAVAILABLE
    assert result.reason == "exact_input_execution_deferred"
    assert result.provenance["execution_deferred"] is True
    assert result.provenance["simulation_advanced"] is False
    assert not (tmp_path / "derived").exists()


def test_source_and_renderer_inputs_change_cache_key() -> None:
    """Cache identity includes source and renderer inputs, not output location."""
    episode = {"episode_id": "episode-cache"}
    digest_a = "a" * 64
    digest_b = "b" * 64

    key_a = build_materialization_cache_key(
        episode,
        source_digest=digest_a,
        render_config={"scene": {"frame_indices": [0]}},
    )
    key_a_other_output = build_materialization_cache_key(
        episode,
        source_digest=digest_a,
        render_config={"scene": {"frame_indices": [0]}},
    )
    key_b = build_materialization_cache_key(
        episode,
        source_digest=digest_b,
        render_config={"scene": {"frame_indices": [0]}},
    )

    assert key_a == key_a_other_output
    assert key_a != key_b


def test_result_envelope_is_explicitly_diagnostic_only() -> None:
    """Public result helpers keep unavailable and successful labels separate."""
    result = materialize.MaterializationResult(
        status="complete",
        materialization_kind=HISTORICAL_ORIGINAL,
        fidelity=FIDELITY_VERIFIED,
        episode_id="episode-envelope",
        cache_key="a" * 64,
        artifacts=({"artifact_id": "recording", "uri": "episode.bin"},),
    )

    assert result.classification == HISTORICAL_ORIGINAL
    assert result.ok is True
    envelope = result.to_dict()
    assert envelope["diagnostic_only"] is True
    assert envelope["classification"] == HISTORICAL_ORIGINAL

    unavailable = materialize.MaterializationResult(
        status="unavailable",
        materialization_kind=UNAVAILABLE,
        fidelity=FIDELITY_UNAVAILABLE,
        episode_id="episode-envelope",
        cache_key="b" * 64,
    )
    assert unavailable.ok is False


def test_invalid_episode_and_request_paths_fail_closed(tmp_path: Path) -> None:
    """Malformed selection, descriptor, URI, and output paths stay unavailable."""
    invalid = materialize_episode(
        {},
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )
    assert invalid.reason == "invalid_episode"
    assert invalid.materialization_kind == UNAVAILABLE

    malformed_descriptor = materialize_episode(
        {"episode_id": "episode-malformed", "recording": "not-an-object"},
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )
    assert malformed_descriptor.diagnostics[0] == "original_recording_declaration_malformed"
    assert malformed_descriptor.reason == "retained_state_missing"

    traversal = materialize_episode(
        {
            "episode_id": "episode-traversal",
            "recording": {"uri": "../outside.bin", "sha256": "0" * 64},
        },
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )
    assert traversal.reason == "retained_state_missing"
    assert "original_recording_unavailable" in traversal.diagnostics[0]

    unsafe_output = materialize_episode(
        {
            "episode_id": "episode-output",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
        output_directory="../escape",
    )
    assert unsafe_output.reason == "materialization_unavailable"
    assert unsafe_output.materialization_kind == UNAVAILABLE


def test_source_root_symlink_and_missing_source_are_rejected(tmp_path: Path) -> None:
    """The source root boundary rejects missing files and symlink components."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = source_root / "link.bin"
    link.symlink_to(outside)
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()

    symlink_result = materialize_episode(
        {
            "episode_id": "episode-symlink",
            "recording": {"uri": "link.bin", "sha256": digest},
        },
        source_root=source_root,
        output_root=tmp_path / "derived",
    )
    assert symlink_result.reason == "retained_state_missing"
    assert "original_recording_unavailable" in symlink_result.diagnostics[0]

    missing_result = materialize_episode(
        {
            "episode_id": "episode-missing",
            "recording": {"uri": "missing.bin", "sha256": digest},
        },
        source_root=source_root,
        output_root=tmp_path / "derived",
    )
    assert missing_result.reason == "retained_state_missing"
    assert "original_recording_unavailable" in missing_result.diagnostics[0]


def test_original_identity_mismatch_and_trace_validation_fail_closed(tmp_path: Path) -> None:
    """Declared identity and canonical trace checks prevent source rebinding."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    original = source_root / "episode.bin"
    original.write_bytes(b"verified-bytes")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    mismatch = materialize_episode(
        {
            "episode_id": "episode-identity",
            "scenario_id": "selected-scenario",
            "recording": {
                "uri": "episode.bin",
                "sha256": digest,
                "scenario_id": "other-scenario",
            },
        },
        source_root=source_root,
        output_root=tmp_path / "derived",
    )
    assert mismatch.reason == "retained_state_missing"
    assert "original_recording_identity_mismatch:scenario_id" in mismatch.diagnostics[0]

    malformed_trace = _trace_payload()
    malformed_trace["source"]["episode_id"] = "other-episode"
    malformed = materialize_episode(
        {
            "episode_id": "fixture_episode_001",
            "retained_trace": malformed_trace,
        },
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )
    assert malformed.reason == "retained_trace_unavailable"
    assert "episode identity" in malformed.diagnostics[-1]


def test_serialized_episode_ref_nested_source_identity_cannot_accept_foreign_commit(
    tmp_path: Path,
) -> None:
    """Canonical EpisodeRef source identity binds an original recording to its commit."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "episode.bin"
    source_bytes = b"serialized-episode-ref"
    source_path.write_bytes(source_bytes)
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    expected_commit = "a" * 40
    foreign_commit = "b" * 40
    row = record_to_dict(
        EpisodeRef(
            campaign_digest="c" * 64,
            source_digest=source_digest,
            execution_id="execution-001",
            scenario_id="scenario-001",
            source=SourceRef(
                artifact_id="episode-recording",
                uri="episode.bin",
                format="video/mp4",
                sha256=source_digest,
                source_commit=expected_commit,
                config_identity="config-001",
            ),
        )
    )
    row["recording"] = {
        "uri": "episode.bin",
        "format": "video/mp4",
        "sha256": source_digest,
        "source_commit": foreign_commit,
    }
    row["retained_states"] = [
        {"time_s": 0.0, "x": 0.0, "y": 0.0},
        {"time_s": 1.0, "x": 1.0, "y": 0.0},
    ]

    result = materialize_episode(
        row,
        source_root=source_root,
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == DERIVED_RENDER
    assert result.reason == ""
    assert "original_recording_identity_mismatch:source_commit" in result.diagnostics


def test_identity_alias_conflict_is_invalid_episode(tmp_path: Path) -> None:
    """Conflicting top-level and nested identity aliases fail before source access."""
    result = materialize_episode(
        {
            "episode_id": "episode-conflicting-identity",
            "source_commit": "a" * 40,
            "source": {"source_commit": "b" * 40},
        },
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert result.reason == "invalid_episode"
    assert "identity aliases conflict" in result.diagnostics[0]


@pytest.mark.parametrize(
    ("field_name", "malformed"),
    [
        ("source_commit", float("nan")),
        ("config_digest", float("inf")),
        ("checkpoint_digest", 123),
        ("environment_digest", {}),
        ("initial_state_digest", ["x"]),
    ],
)
def test_malformed_identity_alias_cannot_admit_verified_original(
    field_name: str, malformed: Any, tmp_path: Path
) -> None:
    """Malformed recognized identity aliases fail before original admission."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "episode.bin"
    source_bytes = b"identity-alias-boundary"
    source_path.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()
    result = materialize_episode(
        {
            "episode_id": "episode-malformed-identity",
            field_name: malformed,
            "recording": {"uri": "episode.bin", "format": "video/mp4", "sha256": digest},
        },
        source_root=source_root,
        output_root=tmp_path / "derived",
    )

    assert result.reason == "invalid_episode"
    assert result.materialization_kind == UNAVAILABLE
    assert field_name in result.diagnostics[0]


@pytest.mark.parametrize(
    "envelope",
    [
        {"source": None},
        {"source": []},
        {"source": {"source_commit": float("nan")}},
        {"source_identity": None},
        {"provenance": []},
    ],
)
def test_malformed_named_identity_envelope_cannot_admit_verified_original(
    envelope: dict[str, Any], tmp_path: Path
) -> None:
    """Malformed named identity containers fail before source admission."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "episode.bin"
    source_bytes = b"identity-envelope-boundary"
    source_path.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()
    row = {
        "episode_id": "episode-malformed-envelope",
        "recording": {"uri": "episode.bin", "format": "video/mp4", "sha256": digest},
        **envelope,
    }

    result = materialize_episode(row, source_root=source_root, output_root=tmp_path / "derived")

    assert result.reason == "invalid_episode"
    assert result.materialization_kind == UNAVAILABLE


def test_retained_replay_validation_and_renderer_failure_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bad retained states and renderer errors do not become historical output."""
    invalid_states = materialize_episode(
        {
            "episode_id": "episode-invalid-states",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 0.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )
    assert invalid_states.reason == "retained_states_unavailable"
    assert invalid_states.materialization_kind == UNAVAILABLE

    monkeypatch.setattr(
        materialize,
        "generate_trajectory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("renderer failed")),
    )
    renderer_failure = materialize_episode(
        {
            "episode_id": "episode-renderer-failure",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived-failure",
    )
    assert renderer_failure.reason == "retained_replay_render_failed"
    assert renderer_failure.materialization_kind == UNAVAILABLE

    monkeypatch.setattr(
        materialize,
        "generate_trajectory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("unexpected renderer error")),
    )
    unexpected_renderer_failure = materialize_episode(
        {
            "episode_id": "episode-unexpected-renderer-failure",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source-unexpected",
        output_root=tmp_path / "derived-unexpected",
    )
    assert unexpected_renderer_failure.reason == "retained_replay_render_failed"
    assert unexpected_renderer_failure.materialization_kind == UNAVAILABLE


def test_trace_renderer_unavailable_and_manifest_failure_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical renderer and manifest publication failures remain unavailable."""
    monkeypatch.setattr(
        materialize.review_scene,
        "run",
        lambda request, **_kwargs: ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason="renderer unavailable",
        ),
    )
    unavailable = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )
    assert unavailable.reason == "retained_trace_renderer_unavailable"
    assert unavailable.materialization_kind == UNAVAILABLE

    monkeypatch.undo()
    monkeypatch.setattr(
        materialize,
        "_exclusive_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk error")),
    )
    manifest_failure = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived-manifest-failure",
    )
    assert manifest_failure.reason == "manifest_write_failed"
    assert manifest_failure.materialization_kind == UNAVAILABLE


def test_trace_renderer_unexpected_exception_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unexpected renderer exceptions are bounded at the materialization boundary."""
    monkeypatch.setattr(
        materialize.review_scene,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("renderer bug")),
    )

    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )

    assert result.reason == "retained_trace_render_failed"
    assert result.materialization_kind == UNAVAILABLE


def test_replay_malformed_renderer_result_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed replay result never escapes as an attribute error."""
    monkeypatch.setattr(materialize, "generate_trajectory", lambda *_args, **_kwargs: None)

    result = materialize_episode(
        {
            "episode_id": "episode-malformed-replay-result",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert result.reason == "retained_replay_render_failed"
    assert result.status == "failed"


def test_trace_empty_complete_renderer_result_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete renderer result with no artifacts cannot publish a manifest."""

    def empty_result(request: Any, *, base: Path, **_kwargs: Any) -> ComponentResult:
        (base / request.output_directory).mkdir()
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="complete",
        )

    monkeypatch.setattr(materialize.review_scene, "run", empty_result)
    output_root = tmp_path / "derived"
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="empty-result",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert result.reason == "retained_trace_render_failed"
    assert not (output_root / "empty-result" / "audit-materialization.v1.json").exists()


def test_trace_malformed_renderer_diagnostics_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed renderer diagnostics are bounded at the materialization edge."""

    def malformed_result(request: Any, **_kwargs: Any) -> ComponentResult:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="complete",
            artifacts=(
                {"artifact_id": "frame.png", "uri": "rendered/frame.png", "sha256": "0" * 64},
            ),
            diagnostics=(None,),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(materialize.review_scene, "run", malformed_result)
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert result.reason == "retained_trace_render_failed"
    assert result.status == "unavailable"


@pytest.mark.parametrize(
    ("status", "artifacts"),
    [
        ("unexpected", ()),
        ("complete", (None,)),
    ],
)
def test_trace_malformed_renderer_status_or_artifacts_are_unavailable(
    status: str,
    artifacts: tuple[Any, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed status and artifact fields stay inside the typed envelope."""

    def malformed_result(request: Any, **_kwargs: Any) -> ComponentResult:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=status,
            artifacts=artifacts,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(materialize.review_scene, "run", malformed_result)

    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )

    assert result.materialization_kind == UNAVAILABLE
    assert result.reason == "retained_trace_render_failed"
    assert result.status == "unavailable"


def test_replay_hardlinked_renderer_bytes_are_copied_to_a_distinct_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay publication cannot preserve an external renderer hardlink alias."""
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.bin"
    original = b"replay-hardlink-source"
    secret.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()

    def hardlink_renderer(_replay: Any, path: Path, **_kwargs: Any) -> FigureArtifact:
        os.link(secret, path)
        return FigureArtifact("trajectory", str(path), "png", digest, "")

    monkeypatch.setattr(materialize, "generate_trajectory", hardlink_renderer)
    output_root = tmp_path / "derived"
    result = materialize_episode(
        {
            "episode_id": "episode-replay-hardlink",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="hardlink",
    )

    output_file = output_root / "hardlink" / "trajectory.png"
    assert result.status == "complete", result.diagnostics
    assert output_file.read_bytes() == original
    assert os.stat(output_file).st_ino != os.stat(secret).st_ino
    secret.write_bytes(b"mutated-outside-source")
    assert output_file.read_bytes() == original
    assert result.artifacts[0]["sha256"] == digest


def test_trace_hardlinked_renderer_bytes_are_copied_to_a_distinct_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trace publication cannot preserve an external renderer hardlink alias."""
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.bin"
    original = b"trace-hardlink-source"
    secret.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()

    def hardlink_renderer(request: Any, *, base: Path, **_kwargs: Any) -> ComponentResult:
        staged = base / request.output_directory
        staged.mkdir()
        os.link(secret, staged / "frame.bin")
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="complete",
            artifacts=(
                {"artifact_id": "frame.bin", "uri": "rendered/frame.bin", "sha256": digest},
            ),
        )

    monkeypatch.setattr(materialize.review_scene, "run", hardlink_renderer)
    output_root = tmp_path / "derived"
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="hardlink",
    )

    output_file = output_root / "hardlink" / "frame.bin"
    assert result.status == "complete", result.diagnostics
    assert output_file.read_bytes() == original
    assert os.stat(output_file).st_ino != os.stat(secret).st_ino
    secret.write_bytes(b"mutated-outside-source")
    assert output_file.read_bytes() == original
    assert result.artifacts[0]["sha256"] == digest


@pytest.mark.parametrize("request_ids", [("foreign-request", "srev09-review-scene"), ("", "")])
def test_trace_renderer_result_ids_must_bind_to_the_request(
    request_ids: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreign or anonymous renderer identities cannot publish a complete result."""
    payload = b"valid-rendered-bytes"
    digest = hashlib.sha256(payload).hexdigest()

    def renderer_with_wrong_identity(
        request: Any, *, base: Path, **_kwargs: Any
    ) -> ComponentResult:
        staged = base / request.output_directory
        staged.mkdir()
        (staged / "frame.bin").write_bytes(payload)
        return ComponentResult(
            request_id=request_ids[0],
            component_id=request_ids[1],
            status="complete",
            artifacts=(
                {"artifact_id": "frame.bin", "uri": "rendered/frame.bin", "sha256": digest},
            ),
        )

    monkeypatch.setattr(materialize.review_scene, "run", renderer_with_wrong_identity)
    output_root = tmp_path / "derived"
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="wrong-identity",
    )

    assert result.status == "unavailable"
    assert result.reason == "retained_trace_render_failed"
    assert not (output_root / "wrong-identity" / "audit-materialization.v1.json").exists()


def test_manifest_publication_rejects_replaced_replay_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published replay directory replacement cannot redirect its manifest."""
    output_root = tmp_path / "derived"
    outside = tmp_path / "outside"
    outside.mkdir()

    real_publish = materialize._exclusive_json

    def publish_then_replace(*args: Any, **kwargs: Any) -> str:
        output = output_root / "race"
        shutil.rmtree(output)
        output.symlink_to(outside, target_is_directory=True)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(materialize, "_exclusive_json", publish_then_replace)
    result = materialize_episode(
        {
            "episode_id": "episode-replay-output-race",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    assert not (outside / "audit-materialization.v1.json").exists()


def test_manifest_publication_rejects_replaced_trace_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published trace directory replacement cannot redirect its manifest."""
    output_root = tmp_path / "derived"
    outside = tmp_path / "outside"
    outside.mkdir()

    real_publish = materialize._publish_staged_files

    def publish_then_replace(lease: Any, staged_output: int) -> None:
        real_publish(lease, staged_output)
        output = output_root / "race"
        shutil.rmtree(output)
        output.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(materialize, "_publish_staged_files", publish_then_replace)
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    assert not (outside / "audit-materialization.v1.json").exists()


def test_replay_child_replacement_before_manifest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced replay artifact cannot yield a stale complete manifest."""
    output_root = tmp_path / "derived"
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.bin"
    secret.write_bytes(b"outside-secret")
    real_publish = materialize._exclusive_json

    def replace_child_before_publish(*args: Any, **kwargs: Any) -> str:
        child = output_root / "race" / "trajectory.png"
        child.unlink()
        child.symlink_to(secret)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(materialize, "_exclusive_json", replace_child_before_publish)
    result = materialize_episode(
        {
            "episode_id": "episode-replay-child-race",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    assert (output_root / "race" / "trajectory.png").is_symlink()
    assert not (output_root / "race" / "audit-materialization.v1.json").exists()


def test_trace_child_replacement_before_manifest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced trace artifact cannot yield a stale complete manifest."""
    output_root = tmp_path / "derived"
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.bin"
    secret.write_bytes(b"outside-secret")
    real_publish = materialize._exclusive_json

    def replace_child_before_publish(*args: Any, **kwargs: Any) -> str:
        child = output_root / "race" / "component-descriptor.json"
        child.unlink()
        child.symlink_to(secret)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(materialize, "_exclusive_json", replace_child_before_publish)
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    assert (output_root / "race" / "component-descriptor.json").is_symlink()
    assert not (output_root / "race" / "audit-materialization.v1.json").exists()


def test_replay_manifest_replacement_after_write_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest replaced after writing cannot yield a stale complete result."""
    output_root = tmp_path / "derived"
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.bin"
    secret.write_bytes(b"outside-secret")
    real_publish = materialize._exclusive_json

    def replace_manifest_after_write(*args: Any, **kwargs: Any) -> str:
        digest = real_publish(*args, **kwargs)
        manifest = output_root / "race" / "audit-materialization.v1.json"
        manifest.unlink()
        manifest.symlink_to(secret)
        return digest

    monkeypatch.setattr(materialize, "_exclusive_json", replace_manifest_after_write)
    result = materialize_episode(
        {
            "episode_id": "episode-replay-manifest-race",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    assert not (output_root / "race" / "audit-materialization.v1.json").is_symlink()
    assert not (output_root / "race" / "audit-materialization.v1.json").exists()
    assert secret.read_bytes() == b"outside-secret"


def test_trace_manifest_replacement_after_write_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace manifest replacement after writing cannot escape its root."""
    output_root = tmp_path / "derived"
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.bin"
    secret.write_bytes(b"outside-secret")
    real_publish = materialize._exclusive_json

    def replace_manifest_after_write(*args: Any, **kwargs: Any) -> str:
        digest = real_publish(*args, **kwargs)
        manifest = output_root / "race" / "audit-materialization.v1.json"
        manifest.unlink()
        manifest.symlink_to(secret)
        return digest

    monkeypatch.setattr(materialize, "_exclusive_json", replace_manifest_after_write)
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    assert not (output_root / "race" / "audit-materialization.v1.json").is_symlink()
    assert not (output_root / "race" / "audit-materialization.v1.json").exists()
    assert secret.read_bytes() == b"outside-secret"


@pytest.mark.parametrize("replacement", ["directory", "file", "symlink"])
def test_trace_publication_does_not_replace_racing_output_entry(
    replacement: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced output entry is never overwritten by publication."""
    output_root = tmp_path / "derived"
    admitted_output = tmp_path / "admitted-output"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_publish = materialize._publish_staged_files

    def replace_output_before_file_publication(lease: Any, staged_output: int) -> None:
        output = output_root / "race"
        output.rename(admitted_output)
        if replacement == "directory":
            output.mkdir()
        elif replacement == "file":
            output.write_bytes(b"racing-file")
        else:
            output.symlink_to(outside, target_is_directory=True)
        real_publish(lease, staged_output)

    monkeypatch.setattr(
        materialize,
        "_publish_staged_files",
        replace_output_before_file_publication,
    )
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    if replacement == "directory":
        assert not any((output_root / "race").iterdir())
    elif replacement == "file":
        assert (output_root / "race").read_bytes() == b"racing-file"
    else:
        assert (output_root / "race").is_symlink()
    assert (admitted_output / "component-descriptor.json").is_file()
    assert not any(outside.iterdir())


def test_trace_private_staging_uses_output_filesystem(tmp_path: Path) -> None:
    """Trace staging succeeds when output_root is on a separate filesystem."""
    shm_root = Path("/dev/shm")
    if not shm_root.is_dir() or not os.access(shm_root, os.W_OK):
        pytest.skip("/dev/shm is unavailable or not writable")
    if os.stat(shm_root).st_dev == os.stat(tmp_path).st_dev:
        pytest.skip("/dev/shm does not provide a separate filesystem here")
    staging_parent = Path(tempfile.mkdtemp(prefix="robot-sf-materialize-", dir=shm_root))
    try:
        result = materialize_episode(
            {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
            source_root=tmp_path / "source",
            output_root=staging_parent / "derived",
            output_directory="cross-fs",
        )
        assert result.status == "complete", result.diagnostics
        assert (staging_parent / "derived" / "cross-fs" / "component-descriptor.json").is_file()
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def test_trace_source_staging_does_not_follow_replaced_output_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced output parent cannot redirect retained source staging bytes."""
    base = tmp_path / "base"
    output_root = base / "derived"
    admitted_base = tmp_path / "base-admitted"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_mkdtemp = materialize.tempfile.mkdtemp

    def replace_parent_before_source_staging(*args: Any, **kwargs: Any) -> str:
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        if prefix == ".audit-materialize-source-":
            base.rename(admitted_base)
            base.symlink_to(outside, target_is_directory=True)
        return real_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(materialize.tempfile, "mkdtemp", replace_parent_before_source_staging)
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    assert not any(outside.rglob("retained-trace.json"))
    assert (admitted_base / "derived" / "race" / "component-descriptor.json").is_file()


def test_replay_renderer_stages_below_retained_root_during_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay bytes stay below the retained root while its parent is replaced."""
    base = tmp_path / "base"
    output_root = base / "derived"
    admitted_base = tmp_path / "base-admitted"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_stage = materialize._open_private_staging_parent
    real_renderer = materialize.generate_trajectory

    def restore_parent() -> None:
        if base.is_symlink():
            base.unlink()
        if admitted_base.exists():
            admitted_base.rename(base)

    def stage_after_parent_swap(root_fd: int, root_identity: tuple[int, int]) -> int:
        base.rename(admitted_base)
        base.symlink_to(outside, target_is_directory=True)
        try:
            return real_stage(root_fd, root_identity)
        except BaseException:
            restore_parent()
            raise

    def render_while_parent_is_replaced(*args: Any, **kwargs: Any) -> Any:
        try:
            figure = real_renderer(*args, **kwargs)
            assert not any(outside.rglob("*"))
            assert any(admitted_base.rglob("trajectory.png"))
            return figure
        finally:
            restore_parent()

    monkeypatch.setattr(materialize, "_open_private_staging_parent", stage_after_parent_swap)
    monkeypatch.setattr(materialize, "generate_trajectory", render_while_parent_is_replaced)
    result = materialize_episode(
        {
            "episode_id": "episode-parent-staging-race",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.status == "complete", result.diagnostics
    assert (output_root / "race" / "trajectory.png").is_file()
    assert not any(outside.rglob("*"))


def test_trace_renderer_stages_below_retained_root_during_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trace bytes stay below the retained root while its parent is replaced."""
    base = tmp_path / "base"
    output_root = base / "derived"
    admitted_base = tmp_path / "base-admitted"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_stage = materialize._open_private_staging_parent
    real_renderer = materialize.review_scene.run

    def restore_parent() -> None:
        if base.is_symlink():
            base.unlink()
        if admitted_base.exists():
            admitted_base.rename(base)

    def stage_after_parent_swap(root_fd: int, root_identity: tuple[int, int]) -> int:
        base.rename(admitted_base)
        base.symlink_to(outside, target_is_directory=True)
        try:
            return real_stage(root_fd, root_identity)
        except BaseException:
            restore_parent()
            raise

    def render_while_parent_is_replaced(request: Any, **kwargs: Any) -> ComponentResult:
        try:
            rendered = real_renderer(request, **kwargs)
            assert not any(outside.rglob("*"))
            assert any(admitted_base.rglob("*.png"))
            return rendered
        finally:
            restore_parent()

    monkeypatch.setattr(materialize, "_open_private_staging_parent", stage_after_parent_swap)
    monkeypatch.setattr(materialize.review_scene, "run", render_while_parent_is_replaced)
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.status == "complete", result.diagnostics
    assert (output_root / "race" / "component-descriptor.json").is_file()
    assert not any(outside.rglob("*"))


def test_replay_renderer_stays_inside_admitted_root_when_root_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root rename/symlink swap cannot redirect replay renderer writes."""
    output_root = tmp_path / "derived"
    admitted_root = tmp_path / "derived-admitted"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_renderer = materialize.generate_trajectory

    def render_after_root_replace(*args: Any, **kwargs: Any) -> Any:
        output_root.rename(admitted_root)
        output_root.symlink_to(outside, target_is_directory=True)
        return real_renderer(*args, **kwargs)

    monkeypatch.setattr(materialize, "generate_trajectory", render_after_root_replace)
    result = materialize_episode(
        {
            "episode_id": "episode-root-output-race",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        },
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    assert (admitted_root / "race" / "trajectory.png").is_file()
    assert not any(outside.iterdir())


def test_trace_renderer_stays_inside_admitted_root_when_root_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root rename/symlink swap cannot redirect canonical trace writes."""
    output_root = tmp_path / "derived"
    admitted_root = tmp_path / "derived-admitted"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_renderer = materialize.review_scene.run

    def render_after_root_replace(request: Any, **kwargs: Any) -> ComponentResult:
        output_root.rename(admitted_root)
        output_root.symlink_to(outside, target_is_directory=True)
        return real_renderer(request, **kwargs)

    monkeypatch.setattr(materialize.review_scene, "run", render_after_root_replace)
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    assert (admitted_root / "race" / "component-descriptor.json").is_file()
    assert not any(outside.iterdir())


def test_trace_renderer_staging_stays_inside_admitted_parent_when_nested_parent_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Private SREV staging cannot follow a replaced nested output parent."""
    output_root = tmp_path / "derived"
    admitted_parent = tmp_path / "derived-nested-admitted"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_stage = materialize.review_scene._stage_outputs

    def stage_after_nested_parent_replace(
        output_dir: Path, request: Any, scene: dict[str, Any], pairs: list[Any]
    ) -> Any:
        nested = output_root / "nested"
        nested.rename(admitted_parent)
        nested.symlink_to(outside, target_is_directory=True)
        return real_stage(output_dir, request, scene, pairs)

    monkeypatch.setattr(
        materialize.review_scene, "_stage_outputs", stage_after_nested_parent_replace
    )
    result = materialize_episode(
        {"episode_id": "fixture_episode_001", "retained_trace": _trace_payload()},
        source_root=tmp_path / "source",
        output_root=output_root,
        output_directory="nested/race",
    )

    assert result.reason == "manifest_write_failed"
    assert result.materialization_kind == UNAVAILABLE
    admitted_output = admitted_parent / "race"
    assert (admitted_output / "component-descriptor.json").is_file()
    assert (admitted_output / "frame-source-map.json").is_file()
    assert any(admitted_output.glob("*-scene_*.png"))
    assert not any(outside.iterdir())


@pytest.mark.parametrize("malformed", [[], "not-an-object", None])
def test_nonmapping_direct_episode_returns_unavailable_envelope(
    malformed: Any, tmp_path: Path
) -> None:
    """Direct malformed episode values never escape as validation exceptions."""
    result = materialize_episode(
        malformed,
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )

    assert result.reason == "invalid_episode"
    assert result.materialization_kind == UNAVAILABLE
    assert result.diagnostics == ("episode must be an object",)


def test_nonmapping_typed_request_returns_unavailable_envelope(tmp_path: Path) -> None:
    """A typed request with malformed episode content is bounded identically."""
    result = materialize_episode(
        materialize.MaterializationRequest(
            episode=[],
            source_root=tmp_path / "source",
            output_root=tmp_path / "derived",
        )
    )

    assert result.reason == "invalid_episode"
    assert result.materialization_kind == UNAVAILABLE
    assert result.diagnostics == ("episode must be an object",)


def test_validation_helpers_cover_strict_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Low-level boundaries reject malformed values before any source read."""
    with pytest.raises(materialize.MaterializationValidationError):
        materialize._bounded_text(None, name="required", required=True)
    with pytest.raises(materialize.MaterializationValidationError):
        materialize._bounded_text(" ", name="required", required=True)
    with pytest.raises(materialize.MaterializationValidationError):
        materialize._bounded_text("x" * 257, name="bounded", required=True)
    with pytest.raises(materialize.MaterializationValidationError):
        materialize._mapping([], name="mapping")
    with pytest.raises(materialize.MaterializationValidationError):
        materialize._canonical_bytes(float("nan"))
    with pytest.raises(materialize.MaterializationValidationError):
        materialize._safe_relative_uri("https://example.invalid/source", name="uri")
    with pytest.raises(materialize.MaterializationValidationError):
        materialize._safe_output_directory("")
    with pytest.raises(materialize.MaterializationValidationError):
        materialize._recording_uri({})

    source_root = tmp_path / "source"
    source_root.mkdir()
    nested = source_root / "nested"
    nested.mkdir()
    source_file = nested / "trace.bin"
    source_file.write_bytes(b"source")
    read = materialize._open_read_bounded(source_root, "nested/trace.bin")
    assert read.payload == b"source"
    with pytest.raises(ValueError, match="regular file"):
        materialize._open_read_bounded(source_root, "nested")
    monkeypatch.setattr(materialize, "MAX_SOURCE_BYTES", 1)
    with pytest.raises(ValueError, match="maximum size"):
        materialize._open_read_bounded(source_root, "nested/trace.bin")


def test_original_trace_and_nested_retained_state_paths_are_supported(tmp_path: Path) -> None:
    """Canonical trace receipts and nested retained-state ownership are explicit."""
    trace_bytes = TRACE_FIXTURE.read_bytes()
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "trace.json"
    source_path.write_bytes(trace_bytes)
    digest = hashlib.sha256(trace_bytes).hexdigest()
    original = materialize_episode(
        {
            "episode_id": "fixture_episode_001",
            "scenario_id": "classic_bottleneck_medium",
            "recording": {
                "uri": "trace.json",
                "format": "simulation_trace_export.v1",
                "sha256": digest,
            },
        },
        source_root=source_root,
        output_root=tmp_path / "derived-original",
    )
    assert original.materialization_kind == HISTORICAL_ORIGINAL
    assert original.fidelity == FIDELITY_VERIFIED

    nested = materialize_episode(
        {
            "episode_id": "fixture_episode_001",
            "retained_state": {"trace": _trace_payload()},
        },
        source_root=tmp_path / "source-other",
        output_root=tmp_path / "derived-nested",
    )
    assert nested.materialization_kind == DERIVED_RENDER

    malformed_path = source_root / "malformed.json"
    malformed_bytes = b"not-json"
    malformed_path.write_bytes(malformed_bytes)
    malformed = materialize_episode(
        {
            "episode_id": "fixture_episode_001",
            "recording": {
                "uri": "malformed.json",
                "format": "simulation_trace_export.v1",
                "sha256": hashlib.sha256(malformed_bytes).hexdigest(),
            },
        },
        source_root=source_root,
        output_root=tmp_path / "derived-malformed",
    )
    assert malformed.reason == "retained_state_missing"
    assert malformed.diagnostics[0] == "original_recording_trace_malformed"


def test_replay_alias_and_numeric_pedestrian_validation(tmp_path: Path) -> None:
    """Replay aliases retain optional pedestrians while rejecting bad numbers."""
    valid = materialize_episode(
        {
            "episode_id": "episode-replay-alias",
            "replay_steps": [
                {
                    "t": 0.0,
                    "x": 0.0,
                    "y": 0.0,
                    "heading": 0.0,
                    "pedestrians": [[1.0, 1.0]],
                },
                {
                    "t": 1.0,
                    "x": 1.0,
                    "y": 0.0,
                    "heading": 0.0,
                    "pedestrians": [{"position": [1.1, 1.0]}],
                },
            ],
        },
        source_root=tmp_path / "source",
        output_root=tmp_path / "derived",
    )
    assert valid.materialization_kind == DERIVED_RENDER

    malformed = materialize_episode(
        {
            "episode_id": "episode-replay-bad-number",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": float("nan"), "y": 0.0},
            ],
        },
        source_root=tmp_path / "source-bad",
        output_root=tmp_path / "derived-bad",
    )
    assert malformed.reason == "retained_states_unavailable"


def test_materialization_request_and_row_adapter_preserve_boundaries(tmp_path: Path) -> None:
    """Typed requests and service row-like objects use the same source-first path."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "recording.bin"
    source.write_bytes(MP4_BYTES)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    request = materialize.MaterializationRequest(
        episode={
            "episode_id": "episode-request",
            "recording": {"uri": "recording.bin", "format": "video/mp4", "sha256": digest},
        },
        source_root=source_root,
        output_root=tmp_path / "derived-request",
    )
    requested = materialize_episode(request)
    assert requested.materialization_kind == HISTORICAL_ORIGINAL

    source_ref_result = materialize_episode(
        {"episode_id": "episode-source-ref"},
        source_root=source_root,
        output_root=tmp_path / "derived-source-ref",
        recording=SourceRef(
            artifact_id="recording",
            uri="recording.bin",
            format="video/mp4",
            sha256=digest,
        ),
    )
    assert source_ref_result.materialization_kind == HISTORICAL_ORIGINAL

    class Row:
        row = {
            "episode_id": "episode-row",
            "retained_states": [
                {"time_s": 0.0, "x": 0.0, "y": 0.0},
                {"time_s": 1.0, "x": 1.0, "y": 0.0},
            ],
        }

    row_result = materialize_episode(
        Row(),
        source_root=tmp_path / "source-row",
        output_root=tmp_path / "derived-row",
    )
    assert row_result.materialization_kind == DERIVED_RENDER


def test_descriptor_backed_path_fails_closed_when_no_descriptor_view_exists() -> None:
    """Path-only renderer adapters never fall back to a visible unresolved path."""

    with pytest.raises(OSError, match="descriptor-backed renderer path is unavailable"):
        materialize._descriptor_path(-1)


def test_darwin_staging_helpers_use_the_private_lease_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Darwin fallback remains bounded when /dev/fd cannot be descended."""

    lease = materialize._OutputLease(
        root=tmp_path,
        relative="render",
        parts=("render",),
        root_fd=-1,
        root_identity=(0, 0),
    )
    monkeypatch.setattr(materialize.sys, "platform", "darwin")

    assert materialize._staging_parent_path(lease, -1) == str(tmp_path)
    assert materialize._renderer_root_path(tmp_path / "render", -1) == tmp_path / "render"


def test_atomic_link_publication_is_no_replace_and_descriptor_relative(tmp_path: Path) -> None:
    """The portable publication path preserves inode completeness and source removal."""

    source_directory = tmp_path / "source"
    destination_directory = tmp_path / "destination"
    source_directory.mkdir()
    destination_directory.mkdir()
    (source_directory / "private.bin").write_bytes(b"complete")
    source_fd = os.open(source_directory, os.O_RDONLY)
    destination_fd = os.open(destination_directory, os.O_RDONLY)
    try:
        materialize._link_noreplace("private.bin", source_fd, "published.bin", destination_fd)
    finally:
        os.close(source_fd)
        os.close(destination_fd)

    assert not (source_directory / "private.bin").exists()
    assert (destination_directory / "published.bin").read_bytes() == b"complete"
