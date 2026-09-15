"""Focused source, timeline, envelope, and publication tests for SREV-10."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import threading
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    component_descriptor_from_dict,
    component_request_from_dict,
    component_result_from_dict,
)
from robot_sf.render import review_encode
from robot_sf.render.review_encode import descriptor, result_payload, run

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/scenario_review/review_encode"
EXPECTED_ORDER = [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 20, 22, 24, 26, 28]


def _sha256(path: Path) -> str:
    """Return the digest used by a fixture source reference."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(tmp_path: Path, *, count: int = 30, variation: int = 0) -> str:
    """Write numbered PNG frames and their integrity-bound manifest."""

    from PIL import Image

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    digests: list[str] = []
    for index in range(count):
        relative = f"frames/frame-{index:03d}.png"
        path = tmp_path / relative
        image = Image.new(
            "RGB",
            (8, 6),
            (
                (index * 17 + variation) % 256,
                (index * 31 + variation * 2) % 256,
                (index * 47 + variation * 3) % 256,
            ),
        )
        image.save(path, format="PNG")
        paths.append(relative)
        digests.append(_sha256(path))
    manifest = {
        "schema_version": "frame-sequence-manifest.v1",
        "source_fps": 10,
        "source_frames": count,
        "frame_paths": paths,
        "frame_digests": digests,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(manifest_path)


def _request(
    tmp_path: Path,
    config_extra: dict | None = None,
    caps: list[str] | None = None,
    *,
    variation: int = 0,
    output: str = "out",
):
    """Create a parsed request backed by actual numbered source frames."""

    manifest_sha = _write_source(tmp_path, variation=variation)
    payload = {
        "schema_version": "component-request.v1",
        "request_id": "t10",
        "component_id": "srev10-review-encode",
        "sources": [
            {
                "artifact_id": "frames",
                "uri": "manifest.json",
                "format": "frame-sequence-manifest.v1",
                "sha256": manifest_sha,
            }
        ],
        "output_directory": output,
        "config": config_extra or {},
        "required_capabilities": caps if caps is not None else ["frame-sequence"],
    }
    return component_request_from_dict(payload)


def _time_map(out: Path) -> dict:
    """Load the emitted presentation-time-map.v1 leaf payload."""

    return json.loads((out / "time-map.json").read_text(encoding="utf-8"))


def test_descriptor_is_shared_contract_valid() -> None:
    """The descriptor must be accepted by SREV-01 before discovery."""

    desc = descriptor()
    component_descriptor_from_dict(desc)
    assert desc["schema_version"] == "component-descriptor.v1"
    assert "frame-sequence" in desc["optional_capabilities"]
    assert "source-clip" in desc["optional_capabilities"]


def test_fixture_plan_produces_expected_frame_order(tmp_path: Path) -> None:
    """Cuts, pauses, and a full-span speed edit preserve source frame identity."""

    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    result = run(_request(tmp_path, config_extra=config), base=tmp_path)
    assert result.status == "complete", result.reason
    assert _time_map(tmp_path / "out")["frame_order_source_indices"] == EXPECTED_ORDER


def test_interval_endpoints_and_first_terminal_frames(tmp_path: Path) -> None:
    """Half-open source intervals retain the first and terminal kept frames."""

    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    result = run(_request(tmp_path, config_extra=config), base=tmp_path)
    assert result.status == "complete"
    mapping = _time_map(tmp_path / "out")
    assert mapping["first_source_frame"] == 0
    assert mapping["terminal_source_frame"] == 28
    assert mapping["segments"][0]["presentation_start_s"] == 0.0
    assert mapping["segments"][-1]["presentation_end_s"] == pytest.approx(2.0)


def test_encoded_video_uses_actual_source_pixels(tmp_path: Path) -> None:
    """The encoder must consume decoded fixture pixels instead of synthetic frames."""

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    result = run(_request(tmp_path, config_extra=config), base=tmp_path)
    assert result.status == "complete"
    with imageio.get_reader(str(tmp_path / "out" / "edit.mp4")) as reader:
        decoded = [np.asarray(frame) for frame in reader]
    assert len(decoded) == len(EXPECTED_ORDER)
    for position in (0, 10, 15, 19):
        with Image.open(
            tmp_path / "frames" / f"frame-{EXPECTED_ORDER[position]:03d}.png"
        ) as source:
            expected = np.asarray(source.resize((160, 120)), dtype=np.uint8)
        assert decoded[position].shape == expected.shape
        assert abs(decoded[position].astype(int) - expected.astype(int)).mean() < 15


def test_pause_repeats_anchor_frame(tmp_path: Path) -> None:
    """A pause inserts repeated decoded anchor frames at presentation time."""

    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    result = run(_request(tmp_path, config_extra=config), base=tmp_path)
    assert result.status == "complete"
    assert _time_map(tmp_path / "out")["frame_order_source_indices"].count(5) == 6


def test_partial_speed_is_split_and_applied(tmp_path: Path) -> None:
    """A speed edit affects only its bounded interval and produces split map segments."""

    result = run(
        _request(
            tmp_path,
            config_extra={
                "edits": [{"op": "speed", "start_s": 0.5, "end_s": 1.5, "factor": 2.0}],
                "preset": dict(review_encode.TINY_PRESET),
            },
        ),
        base=tmp_path,
    )
    assert result.status == "complete", result.reason
    mapping = _time_map(tmp_path / "out")
    assert len(mapping["frame_order_source_indices"]) == 25
    assert [segment["operation"] for segment in mapping["segments"]] == [
        "passthrough",
        "speed",
        "passthrough",
    ]
    assert mapping["segments"][1]["source_start_s"] == pytest.approx(0.5)
    assert mapping["segments"][1]["source_end_s"] == pytest.approx(1.5)


def test_slow_motion_factor_repeats_without_index_error(tmp_path: Path) -> None:
    """Factors below one expand a span deterministically and stay within source bounds."""

    result = run(
        _request(
            tmp_path,
            config_extra={
                "edits": [{"op": "speed", "start_s": 0.0, "end_s": 3.0, "factor": 0.5}],
                "preset": dict(review_encode.TINY_PRESET),
            },
        ),
        base=tmp_path,
    )
    assert result.status == "complete", result.reason
    order = _time_map(tmp_path / "out")["frame_order_source_indices"]
    assert len(order) == 60
    assert order[:4] == [0, 0, 1, 1]
    assert order[-1] == 29


def test_cut_and_speed_keep_operation_identity(tmp_path: Path) -> None:
    """Untouched post-cut spans must not be mislabeled as speed edits."""

    result = run(
        _request(
            tmp_path,
            config_extra={
                "edits": [
                    {"op": "cut", "start_s": 1.0, "end_s": 2.0},
                    {"op": "speed", "start_s": 2.0, "end_s": 2.5, "factor": 2.0},
                ],
                "preset": dict(review_encode.TINY_PRESET),
            },
        ),
        base=tmp_path,
    )
    assert result.status == "complete", result.reason
    assert [segment["operation"] for segment in _time_map(tmp_path / "out")["segments"]] == [
        "cut",
        "speed",
        "cut",
    ]


@pytest.mark.parametrize(("at_s", "duration_s"), [(0.001, 0.001), (3.0, 0.001)])
def test_subframe_and_terminal_pauses_are_retained(
    tmp_path: Path, at_s: float, duration_s: float
) -> None:
    """Sub-frame and terminal pauses round up to at least one output frame."""

    result = run(
        _request(
            tmp_path,
            config_extra={
                "edits": [{"op": "pause", "at_s": at_s, "duration_s": duration_s}],
                "preset": dict(review_encode.TINY_PRESET),
            },
        ),
        base=tmp_path,
    )
    assert result.status == "complete", result.reason
    mapping = _time_map(tmp_path / "out")
    assert any(segment["operation"] == "pause" for segment in mapping["segments"])
    if at_s == 3.0:
        assert mapping["frame_order_source_indices"][-2:] == [29, 29]


def test_terminal_pause_after_fast_speed_is_retained(tmp_path: Path) -> None:
    """A fast resample keeps a pause at the retained source endpoint."""

    result = run(
        _request(
            tmp_path,
            config_extra={
                "edits": [
                    {"op": "speed", "start_s": 0.0, "end_s": 3.0, "factor": 2.0},
                    {"op": "pause", "at_s": 3.0, "duration_s": 0.1},
                ],
                "preset": dict(review_encode.TINY_PRESET),
            },
        ),
        base=tmp_path,
    )
    assert result.status == "complete", result.reason
    mapping = _time_map(tmp_path / "out")
    assert mapping["frame_order_source_indices"][-2:] == [28, 28]
    assert mapping["segments"][-1]["operation"] == "pause"


def test_terminal_pause_after_tail_cut_is_dropped(tmp_path: Path) -> None:
    """A terminal pause cannot anchor after a cut removed the source endpoint."""

    result = run(
        _request(
            tmp_path,
            config_extra={
                "edits": [
                    {"op": "cut", "start_s": 2.0, "end_s": 3.0},
                    {"op": "pause", "at_s": 3.0, "duration_s": 0.1},
                ],
                "preset": dict(review_encode.TINY_PRESET),
            },
        ),
        base=tmp_path,
    )
    assert result.status == "partial"
    assert "dropped_pause" in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_pause_in_cut_region_is_partial_without_artifacts(tmp_path: Path) -> None:
    """A pause with no retained anchor is explicit partial diagnostic output."""

    result = run(
        _request(
            tmp_path,
            config_extra={
                "edits": [
                    {"op": "cut", "start_s": 0.0, "end_s": 1.0},
                    {"op": "pause", "at_s": 0.5, "duration_s": 0.1},
                ],
            },
        ),
        base=tmp_path,
    )
    assert result.status == "partial"
    assert "dropped_pause" in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_crop_clamp_is_partial(tmp_path: Path) -> None:
    """Clamping a valid crop is surfaced as partial rather than complete evidence."""

    result = run(
        _request(
            tmp_path,
            config_extra={
                "edits": [{"op": "crop", "x": 100, "y": 100, "width": 200, "height": 200}],
                "preset": dict(review_encode.TINY_PRESET),
            },
        ),
        base=tmp_path,
    )
    assert result.status == "partial"
    assert "crop_clamped" in result.reason
    assert result.artifacts == ()


def test_crop_provenance_is_recorded_in_receipt_and_time_map(tmp_path: Path) -> None:
    """Normalized crop coordinates make the diagnostic transform auditable."""

    result = run(
        _request(
            tmp_path,
            config_extra={
                "edits": [{"op": "crop", "x": 1.2, "y": 2.7, "width": 3.1, "height": 3.2}],
                "preset": dict(review_encode.TINY_PRESET),
            },
        ),
        base=tmp_path,
    )
    assert result.status == "complete", result.reason
    expected = {"operation": "crop", "box": [1, 2, 5, 6]}
    receipt = json.loads((tmp_path / "out" / "encode-receipt.json").read_text(encoding="utf-8"))
    mapping = _time_map(tmp_path / "out")
    assert receipt["crop"] == mapping["crop"] == expected
    assert receipt["output"]["width"] == 4
    assert receipt["output"]["height"] == 4


def test_required_frame_source_ignores_optional_clip(tmp_path: Path) -> None:
    """One required source family remains authoritative when another is optional."""

    request = _request(tmp_path, config_extra={"preset": dict(review_encode.TINY_PRESET)})
    optional_clip = tmp_path / "optional-clip.bin"
    optional_clip.write_bytes(b"optional source that must not be decoded")
    optional_ref = replace(
        request.sources[0],
        artifact_id="optional-clip",
        uri=optional_clip.name,
        format="source-clip",
        sha256=_sha256(optional_clip),
    )
    request = replace(request, sources=(*request.sources, optional_ref))

    result = run(request, base=tmp_path)

    assert result.status == "complete", result.reason
    receipt = json.loads((tmp_path / "out" / "encode-receipt.json").read_text(encoding="utf-8"))
    assert receipt["source"]["format"] == "frame-sequence-manifest.v1"


def test_source_clip_metadata_rejects_oversized_frame_before_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decoder geometry is bounded before a source frame can be yielded."""

    import imageio.v2 as imageio

    class FakeReader:
        iterated = False

        def __enter__(self) -> FakeReader:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_meta_data(self) -> dict[str, object]:
            return {"fps": 10.0, "size": (review_encode.MAX_OUTPUT_PIXELS + 1, 1)}

        def __iter__(self):
            self.iterated = True
            return iter(())

    reader = FakeReader()
    monkeypatch.setattr(imageio, "get_reader", lambda *_args, **_kwargs: reader)

    with pytest.raises(review_encode._SourceLoadError, match="source_frame_pixels"):
        review_encode._decode_clip_frames(tmp_path / "oversized.mp4")
    assert reader.iterated is False


def test_manifest_decoded_memory_limit_counts_retained_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compressed manifest size cannot bypass the decoded retained-frame ceiling."""

    class DecodedFrame:
        nbytes = review_encode.MAX_SOURCE_BUFFER_BYTES // 2 + 1

    monkeypatch.setattr(review_encode, "_decode_image", lambda _path: DecodedFrame())
    result = run(_request(tmp_path), base=tmp_path)

    assert result.status == "failed"
    assert "decoded_source_buffer_bytes" in result.reason
    assert not (tmp_path / "out").exists()


def test_pause_expansion_is_rejected_before_materialization(tmp_path: Path) -> None:
    """Projected pause cardinality is bounded before allocating repeated frames."""

    result = run(
        _request(
            tmp_path,
            config_extra={
                "edits": [{"op": "pause", "at_s": 0.0, "duration_s": 300.0}],
                "preset": {"width": 16, "height": 16, "fps": 120},
            },
        ),
        base=tmp_path,
    )

    assert result.status == "failed"
    assert result.reason == f"resource_limit: output_frames>{review_encode.MAX_OUTPUT_FRAMES}"
    assert not (tmp_path / "out").exists()


def test_threaded_deadlines_fail_closed_for_decoder_and_encoder(tmp_path: Path) -> None:
    """Worker-thread callers cannot bypass bounded decoder or encoder deadlines."""

    outcomes: dict[str, tuple[str | None, dict]] = {}

    def worker() -> None:
        outcomes["decoder"] = review_encode._decode_frame_count(tmp_path / "missing.mp4")
        outcomes["encoder"] = review_encode._encode_mp4(
            [], tmp_path / "missing-output.mp4", 10.0, {}
        )

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert outcomes["decoder"] == (None, "decoder_timeout")
    assert outcomes["encoder"] == ("encoder_timeout", {})


def test_conflicting_speeds_fail(tmp_path: Path) -> None:
    """Overlapping speed factors with different values make the map ambiguous."""

    edits = [
        {"op": "speed", "start_s": 0.0, "end_s": 2.0, "factor": 2.0},
        {"op": "speed", "start_s": 1.0, "end_s": 3.0, "factor": 0.5},
    ]
    result = run(_request(tmp_path, config_extra={"edits": edits}), base=tmp_path)
    assert result.status == "failed"
    assert "conflicting_time_map" in result.reason


def test_cut_everything_fails(tmp_path: Path) -> None:
    """Removing the whole source is a failed empty timeline, not an empty MP4."""

    result = run(
        _request(tmp_path, config_extra={"edits": [{"op": "cut", "start_s": 0.0, "end_s": 3.0}]}),
        base=tmp_path,
    )
    assert result.status == "failed"
    assert "empty_timeline" in result.reason


def test_corrupt_manifest_fails_closed(tmp_path: Path) -> None:
    """A source changed after request admission fails the declared digest binding."""

    request = _request(tmp_path)
    (tmp_path / "manifest.json").write_text("{nope", encoding="utf-8")
    result = run(request, base=tmp_path)
    assert result.status == "failed"
    assert "source_digest_mismatch" in result.reason


def test_source_digest_change_changes_consumed_output(tmp_path: Path) -> None:
    """Different source pixels produce different encoded output and receipt identity."""

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = run(_request(first_root, variation=0), base=first_root)
    second = run(_request(second_root, variation=3), base=second_root)
    assert first.status == second.status == "complete"
    first_receipt = json.loads((first_root / "out" / "encode-receipt.json").read_text())
    second_receipt = json.loads((second_root / "out" / "encode-receipt.json").read_text())
    assert first_receipt["source"]["frame_sha256"] != second_receipt["source"]["frame_sha256"]
    assert first_receipt["output"]["video_sha256"] != second_receipt["output"]["video_sha256"]


def test_repeated_fixture_runs_have_stable_logical_artifacts(tmp_path: Path) -> None:
    """Repeated identical fixture inputs keep IDs, timing, and logical digests stable."""

    roots = [tmp_path / "first", tmp_path / "second"]
    results = []
    for root in roots:
        root.mkdir()
        results.append(
            run(
                _request(
                    root,
                    config_extra={"preset": dict(review_encode.TINY_PRESET)},
                ),
                base=root,
            )
        )
    assert [result.status for result in results] == ["complete", "complete"]
    first_out, second_out = (root / "out" for root in roots)
    assert (first_out / "encode-receipt.json").read_bytes() == (
        second_out / "encode-receipt.json"
    ).read_bytes()
    assert (first_out / "time-map.json").read_bytes() == (second_out / "time-map.json").read_bytes()
    assert [artifact["artifact_id"] for artifact in results[0].artifacts] == [
        artifact["artifact_id"] for artifact in results[1].artifacts
    ]
    assert [artifact["sha256"] for artifact in results[0].artifacts] == [
        artifact["sha256"] for artifact in results[1].artifacts
    ]


def test_encoder_failure_has_no_complete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An encoder crash is failed closed and cannot publish a complete output."""

    monkeypatch.setattr(
        review_encode,
        "_encode_mp4",
        lambda *_args: ("encoder_failed: injected", {}),
    )
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_source_clip_is_decoded_and_bound(tmp_path: Path) -> None:
    """A clip-only request decodes its frames and does not synthesize a manifest."""

    import imageio.v2 as imageio
    import numpy as np

    clip = tmp_path / "source.mp4"
    frames = [
        np.full((12, 16, 3), (index * 40, 10, 200 - index * 20), dtype=np.uint8)
        for index in range(4)
    ]
    imageio.mimsave(str(clip), frames, fps=10, codec="libx264", macro_block_size=None)
    payload = {
        "schema_version": "component-request.v1",
        "request_id": "clip",
        "component_id": "srev10-review-encode",
        "sources": [
            {
                "artifact_id": "clip",
                "uri": "source.mp4",
                "format": "source-clip",
                "sha256": _sha256(clip),
            }
        ],
        "output_directory": "clip-out",
        "config": {"preset": dict(review_encode.TINY_PRESET)},
        "required_capabilities": ["source-clip"],
    }
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "complete", result.reason
    mapping = _time_map(tmp_path / "clip-out")
    assert mapping["source_frames"] == 4
    assert mapping["frame_order_source_indices"] == [0, 1, 2, 3]


def test_missing_source_family_is_unavailable(tmp_path: Path) -> None:
    """Requests without a usable source family are unavailable."""

    manifest_sha = _write_source(tmp_path)
    payload = {
        "schema_version": "component-request.v1",
        "request_id": "t10",
        "component_id": "srev10-review-encode",
        "sources": [
            {
                "artifact_id": "scores",
                "uri": "manifest.json",
                "format": "exemplar-scores",
                "sha256": manifest_sha,
            }
        ],
        "output_directory": "out",
        "config": {},
        "required_capabilities": ["frame-sequence"],
    }
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "unavailable"


def test_unsupported_capability_is_unavailable(tmp_path: Path) -> None:
    """Unknown capability names cannot be silently downgraded."""

    result = run(_request(tmp_path, caps=["telemetry-xyz"]), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_required_capability" in result.reason


def test_required_storyboard_capability_is_unavailable(tmp_path: Path) -> None:
    """A required storyboard capability cannot complete without a storyboard source."""

    assert "storyboard" not in descriptor()["optional_capabilities"]
    result = run(_request(tmp_path, caps=["storyboard"]), base=tmp_path)
    assert result.status == "unavailable"
    assert result.reason == "unsupported_required_capability: storyboard"
    assert result.artifacts == ()


def test_incompatible_version_fails(tmp_path: Path) -> None:
    """A request requiring a newer major component version fails before encoding."""

    result = run(_request(tmp_path, config_extra={"min_component_version": "9.0.0"}), base=tmp_path)
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_component_identity_is_admitted(tmp_path: Path) -> None:
    """A request for another component cannot be completed by this leaf."""

    result = run(replace(_request(tmp_path), component_id="other-component"), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_direct_api_malformed_request_fails_closed(tmp_path: Path) -> None:
    """Direct dataclass callers receive stable failures for malformed fields."""

    request = _request(tmp_path)
    empty_sources = run(replace(request, sources=[]), base=tmp_path)
    assert empty_sources.status == "failed"
    malformed_digest = run(
        replace(request, sources=(replace(request.sources[0], sha256=123),)),
        base=tmp_path,
    )
    assert malformed_digest.status == "failed"
    missing_digest = run(
        replace(request, sources=(replace(request.sources[0], sha256=""),)),
        base=tmp_path,
    )
    assert missing_digest.status == "failed"


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        ({"edits": [{"op": "cut", "start_s": [0], "end_s": 1}]}, "edit_plan_invalid"),
        (
            {"edits": [{"op": "pause", "at_s": 0.0, "duration_s": float("nan")}]},
            "edit_plan_invalid",
        ),
        (
            {"edits": [{"op": "speed", "start_s": 0.0, "end_s": 1.0, "factor": 1e-12}]},
            "out of range",
        ),
        ({"unknown": True}, "unknown keys"),
        ({"preset": {"width": float("inf"), "height": 120, "fps": 10}}, "config_invalid"),
    ],
)
def test_malformed_config_and_edits_fail_closed(tmp_path: Path, config: dict, reason: str) -> None:
    """Malformed values and unknown keys produce stable failures instead of exceptions."""

    result = run(_request(tmp_path, config_extra=config), base=tmp_path)
    assert result.status == "failed"
    assert reason in result.reason


def test_output_collision_includes_empty_directory(tmp_path: Path) -> None:
    """An existing empty output directory is a collision under no-overwrite semantics."""

    (tmp_path / "out").mkdir()
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_output_symlink_escape_is_rejected(tmp_path: Path) -> None:
    """Output symlinks are rejected before any encoder or file publication."""

    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "out").symlink_to(outside, target_is_directory=True)
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "unsafe_output_path" in result.reason
    assert not list(outside.iterdir())


def test_non_silent_audio_fails(tmp_path: Path) -> None:
    """The v1 silent-audio policy rejects unsupported passthrough requests."""

    result = run(_request(tmp_path, config_extra={"audio_policy": "passthrough"}), base=tmp_path)
    assert result.status == "failed"
    assert "unsupported_audio_policy" in result.reason


def test_resource_limits_reject_huge_manifest_before_frame_load(tmp_path: Path) -> None:
    """A source frame ceiling blocks huge declarations without allocating frames."""

    request = _request(tmp_path)
    manifest = {
        "schema_version": "frame-sequence-manifest.v1",
        "source_fps": 10,
        "source_frames": review_encode.MAX_SOURCE_FRAMES + 1,
        "frame_paths": [],
        "frame_digests": [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    request = replace(
        request, sources=(replace(request.sources[0], sha256=_sha256(manifest_path)),)
    )
    result = run(request, base=tmp_path)
    assert result.status == "failed"
    assert "resource_limit" in result.reason


def test_result_envelope_has_schema_and_artifact_digests(tmp_path: Path) -> None:
    """Complete API output serializes to the shared result envelope with bound files."""

    result = run(_request(tmp_path), base=tmp_path)
    payload = result_payload(result)
    component_result_from_dict(payload)
    assert payload["schema_version"] == "component-result.v1"
    for artifact in payload["artifacts"]:
        assert artifact["sha256"] == _sha256(tmp_path / "out" / artifact["uri"])


def test_cli_emits_schema_valid_status_and_nonzero_failure(tmp_path: Path) -> None:
    """The CLI prints a versioned result and returns nonzero for handled failure."""

    request = _request(tmp_path)
    input_path = tmp_path / "input.json"
    input_payload = {
        "schema_version": "component-request.v1",
        "request_id": request.request_id,
        "component_id": request.component_id,
        "sources": [
            {key: value for key, value in asdict(source).items() if value}
            for source in request.sources
        ],
        "output_directory": request.output_directory,
        "config": request.config,
        "required_capabilities": list(request.required_capabilities),
    }
    input_path.write_text(json.dumps(input_payload, sort_keys=True) + "\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    output_path = tmp_path / "cli-out"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_sf.render.review_encode",
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--base",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    envelope = json.loads(completed.stdout)
    component_result_from_dict(envelope)
    assert envelope["status"] == "complete"
    assert all("sha256" in artifact for artifact in envelope["artifacts"])

    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_sf.render.review_encode",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path.parent / "srev10-cli-outside" / "bad"),
            "--base",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 2
    component_result_from_dict(json.loads(failed.stdout))

    config_path.write_text('{"unknown": true}\n', encoding="utf-8")
    handled = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_sf.render.review_encode",
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "cli-bad"),
            "--base",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert handled.returncode == 1
    handled_envelope = json.loads(handled.stdout)
    component_result_from_dict(handled_envelope)
    assert handled_envelope["status"] == "failed"


def test_module_has_no_simulation_imports() -> None:
    """The leaf remains an offline renderer and cannot import simulator modules."""

    tree = ast.parse(
        (Path(__file__).resolve().parents[2] / "robot_sf/render/review_encode.py").read_bytes()
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(part.name for part in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {name for name in imported if "robot_sf.sim" in name}
