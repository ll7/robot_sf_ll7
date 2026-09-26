"""Render an already-recorded simulation trace without replaying its dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.adversarial import replay_gallery
from robot_sf.benchmark.episode_replay_figure import EpisodeRow, build_replay_from_episode_row

if TYPE_CHECKING:
    from collections.abc import Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_inputs(
    trace_path: Path, provenance_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace = _load_json(trace_path)
    provenance = _load_json(provenance_path)
    _validate_trace_payload(trace, provenance, trace_path)
    _validate_source_metadata(provenance)
    return trace, provenance


def _validate_trace_payload(
    trace: dict[str, Any], provenance: dict[str, Any], trace_path: Path
) -> None:
    """Check the extracted sample schema and its byte-level digest binding."""
    if trace.get("schema_version") != "simulation-step-trace.v1":
        raise ValueError("trace must use simulation-step-trace.v1")
    if provenance.get("schema_version") != "issue_9647_recorded_trace_provenance.v1":
        raise ValueError("provenance must use issue_9647_recorded_trace_provenance.v1")
    trace_binding = provenance.get("trace")
    if not isinstance(trace_binding, dict) or trace_binding.get("sha256") != _sha256(trace_path):
        raise ValueError("trace bytes do not match their provenance checksum")


def _validate_source_metadata(provenance: dict[str, Any]) -> None:
    """Check source artifact identities needed to interpret the stored trace."""
    source = provenance.get("source")
    if not isinstance(source, dict):
        raise ValueError("provenance is missing source artifact digests")
    for field in ("episode_record_sha256", "episode_record_provenance_sha256"):
        digest = source.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"provenance source.{field} must be a 64-character SHA-256 hex digest")
    episode = provenance.get("episode")
    if not isinstance(episode, dict):
        raise ValueError("provenance is missing source episode identity")
    for field in ("episode_id", "scenario_id", "seed", "planner"):
        if field not in episode:
            raise ValueError(f"provenance is missing episode.{field}")
    collision_events = episode.get("collision_events")
    if episode.get("exact_collision") is True and not isinstance(collision_events, list):
        raise ValueError("exact collision is asserted without its source event rows")
    case_id = episode.get("case_id", "recorded_trace")
    if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
        raise ValueError("episode.case_id must be a path-safe stable identifier")


def _validate_trace_steps(trace: dict[str, Any]) -> None:
    """Require enough recorded samples to construct a meaningful figure."""
    steps = trace.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("trace must contain at least two stored samples")


def render_recorded_trace_bundle(
    trace_path: Path,
    provenance_path: Path,
    output_dir: Path,
    *,
    video: bool = True,
    fps: int = 10,
) -> dict[str, Any]:
    """Render figures and optional video strictly from a verified stored trace."""
    if isinstance(fps, bool) or fps < 1:
        raise ValueError("fps must be a positive integer")
    trace_path = trace_path.expanduser().resolve()
    provenance_path = provenance_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    trace, provenance = _validate_inputs(trace_path, provenance_path)
    _validate_trace_steps(trace)
    episode_source = provenance["episode"]
    collision_events = [
        {
            "collision_time": event["event_time_s"],
            "exact_event_source": event["exact_event_source"],
        }
        for event in episode_source.get("collision_events", [])
        if isinstance(event, dict) and "event_time_s" in event and "exact_event_source" in event
    ]
    record = {
        "episode_id": episode_source["episode_id"],
        "scenario_id": episode_source["scenario_id"],
        "seed": episode_source["seed"],
        "algo": episode_source["planner"],
        "git_hash": episode_source.get("runner_revision"),
        "event_ledger": {
            "exact_events": {"collision": episode_source.get("exact_collision") is True},
            "collision_events": collision_events,
        },
        "algorithm_metadata": {"simulation_step_trace": trace},
    }
    case_id = episode_source.get("case_id", "recorded_trace")
    figure_dir = output_dir / "cases" / case_id / "figures"
    replay_dir = output_dir / "cases" / case_id / "replay"
    figure_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    render_result = replay_gallery._render_replay(
        record,
        None,
        replay_dir,
        figure_dir,
        output_dir,
        map_path=None,
    )

    video_result: dict[str, Any] = {"status": "not_requested", "artifact": None}
    if video:
        trace_steps, _critical, _clearance, _continuity = replay_gallery._replay_steps_from_trace(
            trace
        )
        row_payload = {
            **record,
            "replay_steps": trace_steps,
            "replay_dt": trace.get("dt"),
            "replay_map_path": None,
        }
        replay_episode = build_replay_from_episode_row(EpisodeRow.from_dict(row_payload))
        if replay_episode is None:
            video_result = {"status": "unavailable", "reason": "stored_trace_conversion_failed"}
        else:
            video_path = output_dir / "cases" / case_id / "video" / "recorded_trace.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from robot_sf.benchmark.full_classic.render_sim_view import (
                    generate_video_file,
                )

                rendered_video = generate_video_file(
                    replay_episode,
                    str(video_path),
                    fps=fps,
                    max_frames=len(replay_episode.steps),
                )
                if rendered_video.get("status") == "success" and video_path.is_file():
                    video_result = {
                        "status": "rendered_from_stored_trace",
                        "path": video_path.relative_to(output_dir).as_posix(),
                        "sha256": _sha256(video_path),
                        "size_bytes": video_path.stat().st_size,
                        "frames_requested": len(replay_episode.steps),
                        "fps": fps,
                        "renderer_status": rendered_video,
                        "diagnostic_overlays": "not_supported_by_existing_video_renderer",
                    }
                else:
                    video_result = {
                        "status": "unavailable",
                        "reason": rendered_video.get("note") or rendered_video.get("status"),
                        "renderer_status": rendered_video,
                    }
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                video_result = {
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}: {exc}",
                }

    provenance_out: dict[str, Any] = {
        "schema_version": "issue_9647_recorded_trace_render.v1",
        "render_mode": "stored_trace_only",
        "simulation_or_replay_dynamics_executed": False,
        "source_trace": {
            "path": trace_path.as_posix(),
            "sha256": _sha256(trace_path),
            "schema_version": trace["schema_version"],
            "source_extraction_pointer": provenance["trace"].get("source_extraction_pointer"),
        },
        "source_trace_provenance": {
            "path": provenance_path.as_posix(),
            "sha256": _sha256(provenance_path),
            "source_episode_record_sha256": provenance["source"]["episode_record_sha256"],
            "source_episode_record_provenance_sha256": provenance["source"][
                "episode_record_provenance_sha256"
            ],
            "source_runner_revision": episode_source.get("runner_revision"),
        },
        "renderer": {
            "module": "robot_sf.adversarial.replay_gallery._render_replay",
            "revision": _git_revision(),
            "command": "scripts/tools/render_recorded_replay_trace.py --trace <trace> --provenance <provenance> --out <output>",
            "no_map_overlay": "unavailable_no_map_asset_passed_to_renderer",
        },
        "render_result": render_result,
        "video_result": video_result,
        "claim_boundary": (
            "The figures and optional video are renderings of stored trace samples. They are not "
            "a replay, a fresh simulation, a feasibility proof, or a new planner result."
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    provenance_path_out = replay_dir / "render_provenance.json"
    provenance_path_out.write_text(
        json.dumps(provenance_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "render_result": render_result,
        "video_result": video_result,
        "provenance_path": provenance_path_out.as_posix(),
        "provenance_sha256": _sha256(provenance_path_out),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the render-only command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render only persisted trajectory positions and write digest metadata."""
    args = parse_args(argv)
    result = render_recorded_trace_bundle(
        args.trace,
        args.provenance,
        args.out,
        video=not args.no_video,
        fps=args.fps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["render_result"].get("status") == "rendered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
