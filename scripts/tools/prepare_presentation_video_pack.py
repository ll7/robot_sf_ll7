"""Build a small, QA'd presentation pack from benchmark-run videos.

The command consumes existing episode JSONL and native runtime video artifacts. It
does not rerun a benchmark, alter episode records, or promote a video to evidence.
The resulting media is written to a caller-selected directory; the default is
under the repository's ignored ``output/`` directory.

Example::

    uv run python scripts/tools/prepare_presentation_video_pack.py \
        --episodes output/benchmarks/20260304_184717_policy_analysis_ppo/episodes.jsonl \
        --videos output/recordings/20260304_184717_policy_analysis_ppo \
        --output output/presentation_video_pack/20260304_184717_policy_analysis_ppo
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, ImageFont, ImageStat

try:
    from scripts.generate_video_contact_sheet import generate_contact_sheet
except ImportError:  # pragma: no cover - only relevant outside the repository root
    generate_contact_sheet = None


VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
KNOWN_METRICS = (
    "near_misses",
    "min_distance",
    "comfort_exposure",
    "force_exceed_events",
    "ped_force_q95",
    "mean_distance",
)


class VideoPackError(RuntimeError):
    """Raised when the presentation pack cannot be prepared safely."""


@dataclass(frozen=True)
class Candidate:
    """A benchmark episode with a resolvable source video."""

    episode_id: str
    scenario_id: str
    seed: int | str | None
    policy: str
    outcome: str
    source_path: Path
    steps: int | None
    metrics: dict[str, Any]
    score: float
    reasons: tuple[str, ...]
    row: dict[str, Any] = field(repr=False, compare=False)

    @property
    def archetype(self) -> str:
        """Return the scenario archetype when present."""
        params = self.row.get("scenario_params")
        metadata = params.get("metadata") if isinstance(params, dict) else None
        value = metadata.get("archetype") if isinstance(metadata, dict) else None
        return str(value) if value else self.scenario_id.partition("_")[2] or self.scenario_id


@dataclass(frozen=True)
class _ReportData:
    """Inputs used to write the final presentation handoff manifest."""

    output_dir: Path
    ignored_output: bool
    max_clips: int
    source: dict[str, Any]
    snapshot: dict[str, Any]
    warnings: list[str]
    preferred_ids: list[str]
    candidate_count: int
    selected_clips: list[dict[str, Any]]
    qa_failures: list[dict[str, Any]]
    contact_sheet_path: Path | None
    command: list[str] | None


def _as_number(value: Any) -> float | None:
    """Convert finite numeric JSON values to floats."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _nested_value(row: dict[str, Any], key: str) -> Any:
    """Read a metric from the canonical metrics block or the row itself."""
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        return metrics[key]
    return row.get(key)


def _normalize_outcome(row: dict[str, Any]) -> str:
    """Normalize episode termination fields into presentation categories."""
    video_meta = row.get("video") if isinstance(row.get("video"), dict) else {}
    raw = row.get("termination_reason") or row.get("status") or video_meta.get("termination_reason")
    if raw is None:
        outcome = row.get("outcome")
        if isinstance(outcome, dict):
            if outcome.get("collision_event"):
                raw = "collision"
            elif outcome.get("route_complete"):
                raw = "success"
        if raw is None and _nested_value(row, "success") is True:
            raw = "success"
    text = str(raw or "other").strip().lower().replace("-", "_")
    if "collision" in text:
        return "collision"
    if text in {"success", "succeeded", "complete", "completed", "route_complete"}:
        return "success"
    if text in {"timeout", "max_steps", "truncated", "terminated"}:
        return "timeout"
    return "other"


def _policy_name(row: dict[str, Any]) -> str:
    """Resolve the most specific policy name available in an episode row."""
    metadata = row.get("algorithm_metadata")
    video_meta = row.get("video") if isinstance(row.get("video"), dict) else {}
    for value in (
        row.get("algo"),
        row.get("policy"),
        metadata.get("canonical_algorithm") if isinstance(metadata, dict) else None,
        video_meta.get("policy"),
    ):
        if value:
            return str(value)
    return "unknown-policy"


def _seed_value(row: dict[str, Any]) -> int | str | None:
    """Return a JSON-friendly seed value."""
    value = row.get("seed")
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        return value
    return None


def _score_episode(row: dict[str, Any], outcome: str) -> tuple[float, tuple[str, ...]]:
    """Score an episode for visual interest, with human-readable reasons.

    This is a curation heuristic, not a benchmark metric. It favors a mixture of
    outcomes, close interactions, and enough motion to make a useful clip.
    """
    score = {"collision": 60.0, "success": 35.0, "timeout": 25.0, "other": 10.0}[outcome]
    reasons = [outcome]

    near_misses = _as_number(_nested_value(row, "near_misses"))
    if near_misses is not None and near_misses > 0:
        score += min(near_misses, 35.0)
        reasons.append(f"near_misses={near_misses:g}")

    min_distance = _as_number(_nested_value(row, "min_distance"))
    if min_distance is not None:
        if min_distance < 0:
            score += 35.0
            reasons.append(f"negative_clearance={min_distance:.3g}m")
        elif min_distance < 0.25:
            score += 25.0 * (0.25 - min_distance) / 0.25
            reasons.append(f"close_clearance={min_distance:.3g}m")

    comfort = _as_number(_nested_value(row, "comfort_exposure"))
    if comfort is not None and comfort > 0:
        score += min(comfort * 80.0, 20.0)
        reasons.append(f"comfort_exposure={comfort:.3g}")

    force_events = _as_number(_nested_value(row, "force_exceed_events"))
    if force_events is not None and force_events > 0:
        score += min(force_events / 2.0, 10.0)
        reasons.append(f"force_exceed_events={force_events:g}")

    steps = _as_number(row.get("steps"))
    if steps is not None and steps > 0:
        score += min(steps / 30.0, 10.0)

    return score, tuple(reasons)


def _candidate_from_row(row: dict[str, Any], source_path: Path, row_number: int) -> Candidate:
    """Create a scored candidate from one episode row."""
    scenario_id = str(row.get("scenario_id") or row.get("scenario") or "unknown-scenario")
    seed = _seed_value(row)
    episode_id = str(
        row.get("episode_id") or f"{scenario_id}--{seed if seed is not None else row_number}"
    )
    outcome = _normalize_outcome(row)
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    steps_value = row.get("steps")
    steps = (
        int(steps_value)
        if isinstance(steps_value, int) and not isinstance(steps_value, bool)
        else None
    )
    score, reasons = _score_episode(row, outcome)
    return Candidate(
        episode_id=episode_id,
        scenario_id=scenario_id,
        seed=seed,
        policy=_policy_name(row),
        outcome=outcome,
        source_path=source_path,
        steps=steps,
        metrics=metrics,
        score=score,
        reasons=reasons,
        row=row,
    )


def _iter_episode_rows(path: Path) -> list[dict[str, Any]]:
    """Read dictionary rows from an episode JSONL file with line diagnostics."""
    if not path.is_file():
        raise VideoPackError(f"Episode JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VideoPackError(f"Episode row {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise VideoPackError(f"Episode row {line_number} is not a JSON object")
            rows.append(payload)
    return rows


def _video_meta(row: dict[str, Any]) -> dict[str, Any]:
    """Return the optional video metadata block."""
    value = row.get("video")
    return value if isinstance(value, dict) else {}


def _resolve_video_path(
    row: dict[str, Any],
    episodes_path: Path,
    videos_root: Path | None,
) -> Path | None:
    """Resolve an episode's video path, including filename-based legacy fallback."""
    raw_path = _video_meta(row).get("path")
    possible: list[Path] = []
    if isinstance(raw_path, str) and raw_path:
        if _is_url(raw_path):
            raise VideoPackError(
                "URL video references are not supported; provide a local recording path"
            )
        raw = Path(raw_path)
        possible.append(raw if raw.is_absolute() else episodes_path.parent / raw)
        if videos_root is not None:
            possible.append(videos_root / raw)
            possible.append(videos_root / raw.name)
        possible.append(Path.cwd() / raw)

    for path in possible:
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path.resolve()

    if videos_root is None:
        return None
    scenario_id = str(row.get("scenario_id") or row.get("scenario") or "unknown-scenario")
    seed = _seed_value(row)
    if seed is None:
        return None
    matches = sorted(videos_root.glob(f"{scenario_id}_seed{seed}_*.mp4"))
    if not matches:
        return None
    outcome = _normalize_outcome(row)
    exact = [path for path in matches if path.name.endswith(f"_{outcome}.mp4")]
    return (exact[0] if exact else matches[0]).resolve()


def _is_url(value: str) -> bool:
    """Return whether a recording reference is a URI rather than a local path."""
    parsed = urlsplit(value)
    return bool(parsed.scheme or value.startswith("//"))


def _candidate_sort_key(candidate: Candidate) -> tuple[float, str, str, str, str, str]:
    """Return a deterministic descending-interest ordering key."""
    return (
        -candidate.score,
        candidate.scenario_id,
        str(candidate.seed),
        candidate.policy,
        candidate.episode_id,
        candidate.source_path.name,
    )


def _choose_novel_candidate(pool: list[Candidate], selected: list[Candidate]) -> Candidate:
    """Choose the most interesting candidate while rewarding scenario novelty."""
    used_scenarios = {candidate.scenario_id for candidate in selected}
    used_archetypes = {candidate.archetype for candidate in selected}

    def key(candidate: Candidate) -> tuple[float, str, str, str, str, str]:
        novelty = (8.0 if candidate.scenario_id not in used_scenarios else 0.0) + (
            4.0 if candidate.archetype not in used_archetypes else 0.0
        )
        return (
            -(candidate.score + novelty),
            candidate.scenario_id,
            candidate.archetype,
            str(candidate.seed),
            candidate.episode_id,
            candidate.policy,
            candidate.source_path.name,
        )

    return sorted(pool, key=key)[0]


def select_candidates(candidates: list[Candidate], limit: int = 3) -> list[Candidate]:
    """Select a deterministic, outcome-balanced set of candidate episodes."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    remaining = list(candidates)
    selected: list[Candidate] = []
    successes = [candidate for candidate in remaining if candidate.outcome == "success"]
    collisions = [candidate for candidate in remaining if candidate.outcome == "collision"]

    if limit == 1:
        return sorted(remaining, key=_candidate_sort_key)[:1]

    if successes:
        success_target = min(len(successes), max(1, math.ceil(limit / 2)))
        for _ in range(success_target):
            candidate = _choose_novel_candidate(successes, selected)
            selected.append(candidate)
            remaining.remove(candidate)
            successes.remove(candidate)

    if collisions and len(selected) < limit:
        candidate = _choose_novel_candidate(collisions, selected)
        selected.append(candidate)
        remaining.remove(candidate)

    while remaining and len(selected) < limit:
        candidate = _choose_novel_candidate(remaining, selected)
        selected.append(candidate)
        remaining.remove(candidate)

    return selected


def _git_root() -> Path | None:
    """Return the current Git root when the command runs inside a checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


def _path_inside(path: Path, root: Path) -> bool:
    """Return whether path is inside root, including root itself."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git_ignored(path: Path, repo_root: Path | None) -> bool:
    """Check whether a path is covered by the checkout's ignore rules."""
    if repo_root is None or not _path_inside(path, repo_root):
        return False
    relative = path.relative_to(repo_root)
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "--", str(relative)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _ensure_safe_output(output_dir: Path, repo_root: Path | None) -> bool:
    """Reject an in-repository output path that could become tracked media."""
    if repo_root is None or not _path_inside(output_dir, repo_root):
        return False
    if not _git_ignored(output_dir, repo_root):
        raise VideoPackError(
            f"Refusing to write generated media into a non-ignored repository path: {output_dir}"
        )
    return True


def _run_command(command: list[str]) -> tuple[bool, str]:
    """Run a media command and return success plus a compact diagnostic."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""
    diagnostic = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
    return False, diagnostic[-1200:]


def _parse_number(value: Any) -> float | None:
    """Parse a numeric ffprobe field."""
    if value in (None, "", "N/A"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_frame_count(stream: dict[str, Any]) -> int | None:
    """Parse the best available decoded frame count from ffprobe."""
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        try:
            if value not in (None, "", "N/A"):
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _parse_frame_rate(value: Any) -> float | None:
    """Parse an ffprobe ratio such as ``10/1``."""
    if not isinstance(value, str) or "/" not in value:
        return _parse_number(value)
    numerator, denominator = value.split("/", maxsplit=1)
    numerator_value = _parse_number(numerator)
    denominator_value = _parse_number(denominator)
    if numerator_value is None or not denominator_value:
        return None
    return numerator_value / denominator_value


def _probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    """Read compact video metadata with ffprobe."""
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,nb_read_frames,nb_frames,r_frame_rate,avg_frame_rate,duration:format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise VideoPackError(f"Could not run ffprobe: {exc}") from exc
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout).strip()[-1200:]
        raise VideoPackError(f"ffprobe failed for {path.name}: {diagnostic}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoPackError(f"ffprobe returned invalid JSON for {path.name}: {exc}") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    stream = (
        streams[0] if isinstance(streams, list) and streams and isinstance(streams[0], dict) else {}
    )
    format_meta = payload.get("format") if isinstance(payload, dict) else {}
    format_meta = format_meta if isinstance(format_meta, dict) else {}
    duration = _parse_number(stream.get("duration")) or _parse_number(format_meta.get("duration"))
    frame_rate = _parse_frame_rate(stream.get("avg_frame_rate")) or _parse_frame_rate(
        stream.get("r_frame_rate")
    )
    size_value = format_meta.get("size")
    size_bytes = int(size_value) if str(size_value).isdigit() else path.stat().st_size
    return {
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "frame_count": _parse_frame_count(stream),
        "frame_rate": frame_rate,
        "duration_s": duration,
        "size_bytes": size_bytes,
    }


def _sample_positions(duration: float | None) -> list[tuple[str, float]]:
    """Return stable start, middle, and end sample positions."""
    if duration is None or duration <= 0:
        return [("start", 0.0)]
    return [
        ("start", 0.0),
        ("middle", max(0.0, duration * 0.5)),
        ("end", max(0.0, duration * 0.9)),
    ]


def _extract_sample(
    path: Path,
    position: float,
    output_path: Path,
    ffmpeg: str,
) -> tuple[bool, str]:
    """Extract one normalized 16:9 frame."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ss",
        f"{position:.3f}",
        "-frames:v",
        "1",
        "-vf",
        "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2:color=0x101820",
        "-y",
        str(output_path),
    ]
    return _run_command(command)


def _frame_content_stats(path: Path) -> dict[str, float]:
    """Measure whether an extracted frame contains visible scene content."""
    with Image.open(path).convert("RGB") as image:
        pixels = list(image.getdata())
        if not pixels:
            return {"mean_brightness": 0.0, "nonblack_ratio": 0.0}
        nonblack = sum(1 for pixel in pixels if max(pixel) > 12)
        mean_brightness = sum(ImageStat.Stat(image).mean) / 3.0
        return {
            "mean_brightness": round(mean_brightness, 3),
            "nonblack_ratio": round(nonblack / len(pixels), 6),
        }


def _probe_constraints(
    probe: dict[str, Any],
    *,
    min_duration: float,
) -> tuple[list[str], list[str]]:
    """Return hard QA failures and non-fatal presentation warnings."""
    reasons: list[str] = []
    warnings: list[str] = []
    duration = probe.get("duration_s")
    if duration is None or duration < min_duration:
        reasons.append(f"too_short_or_unknown_duration<{min_duration:g}s")
    frame_count = probe.get("frame_count")
    if frame_count is not None and frame_count < 2:
        reasons.append("fewer_than_two_frames")
    width, height = probe.get("width"), probe.get("height")
    if isinstance(width, int) and isinstance(height, int) and height:
        if abs((width / height) - (16 / 9)) > 0.03:
            warnings.append("source_is_not_16_to_9; presentation_copy_will_pad")
    return reasons, warnings


def _capture_sample(
    path: Path,
    label: str,
    position: float,
    temp_root: Path,
    sample_dir: Path | None,
    sample_prefix: str,
    ffmpeg: str,
) -> dict[str, Any]:
    """Extract and optionally retain one sampled frame."""
    temp_sample = temp_root / f"{label}.png"
    extracted, diagnostic = _extract_sample(path, position, temp_sample, ffmpeg)
    sample: dict[str, Any] = {"label": label, "position_s": round(position, 3)}
    if not extracted or not temp_sample.is_file():
        sample["status"] = "failed"
        sample["error"] = diagnostic or "frame_not_written"
        return sample
    sample.update(_frame_content_stats(temp_sample))
    sample["status"] = "pass"
    if sample_dir is not None:
        destination = sample_dir / f"{sample_prefix}_{label}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temp_sample, destination)
        sample["path"] = str(destination)
    return sample


def _capture_samples(
    path: Path,
    duration: float | None,
    *,
    ffmpeg: str,
    sample_dir: Path | None,
    sample_prefix: str,
) -> list[dict[str, Any]]:
    """Extract stable start, middle, and end samples for one video."""
    samples: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="robot-sf-video-qa-") as temp_dir:
        temp_root = Path(temp_dir)
        for label, position in _sample_positions(duration):
            samples.append(
                _capture_sample(
                    path,
                    label,
                    position,
                    temp_root,
                    sample_dir,
                    sample_prefix,
                    ffmpeg,
                )
            )
    return samples


def _qa_video(
    path: Path,
    *,
    ffmpeg: str,
    ffprobe: str,
    min_duration: float,
    sample_dir: Path | None = None,
    sample_prefix: str = "sample",
) -> dict[str, Any]:
    """Decode and content-check a video."""
    result: dict[str, Any] = {"path": str(path), "status": "failed", "reasons": [], "samples": []}
    if not path.is_file():
        result["reasons"].append("missing_file")
        return result
    try:
        probe = _probe_video(path, ffprobe)
    except VideoPackError as exc:
        result["reasons"].append(str(exc))
        return result
    result["probe"] = probe

    decoded, diagnostic = _run_command(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "null", "-"]
    )
    if not decoded:
        result["reasons"].append(f"decode_failed: {diagnostic}")

    reasons, warnings = _probe_constraints(probe, min_duration=min_duration)
    result["reasons"].extend(reasons)
    if warnings:
        result["warnings"] = warnings
    result["samples"] = _capture_samples(
        path,
        probe.get("duration_s"),
        ffmpeg=ffmpeg,
        sample_dir=sample_dir,
        sample_prefix=sample_prefix,
    )

    visible_samples = [
        sample
        for sample in result["samples"]
        if sample.get("status") == "pass" and sample.get("nonblack_ratio", 0.0) > 0.01
    ]
    if not visible_samples:
        result["reasons"].append("all_sampled_frames_are_blank_or_dark")
    if any(sample.get("status") != "pass" for sample in result["samples"]):
        result["reasons"].append("sampled_frame_decode_failed")
    required_visible = max(1, math.ceil(len(result["samples"]) * 2 / 3))
    if len(visible_samples) < required_visible:
        result["reasons"].append(
            f"insufficient_visible_samples<{required_visible}_of_{len(result['samples'])}"
        )
    if not result["reasons"]:
        result["status"] = "pass"
    return result


def _escape_drawtext(value: str) -> str:
    """Escape text for an ffmpeg drawtext filter value."""
    return (
        value.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace("\n", " ")
    )


def _presentation_filter(candidate: Candidate, *, with_overlay: bool) -> str:
    """Build a 16:9 scale/pad filter with optional readable labels."""
    base = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x101820"
    if not with_overlay:
        return base
    seed = str(candidate.seed) if candidate.seed is not None else "?"
    title = _escape_drawtext(f"{candidate.scenario_id} - {candidate.policy} - seed {seed}")
    timing = candidate.row.get("timing") if isinstance(candidate.row.get("timing"), dict) else {}
    duration = _as_number(timing.get("duration_s"))
    duration_text = f"{duration:.1f}s" if duration is not None else "duration unknown"
    outcome = _escape_drawtext(
        f"{candidate.outcome.upper()} - {candidate.steps or '?'} steps - {duration_text}"
    )
    return (
        f"{base},drawtext=font=Sans:text='{title}':x=24:y=24:fontsize=28:fontcolor=white:"
        "box=1:boxcolor=0x101820CC:boxborderw=12,"
        f"drawtext=font=Sans:text='{outcome}':x=24:y=h-th-28:fontsize=24:fontcolor=white:"
        "box=1:boxcolor=0x101820CC:boxborderw=10"
    )


def _polish_video(
    candidate: Candidate,
    output_path: Path,
    ffmpeg: str,
    *,
    no_polish: bool,
) -> dict[str, Any]:
    """Write a presentation copy with stable dimensions and labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if no_polish:
        shutil.copy2(candidate.source_path, output_path)
        return {"status": "copied", "overlay": False}

    def encode(with_overlay: bool) -> tuple[bool, str]:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(candidate.source_path),
            "-vf",
            _presentation_filter(candidate, with_overlay=with_overlay),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output_path),
        ]
        return _run_command(command)

    encoded, diagnostic = encode(with_overlay=True)
    if encoded:
        return {"status": "encoded", "overlay": True}
    fallback, fallback_diagnostic = encode(with_overlay=False)
    if fallback:
        return {
            "status": "encoded_with_warning",
            "overlay": False,
            "warning": f"overlay_failed: {diagnostic or fallback_diagnostic}",
        }
    raise VideoPackError(
        f"Could not encode presentation copy for {candidate.source_path.name}: {fallback_diagnostic}"
    )


def _font(size: int) -> ImageFont.ImageFont:
    """Load a readable local font, falling back to Pillow's default."""
    candidates = (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _truncate(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, width: int) -> str:
    """Truncate a caption to the available pixel width."""
    if draw.textbbox((0, 0), text, font=font)[2] <= width:
        return text
    suffix = "..."
    while text and draw.textbbox((0, 0), text + suffix, font=font)[2] > width:
        text = text[:-1]
    return text + suffix


def _annotate_poster(source: Path, destination: Path, candidate: Candidate, index: int) -> None:
    """Add a compact caption banner to a contact-sheet poster."""
    with Image.open(source).convert("RGB") as image:
        banner_height = 66
        canvas = Image.new("RGB", (image.width, image.height + banner_height), color=(16, 24, 32))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        accent = (46, 190, 120) if candidate.outcome == "success" else (220, 80, 80)
        draw.rectangle(
            (0, image.height, image.width, image.height + banner_height), fill=(16, 24, 32)
        )
        draw.rectangle((0, image.height, 8, image.height + banner_height), fill=accent)
        title_font = _font(20)
        detail_font = _font(17)
        title = _truncate(
            f"{index:02d}  {candidate.scenario_id}", draw, title_font, image.width - 24
        )
        detail = _truncate(
            f"{candidate.outcome} | seed {candidate.seed if candidate.seed is not None else '?'} | {candidate.policy}",
            draw,
            detail_font,
            image.width - 24,
        )
        draw.text((16, image.height + 8), title, font=title_font, fill=(245, 248, 250))
        draw.text((16, image.height + 36), detail, font=detail_font, fill=(190, 205, 215))
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)


def _sha256(path: Path) -> str:
    """Hash a generated or source artifact in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    """Return a filesystem-safe, bounded artifact stem."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (slug or "clip")[:96]


def _portable_source_name(path: Path, videos_root: Path | None) -> str:
    """Return a stable source label without exposing machine-specific directories."""
    if videos_root is not None:
        try:
            return (Path("videos") / path.relative_to(videos_root)).as_posix()
        except ValueError:
            pass
    return path.name


def _portable_qa(qa: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Remove absolute local paths from a QA record while retaining diagnostics."""
    portable = copy.deepcopy(qa)
    if "path" in portable:
        portable["path"] = Path(str(portable["path"])).name
    for sample in portable.get("samples", []):
        if "path" in sample:
            try:
                sample["path"] = (
                    Path(str(sample["path"])).resolve().relative_to(output_dir).as_posix()
                )
            except ValueError:
                sample["path"] = Path(str(sample["path"])).name
    return portable


def _candidate_record(candidate: Candidate, videos_root: Path | None) -> dict[str, Any]:
    """Serialize the useful provenance and curation fields for one candidate."""
    video_meta = _video_meta(candidate.row)
    metrics = {key: candidate.metrics[key] for key in KNOWN_METRICS if key in candidate.metrics}
    return {
        "episode_id": candidate.episode_id,
        "scenario_id": candidate.scenario_id,
        "seed": candidate.seed,
        "policy": candidate.policy,
        "outcome": candidate.outcome,
        "steps": candidate.steps,
        "selection_score": round(candidate.score, 3),
        "selection_reasons": list(candidate.reasons),
        "metrics": metrics,
        "source_file": _portable_source_name(candidate.source_path, videos_root),
        "source_renderer": video_meta.get("renderer"),
        "recorded_frame_count": video_meta.get("frames"),
    }


def _source_provenance(rows: list[dict[str, Any]], episodes_path: Path) -> dict[str, Any]:
    """Summarize provenance fields carried by the input episode records."""
    git_hashes = sorted({str(row["git_hash"]) for row in rows if row.get("git_hash")})
    config_hashes = sorted({str(row["config_hash"]) for row in rows if row.get("config_hash")})
    return {
        "episodes_file": episodes_path.name,
        "episode_count": len(rows),
        "source_git_hashes": git_hashes,
        "source_config_hashes": config_hashes,
        "source_git_hash_status": "complete" if git_hashes else "missing",
        "rights": {
            "status": "redistribution-unknown",
            "basis": "local-only-byo",
            "note": "Local staging and checksums do not establish redistribution or publication rights.",
        },
    }


def _git_snapshot(repo_root: Path | None) -> dict[str, Any]:
    """Capture the current checkout identity without mutating it."""
    if repo_root is None:
        return {"repository": None, "head": None, "dirty": None}
    head_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    return {
        "repository": None,
        "head": head_result.stdout.strip() if head_result.returncode == 0 else None,
        "dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic, human-readable JSON metadata."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    """Write a short presentation handoff note next to the pack."""
    clips = report.get("clips", [])
    lines = [
        "# Presentation video pack",
        "",
        "This pack was prepared from existing benchmark-run videos; it did not rerun the benchmark.",
        "The videos are for presentation only, not benchmark or paper-facing evidence.",
        "",
        f"Status: **{report.get('status', 'unknown')}**",
        f"Source episodes file: `{report['source']['episodes_file']}`",
        f"Source commit(s): `{', '.join(report['source'].get('source_git_hashes', [])) or 'missing'}`",
        "",
        "## Clips",
        "",
    ]
    if not clips:
        lines.append("No clip passed media QA. See `presentation_video_pack.json` for diagnostics.")
    for clip in clips:
        lines.append(
            f"- `{clip['presentation_path']}` — {clip['scenario_id']}, {clip['outcome']}, "
            f"seed {clip['seed']}, {clip['qa']['probe'].get('duration_s', '?')} s"
        )
    lines.extend(
        [
            "",
            "Open `contact_sheet.png` for a quick visual check. `stills/` contains the sampled frames used by QA.",
            "",
            "Generated media is intentionally local and untracked; the output directory is covered by the repository's ignore policy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_candidates(
    rows: list[dict[str, Any]],
    episodes_path: Path,
    videos_root: Path | None,
) -> tuple[list[Candidate], int]:
    """Resolve episode rows to source videos and count unresolved rows."""
    candidates: list[Candidate] = []
    missing_video_rows = 0
    for row_number, row in enumerate(rows, start=1):
        source_path = _resolve_video_path(row, episodes_path, videos_root)
        if source_path is None:
            missing_video_rows += 1
            continue
        candidates.append(_candidate_from_row(row, source_path, row_number))
    return candidates, missing_video_rows


def _build_warnings(
    source: dict[str, Any],
    snapshot: dict[str, Any],
    missing_video_rows: int,
    ffmpeg: str | None,
    ffprobe: str | None,
) -> list[str]:
    """Build provenance and environment warnings for the handoff manifest."""
    warnings: list[str] = []
    if source["source_git_hashes"] and snapshot.get("head") not in source["source_git_hashes"]:
        warnings.append(
            "source benchmark commit differs from current checkout; no rerun was performed"
        )
    if not source["source_git_hashes"]:
        warnings.append("episode records do not carry a source git hash")
    if missing_video_rows:
        warnings.append(f"{missing_video_rows} episode rows had no resolvable video path")
    if not ffmpeg or not ffprobe:
        warnings.append("ffmpeg and ffprobe are required for media QA")
    return warnings


def _failure_record(
    candidate: Candidate,
    qa: dict[str, Any],
    *,
    output_dir: Path,
    videos_root: Path | None,
) -> dict[str, Any]:
    """Serialize one candidate that failed source or presentation QA."""
    return {
        "episode_id": candidate.episode_id,
        "source_file": _portable_source_name(candidate.source_path, videos_root),
        "qa": _portable_qa(qa, output_dir),
    }


def _process_candidate(  # noqa: PLR0913 - explicit media-processing context
    candidate: Candidate,
    index: int,
    *,
    clips_dir: Path,
    stills_dir: Path,
    ffmpeg: str,
    ffprobe: str,
    min_duration: float,
    no_polish: bool,
    videos_root: Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """QA, polish, and serialize one candidate, returning a failure if needed."""
    source_qa = _qa_video(
        candidate.source_path,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        min_duration=min_duration,
    )
    if source_qa["status"] != "pass":
        return None, _failure_record(
            candidate, source_qa, output_dir=clips_dir.parent, videos_root=videos_root
        )

    stem = _slug(f"{candidate.scenario_id}_seed{candidate.seed}_{candidate.outcome}")
    presentation_path = clips_dir / f"{index:02d}_{stem}.mp4"
    try:
        encoding = _polish_video(candidate, presentation_path, ffmpeg, no_polish=no_polish)
    except VideoPackError as exc:
        return None, _failure_record(
            candidate,
            {"status": "failed", "reasons": [str(exc)]},
            output_dir=clips_dir.parent,
            videos_root=videos_root,
        )

    prefix = f"{index:02d}_{stem}"
    presentation_qa = _qa_video(
        presentation_path,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        min_duration=min_duration,
        sample_dir=stills_dir,
        sample_prefix=prefix,
    )
    if presentation_qa["status"] != "pass":
        failure = _failure_record(
            candidate, presentation_qa, output_dir=clips_dir.parent, videos_root=videos_root
        )
        failure["presentation_path"] = str(presentation_path.relative_to(clips_dir.parent))
        return None, failure

    middle_sample = next(
        (
            sample.get("path")
            for sample in presentation_qa["samples"]
            if sample["label"] == "middle"
        ),
        None,
    )
    if not middle_sample:
        return None, _failure_record(
            candidate,
            {"status": "failed", "reasons": ["middle_sample_missing"]},
            output_dir=clips_dir.parent,
            videos_root=videos_root,
        )
    poster_path = stills_dir / f"{prefix}_poster.png"
    _annotate_poster(Path(middle_sample), poster_path, candidate, index)
    clip_record = _candidate_record(candidate, videos_root)
    clip_record.update(
        {
            "source_sha256": _sha256(candidate.source_path),
            "source_qa": _portable_qa(source_qa, clips_dir.parent),
            "encoding": encoding,
            "presentation_path": str(presentation_path.relative_to(clips_dir.parent)),
            "presentation_sha256": _sha256(presentation_path),
            "qa": _portable_qa(presentation_qa, clips_dir.parent),
            "poster_path": str(poster_path.relative_to(stills_dir.parent)),
        }
    )
    return clip_record, None


def _process_candidates(
    process_order: list[Candidate],
    *,
    output_dir: Path,
    max_clips: int,
    min_duration: float,
    no_polish: bool,
    ffmpeg: str | None,
    ffprobe: str | None,
    videos_root: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process preferred candidates and deterministic fallbacks until full."""
    selected_clips: list[dict[str, Any]] = []
    qa_failures: list[dict[str, Any]] = []
    if not ffmpeg or not ffprobe:
        return selected_clips, qa_failures
    for candidate in process_order:
        if len(selected_clips) >= max_clips:
            break
        clip, failure = _process_candidate(
            candidate,
            len(selected_clips) + 1,
            clips_dir=output_dir / "clips",
            stills_dir=output_dir / "stills",
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            videos_root=videos_root,
            min_duration=min_duration,
            no_polish=no_polish,
        )
        if clip is not None:
            selected_clips.append(clip)
        if failure is not None:
            qa_failures.append(failure)
    return selected_clips, qa_failures


def _make_contact_sheet(output_dir: Path, clips: list[dict[str, Any]]) -> Path | None:
    """Assemble the annotated posters with the canonical contact-sheet helper."""
    if not clips or generate_contact_sheet is None:
        return None
    sources = [
        {
            "episode_id": clip["episode_id"],
            "frame_paths": [str(output_dir / clip["poster_path"])],
        }
        for clip in clips
    ]
    contact_sheet_path = output_dir / "contact_sheet.png"
    with tempfile.TemporaryDirectory(prefix="robot-sf-contact-sheet-") as temp_dir:
        sources_path = Path(temp_dir) / "sources.jsonl"
        sources_path.write_text(
            "\n".join(json.dumps(source_row) for source_row in sources) + "\n", encoding="utf-8"
        )
        generate_contact_sheet(sources_path, contact_sheet_path, columns=min(3, len(clips)))
    return contact_sheet_path


def _build_report(data: _ReportData) -> dict[str, Any]:
    """Create the machine-readable presentation handoff manifest."""
    status = (
        "ready"
        if len(data.selected_clips) == data.max_clips
        else "partial"
        if data.selected_clips
        else "blocked"
    )
    return {
        "schema_version": "presentation-video-pack.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "command": "prepare_presentation_video_pack",
        "status": status,
        "source": data.source,
        "current_checkout": data.snapshot,
        "artifact_policy": {
            "claim_scope": "presentation_only_not_benchmark_evidence",
            "media_disposition": "local_presentation_only",
            "videos_tracked": False,
            "output_dir_git_ignored": data.ignored_output,
        },
        "selection": {
            "requested_clips": data.max_clips,
            "available_video_candidates": data.candidate_count,
            "preferred_candidate_ids": data.preferred_ids,
            "selected_episode_ids": [clip["episode_id"] for clip in data.selected_clips],
        },
        "clips": data.selected_clips,
        "qa_failures": data.qa_failures,
        "contact_sheet": str(data.contact_sheet_path.relative_to(data.output_dir))
        if data.contact_sheet_path
        else None,
        "warnings": data.warnings,
    }


def prepare_pack(
    episodes_path: Path,
    *,
    videos_root: Path | None = None,
    output_dir: Path | None = None,
    max_clips: int = 3,
    min_duration: float = 1.0,
    no_polish: bool = False,
    command: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Prepare the presentation pack and return its manifest path and payload."""
    if max_clips <= 0:
        raise ValueError("max_clips must be positive")
    if min_duration < 0:
        raise ValueError("min_duration cannot be negative")
    episodes_path = episodes_path.resolve()
    repo_root = _git_root()
    if output_dir is None:
        if repo_root is not None:
            output_dir = repo_root / "output" / "presentation_video_pack" / episodes_path.stem
        else:
            output_dir = Path(tempfile.mkdtemp(prefix="robot-sf-presentation-video-pack-"))
    output_dir = output_dir.resolve()
    ignored_output = _ensure_safe_output(output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if videos_root is not None:
        videos_root = videos_root.resolve()
        if not videos_root.is_dir():
            raise VideoPackError(f"Video directory not found: {videos_root}")
    rows = _iter_episode_rows(episodes_path)
    candidates, missing_video_rows = _discover_candidates(rows, episodes_path, videos_root)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    source = _source_provenance(rows, episodes_path)
    snapshot = _git_snapshot(repo_root)
    warnings = _build_warnings(source, snapshot, missing_video_rows, ffmpeg, ffprobe)

    preferred = select_candidates(candidates, limit=max_clips) if candidates else []
    preferred_ids = [candidate.episode_id for candidate in preferred]
    fallback = [
        candidate
        for candidate in sorted(candidates, key=_candidate_sort_key)
        if candidate not in preferred
    ]
    process_order = preferred + fallback
    selected_clips, qa_failures = _process_candidates(
        process_order,
        output_dir=output_dir,
        max_clips=max_clips,
        min_duration=min_duration,
        no_polish=no_polish,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        videos_root=videos_root,
    )
    contact_sheet_path = _make_contact_sheet(output_dir, selected_clips)
    if selected_clips and contact_sheet_path is None:
        warnings.append("contact-sheet helper could not be imported")
    report = _build_report(
        _ReportData(
            output_dir=output_dir,
            ignored_output=ignored_output,
            max_clips=max_clips,
            source=source,
            snapshot=snapshot,
            warnings=warnings,
            preferred_ids=preferred_ids,
            candidate_count=len(candidates),
            selected_clips=selected_clips,
            qa_failures=qa_failures,
            contact_sheet_path=contact_sheet_path,
            command=command,
        )
    )
    manifest_path = output_dir / "presentation_video_pack.json"
    _write_json(manifest_path, report)
    _write_readme(output_dir / "README.md", report)
    return manifest_path, report


def _build_parser() -> argparse.ArgumentParser:
    """Build the presentation pack CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", required=True, type=Path, help="Episode JSONL from the benchmark run"
    )
    parser.add_argument(
        "--videos",
        type=Path,
        help="Directory containing native runtime videos when episode paths need fallback resolution",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ignored or external output directory; defaults to output/presentation_video_pack/<episode stem>",
    )
    parser.add_argument(
        "--max-clips", type=int, default=3, help="Number of clips to prepare (default: 3)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum accepted presentation duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--no-polish",
        action="store_true",
        help="Copy accepted source videos without scale/pad/label re-encoding",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the presentation pack CLI."""
    args = _build_parser().parse_args(argv)
    if args.max_clips <= 0:
        print("error: --max-clips must be positive", file=sys.stderr)
        return 2
    if args.min_duration < 0:
        print("error: --min-duration cannot be negative", file=sys.stderr)
        return 2
    try:
        manifest_path, report = prepare_pack(
            args.episodes,
            videos_root=args.videos,
            output_dir=args.output,
            max_clips=args.max_clips,
            min_duration=args.min_duration,
            no_polish=args.no_polish,
            command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
        )
    except (VideoPackError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "manifest": str(manifest_path)}))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
