"""Tests for the static scenario/planner gallery (issue #5796).

These tests assert the acceptance contract from the issue: ``gallery build``
produces a static page listing all scenarios with thumbnail + metadata + a
runnable command, and the generated page references every scenario in the
manifest. They run CPU-only with headless matplotlib (configured in conftest).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.cli import main as robot_sf_main
from robot_sf.gallery.builder import (
    GALLERY_HTML_SCHEMA_VERSION,
    GalleryBuildResult,
    build_gallery,
    estimate_expected_runtime_seconds,
    resolve_supported_planners,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# A small, representative abstract scenario matrix mirroring
# configs/baselines/example_matrix.yaml (kept inline so the test is hermetic).
SAMPLE_SCENARIOS = [
    {
        "id": "gallery-test-uni-low-open",
        "density": "low",
        "flow": "uni",
        "obstacle": "open",
        "groups": 0.0,
        "speed_var": "low",
        "goal_topology": "point",
        "robot_context": "embedded",
        "repeats": 1,
    },
    {
        "id": "gallery-test-bi-med-bottleneck",
        "density": "med",
        "flow": "bi",
        "obstacle": "bottleneck",
        "groups": 0.2,
        "speed_var": "high",
        "goal_topology": "point",
        "robot_context": "embedded",
        "repeats": 1,
    },
    {
        "id": "gallery-test-cross-high-maze",
        "density": "high",
        "flow": "cross",
        "obstacle": "maze",
        "groups": 0.4,
        "speed_var": "high",
        "goal_topology": "swap",
        "robot_context": "ahead",
        "repeats": 1,
    },
]


def _build(tmp_path: Path, *, render_thumbnails: bool = True) -> GalleryBuildResult:
    """Build a gallery from the inline sample scenarios into a tmp dir."""
    return build_gallery(
        SAMPLE_SCENARIOS,
        matrix_path="configs/baselines/example_matrix.yaml",
        out_dir=tmp_path / "gallery",
        base_seed=0,
        horizon_steps=100,
        render_thumbnails=render_thumbnails,
        embed_thumbnails=True,
    )


def test_build_gallery_generates_html_and_manifest(tmp_path: Path) -> None:
    """Write portable HTML and a machine-readable manifest for offline inspection."""
    result = _build(tmp_path)

    assert result.html_path.is_file()
    assert result.manifest_path.is_file()
    assert result.schema_version == GALLERY_HTML_SCHEMA_VERSION
    assert len(result.cards) == len(SAMPLE_SCENARIOS)


def test_generated_page_references_every_scenario(tmp_path: Path) -> None:
    """Keep every manifest scenario visible so gallery discovery does not silently omit rows."""
    result = _build(tmp_path)
    html_text = result.html_path.read_text(encoding="utf-8")

    for scenario in SAMPLE_SCENARIOS:
        assert scenario["id"] in html_text, f"scenario id missing from page: {scenario['id']}"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest_ids = {card["scenario_id"] for card in manifest["cards"]}
    assert manifest_ids == {s["id"] for s in SAMPLE_SCENARIOS}
    assert manifest["scenario_count"] == len(SAMPLE_SCENARIOS)


def test_each_card_has_required_metadata_fields(tmp_path: Path) -> None:
    """Expose actionable metadata and a runnable per-scenario command on every card."""
    result = _build(tmp_path)
    for card in result.cards:
        # A run-this command is present and references the matrix + algo.
        assert "robot_sf_bench run" in card.run_command
        assert "simple_policy" in card.run_command
        assert "--out" in card.run_command
        assert f"--scenario-id {card.scenario_id}" in card.run_command
        # Pedestrian count is a non-negative int derived from density.
        assert isinstance(card.pedestrian_count, int)
        assert card.pedestrian_count >= 0
        # Supported planners is the documented non-empty constant set.
        planners = card.supported_planners
        assert len(planners) >= 1
        assert planners == resolve_supported_planners()
        # Expected runtime is a positive estimate (not a benchmark claim).
        assert card.expected_runtime_seconds > 0
        # Difficulty band/score are populated.
        assert card.difficulty_band in {"low", "medium", "high"}
        assert 0.0 <= card.difficulty_score <= 1.0
        # Map name is a non-empty string.
        assert isinstance(card.map_name, str) and card.map_name
        # Thumbnail was rendered and embedded (we requested thumbnails).
        assert card.thumbnail_relpath is not None


def test_embedded_thumbnails_make_page_self_contained(tmp_path: Path) -> None:
    """Embed images so copied HTML remains usable without external thumbnail files."""
    result = _build(tmp_path, render_thumbnails=True)
    html_text = result.html_path.read_text(encoding="utf-8")
    # As many data-URI images as rendered cards.
    assert html_text.count("data:image/png;base64,") == len(result.cards)
    # The thumbnails directory exists with one PNG per unique scenario.
    pngs = list(result.thumbnail_dir.glob("*.png"))
    assert len(pngs) == len(SAMPLE_SCENARIOS)


def test_no_thumbnails_mode_emits_placeholders(tmp_path: Path) -> None:
    """Keep headless gallery builds useful when thumbnail rendering is unavailable."""
    result = _build(tmp_path, render_thumbnails=False)
    assert result.html_path.is_file()
    html_text = result.html_path.read_text(encoding="utf-8")
    # No embedded image data when thumbnails are skipped.
    assert "data:image/png;base64," not in html_text
    # Placeholder CSS class is present.
    assert "thumb-missing" in html_text
    # Every scenario is still listed.
    for scenario in SAMPLE_SCENARIOS:
        assert scenario["id"] in html_text
    # No thumbnail directory contents were created.
    assert not result.thumbnail_dir.exists() or not list(result.thumbnail_dir.glob("*.png"))


def test_empty_scenarios_fail_closed(tmp_path: Path) -> None:
    """Reject empty input so users never mistake a blank gallery for valid output."""
    with pytest.raises(ValueError):
        build_gallery([], matrix_path="x.yaml", out_dir=tmp_path / "g")


def test_runtime_estimate_is_deterministic_and_positive() -> None:
    """Keep estimates stable and reject invalid horizons rather than inventing input values."""
    low = estimate_expected_runtime_seconds(10, horizon_steps=100)
    high = estimate_expected_runtime_seconds(40, horizon_steps=100)
    assert low > 0
    assert high > low
    # Deterministic: same inputs -> same output.
    assert estimate_expected_runtime_seconds(40, horizon_steps=100) == high
    # Zero/invalid pedestrian counts are floored, not negative.
    assert estimate_expected_runtime_seconds(0) >= 0.1
    assert estimate_expected_runtime_seconds(-5) >= 0.1
    with pytest.raises(ValueError, match="horizon_steps"):
        estimate_expected_runtime_seconds(10, horizon_steps=0)


def test_sanitized_scenario_id_collisions_fail_closed() -> None:
    """Reject distinct labels that would overwrite each other in gallery artifacts."""
    with pytest.raises(ValueError, match="both sanitize"):
        build_gallery(
            [{"id": "a/b"}, {"id": "a_b"}],
            matrix_path="matrix.yaml",
            render_thumbnails=False,
        )


def test_cli_gallery_build_against_real_example_matrix(tmp_path: Path) -> None:
    """The ``robot-sf gallery build`` CLI builds a page from the shipped example matrix.

    This exercises the real load_scenario_matrix path end-to-end (acceptance:
    ``gallery build`` produces a static page listing all scenarios).
    """
    matrix = REPO_ROOT / "configs/baselines/example_matrix.yaml"
    if not matrix.is_file():
        pytest.skip(f"shipped matrix not present in this checkout: {matrix}")
    out_dir = tmp_path / "cli_gallery"

    rc = robot_sf_main(
        [
            "gallery",
            "build",
            "--matrix",
            str(matrix),
            "--out-dir",
            str(out_dir),
            "--base-seed",
            "0",
            "--horizon",
            "100",
            "--link-thumbnails",  # avoid large base64 blobs in CLI test
        ]
    )
    assert rc == 0

    html = out_dir / "index.html"
    manifest = out_dir / "gallery_manifest.json"
    assert html.is_file()
    assert manifest.is_file()

    # Parse the stdout JSON summary.
    # (main() prints to stdout; we re-read the manifest which carries the same facts.)
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["schema_version"] == GALLERY_HTML_SCHEMA_VERSION
    assert manifest_data["scenario_count"] >= 1

    html_text = html.read_text(encoding="utf-8")
    for card in manifest_data["cards"]:
        assert card["scenario_id"] in html_text


def test_cli_gallery_build_bad_matrix_returns_2(tmp_path: Path) -> None:
    """Return a clean CLI error for bad input instead of exposing a traceback."""
    rc = robot_sf_main(
        [
            "gallery",
            "build",
            "--matrix",
            str(tmp_path / "definitely-not-a-matrix.yaml"),
            "--out-dir",
            str(tmp_path / "nope"),
        ]
    )
    assert rc == 2


def test_sample_rollout_link_discovered_when_present(tmp_path: Path) -> None:
    """Stage external rollouts inside the gallery before emitting safe relative links."""
    rollout_root = tmp_path / "rollouts"
    rollout_root.mkdir()
    sid = "gallery-test-uni-low-open"
    rollout_file = rollout_root / f"{sid}.mp4"
    rollout_file.write_bytes(b"\x00\x00\x00\x00")  # sentinel bytes

    result = build_gallery(
        SAMPLE_SCENARIOS,
        matrix_path="configs/baselines/example_matrix.yaml",
        out_dir=tmp_path / "gallery",
        sample_rollout_root=rollout_root,
        render_thumbnails=False,
    )
    card = next(c for c in result.cards if c.scenario_id == sid)
    assert card.sample_rollout_relpath is not None
    assert card.sample_rollout_relpath == f"rollouts/{sid}.mp4"
    assert (result.html_path.parent / card.sample_rollout_relpath).is_file()
    html_text = result.html_path.read_text(encoding="utf-8")
    assert "view sample rollout" in html_text
