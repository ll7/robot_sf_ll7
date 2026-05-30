"""Tests for publication-bundle planner tradeoff plotting."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.planner_tradeoff_plot import (
    bootstrap_seed_mean_ci,
    build_tradeoff_points,
    save_planner_tradeoff_figure,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL rows for a tiny publication-bundle fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _episode(planner: str, seed: int, success: bool, collision: bool) -> dict:
    """Build a minimal episode row with April-2026 collision schema semantics."""
    return {
        "episode_id": f"{planner}-{seed}-{success}-{collision}",
        "scenario_id": "tradeoff-smoke",
        "seed": seed,
        "termination_reason": "success" if success else "max_steps",
        "outcome": {"collision_event": collision},
        "scenario_params": {"algo": planner},
        "metrics": {
            "success": success,
            # Deliberately zero to prove plotting uses outcome.collision_event.
            "collisions": 0,
        },
    }


def _write_bundle(root: Path) -> Path:
    """Create a small publication-bundle fixture."""
    bundle = root / "tiny_publication_bundle"
    reports = bundle / "payload" / "reports"
    reports.mkdir(parents=True)
    with (reports / "campaign_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["planner_key", "success_mean", "collisions_mean"],
        )
        writer.writeheader()
        writer.writerow({"planner_key": "orca", "success_mean": "0.75", "collisions_mean": "0.0"})
        writer.writerow({"planner_key": "ppo", "success_mean": "0.5", "collisions_mean": "0.5"})
        writer.writerow({"planner_key": "goal", "success_mean": "0.25", "collisions_mean": "0.0"})

    _write_jsonl(
        bundle / "payload" / "runs" / "orca__differential_drive" / "episodes.jsonl",
        [
            _episode("orca", 1, True, False),
            _episode("orca", 1, True, False),
            _episode("orca", 2, True, False),
            _episode("orca", 2, False, False),
        ],
    )
    _write_jsonl(
        bundle / "payload" / "runs" / "ppo__differential_drive" / "episodes.jsonl",
        [
            _episode("ppo", 1, True, True),
            _episode("ppo", 1, False, True),
            _episode("ppo", 2, True, False),
            _episode("ppo", 2, False, False),
        ],
    )
    _write_jsonl(
        bundle / "payload" / "runs" / "goal__differential_drive" / "episodes.jsonl",
        [
            _episode("goal", 1, False, False),
            _episode("goal", 2, True, False),
        ],
    )
    (bundle / "publication_manifest.json").write_text(
        json.dumps({"provenance": {"run_id": "tiny-run"}}) + "\n",
        encoding="utf-8",
    )
    return bundle


def test_build_tradeoff_points_uses_outcome_collision_event(tmp_path: Path) -> None:
    """Planner CIs should use outcome.collision_event, not stale metrics.collisions."""
    bundle = _write_bundle(tmp_path)
    points = build_tradeoff_points(bundle, bootstrap_samples=50, bootstrap_seed=7)
    by_key = {point.planner_key: point for point in points}

    assert by_key["ppo"].role == "headline"
    assert by_key["goal"].role == "control"
    assert by_key["ppo"].collision_mean == 0.5
    assert by_key["ppo"].collision_ci is not None
    assert by_key["ppo"].collision_ci[1] > 0.0


def test_bootstrap_seed_mean_ci_is_deterministic(tmp_path: Path) -> None:
    """Bootstrap CIs should be reproducible for a fixed seed."""
    bundle = _write_bundle(tmp_path)
    episodes = bundle / "payload" / "runs" / "ppo__differential_drive" / "episodes.jsonl"

    ci1 = bootstrap_seed_mean_ci(episodes, "collisions", samples=100, seed=123)
    ci2 = bootstrap_seed_mean_ci(episodes, "collisions", samples=100, seed=123)

    assert ci1 == ci2
    assert ci1[0] <= ci1[1]


def test_save_planner_tradeoff_figure_and_cli(tmp_path: Path, capsys) -> None:
    """Both the library helper and CLI should write PNG/PDF plus metadata."""
    bundle = _write_bundle(tmp_path)
    out_png = tmp_path / "tradeoff.png"
    out_pdf = tmp_path / "tradeoff.pdf"
    meta = save_planner_tradeoff_figure(
        bundle,
        out_png=out_png,
        out_pdf=out_pdf,
        bootstrap_samples=20,
    )

    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert meta["run_id"] == "tiny-run"

    cli_png = tmp_path / "cli_tradeoff.png"
    cli_meta = tmp_path / "cli_tradeoff.json"
    rc = cli_main(
        [
            "plot-planner-tradeoff",
            "--bundle-path",
            str(bundle),
            "--out",
            str(cli_png),
            "--metadata-out",
            str(cli_meta),
            "--bootstrap-samples",
            "20",
        ]
    )
    cap = capsys.readouterr()
    assert rc == 0, f"plot-planner-tradeoff failed: {cap.err}"
    assert cli_png.exists() and cli_png.stat().st_size > 0
    payload = json.loads(cli_meta.read_text(encoding="utf-8"))
    assert payload["run_id"] == "tiny-run"
