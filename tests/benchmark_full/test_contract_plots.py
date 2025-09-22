"""Contract test T012 for `generate_plots`.

Expectation (final):
  - In smoke mode produces a subset of PDF plots (e.g., distributions_*.pdf) in output dir.
  - Returns list of PlotArtifact objects with status 'generated' or 'skipped'.

Current state: After T035 basic implementation, expect at least one PDF plot generated in smoke mode.
"""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.full_classic.plots import generate_plots


def test_generate_plots_smoke(temp_results_dir, synthetic_episode_record):
    out_dir = Path(temp_results_dir) / "plots"
    records = [
        synthetic_episode_record(
            episode_id="ep1",
            scenario_id="scenario_a",
            seed=1,
        )
    ]

    class _Metric:
        def __init__(self, name, mean):
            self.name = name
            self.mean = mean
            self.median = mean
            self.p95 = mean
            self.mean_ci = None
            self.median_ci = None

    class _Group:
        def __init__(self):
            self.archetype = "crossing"
            self.density = "low"
            self.count = 1
            self.metrics = {"collision_rate": _Metric("collision_rate", 0.0)}

    groups = [_Group()]

    class _Cfg:
        smoke = True

    artifacts = generate_plots(groups, records, str(out_dir), _Cfg())
    # Basic assertions: returns list, creates directory, at least one PDF file
    assert isinstance(artifacts, list)
    assert out_dir.exists()
    pdfs = list(out_dir.glob("*.pdf"))
    assert pdfs, "Expected at least one PDF plot in smoke mode"
    # Each artifact expected to contain minimally these keys/attributes
    for art in artifacts:
        # Accept both dict (future) or simple object with attributes
        kind = getattr(art, "kind", None) or (art.get("kind") if isinstance(art, dict) else None)
        status = getattr(art, "status", None) or (
            art.get("status") if isinstance(art, dict) else None
        )
        assert kind in {"distribution", "trajectory", "kde", "pareto", "force_heatmap"}
        assert status in {"generated", "skipped"}
