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
    """Verify smoke-mode plotting writes a PDF and reports artifact kind/status.

    Args:
        temp_results_dir: Temporary directory receiving the plot output.
        synthetic_episode_record: Factory building the single episode record.
    """
    out_dir = Path(temp_results_dir) / "plots"
    records = [
        synthetic_episode_record(
            episode_id="ep1",
            scenario_id="scenario_a",
            seed=1,
        ),
    ]

    class _Metric:
        """Minimal metric double carrying mean and percentile fields."""

        def __init__(self, name, mean):
            """Store the metric identity and mean used across all aggregates.

            Args:
                name: Metric identifier.
                mean: Central value mirrored into median and p95.
            """
            self.name = name
            self.mean = mean
            self.median = mean
            self.p95 = mean
            self.mean_ci = None
            self.median_ci = None

    class _Group:
        """Minimal aggregate-group double with rate metrics for one archetype."""

        def __init__(self):
            """Initialize a crossing/low group with collision and success metrics."""
            self.archetype = "crossing"
            self.density = "low"
            self.count = 1
            self.metrics = {
                "collision_rate": _Metric("collision_rate", 0.0),
                "success_rate": _Metric("success_rate", 1.0),
            }

    groups = [_Group()]

    class _Cfg:
        """Minimal config stub enabling smoke mode."""

        smoke = True

    artifacts = generate_plots(groups, records, str(out_dir), _Cfg())
    # Basic assertions: returns list, creates directory, at least one PDF file
    assert isinstance(artifacts, list)
    assert out_dir.exists()
    pdfs = list(out_dir.glob("*.pdf"))
    assert pdfs, "Expected at least one PDF plot in smoke mode"
    pdf_names = {path.name for path in pdfs}
    assert "success_collision_scatter.pdf" in pdf_names
    assert "pareto_placeholder.pdf" not in pdf_names
    # Each artifact expected to contain minimally these keys/attributes
    for art in artifacts:
        # Accept both dict (future) or simple object with attributes
        kind = getattr(art, "kind", None) or (art.get("kind") if isinstance(art, dict) else None)
        status = getattr(art, "status", None) or (
            art.get("status") if isinstance(art, dict) else None
        )
        assert kind in {
            "distribution",
            "trajectory",
            "path_efficiency",
            "success_collision_scatter",
            "episode_lengths",
        }
        assert status in {"generated", "skipped"}
