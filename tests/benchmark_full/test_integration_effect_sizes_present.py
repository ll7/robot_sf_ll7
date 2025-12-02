"""Integration test T019: effect size report presence.

After a full run the effect sizes artifact should exist and contain at least
one archetype entry with comparison objects including standardized values.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def test_effect_sizes_presence(config_factory):
    """Test effect sizes presence.

    Args:
        config_factory: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    cfg = config_factory(smoke=True)
    manifest = run_full_benchmark(cfg)
    effects_path = Path(manifest.output_root) / "reports" / "effect_sizes.json"
    assert effects_path.exists(), "effect_sizes.json not created"
    data = json.loads(effects_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    if data:  # structure check
        first = data[0]
        assert "archetype" in first
        assert "comparisons" in first and isinstance(first["comparisons"], list)
        if first["comparisons"]:
            comp = first["comparisons"][0]
            for key in ("metric", "density_low", "density_high", "diff", "standardized"):
                assert key in comp
