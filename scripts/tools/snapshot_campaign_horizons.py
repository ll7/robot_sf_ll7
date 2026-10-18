"""Capture admission and authored simulator limits without constructing an environment.

Generate the historical fixture in a standalone checkout of main 93ba0d75:
PYTHONPATH=$PWD uv run --no-sync python /path/to/snapshot_campaign_horizons.py \
    --output /path/to/campaign_horizons_main_93ba0d75.json
The script imports the checkout on PYTHONPATH, not its own source checkout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config


def tracked_campaign_yaml(root: Path) -> list[str]:
    """Return every tracked benchmark YAML, including non-campaign inputs."""
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    return sorted(p for p in paths if p.startswith("configs/benchmarks/") and p.endswith(".yaml"))


def campaign_admission(path: Path, root: Path) -> dict:
    """Observe production admission and each resolved simulator limit, without stepping."""
    try:
        cfg = load_campaign_config(path, repository_root=root)
        scenarios = _load_campaign_scenarios(cfg, repository_root=root)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"admitted": False, "error": type(exc).__name__}
    return {
        "admitted": True,
        "limits": [
            {
                "scenario": str(s.get("name") or s.get("scenario_id") or s.get("id")),
                "max_episode_steps": s.get("simulation_config", {}).get("max_episode_steps"),
            }
            for s in scenarios
        ],
    }


def main() -> None:
    """Write a revision-bound, deterministic production-loader snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    payload = {
        "source_revision": revision,
        "generator_command": (
            "PYTHONPATH=$PWD uv run --no-sync python "
            "$FIX_CHECKOUT/scripts/tools/snapshot_campaign_horizons.py "
            "--output $FIX_CHECKOUT/tests/benchmark/fixtures/"
            "campaign_horizons_main_93ba0d75.json"
        ),
        "configs": {p: campaign_admission(root / p, root) for p in tracked_campaign_yaml(root)},
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{revision}: {len(payload['configs'])} tracked YAML inputs")


if __name__ == "__main__":
    main()
