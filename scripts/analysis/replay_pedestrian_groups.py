"""Export development-only zero-step reset memberships without changing episode inputs."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_identity import _scenario_with_episode_seed_defaults
from robot_sf.benchmark.result_provenance import capture_pedestrian_group_provenance
from robot_sf.benchmark.runtime_seed_guard import SEED_FIELDS
from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import load_scenarios


def _require_development_seed(seed: Any) -> None:
    """Reject unsupported bands without including protected values in error messages."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 1001 <= seed <= 1200:
        raise ValueError("reset replay supports development seeds 1001..1200 only")


def export_reset_groups(
    episodes: Path,
    matrix: Path,
    output: Path,
    *,
    authorize_dev_reset_replay: bool = False,
) -> int:
    """Replay only authorized development resets; emit a separate diagnostic sidecar.

    The supplied matrix and current simulator are the replay inputs. This does not
    establish equivalence to a historical source, even if commit labels agree.
    Validate the whole batch before constructing an environment; never step a planner.

    Returns:
        Number of derived provenance rows written to the new output file.
    """
    if not authorize_dev_reset_replay:
        raise ValueError("explicit --authorize-dev-reset-replay is required")
    if (
        output.exists()
        or output.is_symlink()
        or output.resolve() in {episodes.resolve(), matrix.resolve()}
    ):
        raise ValueError("derived output must be a new file separate from frozen inputs")
    input_bytes = episodes.read_bytes()
    rows = [json.loads(line) for line in input_bytes.splitlines() if line.strip()]
    for row in rows:
        _require_development_seed(row.get("seed"))
    scenarios = {s["name"]: dict(s) for s in load_scenarios(matrix)}
    prepared = []
    for row in rows:
        scenario = _scenario_with_episode_seed_defaults(
            scenarios[row["scenario_id"]], seed=row["seed"]
        )
        inputs: list[dict[str, str]] = []
        config = build_env_config(scenario, scenario_path=matrix, runtime_input_records=inputs)
        for candidate in (config, config.sim_config):
            for field in SEED_FIELDS:
                value = getattr(candidate, field, None)
                if value is not None:
                    _require_development_seed(value)
        prepared.append((row, scenario, config, inputs))
    derived = []
    for row, scenario, config, inputs in prepared:
        env = make_robot_env(config=config, seed=row["seed"], debug=False)
        try:
            env.reset(seed=row["seed"])
            snapshot = capture_pedestrian_group_provenance(env.simulator)
            if snapshot["status"] != "recorded":
                raise ValueError("reset replay could not record runtime memberships")
        finally:
            env.close()
        derived.append(
            {
                "schema_version": "derived_pedestrian_group_provenance.v1",
                "evidence_tier": "diagnostic-only",
                "historical_equivalence": "not_established",
                "authorization": "explicit_development_reset_replay",
                "episode_id": row["episode_id"],
                "scenario_id": row["scenario_id"],
                "seed": row["seed"],
                "input_sha256": sha256(input_bytes).hexdigest(),
                "input_row_sha256": sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "input_git_hash": row.get("git_hash"),
                "replay_git_hash": _git_hash_fallback(),
                "replay_scenario_hash": _config_hash(scenario),
                "matrix_sha256": sha256(matrix.read_bytes()).hexdigest(),
                "runtime_input_records": inputs,
                "pedestrian_group_provenance": snapshot,
            }
        )
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in derived)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return len(derived)


def main() -> None:
    """Parse the explicit replay authorization and write development provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--authorize-dev-reset-replay", action="store_true")
    args = parser.parse_args()
    count = export_reset_groups(
        args.episodes,
        args.matrix,
        args.out,
        authorize_dev_reset_replay=args.authorize_dev_reset_replay,
    )
    print(
        f"Wrote {count} development reset provenance rows; historical equivalence not established."
    )


if __name__ == "__main__":
    main()
