"""One producer/consumer contract for every repeated episode identity claim.

Only fresh producers publish copies. Validators never repair a saved record.
Field authorities are campaign configuration, the campaign commit, or the
selected source scenario; trace/output artifact digests are not runtime inputs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from robot_sf.benchmark.utils import _config_hash

RUNTIME_IDENTITY_FIELDS = frozenset(
    {
        "algo",
        "algo_config_hash",
        "run_horizon",
        "run_dt",
        "record_forces",
        "observation_mode",
        "observation_level",
        "benchmark_track",
        "track_schema_version",
        "observation_noise_profile",
        "observation_noise_hash",
        "tracking_precision",
        "tracking_precision_hash",
        "synthetic_actuation_profile",
        "latency_stress_profile",
        "safety_wrapper",
        "cbf_safety_filter",
        "record_planner_decision_trace",
        "record_simulation_step_trace",
    }
)


@dataclass(frozen=True)
class IdentityField:
    """Declared producer copies of one identity value and its trusted authority."""

    name: str
    copies: tuple[tuple[str, ...], ...]
    authority: str = "campaign_configuration"
    conditional_copies: tuple[tuple[str, ...], ...] = ()


def _mirrors(name: str) -> tuple[tuple[str, ...], ...]:
    return ((name,), ("provenance", name), ("result_provenance", name))


# This is the sole inventory. Every declared copy is mandatory, even when its
# value is None. Different hash scopes have different named fields/authorities.
IDENTITY_FIELDS = (
    IdentityField(
        "git_hash",
        (("git_hash",), ("provenance", "git_hash"), ("result_provenance", "repo_commit")),
        "campaign_commit",
        (("algorithm_metadata", "analysis_trace", "git_hash"),),
    ),
    IdentityField("config_hash", _mirrors("config_hash")),
    IdentityField(
        "trace_config_hash",
        (),
        "selected_source_scenario",
        (("algorithm_metadata", "analysis_trace", "config_hash"),),
    ),
    IdentityField("scenario_id", (("scenario_id",), ("result_provenance", "scenario_id"))),
    IdentityField("seed", (("seed",), ("result_provenance", "seed"))),
    IdentityField("episode_id", (("episode_id",), ("result_provenance", "episode_id"))),
    IdentityField("telemetry_profile", _mirrors("telemetry_profile")),
    IdentityField("simulator_settings", (("result_provenance", "simulator_settings"),)),
    IdentityField(
        "horizon", _mirrors("horizon") + (("result_provenance", "simulator_settings", "horizon"),)
    ),
    IdentityField("dt", _mirrors("dt") + (("result_provenance", "simulator_settings", "dt"),)),
    IdentityField("observation_noise", _mirrors("observation_noise")),
    *(
        IdentityField(
            name,
            _mirrors(name)
            + (
                (("result_provenance", "simulator_settings", name),)
                if name
                in {
                    "record_forces",
                    "observation_mode",
                    "observation_level",
                    "observation_noise_hash",
                    "tracking_precision_hash",
                }
                else ()
            ),
        )
        for name in sorted(RUNTIME_IDENTITY_FIELDS)
    ),
)

# Single-block annotations do not select runtime configuration. Unknown fields
# in either provenance block fail closed until the schema declares their role.
PROVENANCE_ENVELOPE_FIELDS = {
    "provenance": frozenset(
        {"artifact_uri", "artifact_sha256", "map_digest", "scenario_digest", "planner_commit"}
    ),
    "result_provenance": frozenset(
        {"schema_version", "postprocessing", "raw_artifact", "jsonl_line"}
    ),
}


def publish_episode_identity(record: dict[str, Any], *, simulator_settings: dict[str, Any]) -> None:
    """Publish the declared copies on a fresh producer record, never an admitted row."""
    params = record["scenario_params"]
    values = {name: deepcopy(params.get(name)) for name in RUNTIME_IDENTITY_FIELDS}
    values.update(
        {
            "git_hash": record["git_hash"],
            "config_hash": _config_hash(params),
            "scenario_id": record["scenario_id"],
            "seed": record["seed"],
            "episode_id": record["episode_id"],
            "telemetry_profile": record.get("algorithm_metadata", {}).get("telemetry"),
            "simulator_settings": simulator_settings,
            "horizon": simulator_settings["horizon"],
            "dt": simulator_settings["dt"],
            "observation_noise": record["observation_noise"],
            "observation_noise_hash": record["observation_noise_hash"],
            "tracking_precision": record["tracking_precision"],
            "tracking_precision_hash": record["tracking_precision_hash"],
        }
    )
    for field in IDENTITY_FIELDS:
        for path in field.copies:
            target = record
            for part in path[:-1]:
                target = target.setdefault(part, {})
            target[path[-1]] = deepcopy(values[field.name])
