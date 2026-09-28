#!/usr/bin/env python3
"""Build the exact 72-row pedestrian-speed activation canary without running episodes.

The #6561 protocol owns scenarios, planners, regimes, and the registered grid.
The closed #8888 preparation owner already froze disjoint seeds 311–314. The
current #8871 canary selects their lowest seed before any outcome, then includes
the legacy reference and both interventions. This packet is smoke evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.validation.check_issue_6561_pedestrian_speed_protocol import load_protocol  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs/benchmarks/issue_8871_pedestrian_speed_canary_v1.yaml"
SCHEMA_VERSION = "robot_sf.issue_8871_pedestrian_speed_canary.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _repo_file(raw: Any, field: str) -> Path:
    _require(isinstance(raw, str) and raw, f"{field} must be a path")
    relative = Path(raw)
    _require(
        not relative.is_absolute() and ".." not in relative.parts, f"{field} leaves repository"
    )
    path = REPO_ROOT / relative
    _require(path.is_file(), f"{field} missing: {raw}")
    return path


def load_canary_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the frozen canary selection and reject authority-boundary drift."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "canary config must be a mapping")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "canary schema drifted")
    _require(
        (payload.get("owner_issue"), payload.get("parent_issue"), payload.get("seed_owner_issue"))
        == (8871, 6561, 8888),
        "issue ownership drifted",
    )
    _require(
        payload.get("execution_boundary") == "activation_canary_only", "canary boundary drifted"
    )
    _require(payload.get("registered_rows_allowed") is False, "registered rows are forbidden")
    _require(payload.get("publication_authorized") is False, "publication is forbidden")
    return payload


def build_manifest(config: dict[str, Any], *, source_commit: str) -> dict[str, Any]:
    """Construct the unique disjoint 72-row identity set from frozen inputs."""
    _require(
        len(source_commit) == 40 and all(ch in "0123456789abcdef" for ch in source_commit),
        "source commit must be a 40-character SHA",
    )
    protocol_path = _repo_file(config.get("protocol_config"), "protocol_config")
    seed_owner_path = _repo_file(config.get("seed_owner_config"), "seed_owner_config")
    _require(_sha256(protocol_path) == config.get("protocol_sha256"), "protocol byte hash drifted")
    _require(
        _sha256(seed_owner_path) == config.get("seed_owner_sha256"), "seed-owner byte hash drifted"
    )
    protocol = load_protocol(protocol_path)
    _require(
        protocol["manifest_contract"]["manifest_hash"] == config.get("production_manifest_hash"),
        "registered manifest hash drifted",
    )
    seed_owner = yaml.safe_load(seed_owner_path.read_text(encoding="utf-8"))
    _require(seed_owner.get("child_issue") == 8888, "seed-owner issue drifted")
    frozen_seeds = [int(seed) for seed in seed_owner["preflight_seed_block"]["seeds"]]
    _require(
        frozen_seeds and len(set(frozen_seeds)) == len(frozen_seeds), "seed-owner block invalid"
    )
    _require(
        config.get("seed_selection_rule") == "lowest_frozen_disjoint_seed_from_issue_8888",
        "seed selection rule drifted",
    )
    selected_seed = int(config.get("selected_seed"))
    _require(
        selected_seed == min(frozen_seeds), "selected canary seed is not the frozen lowest seed"
    )
    registered = {int(seed) for seed in protocol["seed_contract"]["seeds"]}
    _require(selected_seed not in registered, "canary seed overlaps registered seeds")

    baseline = protocol["baseline_protocol"]
    identities: list[dict[str, Any]] = []
    for scenario in protocol["scenario_contract"]["selected_scenarios"]:
        for regime in protocol["pedestrian_speed_contract"]["regimes"]:
            for planner in protocol["planner_contract"]["roster"]:
                identities.append(
                    {
                        "identity_key": f"{scenario['scenario_id']}__{regime['regime_id']}__{planner['planner_id']}__{selected_seed}",
                        "scenario_id": scenario["scenario_id"],
                        "scenario_source_sha256": scenario["source_sha256"],
                        "regime_id": regime["regime_id"],
                        "runtime_controls": dict(regime["runtime_controls"]),
                        "planner_id": planner["planner_id"],
                        "planner_config_sha256": planner["config_sha256"],
                        "seed": selected_seed,
                        "horizon_steps": int(baseline["horizon_steps"]),
                        "dt_seconds": float(baseline["dt_seconds"]),
                        "robot_speed_cap_m_s": float(baseline["robot_speed_cap_m_s"]),
                        "execution_mode": baseline["execution_mode"],
                        "registered": False,
                        "canary": True,
                    }
                )
    keys = [row["identity_key"] for row in identities]
    _require(
        config.get("expected_rows") == len(identities) == 72, "canary must have exactly 72 rows"
    )
    _require(len(set(keys)) == 72, "canary identities are not unique")
    _require(
        {row["regime_id"] for row in identities}
        == {"legacy_default", "slow_distributed", "typical_distributed"},
        "canary regime roster drifted",
    )
    core = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": source_commit,
        "protocol_sha256": config["protocol_sha256"],
        "production_manifest_hash": config["production_manifest_hash"],
        "selected_seed": selected_seed,
        "identities": identities,
    }
    return {
        **core,
        "expected_rows": 72,
        "manifest_hash": _canonical_hash(core),
        "claim_boundary": "activation canary only; no registered or publication evidence",
    }


def main() -> int:
    """Print or write an outcome-free packet for later native admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", action="store_true", help="print the complete 72-row packet")
    parser.add_argument("--output", type=Path, help="write the complete packet to an output file")
    args = parser.parse_args()
    config = load_canary_config(args.config)
    source_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = build_manifest(config, source_commit=source_commit)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.manifest:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {key: value for key, value in manifest.items() if key != "identities"},
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
