"""Explicit projection of verified publication rows into the offline auditor.

The release loader supplies the arm and member identity. Episode and planner
configuration hashes have different scopes in these rows; they must never be
advertised as one campaign configuration hash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def project_release_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep publication identities scoped and namespace episode IDs by release arm.

    Unmarked generic campaign rows are returned unchanged. This projection does
    not attest execution or scientific validity; ordinary admission still runs.

    Returns:
        An explicitly scoped release projection, or the unmarked source row.
    """
    if "_release_arm" not in row:
        return row
    arm = row.get("_release_arm")
    member = row.get("_source_member")
    episode = row.get("episode_id")
    config_hash = row.get("config_hash")
    if not all(isinstance(x, str) and x.strip() for x in (arm, member, episode, config_hash)):
        raise ValueError("release row identity is incomplete")
    if member != f"payload/runs/{arm}__differential_drive/episodes.jsonl":
        raise ValueError("release arm does not match its source member")
    projected = dict(row)
    projected["episode_id"] = f"{arm}::{episode}"
    projected["planner_id"] = arm
    metadata = row.get("algorithm_metadata")
    planner_hash = metadata.get("config_hash") if isinstance(metadata, Mapping) else None
    if not isinstance(planner_hash, str) or not planner_hash.strip():
        raise ValueError("release row planner config hash is missing")
    # The episode hash includes seed/reset state. Cohort comparison uses the
    # planner configuration plus the separately keyed scenario and outcome.
    projected["config"] = {"config_id": planner_hash}
    projected["config_digest"] = hashlib.sha256(config_hash.encode()).hexdigest()
    projected["release_row_identity"] = {
        "episode_id": episode,
        "release_arm": arm,
        "source_member": member,
        "episode_config_hash": config_hash,
        "adapter_version": "1.0.0",
    }
    projected["episode_config_hash"] = projected.pop("config_hash")
    for key in ("provenance", "result_provenance"):
        value = row.get(key)
        if isinstance(value, Mapping):
            value = dict(value)
            if "config_hash" in value:
                if value["config_hash"] != config_hash:
                    raise ValueError("release episode config hashes conflict")
                value["episode_config_hash"] = value.pop("config_hash")
            projected[key] = value
    # Planner-hash validation above already established this mapping and key.
    metadata = dict(metadata)
    metadata["planner_config_hash"] = metadata.pop("config_hash")
    projected["algorithm_metadata"] = metadata
    return projected
