"""Scientific identity checks for the disjoint pedestrian-speed canary."""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.validation.build_issue_8871_pedestrian_speed_canary import (
    build_manifest,
    load_canary_config,
)


def test_canary_is_exactly_one_disjoint_seed_and_all_frozen_arms() -> None:
    packet = build_manifest(load_canary_config(), source_commit="a" * 40)
    rows = packet["identities"]
    assert packet["expected_rows"] == len(rows) == len({row["identity_key"] for row in rows}) == 72
    assert {row["seed"] for row in rows} == {311}
    assert {row["regime_id"] for row in rows} == {
        "legacy_default",
        "slow_distributed",
        "typical_distributed",
    }
    assert len({row["scenario_id"] for row in rows}) == 6
    assert len({row["planner_id"] for row in rows}) == 4
    assert all(row["canary"] is True and row["registered"] is False for row in rows)
    assert all(row["execution_mode"] == "native" for row in rows)


def test_canary_rejects_seed_or_protocol_drift() -> None:
    config = load_canary_config()
    wrong_seed = deepcopy(config)
    wrong_seed["selected_seed"] = 111
    with pytest.raises(ValueError, match="lowest seed"):
        build_manifest(wrong_seed, source_commit="a" * 40)

    wrong_protocol = deepcopy(config)
    wrong_protocol["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol byte hash drifted"):
        build_manifest(wrong_protocol, source_commit="a" * 40)


def test_manifest_hash_binds_source_commit() -> None:
    config = load_canary_config()
    first = build_manifest(config, source_commit="a" * 40)
    second = build_manifest(config, source_commit="b" * 40)
    assert first["manifest_hash"] != second["manifest_hash"]
