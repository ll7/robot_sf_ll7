"""Contract tests for the issue #6561 pedestrian-speed protocol."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.validation import check_issue_6561_pedestrian_speed_protocol as checker


def _payload() -> dict[str, object]:
    """Load a fresh copy of the tracked protocol payload."""
    return yaml.safe_load(checker.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_protocol_compiles_exact_2160_unique_native_identities() -> None:
    """The check-only compiler emits the frozen 6 x 3 x 4 x 30 grid."""
    payload = _payload()
    checker.validate_protocol(payload)

    manifest = checker.compile_manifest(payload)

    assert manifest["identity_count"] == 2160
    assert manifest["unique_identity_count"] == 2160
    assert manifest["manifest_hash"] == (
        "371f1a0160ec7faf1ade531691f104e2a1c92f7c34857e887ba1ba539e1b5238"
    )
    assert all(row["execution_mode"] == "native" for row in manifest["identities"])
    assert all(row["registered"] is True for row in manifest["identities"])


def test_manifest_pairs_regimes_and_planners_by_seed() -> None:
    """Each seed has one identity for every regime/planner/scenario combination."""
    manifest = checker.compile_manifest(_payload())
    identities = manifest["identities"]

    assert identities[0]["identity_key"] == (
        "classic_head_on_corridor_medium__legacy_default__"
        "scenario_adaptive_hybrid_orca_v2_collision_guard__111"
    )
    assert identities[-1]["identity_key"] == (
        "classic_station_platform_medium__typical_distributed__prediction_planner__140"
    )
    assert {row["regime_id"] for row in identities} == {
        "legacy_default",
        "slow_distributed",
        "typical_distributed",
    }
    assert {row["planner_id"] for row in identities} == set(checker.EXPECTED_PLANNERS)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("scenario_contract", "selected_scenarios", 0, "source_sha256"),
            "0" * 64,
            "source hash drifted",
        ),
        (
            ("planner_contract", "roster", 1, "config_sha256"),
            "0" * 64,
            "config hash drifted",
        ),
        (
            ("seed_contract", "seeds", 29),
            141,
            "seed set drifted",
        ),
        (
            ("baseline_protocol", "seed_set_sha256"),
            "0" * 64,
            "seed set hash drifted",
        ),
    ],
)
def test_checker_rejects_frozen_identity_drift(
    path: tuple[object, ...], value: object, message: str
) -> None:
    """Scenario, planner, and seed changes fail before any campaign path exists."""
    payload = _payload()
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        checker.validate_protocol(payload)


def test_checker_rejects_protocol_production_enablement() -> None:
    """The protocol cannot silently authorize production rows."""
    payload = _payload()
    payload["execution_boundary"]["registered_campaign_in_this_pr"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="registered_campaign_in_this_pr drifted"):
        checker.validate_protocol(payload)


def test_checker_rejects_transient_scheduler_state() -> None:
    """Queue or host details do not belong in a reproducible protocol manifest."""
    payload = deepcopy(_payload())
    payload["runtime"] = {"job_id": 123}  # type: ignore[index]

    with pytest.raises(ValueError, match="transient routing state"):
        checker.validate_protocol(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("activation_contract", "rule"), "activation is optional"),
        (("inference_contract", "interval_method"), "wilson_score"),
        (
            ("inference_contract", "multiplicity", "family"),
            "primary_metrics_only",
        ),
        (("scheduler",), {"partition": "gpu"}),
    ],
)
def test_compile_manifest_rejects_semantic_and_transient_drift(
    path: tuple[object, ...], value: object
) -> None:
    """Manifest compilation cannot bypass the complete frozen semantic contract."""
    payload = deepcopy(_payload())
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError):
        checker.compile_manifest(payload)


def test_checker_requires_spawn_speed_and_activation_rule() -> None:
    """The spawn boundary and desired-speed activation gate are immutable."""
    payload = _payload()
    payload["pedestrian_speed_contract"]["spawn"]["initial_speed_m_s"] = 0.65  # type: ignore[index]

    with pytest.raises(ValueError, match="initial spawn speed"):
        checker.validate_protocol(payload)


def test_protocol_paths_are_repository_relative() -> None:
    """The frozen packet cannot encode a private machine path."""
    payload = _payload()
    payload["baseline_protocol"]["scenario_matrix"] = str(Path("/tmp/private.yaml"))  # type: ignore[index]

    with pytest.raises(ValueError, match="must be repository-relative"):
        checker.validate_protocol(payload)
