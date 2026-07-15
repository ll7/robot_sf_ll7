"""Fixture/shape tests for the runnable Robot SF -> SocNavBench cross-suite canary (#5783).

These tests run **without licensed data**: they exercise the export materialization, the
fail-closed asset gate, the policy-identity fallback detector, the denominator guard, and the
joint-receipt shape. Only the real-asset SocNavBench execution path (the staged-ETH branch) is
skipped, because the licensed SocNavBench ETH asset is never staged in-repo.

Claim boundary: these tests prove the canary *mechanics* (determinism, fail-closed behavior,
identity matching, receipt shape). They are not benchmark, equivalence, or campaign evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.socnavbench_canary import (
    CANARY_NAME,
    CANARY_ROBOT_SF_SCENARIO_ID,
    CANARY_SEED,
    CANARY_SOCNAVBENCH_SCENARIO_ID,
    CANARY_VERSION,
    EXPORT_SCHEMA_VERSION,
    PINNED_POLICY_ALGO,
    PINNED_POLICY_ALGO_CONFIG,
    PINNED_POLICY_ID,
    PINNED_POLICY_VERSION,
    RECEIPT_SCHEMA_VERSION,
    CanaryError,
    PinnedPolicy,
    ScenarioMapping,
    materialize_socnavbench_export,
    resolve_pinned_policy,
    resolve_scenario_mapping,
    run_canary,
    run_robot_sf_receipt,
    run_socnavbench_receipt,
)
from robot_sf.data.external.socnavbench_eth import ASSET_ID as ETH_ASSET_ID

if TYPE_CHECKING:
    from pathlib import Path


def test_pinned_policy_resolves_concrete_identity() -> None:
    """The pinned policy must resolve to concrete, non-placeholder identity fields."""
    policy = resolve_pinned_policy()
    assert policy.policy_id == PINNED_POLICY_ID
    assert policy.version == PINNED_POLICY_VERSION
    assert policy.algo == PINNED_POLICY_ALGO
    # algo_config is repo-relative so the identity block is portable across checkouts.
    assert policy.algo_config == "configs/algos/social_force_holonomic_tuned_tau_low.yaml"
    # Source commit and config digest must be non-empty (no placeholder).
    assert policy.source_commit
    assert policy.config_digest_sha256
    assert len(policy.config_digest_sha256) == 64


def test_pinned_policy_config_file_exists() -> None:
    """The pinned policy config must point at a real, tracked config file (not TBD)."""
    assert PINNED_POLICY_ALGO_CONFIG.is_file(), (
        f"Pinned policy config missing: {PINNED_POLICY_ALGO_CONFIG}"
    )


def test_scenario_mapping_is_concrete_and_documents_limitations() -> None:
    """The scenario mapping must pin concrete IDs, a seed, an asset, and limitation flags."""
    mapping = resolve_scenario_mapping()
    assert isinstance(mapping, ScenarioMapping)
    assert mapping.robot_sf_scenario_id == CANARY_ROBOT_SF_SCENARIO_ID
    assert mapping.socnavbench_scenario_id == CANARY_SOCNAVBENCH_SCENARIO_ID
    assert mapping.seed == CANARY_SEED
    assert mapping.external_asset_id == ETH_ASSET_ID
    assert mapping.mapping_quality
    # Non-equivalence limitations must be stated explicitly (not a silent equivalence claim).
    assert mapping.limitation_flags
    assert any("equivalence" in flag for flag in mapping.limitation_flags)


def test_materialize_socnavbench_export_writes_real_scenario(tmp_path: Path) -> None:
    """The exporter must write a concrete SocNavBench scenario, not a preview-only artifact."""
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    export_path = materialize_socnavbench_export(policy=policy, mapping=mapping, out_dir=tmp_path)
    assert export_path.is_file()

    import json

    export = json.loads(export_path.read_text(encoding="utf-8"))
    assert export["schema_version"] == EXPORT_SCHEMA_VERSION
    assert export["canary"] == CANARY_NAME
    assert export["canary_version"] == CANARY_VERSION
    # The export references the staged ETH asset via the registry id (not a repo path).
    assert export["external_asset_id"] == ETH_ASSET_ID
    assert export["map"]["asset_id"] == ETH_ASSET_ID
    # Robot start/goal and at least one pedestrian must be concrete.
    assert len(export["robot"]["start"]) == 2
    assert len(export["robot"]["goal"]) == 2
    assert export["pedestrians"]
    # Policy identity is embedded so a downstream runner records the identical identity.
    assert export["policy_identity"]["policy_id"] == PINNED_POLICY_ID


def test_robot_sf_receipt_computes_real_metric() -> None:
    """The Robot SF receipt must compute the real vendored-style path-length-ratio metric."""
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    receipt = run_robot_sf_receipt(policy=policy, mapping=mapping)
    assert receipt["suite"] == "Robot SF"
    assert receipt["metric_id"] == "socnavbench.path_length_ratio"
    # value is a finite float computed from the deterministic trajectory.
    assert isinstance(receipt["value"], float)
    assert receipt["value"] == receipt["value"]  # not NaN
    # Suite-specific denominator is recorded (not silently shared/changed).
    assert receipt["denominator"] > 0.0
    assert receipt["denominator_kind"] == "start_to_goal_displacement_m"
    assert receipt["policy_identity"]["policy_id"] == PINNED_POLICY_ID


def test_robot_sf_receipt_is_deterministic() -> None:
    """The canary must be a pure function of the seed (reproducible across runs)."""
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    first = run_robot_sf_receipt(policy=policy, mapping=mapping)
    second = run_robot_sf_receipt(policy=policy, mapping=mapping)
    assert first["value"] == second["value"]
    assert first["denominator"] == second["denominator"]


def test_socnavbench_receipt_runs_without_licensed_data_via_test_flag(tmp_path: Path) -> None:
    """The SocNavBench receipt runs the vendored metric when the test-only flag is set.

    This is the no-licensed-data fixture path: it skips ONLY the staged-asset gate while still
    executing the real vendored SocNavBench ``path_length_ratio`` metric. The fallback is
    recorded in the receipt.
    """
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    export_path = materialize_socnavbench_export(policy=policy, mapping=mapping, out_dir=tmp_path)
    receipt = run_socnavbench_receipt(
        export_path=export_path,
        policy=policy,
        mapping=mapping,
        allow_synthetic_traversible=True,
    )
    assert receipt["suite"] == "SocNavBench"
    assert receipt["metric_id"] == "socnavbench.path_length_ratio"
    assert isinstance(receipt["value"], float)
    assert receipt["value"] == receipt["value"]  # not NaN
    assert receipt["external_asset_id"] == ETH_ASSET_ID
    # Whether or not the licensed asset is staged here, the receipt records the truth.
    assert "external_asset_staged" in receipt


def test_socnavbench_receipt_fails_closed_without_asset_or_test_flag(tmp_path: Path) -> None:
    """Without the test-only flag, a missing licensed asset must fail closed (no fallback)."""
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    export_path = materialize_socnavbench_export(policy=policy, mapping=mapping, out_dir=tmp_path)
    with pytest.raises(CanaryError, match="Licensed SocNavBench ETH asset not staged"):
        run_socnavbench_receipt(
            export_path=export_path,
            policy=policy,
            mapping=mapping,
            allow_synthetic_traversible=False,
        )


def test_socnavbench_receipt_fails_closed_on_missing_export() -> None:
    """A missing export file must fail closed rather than silently degrade."""
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    with pytest.raises(CanaryError, match="SocNavBench export missing"):
        run_socnavbench_receipt(
            export_path=__import__("pathlib").Path("/nonexistent/export.json"),
            policy=policy,
            mapping=mapping,
            allow_synthetic_traversible=True,
        )


def test_run_canary_emits_joint_receipt_with_test_flag(tmp_path: Path) -> None:
    """The full canary emits a machine-checkable joint receipt without licensed data."""
    receipt = run_canary(out_dir=tmp_path, allow_synthetic_traversible=True)
    assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert receipt["canary"] == CANARY_NAME
    assert receipt["canary_version"] == CANARY_VERSION
    assert receipt["policy_identity_match"] is True
    assert receipt["denominators_preserved"] is True
    assert receipt["fallback_forbidden"] is True
    assert receipt["seed"] == CANARY_SEED
    # Both suites are present with their own receipts.
    suite_names = {suite["suite"] for suite in receipt["suites"]}
    assert suite_names == {"Robot SF", "SocNavBench"}
    # Per-suite denominators and metric values are recorded separately.
    assert set(receipt["per_suite_denominators"]) == {"Robot SF", "SocNavBench"}
    assert set(receipt["metric_values"]) == {"Robot SF", "SocNavBench"}
    # Limitation flags survive into the receipt mapping block.
    assert receipt["scenario_mapping"]["limitation_flags"]
    # The receipt file was actually written.
    assert __import__("pathlib").Path(receipt["receipt_path"]).is_file()


def test_run_canary_fails_closed_without_test_flag(tmp_path: Path) -> None:
    """Without the test-only flag the canary must fail closed on a missing licensed asset."""
    with pytest.raises(CanaryError, match="Licensed SocNavBench ETH asset not staged"):
        run_canary(out_dir=tmp_path, allow_synthetic_traversible=False)


def test_policy_identity_match_is_exact() -> None:
    """A differing identity field must be detected as a fallback."""
    base = resolve_pinned_policy()
    tampered = PinnedPolicy(
        policy_id=base.policy_id,
        version="different_version",
        algo=base.algo,
        algo_config=base.algo_config,
        config_digest_sha256=base.config_digest_sha256,
        source_commit=base.source_commit,
    )
    assert not base.matches(tampered)
    assert base.matches(base)


def test_pinned_policy_resolves_asset_via_external_data_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scenario mapping must reference the canonical external-data registry asset id."""
    mapping = resolve_scenario_mapping()
    # The asset id must be a real entry in the manage_external_data registry.
    from scripts.tools.manage_external_data import list_assets

    asset_ids = {asset.asset_id for asset in list_assets()}
    assert mapping.external_asset_id in asset_ids


def test_external_data_root_env_is_honored_by_asset_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ROBOT_SF_EXTERNAL_DATA_ROOT must redirect asset resolution (no hardcoded repo path)."""
    from robot_sf.data.external import socnavbench_eth
    from robot_sf.data.external.paths import EXTERNAL_DATA_ROOT_ENV

    fake_root = tmp_path / "external_data"
    fake_root.mkdir()
    monkeypatch.setenv(EXTERNAL_DATA_ROOT_ENV, str(fake_root))
    resolved = socnavbench_eth.resolve_root()
    # When the env root is set, resolution must point under it, not under the repo tree.
    assert resolved == fake_root / "socnavbench"


def test_cli_canary_runner_emits_receipt_with_test_flag(tmp_path: Path) -> None:
    """The CLI runner produces a joint receipt without licensed data via the test flag."""
    from scripts.tools.cross_benchmark_canary import main

    out_dir = tmp_path / "canary"
    exit_code = main(["--out-dir", str(out_dir), "--allow-synthetic-traversible"])
    assert exit_code == 0
    receipt_path = out_dir / "cross_suite_canary_receipt.json"
    assert receipt_path.is_file()


def test_cli_canary_runner_fails_closed_without_asset(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Without the test-only flag the CLI must exit nonzero and report the blocked asset gate."""
    from scripts.tools.cross_benchmark_canary import main

    exit_code = main(["--out-dir", str(tmp_path / "canary")])
    assert exit_code == 1
    assert "Licensed SocNavBench ETH asset not staged" in capsys.readouterr().err
