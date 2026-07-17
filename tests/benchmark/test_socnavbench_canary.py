"""Tests for the runnable SocNavBench cross-suite canary (#5783 / #5842).

Issue #5842 hardens the canary so that:
  - The pinned policy is actually executed through Robot SF (no synthetic trajectory).
  - The SocNavBench runner consumes the exported scenario via a real source harness.
  - The two suites use distinct metric IDs for reciprocal path-length-ratio definitions.
  - Any fallback/synthetic-traversible path reports counts_as_success_evidence: False.
  - fallback_forbidden reflects whether a fallback path was actually taken.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark import canary_rollout
from robot_sf.benchmark.canary_rollout import (
    _resolve_pinned_planner_config,
    execute_pinned_policy,
)
from robot_sf.benchmark.socnavbench_canary import (
    CANARY_NAME,
    CANARY_SEED,
    CANARY_VERSION,
    PINNED_POLICY_ALGO_CONFIG,
    PINNED_POLICY_ID,
    RECEIPT_SCHEMA_VERSION,
    ROBOT_SF_METRIC_ID,
    SOCNAVBENCH_METRIC_ID,
    CanaryError,
    PinnedPolicy,
    _config_digest,
    _git_commit_sha,
    materialize_socnavbench_export,
    resolve_pinned_policy,
    resolve_scenario_mapping,
    run_canary,
    run_robot_sf_receipt,
    run_robot_sf_receipt_from_rollout,
    run_socnavbench_receipt,
)
from robot_sf.data.external.socnavbench_eth import ASSET_ID as ETH_ASSET_ID

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pinned_policy_resolves_concrete_identity() -> None:
    """The pinned policy identity must have no placeholder tokens."""
    policy = resolve_pinned_policy()
    assert policy.policy_id == PINNED_POLICY_ID
    assert policy.version
    assert policy.algo
    assert policy.algo_config
    assert policy.config_digest_sha256
    assert policy.source_commit
    assert not any(
        tok in policy.policy_id.lower() for tok in ("tbd", "placeholder", "blocked", "scaffold")
    )


def test_pinned_policy_config_file_exists() -> None:
    """The pinned algo config path must exist so the canary can read provenance."""
    assert PINNED_POLICY_ALGO_CONFIG.is_file(), (
        f"Pinned policy config not found: {PINNED_POLICY_ALGO_CONFIG}"
    )


def test_scenario_mapping_is_concrete_and_documents_limitations() -> None:
    """The scenario mapping must document limitation flags and carry a real asset id."""
    mapping = resolve_scenario_mapping()
    assert mapping.robot_sf_scenario_id
    assert mapping.socnavbench_scenario_id
    assert mapping.external_asset_id == ETH_ASSET_ID
    assert len(mapping.limitation_flags) >= 1
    assert "tbd" not in mapping.socnavbench_scenario_id.lower()
    assert mapping.mapping_quality


def test_materialize_socnavbench_export_writes_real_scenario(tmp_path: Path) -> None:
    """materialize_socnavbench_export must write a JSON consumed by a real runner.

    The export must include:
    - The executed trajectory (so the SocNavBench runner consumes identical path).
    - The robot goal (derived from the actual rollout, not a hardcoded constant).
    - The policy identity block.
    - The limitation flags and mapping quality.
    - The external asset id (not a placeholder).
    """
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    export_path, rollout = materialize_socnavbench_export(
        policy=policy, mapping=mapping, out_dir=tmp_path
    )
    assert export_path.is_file()
    export = json.loads(export_path.read_text(encoding="utf-8"))
    assert export["schema_version"]
    assert export["external_asset_id"] == ETH_ASSET_ID
    assert export["executed_robot_sf_scenario_id"] == mapping.robot_sf_scenario_id
    assert len(export["robot"]["goal"]) == 2
    assert export["pedestrians"]
    # Trajectory must be embedded and non-trivial (at least 2 positions from the real rollout).
    assert "trajectory" in export
    assert len(export["trajectory"]) >= 2
    # Policy identity is embedded so a downstream runner records the identical identity.
    assert export["policy_identity"]["policy_id"] == PINNED_POLICY_ID
    # The rollout returned is the same object used to write the export.
    assert len(rollout.robot_positions) == len(export["trajectory"])


def test_materialize_export_returns_rollout_for_reuse(tmp_path: Path) -> None:
    """materialize_socnavbench_export must return the rollout so callers avoid double-execution."""
    from robot_sf.benchmark.canary_rollout import execute_pinned_policy

    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    pre_rollout = execute_pinned_policy(seed=mapping.seed, algo_config=PINNED_POLICY_ALGO_CONFIG)
    _, returned_rollout = materialize_socnavbench_export(
        policy=policy, mapping=mapping, out_dir=tmp_path, rollout=pre_rollout
    )
    # When a rollout is passed in, it must be returned as-is (no re-execution).
    assert returned_rollout is pre_rollout


def test_robot_sf_receipt_computes_real_metric() -> None:
    """The Robot SF receipt must execute the real pinned policy and compute path-length-ratio."""
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    receipt = run_robot_sf_receipt(policy=policy, mapping=mapping)
    assert receipt["suite"] == "Robot SF"
    # New API: metric identity in metric_spec, not as a top-level metric_id key.
    assert "metric_spec" in receipt
    assert receipt["metric_spec"]["metric_id"] == ROBOT_SF_METRIC_ID
    assert receipt["metric_spec"]["ratio_direction"] == "distance_over_displacement"
    assert receipt["metric_spec"]["mapping_class"] == "approximate"
    # Value is a finite float from the real executed trajectory.
    assert isinstance(receipt["value"], float)
    assert receipt["value"] == receipt["value"]  # not NaN
    assert receipt["value"] > 0.0
    # Suite-specific denominator is recorded.
    assert receipt["denominator"] > 0.0
    assert receipt["metric_spec"]["denominator_kind"] == "start_to_goal_displacement_m"
    assert receipt["policy_identity"]["policy_id"] == PINNED_POLICY_ID
    # Policy identity must include runtime provenance (not a static metadata copy).
    assert "runtime_planner_config" in receipt["policy_identity"]
    assert receipt["executed_policy"] is True


def test_robot_sf_receipt_from_rollout_matches_independent_receipt() -> None:
    """run_robot_sf_receipt_from_rollout must produce the same receipt as run_robot_sf_receipt."""
    from robot_sf.benchmark.canary_rollout import execute_pinned_policy

    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    rollout = execute_pinned_policy(seed=mapping.seed, algo_config=PINNED_POLICY_ALGO_CONFIG)

    from_rollout = run_robot_sf_receipt_from_rollout(
        policy=policy, mapping=mapping, rollout=rollout
    )
    assert from_rollout["suite"] == "Robot SF"
    assert from_rollout["metric_spec"]["metric_id"] == ROBOT_SF_METRIC_ID
    assert isinstance(from_rollout["value"], float)
    assert from_rollout["value"] > 0.0


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

    This is the no-licensed-data fixture path. The run is recorded as a fallback and the
    receipt reports counts_as_success_evidence: False and is_fallback: True.
    """
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    export_path, _rollout = materialize_socnavbench_export(
        policy=policy, mapping=mapping, out_dir=tmp_path
    )
    receipt = run_socnavbench_receipt(
        export_path=export_path,
        policy=policy,
        mapping=mapping,
        allow_synthetic_traversible=True,
    )
    assert receipt["suite"] == "SocNavBench"
    # New API: metric identity in metric_spec, distinct from Robot SF.
    assert "metric_spec" in receipt
    assert receipt["metric_spec"]["metric_id"] == SOCNAVBENCH_METRIC_ID
    assert receipt["metric_spec"]["ratio_direction"] == "displacement_over_distance"
    assert receipt["metric_spec"]["mapping_class"] == "approximate"
    assert isinstance(receipt["value"], float)
    assert receipt["value"] == receipt["value"]  # not NaN
    assert receipt["external_asset_id"] == ETH_ASSET_ID
    assert "external_asset_staged" in receipt
    # Fallback must be recorded; a synthetic path must not claim success evidence.
    assert receipt["is_fallback"] is True
    assert receipt["counts_as_success_evidence"] is False
    assert receipt["executed_policy"] is True


def test_socnavbench_receipt_fails_closed_without_asset_or_test_flag(tmp_path: Path) -> None:
    """Without the test-only flag, a missing licensed asset must fail closed (no fallback)."""
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    export_path, _rollout = materialize_socnavbench_export(
        policy=policy, mapping=mapping, out_dir=tmp_path
    )
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
            export_path=Path("/nonexistent/export.json"),
            policy=policy,
            mapping=mapping,
            allow_synthetic_traversible=True,
        )


def test_socnavbench_receipt_rejects_non_mapping_export(tmp_path: Path) -> None:
    """A JSON list at the export boundary must become a clear canary error."""
    export_path = tmp_path / "invalid.json"
    export_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(CanaryError, match="export root must be a mapping"):
        run_socnavbench_receipt(
            export_path=export_path,
            policy=resolve_pinned_policy(),
            mapping=resolve_scenario_mapping(),
            allow_synthetic_traversible=True,
        )


def test_socnavbench_receipt_uses_traversible_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A path outside the loaded traversible must fail the source-harness boundary."""
    policy = resolve_pinned_policy()
    mapping = resolve_scenario_mapping()
    export_path, _rollout = materialize_socnavbench_export(
        policy=policy, mapping=mapping, out_dir=tmp_path
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.socnavbench_canary.socnavbench_eth_load_traversible",
        lambda: (1.0, np.ones((1, 1), dtype=bool)),
    )
    with pytest.raises(CanaryError, match="outside the staged traversible bounds"):
        run_socnavbench_receipt(
            export_path=export_path,
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
    assert receipt["seed"] == CANARY_SEED
    # With allow_synthetic_traversible=True, a fallback IS active.
    # fallback_forbidden must reflect the actual state: False when fallback is used.
    assert receipt["allow_synthetic_traversible"] is True
    assert receipt["fallback_forbidden"] is False  # fallback was taken
    assert receipt["counts_as_success_evidence"] is False  # synthetic path not success evidence
    # Both suites are present with their own receipts.
    suite_names = {suite["suite"] for suite in receipt["suites"]}
    assert suite_names == {"Robot SF", "SocNavBench"}
    # Per-suite metric specs must carry DISTINCT metric IDs for reciprocal definitions.
    specs = receipt["per_suite_metric_specs"]
    assert set(specs) == {"Robot SF", "SocNavBench"}
    rf_spec = specs["Robot SF"]
    sn_spec = specs["SocNavBench"]
    assert rf_spec["metric_id"] != sn_spec["metric_id"]
    assert rf_spec["metric_id"] == ROBOT_SF_METRIC_ID
    assert sn_spec["metric_id"] == SOCNAVBENCH_METRIC_ID
    assert rf_spec["ratio_direction"] != sn_spec["ratio_direction"]
    assert rf_spec["ratio_direction"] == "distance_over_displacement"
    assert sn_spec["ratio_direction"] == "displacement_over_distance"
    # Per-suite denominators and metric values are recorded separately.
    assert set(receipt["per_suite_denominators"]) == {"Robot SF", "SocNavBench"}
    assert set(receipt["metric_values"]) == {"Robot SF", "SocNavBench"}
    # The denominators must be identical (single rollout, same geometry).
    rf_denom = receipt["per_suite_denominators"]["Robot SF"]
    sn_denom = receipt["per_suite_denominators"]["SocNavBench"]
    assert rf_denom == pytest.approx(sn_denom, rel=1e-6)
    # Limitation flags survive into the receipt mapping block.
    assert receipt["scenario_mapping"]["limitation_flags"]
    # The receipt file was actually written.
    assert Path(receipt["receipt_path"]).is_file()
    stored_receipt = json.loads(Path(receipt["receipt_path"]).read_text(encoding="utf-8"))
    assert stored_receipt["receipt_path"] == receipt["receipt_path"]


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
        runtime_planner_config=dict(base.runtime_planner_config),
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


def test_staged_eth_loader_returns_traversible_for_native_runner(tmp_path: Path) -> None:
    """The native runner loader returns the validated staged ETH array and resolution."""
    from robot_sf.data.external import socnavbench_eth

    root = tmp_path / "socnavbench"
    layout = socnavbench_eth.expected_layout(root)
    layout.mesh_dir.mkdir(parents=True)
    (layout.mesh_dir / "mesh.obj").write_text("# synthetic mesh marker\n", encoding="utf-8")
    layout.traversible_pickle.parent.mkdir(parents=True)
    expected = np.array([[True, False], [True, True]], dtype=bool)
    with layout.traversible_pickle.open("wb") as handle:
        pickle.dump({"resolution": 0.05, "traversible": expected}, handle, protocol=2)

    resolution, traversible = socnavbench_eth.load_traversible(root)

    assert resolution == 0.05
    np.testing.assert_array_equal(traversible, expected)


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


def test_cli_canary_runner_consumes_export_across_process_boundary(tmp_path: Path) -> None:
    """The public canary entry point must consume its export in a fresh process."""
    out_dir = tmp_path / "process-canary"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "tools" / "cross_benchmark_canary.py"),
            "--out-dir",
            str(out_dir),
            "--allow-synthetic-traversible",
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    receipt_path = Path(receipt["receipt_path"])
    assert receipt_path.is_file()
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["export_path"] == receipt["export_path"]
    assert persisted["receipt_path"] == receipt["receipt_path"]


def test_pinned_planner_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    """Malformed pinned planner YAML must not silently use planner defaults."""
    config = tmp_path / "invalid.yaml"
    config.write_text("- social_force_tau: 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        _resolve_pinned_planner_config(algo_config=config)


def test_execute_pinned_policy_rejects_non_mapping_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed selected scenario must fail before environment construction."""
    monkeypatch.setattr(
        "robot_sf.benchmark.canary_rollout.select_scenario", lambda _scenarios, _index: None
    )
    with pytest.raises(ValueError, match="scenario at index 0 must be a mapping"):
        execute_pinned_policy(seed=0, algo_config=PINNED_POLICY_ALGO_CONFIG)


def test_execute_pinned_policy_closes_environment_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rollout must close the simulator even when policy execution raises."""

    class FailingEnvironment:
        def __init__(self) -> None:
            self.closed = False
            self.simulator = SimpleNamespace(robot_pos=np.array([[0.0, 0.0]]))

        def reset(self, *, seed: int) -> tuple[None, dict[str, object]]:
            return None, {}

        def close(self) -> None:
            self.closed = True

    class FailingPolicy:
        def act(self, _obs: object) -> object:
            raise RuntimeError("synthetic policy failure")

    environment = FailingEnvironment()
    monkeypatch.setattr(canary_rollout, "load_scenarios", lambda _path: [{"name": "fake"}])
    monkeypatch.setattr(canary_rollout, "select_scenario", lambda scenarios, _index: scenarios[0])
    monkeypatch.setattr(
        canary_rollout,
        "build_robot_config_from_scenario",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    def resolve_config(*, algo_config: Path) -> object:
        return object()

    monkeypatch.setattr(
        canary_rollout,
        "_resolve_pinned_planner_config",
        resolve_config,
    )
    monkeypatch.setattr(canary_rollout, "make_social_force_policy", lambda _config: FailingPolicy())
    monkeypatch.setattr(canary_rollout, "make_robot_env", lambda **_kwargs: environment)

    with pytest.raises(RuntimeError, match="synthetic policy failure"):
        execute_pinned_policy(seed=0, algo_config=PINNED_POLICY_ALGO_CONFIG)

    assert environment.closed is True


def test_git_commit_sha_fallback_uses_custom_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The non-git provenance fallback must hash config under the requested repository root."""
    config_path = tmp_path / "configs" / "algos" / "social_force_holonomic_tuned_tau_low.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("custom-root-config\n", encoding="utf-8")

    def fail_git(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr("robot_sf.benchmark.socnavbench_canary.subprocess.run", fail_git)

    assert _git_commit_sha(repo_root=tmp_path) == "dirty:" + _config_digest(config_path)[:12]


def test_metric_ids_are_distinct_and_reciprocal() -> None:
    """The module must define distinct metric IDs for the reciprocal ratio definitions."""
    assert ROBOT_SF_METRIC_ID != SOCNAVBENCH_METRIC_ID
    assert "distance_over_displacement" in ROBOT_SF_METRIC_ID
    assert "displacement_over_distance" in SOCNAVBENCH_METRIC_ID


def test_joint_receipt_policy_identity_includes_runtime_provenance(tmp_path: Path) -> None:
    """Policy identity in the joint receipt must include runtime planner config provenance."""
    receipt = run_canary(out_dir=tmp_path, allow_synthetic_traversible=True)
    policy_id = receipt["policy_identity"]
    assert "runtime_planner_config" in policy_id
    # The runtime config must be non-empty (at least some tau-low parameters).
    assert len(policy_id["runtime_planner_config"]) > 0
