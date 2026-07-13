"""Tests for issue #5355 prediction-MPC 2x2 factorial design.

Validates config parsing, toggle behavior, and preregistration harness
for the prediction on/off x constraint-handling on/off factorial.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.planner.nmpc_social import _RolloutContext
from robot_sf.planner.prediction_mpc import (
    NullPedestrianPredictor,
    PredictionMPCConfig,
    PredictionMPCPlannerAdapter,
    build_prediction_mpc_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _minimal_observation(
    *,
    ped_positions: list[list[float]] | None = None,
    ped_velocities: list[list[float]] | None = None,
) -> dict:
    """Build a minimal SocNav observation for testing."""
    if ped_positions is None:
        ped_positions = [[1.0, 0.0]]
    if ped_velocities is None:
        ped_velocities = [[0.0, 0.0]]
    return {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "speed": [0.1],
            "radius": [0.25],
        },
        "goal": {"current": [5.0, 0.0], "next": [5.0, 0.0]},
        "pedestrians": {
            "positions": [coord for pos in ped_positions for coord in pos],
            "velocities": [coord for vel in ped_velocities for coord in vel],
            "count": [len(ped_positions)],
            "radius": [0.25],
        },
    }


class TestPredictionMPCFactorialConfig:
    """Test config parsing for the 2x2 factorial arms."""

    def test_default_config_has_factorial_fields(self):
        config = PredictionMPCConfig()
        assert config.hard_pedestrian_constraints_enabled is True
        assert config.pedestrian_clearance_weight == 0.0

    def test_build_config_from_yaml_A0_B0(self):
        cfg = {
            "predictor_backend": "none",
            "hard_pedestrian_constraints_enabled": False,
            "pedestrian_clearance_weight": 4.5,
        }
        config = build_prediction_mpc_config(cfg)
        assert config.predictor_backend == "none"
        assert config.hard_pedestrian_constraints_enabled is False
        assert config.pedestrian_clearance_weight == 4.5

    def test_build_config_from_yaml_A1_B1(self):
        cfg = {
            "predictor_backend": "constant_velocity",
            "hard_pedestrian_constraints_enabled": True,
            "pedestrian_clearance_weight": 0.0,
        }
        config = build_prediction_mpc_config(cfg)
        assert config.predictor_backend == "constant_velocity"
        assert config.hard_pedestrian_constraints_enabled is True
        assert config.pedestrian_clearance_weight == 0.0

    def test_build_config_string_bool_parsing(self):
        cfg = {
            "hard_pedestrian_constraints_enabled": "true",
            "pedestrian_clearance_weight": "0.0",
        }
        config = build_prediction_mpc_config(cfg)
        assert config.hard_pedestrian_constraints_enabled is True
        assert config.pedestrian_clearance_weight == 0.0

    def test_soft_factor_b_requires_positive_pedestrian_clearance_weight(self):
        with pytest.raises(ValueError, match="soft pedestrian clearance"):
            PredictionMPCConfig(hard_pedestrian_constraints_enabled=False)


class TestNullPedestrianPredictor:
    """Test the null predictor for Factor A OFF."""

    def test_null_predictor_holds_current_positions_without_predicting_velocity(self):
        predictor = NullPedestrianPredictor()
        obs = _minimal_observation()
        futures = predictor.predict(obs, horizon_steps=6, dt=0.25)
        assert futures.positions_world.shape == (1, 6, 2)
        assert np.all(futures.positions_world == [[1.0, 0.0]])
        assert futures.mask.tolist() == [1.0]
        assert futures.source == "current_position_hold"

    def test_null_predictor_zero_pedestrians(self):
        predictor = NullPedestrianPredictor()
        obs = _minimal_observation(ped_positions=[], ped_velocities=[])
        futures = predictor.predict(obs, horizon_steps=4, dt=0.25)
        assert futures.positions_world.shape[0] == 0


class TestFactorAToggle:
    """Test prediction ON/OFF toggle behavior."""

    def test_none_backend_uses_null_predictor(self):
        config = PredictionMPCConfig(predictor_backend="none")
        adapter = PredictionMPCPlannerAdapter(config=config)
        assert isinstance(adapter._future_predictor, NullPedestrianPredictor)

    def test_null_backend_uses_null_predictor(self):
        config = PredictionMPCConfig(predictor_backend="null")
        adapter = PredictionMPCPlannerAdapter(config=config)
        assert isinstance(adapter._future_predictor, NullPedestrianPredictor)

    def test_cv_backend_uses_cv_predictor(self):
        config = PredictionMPCConfig(predictor_backend="constant_velocity")
        adapter = PredictionMPCPlannerAdapter(config=config)
        assert not isinstance(adapter._future_predictor, NullPedestrianPredictor)

    def test_prediction_off_preserves_hard_constraints_from_factor_b(self):
        config = PredictionMPCConfig(
            predictor_backend="none",
            hard_pedestrian_constraints_enabled=True,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        context = _RolloutContext(
            robot_pos=np.array([0.0, 0.0]),
            heading=0.0,
            current_speed=0.1,
            goal=np.array([5.0, 0.0]),
            ped_positions=np.array([[1.0, 0.0]]),
            ped_velocities=np.array([[0.0, 0.0]]),
            robot_radius=0.25,
            ped_radius=0.25,
            pedestrian_uncertainty_envelope_enabled=False,
            pedestrian_uncertainty_alpha_mps=0.0,
            observation=obs,
            speed_cap=0.9,
        )
        assert len(adapter._optimizer_constraints(context)) == 1

    def test_prediction_off_freezes_soft_cost_pedestrians(self):
        frozen_adapter = PredictionMPCPlannerAdapter(
            config=PredictionMPCConfig(
                predictor_backend="none",
                hard_pedestrian_constraints_enabled=False,
                pedestrian_clearance_weight=4.5,
            )
        )
        moving_adapter = PredictionMPCPlannerAdapter(
            config=PredictionMPCConfig(
                predictor_backend="constant_velocity",
                hard_pedestrian_constraints_enabled=False,
                pedestrian_clearance_weight=4.5,
            )
        )
        positions = np.array([[1.0, 0.0]])
        velocities = np.array([[0.4, 0.0]])

        assert np.array_equal(
            frozen_adapter._predict_pedestrians(positions, velocities, step_idx=2), positions
        )
        assert np.array_equal(
            moving_adapter._predict_pedestrians(positions, velocities, step_idx=2),
            np.array([[1.3, 0.0]]),
        )


class TestFactorBToggle:
    """Test hard-vs-soft pedestrian-clearance handling."""

    @pytest.mark.parametrize("predictor_backend", ["none", "constant_velocity"])
    def test_constraints_off_arms_remain_functional(self, predictor_backend: str):
        """B0 arms still make finite progress toward an unobstructed goal."""
        config = PredictionMPCConfig(
            predictor_backend=predictor_backend,
            hard_pedestrian_constraints_enabled=False,
            pedestrian_clearance_weight=4.5,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)

        linear, angular = adapter.plan(_minimal_observation(ped_positions=[], ped_velocities=[]))

        assert np.isfinite(linear)
        assert np.isfinite(angular)
        assert linear > 0.0
        assert adapter.diagnostics()["nonzero_command_count"] == 1

    def test_constraints_disabled_returns_empty(self):
        config = PredictionMPCConfig(
            predictor_backend="constant_velocity",
            hard_pedestrian_constraints_enabled=False,
            pedestrian_clearance_weight=4.5,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        assert diag["factorial_toggles"]["hard_pedestrian_constraints_enabled"] is False
        assert diag["factorial_toggles"]["pedestrian_clearance_weight"] == 4.5

    def test_constraints_enabled_returns_nonempty(self):
        config = PredictionMPCConfig(
            predictor_backend="constant_velocity",
            hard_pedestrian_constraints_enabled=True,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        assert diag["factorial_toggles"]["hard_pedestrian_constraints_enabled"] is True


class TestFactorialDiagnostics:
    """Test diagnostics payload includes factorial toggle state."""

    def test_diagnostics_has_factorial_toggles(self):
        config = PredictionMPCConfig(predictor_backend="none")
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        assert "factorial_toggles" in diag
        toggles = diag["factorial_toggles"]
        assert "prediction_enabled" in toggles
        assert "hard_pedestrian_constraints_enabled" in toggles
        assert "pedestrian_clearance_weight" in toggles

    def test_diagnostics_A0_B0_arm(self):
        config = PredictionMPCConfig(
            predictor_backend="none",
            hard_pedestrian_constraints_enabled=False,
            pedestrian_clearance_weight=4.5,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        toggles = diag["factorial_toggles"]
        assert toggles["prediction_enabled"] is False
        assert toggles["hard_pedestrian_constraints_enabled"] is False
        assert toggles["pedestrian_clearance_weight"] == 4.5

    def test_diagnostics_A1_B1_arm(self):
        config = PredictionMPCConfig(
            predictor_backend="constant_velocity",
            hard_pedestrian_constraints_enabled=True,
            pedestrian_clearance_weight=0.0,
        )
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        toggles = diag["factorial_toggles"]
        assert toggles["prediction_enabled"] is True
        assert toggles["hard_pedestrian_constraints_enabled"] is True
        assert toggles["pedestrian_clearance_weight"] == 0.0


class TestFactorialArmConfigs:
    """Test that the four YAML arm configs build valid adapters."""

    ARM_CONFIGS = [
        "configs/algos/prediction_mpc_factorial_A0_B0.yaml",
        "configs/algos/prediction_mpc_factorial_A0_B1.yaml",
        "configs/algos/prediction_mpc_factorial_A1_B0.yaml",
        "configs/algos/prediction_mpc_factorial_A1_B1.yaml",
    ]

    @pytest.mark.parametrize("config_path", ARM_CONFIGS)
    def test_arm_config_loads_and_builds(self, config_path: str):
        path = REPO_ROOT / config_path
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = build_prediction_mpc_config(cfg)
        adapter = PredictionMPCPlannerAdapter(config=config)
        obs = _minimal_observation()
        adapter.plan(obs)
        diag = adapter.diagnostics()
        assert "factorial_toggles" in diag

    def test_arm_configs_cover_all_four_cells(self):
        configs = {}
        for path_str in self.ARM_CONFIGS:
            path = REPO_ROOT / path_str
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            config = build_prediction_mpc_config(cfg)
            arm_key = path.stem.replace("prediction_mpc_factorial_", "")
            configs[arm_key] = config

        assert configs["A0_B0"].predictor_backend == "none"
        assert configs["A0_B0"].hard_pedestrian_constraints_enabled is False
        assert configs["A0_B0"].pedestrian_clearance_weight == 4.5

        assert configs["A0_B1"].predictor_backend == "none"
        assert configs["A0_B1"].hard_pedestrian_constraints_enabled is True
        assert configs["A0_B1"].pedestrian_clearance_weight == 0.0

        assert configs["A1_B0"].predictor_backend == "constant_velocity"
        assert configs["A1_B0"].hard_pedestrian_constraints_enabled is False
        assert configs["A1_B0"].pedestrian_clearance_weight == 4.5

        assert configs["A1_B1"].predictor_backend == "constant_velocity"
        assert configs["A1_B1"].hard_pedestrian_constraints_enabled is True
        assert configs["A1_B1"].pedestrian_clearance_weight == 0.0


class TestPreregistrationHarness:
    """Test the preregistration harness module."""

    def test_validate_arm_configs(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            validate_arm_configs,
        )

        config_path = REPO_ROOT / "configs/research/prediction_mpc_factorial_v1.yaml"
        results = validate_arm_configs(config_path)
        assert len(results) == 4
        for arm_key, result in results.items():
            assert result["valid"] is True, f"{arm_key}: {result.get('error')}"

        assert results["A0_B0"]["pedestrian_clearance_weight"] == 4.5
        assert results["A1_B0"]["pedestrian_clearance_weight"] == 4.5
        assert results["A0_B1"]["pedestrian_clearance_weight"] == 0.0
        assert results["A1_B1"]["pedestrian_clearance_weight"] == 0.0

    def test_preregistration_config_is_pinned_in_evidence_registry(self):
        registry_path = (
            REPO_ROOT
            / "docs/context/evidence/issue_5355_prediction_mpc_factorial_preregistration"
            / "preregistration_config_registry.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

        assert registry["campaign_id"] == "issue_5355_prediction_mpc_factorial_v1"
        config_path = REPO_ROOT / registry["config_path"]
        assert config_path.is_file()
        assert registry["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()

    def test_check_planned_rows_complete(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            EXPECTED_FACTORIAL_ARMS,
            check_planned_rows,
        )

        rows = []
        for scenario_id in ["s1", "s2"]:
            for seed in [111, 112]:
                for arm in EXPECTED_FACTORIAL_ARMS:
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "seed": seed,
                            "factorial_arm": arm,
                        }
                    )
        result = check_planned_rows(rows)
        assert result["complete"] is True
        assert result["row_count"] == 16
        assert result["pair_count"] == 4

    def test_check_planned_rows_incomplete(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            check_planned_rows,
        )

        rows = [
            {"scenario_id": "s1", "seed": 111, "factorial_arm": "A0_B0"},
            {"scenario_id": "s1", "seed": 111, "factorial_arm": "A1_B1"},
        ]
        result = check_planned_rows(rows)
        assert result["complete"] is False
        assert len(result["incomplete_pairs"]) > 0


class TestDependencyStateReconciliationIssue5483:
    """Regression coverage that the pinned factorial config stays reconciled (#5483).

    #5353 (matched-capability fairness contract) is resolved, while #5351
    (hierarchical paired analysis) remains open. The consistency guard protects the
    reconciled #5353 state from silently drifting back to ``open``.
    """

    CONFIG_PATH = REPO_ROOT / "configs/research/prediction_mpc_factorial_v1.yaml"
    REGISTRY_PATH = (
        REPO_ROOT
        / "docs/context/evidence/issue_5355_prediction_mpc_factorial_preregistration"
        / "preregistration_config_registry.json"
    )

    def test_landed_config_declares_5351_open_and_5353_resolved(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            load_factorial_preregistration_config,
        )

        config = load_factorial_preregistration_config(self.CONFIG_PATH)
        by_issue = {int(d["issue"]): d for d in config.get("dependencies", [])}
        assert by_issue[5351]["status"] == "open"
        assert by_issue[5353]["status"] == "resolved"

    def test_landed_config_passes_dependency_consistency_guard(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            check_dependency_state_consistency,
            load_factorial_preregistration_config,
        )

        config = load_factorial_preregistration_config(self.CONFIG_PATH)
        report = check_dependency_state_consistency(config.get("dependencies"))
        assert report["consistent"] is True
        assert report["inconsistent"] == []

    def test_consistency_guard_flags_reopened_closed_dependency(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            check_dependency_state_consistency,
        )

        drifted = [
            {"issue": 5351, "status": "open", "blocking": "analysis machinery"},
            {"issue": 5353, "status": "open", "blocking": "capability matrix"},
        ]
        report = check_dependency_state_consistency(drifted)
        assert report["consistent"] is False
        assert any("#5353" in item for item in report["inconsistent"])

    def test_registry_digest_matches_landed_config(self):
        config_bytes = self.CONFIG_PATH.read_bytes()
        registry = json.loads(self.REGISTRY_PATH.read_text(encoding="utf-8"))
        assert registry["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()

    def test_readiness_gate_still_blocks_on_5351_after_reconciliation(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            assess_campaign_readiness,
        )

        report = assess_campaign_readiness(self.CONFIG_PATH)
        assert report["criteria"]["dependencies_resolved"]["ready"] is False
        assert any("#5351" in blocker for blocker in report["blockers"])

    def test_readiness_gate_consumes_consistency_guard(self, tmp_path):
        """A non-blocking drift on reconciled #5353 still fails the gate."""
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            assess_campaign_readiness,
        )

        config = yaml.safe_load(self.CONFIG_PATH.read_text(encoding="utf-8"))
        for dependency in config["dependencies"]:
            if int(dependency["issue"]) == 5353:
                dependency["status"] = "open"
                dependency["blocking"] = ""
        cfg = tmp_path / "drifted.yaml"
        cfg.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        registry = tmp_path / "registry.json"
        registry.write_text(
            json.dumps(
                {
                    "campaign_id": "issue_5355_prediction_mpc_factorial_v1",
                    "config_path": str(cfg),
                    "config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )

        report = assess_campaign_readiness(cfg, registry_path=registry)
        assert report["ready"] is False
        assert any("#5353" in blocker for blocker in report["blockers"])


class TestDependencyBlockers:
    """Test the pure dependency-resolution helper (prereg §6 gate input)."""

    def test_no_dependencies_block_is_empty(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            dependency_blockers,
        )

        assert dependency_blockers(None) == []
        assert dependency_blockers([]) == []

    def test_open_blocking_dependency_is_a_blocker(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            dependency_blockers,
        )

        deps = [{"issue": 5351, "status": "open", "blocking": "analysis machinery"}]
        blockers = dependency_blockers(deps)
        assert len(blockers) == 1
        assert "#5351" in blockers[0]
        assert "analysis machinery" in blockers[0]

    def test_resolved_dependency_is_not_a_blocker(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            dependency_blockers,
        )

        for state in ("resolved", "closed", "merged", "done"):
            deps = [{"issue": 5351, "status": state, "blocking": "analysis machinery"}]
            assert dependency_blockers(deps) == []

    def test_non_blocking_dependency_is_ignored(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            dependency_blockers,
        )

        # No ``blocking`` reason => informational only, never blocks submission.
        assert dependency_blockers([{"issue": 42, "status": "open"}]) == []

    def test_none_dependency_fields_are_treated_as_missing(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            dependency_blockers,
        )

        assert dependency_blockers([{"issue": 42, "blocking": None, "status": None}]) == []
        blockers = dependency_blockers([{"issue": 42, "blocking": "analysis", "status": None}])
        assert blockers == ["dependency #42 unresolved (status=unset): analysis"]

    def test_malformed_dependencies_raise(self):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            dependency_blockers,
        )

        with pytest.raises(ValueError):
            dependency_blockers("not-a-list")
        with pytest.raises(ValueError):
            dependency_blockers([["not", "a", "mapping"]])


class TestCampaignReadinessGate:
    """Test the fail-closed campaign-readiness gate for issue #5355."""

    CONFIG_PATH = REPO_ROOT / "configs/research/prediction_mpc_factorial_v1.yaml"

    def test_real_config_is_blocked_only_on_open_dependencies(self):
        """After #5483 reconciliation the packet is campaign-ready except for #5351.

        #5353 (capability-matrix contract) is resolved; #5351 (hierarchical
        paired analysis) remains the sole open dependency blocker.
        """
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            assess_campaign_readiness,
        )

        report = assess_campaign_readiness(self.CONFIG_PATH)
        assert report["ready"] is False
        criteria = report["criteria"]
        assert criteria["preregistration_config_valid"]["ready"] is True
        assert criteria["arm_configs_valid"]["ready"] is True
        assert criteria["evidence_registry_pinned"]["ready"] is True
        # ...and the sole remaining gate is the single declared open dependency (#5351).
        assert criteria["dependencies_resolved"]["ready"] is False
        joined = " ".join(report["blockers"])
        assert "#5351" in joined
        assert "#5353" not in joined

    def test_missing_registry_blocks_readiness(self, tmp_path):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            assess_campaign_readiness,
        )

        report = assess_campaign_readiness(
            self.CONFIG_PATH, registry_path=tmp_path / "does_not_exist.json"
        )
        assert report["ready"] is False
        assert report["criteria"]["evidence_registry_pinned"]["ready"] is False

    def test_non_object_registry_blocks_readiness(self, tmp_path):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            assess_campaign_readiness,
        )

        registry = tmp_path / "registry.json"
        registry.write_text("[]", encoding="utf-8")
        report = assess_campaign_readiness(self.CONFIG_PATH, registry_path=registry)
        assert report["ready"] is False
        detail = report["criteria"]["evidence_registry_pinned"]["detail"]
        assert "JSON object/dictionary" in detail

    def test_invalid_config_fails_closed(self, tmp_path):
        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            assess_campaign_readiness,
        )

        bad = tmp_path / "bad.yaml"
        bad.write_text("issue: 5355\nschema_version: wrong\n", encoding="utf-8")
        report = assess_campaign_readiness(bad)
        assert report["ready"] is False
        assert report["criteria"]["preregistration_config_valid"]["ready"] is False

    def test_ready_when_dependencies_resolved_and_registry_matches(self, tmp_path):
        """End-to-end ready verdict once the declared dependencies are marked resolved."""
        import hashlib
        import json

        from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
            assess_campaign_readiness,
        )

        # Copy the real config verbatim but mark the blocking dependencies resolved.
        source = self.CONFIG_PATH.read_text(encoding="utf-8")
        resolved_text = source.replace("status: open", "status: resolved")
        assert "status: open" not in resolved_text
        cfg = tmp_path / "prediction_mpc_factorial_resolved.yaml"
        cfg.write_text(resolved_text, encoding="utf-8")

        # Pin the copy with a matching sha256 registry (arm/scenario paths resolve via cwd).
        registry = tmp_path / "registry.json"
        registry.write_text(
            json.dumps(
                {
                    "campaign_id": "issue_5355_prediction_mpc_factorial_v1",
                    "config_path": str(cfg),
                    "config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )

        report = assess_campaign_readiness(cfg, registry_path=registry)
        assert report["ready"] is True, report["blockers"]
        assert all(c["ready"] for c in report["criteria"].values())
