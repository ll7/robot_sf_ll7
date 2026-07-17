"""Test for issue #5948 doorway butterfly trace re-export config provenance."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestDoorwayButterflyReexportConfig:
    """Test the reconstructed config for job 13483 (issue #5948)."""

    @pytest.fixture
    def config_path(self) -> Path:
        return REPO_ROOT / "configs" / "benchmarks" / "doorway_butterfly_trace_reexport.yaml"

    @pytest.fixture
    def provenance_path(self) -> Path:
        return (
            REPO_ROOT
            / "docs"
            / "context"
            / "evidence"
            / "issue_5948_doorway_provenance_2026-07-17"
            / "provenance_manifest.json"
        )

    @pytest.fixture
    def config_hash_path(self) -> Path:
        """Return the committed SHA-256 record for the reconstructed config."""
        return (
            REPO_ROOT
            / "docs"
            / "context"
            / "evidence"
            / "issue_5948_doorway_provenance_2026-07-17"
            / "config_sha256.txt"
        )

    def test_config_file_exists(self, config_path: Path) -> None:
        """Verify the reconstructed config file exists."""
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_config_valid_yaml(self, config_path: Path) -> None:
        """Verify the config is valid YAML."""
        with config_path.open() as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)

    def test_config_has_required_fields(self, config_path: Path) -> None:
        """Verify the config has the key reconstruction fields."""
        with config_path.open() as f:
            config = yaml.safe_load(f)

        # Core identification
        assert config["name"] == "doorway_butterfly_trace_reexport"
        assert config["paper_facing"] is False

        # Scenario
        assert (
            config["scenario_matrix"] == "configs/scenarios/classic_interactions_francis2023.yaml"
        )
        assert config["scenario_candidates"] == ["classic_doorway_medium"]

        # Seeds (30-seed set 111-140, not the minimal [113, 114])
        assert config["seed_policy"]["mode"] == "fixed-list"
        expected_seeds = list(range(111, 141))
        assert config["seed_policy"]["seeds"] == expected_seeds

        # Runtime
        assert config["horizon"] == 600
        assert config["dt"] == 0.1
        assert config["workers"] == 1

        # SNQI
        assert "snqi_weights" in config
        assert "snqi_baseline" in config
        assert config["snqi_contract"]["enabled"] is True
        assert config["snqi_contract"]["enforcement"] == "warn"

        # AMV
        assert config["amv_contract"]["enabled"] is True
        assert config["amv_contract"]["status"] == "pass"
        assert config["amv_contract"]["profile"] == "amv-paper-v1"

        # Observation noise disabled
        assert config["observation_noise"]["enabled"] is False
        assert config["doi"] is None

        # Planner
        assert len(config["planners"]) == 1
        assert config["planners"][0]["key"] == "ppo"
        assert config["planners"][0]["algo"] == "ppo"
        assert config["planners"][0]["algo_config"] == "configs/baselines/ppo_15m_grid_socnav.yaml"

    def test_provenance_manifest_exists(self, provenance_path: Path) -> None:
        """Verify the provenance manifest file exists."""
        assert provenance_path.exists(), f"Provenance manifest not found: {provenance_path}"

    def test_provenance_manifest_valid_json(self, provenance_path: Path) -> None:
        """Verify the provenance manifest is valid JSON."""
        with provenance_path.open() as f:
            manifest = json.load(f)
        assert isinstance(manifest, dict)

    def test_provenance_manifest_has_required_fields(self, provenance_path: Path) -> None:
        """Verify the provenance manifest has the key fields."""
        with provenance_path.open() as f:
            manifest = json.load(f)

        assert manifest["schema"] == "robot_sf_run_config_provenance.v1"
        assert manifest["issue"] == 5948
        assert manifest["run_identification"]["slurm_job_id"] == 13483
        assert manifest["run_identification"]["config_hash_from_run"] == "846e99aaba7dff51"
        assert manifest["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
        assert (
            manifest["durable_bundle"]["sha256"]
            == "1a434946d774a5550ec3791ec6a829768fab5ce058c7a9ec4db9d43711780dff"
        )
        assert "rule_established" in manifest

    def test_config_matches_provenance(self, config_path: Path, provenance_path: Path) -> None:
        """Verify the config matches the provenance manifest reconstruction."""
        with config_path.open() as f:
            config = yaml.safe_load(f)
        with provenance_path.open() as f:
            manifest = json.load(f)

        # Check key parameters match
        recon = manifest["config_reconstruction"]["key_parameters"]
        assert config["scenario_matrix"] == recon["scenario_matrix"]
        assert config["scenario_candidates"] == [recon["scenario_candidate"]]
        assert config["planners"][0]["algo_config"] == recon["planner_config"]
        assert config["seed_policy"]["seeds"] == recon["seeds"]
        assert config["horizon"] == recon["horizon"]
        assert config["dt"] == recon["dt"]
        assert config["workers"] == recon["workers"]
        assert config["snqi_weights"] == recon["snqi_weights"]
        assert config["snqi_contract"]["enforcement"] == recon["snqi_enforcement"]
        assert config["amv_contract"]["profile"] == recon["amv_profile"]
        assert config["amv_contract"]["status"] == recon["amv_status"]
        assert config["observation_noise"]["enabled"] == recon["observation_noise_enabled"]
        assert config["paper_facing"] == recon["paper_facing"]

    def test_config_hash_record_matches_config(
        self, config_path: Path, config_hash_path: Path
    ) -> None:
        """Verify the README-listed config hash is bound to the committed bytes."""
        assert config_hash_path.exists()
        lines = config_hash_path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "# AI-GENERATED NEEDS-REVIEW"
        recorded_hash, recorded_path = lines[1].split(maxsplit=1)
        assert recorded_path == "configs/benchmarks/doorway_butterfly_trace_reexport.yaml"
        actual_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert recorded_hash == actual_hash
