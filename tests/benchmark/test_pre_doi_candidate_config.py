"""Publication-free hashes must leave every existing campaign hash unchanged."""

import hashlib
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark import release_protocol
from robot_sf.benchmark.camera_ready import _preflight, campaign
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig
from robot_sf.benchmark.camera_ready._util import (
    _config_hash_payload,
    _jsonable_repo_relative,
)


def test_default_campaign_hash_payload_is_byte_compatible() -> None:
    cfg = CampaignConfig(name="synthetic", scenario_matrix_path=Path("synthetic.yaml"), planners=())
    prior = asdict(cfg)
    prior.pop("publication_identity_mode")
    prior.pop("tuning_run_provenance")
    assert _config_hash_payload(cfg) == _jsonable_repo_relative(prior)


def test_pre_doi_scientific_projection_removes_only_publication_coordinates() -> None:
    cfg = CampaignConfig(name="synthetic", scenario_matrix_path=Path("synthetic.yaml"), planners=())
    candidate = replace(
        cfg, publication_identity_mode="scientific_candidate", release_tag="", doi=""
    )
    bound = _config_hash_payload(cfg)
    scientific = _config_hash_payload(candidate)
    bound.pop("release_tag")
    bound.pop("doi")
    assert scientific == bound
    assert "publication_identity_mode" not in scientific


def test_candidate_admission_rejects_wrong_template_source_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / (
        "paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_template.yaml"
    )
    template.write_text("synthetic: true\n", encoding="utf-8")
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    cfg = CampaignConfig(
        name="synthetic",
        scenario_matrix_path=Path("synthetic.yaml"),
        planners=(),
        release_tag="",
        doi="",
        export_publication_bundle=False,
        publication_identity_mode="scientific_candidate",
        source_config_path=template,
        source_config_sha256=digest,
    )
    kwargs = {
        "cfg": cfg,
        "baseline_manifest": {},
        "source_sha": "a" * 40,
        "checkpoint_receipt": {"submit_safe": False},
        "checkpoint_receipt_sha256": "b" * 64,
        "repository_root": tmp_path,
    }
    with pytest.raises(ValueError, match="exact dedicated campaign template"):
        release_protocol.build_scientific_candidate_identity(**kwargs)

    monkeypatch.setattr(release_protocol, "SCIENTIFIC_CANDIDATE_TEMPLATE_SHA256", digest)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", template.name], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    with pytest.raises(ValueError, match="source_commit"):
        release_protocol.build_scientific_candidate_identity(**kwargs)

    monkeypatch.setattr(release_protocol, "_require_clean_exact_checkout", lambda *_, **__: None)
    with pytest.raises(ValueError, match="submit-safe checkpoint receipt"):
        release_protocol.build_scientific_candidate_identity(**kwargs)


def test_candidate_preflight_and_summary_omit_publication_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = CampaignConfig(
        name="synthetic",
        scenario_matrix_path=Path("synthetic.yaml"),
        planners=(),
        release_tag="",
        doi="",
        publication_identity_mode="scientific_candidate",
    )
    ledger = {
        "schema_version": "fixture",
        "record_schema_version": "fixture",
        "ledger_sha256": "a" * 64,
        "records": [],
        "summary": {},
        "policy": {},
    }
    preflight = _preflight._build_manifest_execution_block(
        cfg, [], tuning_ledger=ledger, tuning_ledger_path=tmp_path / "ledger.json"
    )
    assert "release_tag" not in preflight and "doi" not in preflight

    monkeypatch.setattr(campaign, "_build_campaign_execution_metadata", lambda *_, **__: {})
    paths = SimpleNamespace(campaign_id="synthetic", manifest_payload={})
    snqi = SimpleNamespace(
        weights_sha256="a" * 64,
        baseline_sha256="b" * 64,
        contract_eval=SimpleNamespace(
            status="pass",
            rank_alignment_spearman=None,
            outcome_separation=None,
            dominant_component=None,
            dominant_component_mean_abs=None,
        ),
        positioning={},
    )
    summary = campaign._build_campaign_metadata_section(
        cfg,
        paths=paths,
        outcome=object(),
        snqi=snqi,
        run_entries=[],
        kinematics_matrix=("differential_drive",),
        invoked_command=None,
    )
    assert (
        not {"release_tag", "doi", "doi_url", "release_url", "release_asset_url"} & summary.keys()
    )

    class Paths:
        campaign_id = "synthetic"
        campaign_root = tmp_path
        reports_dir = tmp_path / "reports"

        def __getattr__(self, name: str) -> Path:
            return tmp_path / name

    class Tables(dict):
        def __missing__(self, name: str) -> Path:
            return tmp_path / name

    artifacts = campaign._build_campaign_artifacts_section(
        Paths(),
        SimpleNamespace(
            **{
                name: tmp_path / name
                for name in (
                    "snqi_diagnostics_json_path",
                    "snqi_diagnostics_md_path",
                    "snqi_sensitivity_csv_path",
                )
            }
        ),
        Tables(),
        publication_identity_mode="scientific_candidate",
    )
    assert (
        not {"expected_release_archive", "release_url", "release_asset_url", "doi_url"}
        & artifacts.keys()
    )
