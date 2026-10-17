"""D-086 defence-in-depth witnesses; marked bytes and no environment execution.

Test-value gate: protect renamed-marker admission, forged development inventories,
and marker-gated SNQI commitment checks. The nearest rehearsal tests use the
original marker and resolver-validated seeds; the SNQI receipt test is non-diagnostic.
Each witness fails on main for its stated defect, with no production test seam.
"""
# seed-holdout: synthetic-fixture begin

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.benchmark import (
    artifact_publication,
    release_acceptance,
    release_doctor,
    release_protocol,
    release_publication_contract,
    runtime_smoke_admission,
    zenodo_publisher,
)
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, SeedPolicy
from robot_sf.evidence.writers import write_json
from scripts.tools import run_benchmark_release as runner
from tests.benchmark.test_release_campaign_authority import no_execution as _no_execution

no_execution = _no_execution
ROOT = Path(__file__).resolve().parents[2]
PATCHED_KIND = "renamed-development-rehearsal"


@pytest.fixture
def renamed_marker(tmp_path, monkeypatch):
    """Patch the canonical constant and its from-import bindings, then write real bytes."""
    for module in (
        release_protocol,
        release_acceptance,
        release_doctor,
        release_publication_contract,
        runtime_smoke_admission,
        artifact_publication,
        zenodo_publisher,
    ):
        # Main lacks some imports: keep the witness runnable there, where the
        # unchanged literal comparison must fail the behavioral assertion.
        monkeypatch.setattr(module, "DEVELOPMENT_REHEARSAL_KIND", PATCHED_KIND, raising=False)
    path = tmp_path / "release/release_result.json"
    path.parent.mkdir(parents=True)
    write_json(
        path,
        {
            "release_kind": PATCHED_KIND,
            "release_tag": "ordinary-source-tag",
            "metadata_path": str(path),
            "metadata_sha256": "a" * 64,
            "concept_doi": "10.5281/zenodo.99000001",
            "version_doi": "10.5281/zenodo.99000002",
        },
    )
    return path, SimpleNamespace(**json.loads(path.read_text()))


@pytest.mark.parametrize(
    "boundary",
    ["acceptance", "mint", "publication", "runtime-smoke", "doi", "export", "marker", "preflight"],
)
def test_renamed_marker_remains_non_releasable(  # noqa: C901, PLR0915 - eight boundary witnesses
    renamed_marker, tmp_path, monkeypatch, boundary
):
    """Every one of the eight boundary sites recognizes a renamed canonical marker."""
    path, manifest = renamed_marker
    if boundary == "acceptance":
        reached = []

        def downstream(*args, **kwargs):
            reached.append((args, kwargs))
            return {"status": "valid", "benchmark_success": True, "blockers": []}

        monkeypatch.setattr(
            release_acceptance, "_validate_benchmark_campaign_acceptance", downstream
        )
        report = release_acceptance.validate_full_benchmark_release_acceptance(
            tmp_path, manifest=manifest
        )
        assert report["blockers"] == [
            "development rehearsal cannot satisfy full release acceptance"
        ]
        assert report["benchmark_success"] is False
        assert reached == []
    elif boundary == "mint":
        calls = []

        def load(*args, **kwargs):
            calls.append((args, kwargs))
            return manifest

        monkeypatch.setattr(release_doctor, "load_release_manifest", load)
        manifest.canonical_campaign_config_path = path

        def downstream(*args, **kwargs):
            calls.append((args, kwargs))
            raise ValueError("mint validator reached")

        monkeypatch.setattr(release_doctor, "load_campaign_config", downstream)
        report, loaded, cfg = release_doctor._manifest_check(path, 20160)
        assert report.summary == "development rehearsal cannot be minted as a release"
        assert report.status == "fail"
        assert loaded is manifest and cfg is None and len(calls) == 1
    elif boundary == "publication":
        from tests.benchmark.test_release_publication_contract import _build_fixture

        campaign, bundle = _build_fixture(tmp_path)
        result_path = campaign / "release/release_result.json"
        result = json.loads(result_path.read_text())
        result["release_kind"] = PATCHED_KIND
        write_json(result_path, result)
        report = release_publication_contract.validate_release_publication_contract(
            campaign, bundle, expected_release_tag="v1"
        )
        assert report["blockers"] == ["development rehearsal cannot be published as a release"]
    elif boundary == "runtime-smoke":
        with pytest.raises(
            runtime_smoke_admission.RuntimeSmokeAdmissionError,
            match="development rehearsal cannot satisfy release runtime smoke",
        ):
            runtime_smoke_admission.validate_runtime_smoke_result(
                path, repo_root=tmp_path, expected_source_commit="a" * 40, expected_planner_keys=()
            )
    elif boundary == "doi":
        # A neutral tag isolates the marker check from the separate tag-prefix guard.
        with pytest.raises(
            zenodo_publisher.ZenodoPublisherError, match="development rehearsal cannot"
        ):
            zenodo_publisher.build_release_binding(manifest)
    elif boundary == "marker":
        (tmp_path / "release").mkdir(exist_ok=True)
        write_json(
            tmp_path / "release/release_result.json",
            {
                "benchmark_release": {"release_kind": PATCHED_KIND},
                "release_eligible": False,
                "release_benchmark_success": False,
            },
        )
        violations = []
        artifact_publication._preflight_check_development_marker(
            tmp_path, {"release_eligible": False}, violations
        )
        assert violations == ["development rehearsal marker differs between payload and bundle"]
    else:
        from tests.benchmark.test_artifact_publication import _make_run

        run = tmp_path / "run"
        _make_run(run, with_video=False)
        (run / "episodes/episodes.jsonl").unlink()
        (run / "runs/goal__differential_drive").mkdir(parents=True)
        (run / "release").mkdir()
        write_json(
            run / "runs/goal__differential_drive/episodes.jsonl",
            {"episode_id": "ep-1", "event_ledger": {"software_commit": "abc123"}},
            indent=None,
        )
        write_json(
            run / "release/release_result.json",
            {
                "benchmark_release": {"release_kind": PATCHED_KIND},
                "release_eligible": False,
                "release_benchmark_success": False,
            },
        )
        bundle = artifact_publication.export_publication_bundle(
            run, tmp_path / "publication", release_tag="v1", doi="10.5281/zenodo.99000002"
        )
        if boundary == "export":
            payload = json.loads(bundle.manifest_path.read_text())
            assert payload.get("release_kind") == PATCHED_KIND
            assert payload["release_eligible"] is False
        else:
            # Independently provide the signed manifest marker to isolate the
            # final preflight-report boundary from exporter marker propagation.
            payload = json.loads(bundle.manifest_path.read_text())
            payload.update(release_kind=PATCHED_KIND, release_eligible=False)
            write_json(bundle.manifest_path, payload)
            report = artifact_publication.verify_publication_bundle_preflight(
                bundle.bundle_dir, require_release_reconciliation=False
            )
            assert report["status"] == "pass"
            assert report.get("release_eligible") is False


@pytest.mark.parametrize("seeds", [(1031,), (1000,), (1001, 1031), (111,), (50036,)])
def test_handwritten_marker_seed_policy_refuses_non_development_seeds(tmp_path, seeds):
    """A hand-written manifest cannot bypass dev inventory validation with its marker."""
    source = ROOT / "configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml"
    payload = yaml.safe_load(source.read_text())
    for key in ("canonical_campaign_config", "citation_path", "release_checklist_path"):
        payload[key] = str((source.parent / payload[key]).resolve())
    payload["scenario"]["matrix_path"] = str(
        (source.parent / payload["scenario"]["matrix_path"]).resolve()
    )
    for key in ("snqi_weights_path", "snqi_baseline_path"):
        payload["metrics"][key] = str((source.parent / payload["metrics"][key]).resolve())
    payload["release_kind"] = "development_rehearsal"
    payload["seed_policy"] = {"mode": "fixed-list", "seed_set": None, "seeds": list(seeds)}
    path = tmp_path / "handwritten-marker.json"
    write_json(path, payload)
    manifest = release_protocol.load_release_manifest(path)
    cfg = release_protocol.load_campaign_config(manifest.canonical_campaign_config_path)
    cfg = replace(
        cfg, seed_policy=replace(cfg.seed_policy, mode="fixed-list", seed_set=None, seeds=seeds)
    )
    payload["seed_policy"]["seed_sets_path"] = str(cfg.seed_policy.seed_sets_path)
    write_json(path, payload)
    manifest = release_protocol.load_release_manifest(path)
    problems = []
    release_protocol._validate_release_seed_policy(manifest, cfg, problems)
    assert "development rehearsal accepts only unique development seeds 1001-1030" in problems


@pytest.mark.parametrize(
    ("diagnostic", "marker", "checked"),
    [
        (True, None, True),
        (True, "development_rehearsal", False),
        (False, "development_rehearsal", True),
    ],
)
def test_receipt_skips_commitment_only_for_marked_diagnostic(tmp_path, diagnostic, marker, checked):
    """Diagnostic specs still check commitments unless the manifest carries D-086."""
    matrix = tmp_path / "scenarios.json"
    write_json(
        matrix,
        {
            "scenarios": [
                {
                    "name": "static-only",
                    "map_file": str(ROOT / "maps/svg_maps/classic_crossing.svg"),
                }
            ]
        },
    )
    calls = []
    spec = SimpleNamespace(diagnostic=diagnostic, validate_evaluation_commitment=calls.append)
    cfg = CampaignConfig(
        "static-seed-receipt",
        matrix,
        (),
        seed_policy=SeedPolicy(mode="fixed-list", seeds=(1001,)),
        snqi_v2_spec=spec,
    )
    # Without a marker, exercise the existing public helper signature on main.
    args = {} if marker is None else {"manifest": SimpleNamespace(release_kind=marker)}
    receipt = runner._snqi_v2_evaluation_seed_receipt(cfg, **args)
    assert calls == ([[1001]] if checked else [])
    assert receipt == {
        "snqi_v2_evaluation_seeds_sha256": "1ba07180d25194b104a42b91ad2cef4bf53b3d9f62b0a77fab502e87fd640d19"
    }


# seed-holdout: synthetic-fixture end
