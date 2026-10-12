"""Publication bundles must retain the exact file inputs named by v2 diagnostics."""

import hashlib
import json
from itertools import product
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.artifact_publication import (
    PublicationPreflightError,
    export_publication_bundle,
    verify_publication_bundle_preflight,
)


def _valid_asset_contents():
    """Build synthetic calibrated metadata; no calibration or simulation is executed."""
    from robot_sf.benchmark.metric_definitions import metric_definitions_sha256
    from robot_sf.benchmark.robot_force_contract import declared_force_source_contract
    from robot_sf.benchmark.snqi.evaluation_seeds import SEALED_EVALUATION_SEEDS_SHA256
    from robot_sf.benchmark.snqi.v2_spec import FAMILY, SIMULATED_FORCE, WEIGHTS

    arms = [f"synthetic-arm-{i}" for i in range(14)]
    scenarios = [f"synthetic-scenario-{i}" for i in range(48)]
    seeds = [1001, 1002]
    grid_hash = hashlib.sha256(
        json.dumps(sorted(product(arms, scenarios, seeds)), separators=(",", ":")).encode()
    ).hexdigest()
    files = {f"runs/{arm}__differential_drive/episodes.jsonl": "a" * 64 for arm in arms}
    anchors = {
        "version": "SNQI-v2.0",
        "status": "frozen",
        "metric_schema_version": "robot-sf-metrics.v2",
        "metric_definitions_sha256": metric_definitions_sha256(),
        "evaluation_seeds_sha256": SEALED_EVALUATION_SEEDS_SHA256,
        "anchors": {
            term: {
                "lower": 0,
                "upper": value,
                "type": "normative" if term in ("T", "N") else "calibration_p95",
            }
            for term, value in {"T": 3, "N": 0.25, "F": 10, "J": 2, "K": 4}.items()
        },
        "force_decision": {
            "source": SIMULATED_FORCE,
            "selected_source_contract": declared_force_source_contract(SIMULATED_FORCE),
            "spearman_rho_F_N": 0.8,
            "threshold_absolute_rho": 0.90,
            "N": "clip(near_misses/steps/0.25)",
            "selected_source_coverage": 1344,
        },
        "calibration": {
            "arms": arms,
            "scenarios": scenarios,
            "seeds": seeds,
            "episode_count": 1344,
            "quantile_method": "linear",
            "benchmark_execution": "nonfallback",
            "command_mode_counts": {arm: {"native": 96} for arm in arms},
            "source_commit": "a" * 40,
            "run_id": "synthetic-test-only",
            "grid_sha256": grid_hash,
            "split_id": f"snqi-v2-dev1001-1002-{grid_hash[:12]}",
            "scenario_horizons": dict.fromkeys(scenarios, 600),
            "episode_files_sha256": files,
            "producer_sidecars_sha256": {name + ".provenance.json": "b" * 64 for name in files},
            "episodes_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "episodes_hash_rule": "sha256(sorted compact JSON relative-path-to-file-sha256 map)",
            "campaign_config_hash": "c" * 64,
            "campaign_config_identity": "c" * 16,
            "campaign_manifest_sha256": "d" * 64,
        },
    }
    return {
        "weights": json.dumps(
            {
                "version": "SNQI-v2.0",
                "weights": {
                    "w_" + term: {"value": value, "rationale": "synthetic test fixture"}
                    for term, value in WEIGHTS.items()
                },
            }
        ),
        "anchors": json.dumps(anchors),
        "family": yaml.safe_dump(FAMILY),
    }


def _set_diagnostics(run, provenance):
    """Update a report without changing its expected schema."""
    (run / "reports/snqi_v2_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": "snqi-v2-diagnostics.v1",
                "episode_count": 1,
                "provenance": provenance,
            }
        )
    )


def _declare_campaign(repo, run, provenance):
    """Bind the v2 asset pins in the campaign as well as the report."""
    schedule = json.loads((repo / provenance["snqi_v2_anchors_path"]).read_text())["calibration"][
        "scenario_horizons"
    ]
    (run / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "legacy_snqi": "excluded",
                "metrics": {"snqi_v2_version": "SNQI-v2", **provenance},
                "scenario_horizons": schedule,
            }
        )
    )
    (run / "preflight").mkdir(exist_ok=True)
    for name in ("validate_config.json", "preview_scenarios.json"):
        (run / "preflight" / name).write_text("{}")


def _resign(bundle):
    """Rebuild the complete inventory, isolating semantic admission from checksum errors."""
    payload = bundle.bundle_dir / "payload"
    manifest = json.loads(bundle.manifest_path.read_text())
    manifest["files"] = [
        {
            "path": file.relative_to(payload).as_posix(),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
        for file in sorted(payload.rglob("*"))
        if file.is_file()
    ]
    bundle.manifest_path.write_text(json.dumps(manifest))
    bundle.checksums_path.write_text(
        "".join(f"{e['sha256']}  payload/{e['path']}\n" for e in manifest["files"])
    )


def _run(tmp_path, monkeypatch):
    """Make a small public export with repository-relative diagnostic inputs."""
    repo = tmp_path / "source"
    repo.mkdir()
    monkeypatch.setattr("robot_sf.benchmark.artifact_publication.get_repository_root", lambda: repo)
    run = repo / "output/campaign"
    (run / "reports").mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"repo": {"commit": "abc123"}}))
    (run / "runs/goal").mkdir(parents=True)
    (run / "runs/goal/episodes.jsonl").write_text(
        json.dumps(
            {
                "event_ledger": {"software_commit": "abc123"},
                "metrics": {
                    "success": True,
                    "total_collision_count": 0,
                    "snqi_v2": 1.0,
                    "snqi_v2_terms": {
                        "S": 1.0,
                        "C": 0.0,
                        "T": 0.0,
                        "N": 0.0,
                        "F": 0.0,
                        "J": 0.0,
                        "K": 0.0,
                    },
                },
            }
        )
        + "\n"
    )
    provenance = {}
    contents = _valid_asset_contents()
    for name, rel in [
        ("weights", "configs/weights.json"),
        ("anchors", "output/calibration/anchors.json"),
        ("family", "configs/family.yaml"),
        ("extra", "configs/input.json"),
    ]:
        source = repo / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(contents.get(name, "fixture-" + name))
        provenance[f"snqi_v2_{name}_path"] = rel
        provenance[f"snqi_v2_{name}_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "reports/snqi_v2_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": "snqi-v2-diagnostics.v1",
                "episode_count": 1,
                "provenance": provenance,
            }
        )
    )
    return repo, run, provenance


def _export(run, tmp_path):
    return export_publication_bundle(
        run, tmp_path / "bundles", release_tag="test-v2", doi="10.5281/zenodo.1234567"
    )


def test_export_includes_every_path_digest_diagnostic_input(tmp_path, monkeypatch):
    """Anchor paths outside the campaign and future additional inputs survive a cold export."""
    repo, run, provenance = _run(tmp_path, monkeypatch)
    bundle = _export(run, tmp_path)
    manifest = json.loads(bundle.manifest_path.read_text())
    entries = {e["path"]: e["sha256"] for e in manifest["files"]}
    for key, rel in provenance.items():
        if not key.endswith("_path"):
            continue
        bundled = "diagnostic_inputs/" + rel
        assert (bundle.bundle_dir / "payload" / bundled).read_bytes() == (repo / rel).read_bytes()
        assert entries[bundled] == provenance[key[:-5] + "_sha256"]
        assert f"{entries[bundled]}  payload/{bundled}" in bundle.checksums_path.read_text()


@pytest.mark.parametrize("mutation", ["missing", "wrong_digest", "unsafe", "no_digest", "symlink"])
def test_export_refuses_unrecoverable_diagnostic_input(tmp_path, monkeypatch, mutation):
    """Exporter cannot produce a future bundle with missing or untrusted input bytes."""
    repo, run, provenance = _run(tmp_path, monkeypatch)
    source = repo / provenance["snqi_v2_anchors_path"]
    if mutation == "missing":
        source.unlink()
    elif mutation == "wrong_digest":
        source.write_text("changed")
    elif mutation == "unsafe":
        provenance["snqi_v2_anchors_path"] = "../outside.json"
    elif mutation == "no_digest":
        del provenance["snqi_v2_anchors_sha256"]
    else:
        source.unlink()
        source.symlink_to(repo / "configs/weights.json")
    (run / "reports/snqi_v2_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": "snqi-v2-diagnostics.v1",
                "episode_count": 1,
                "provenance": provenance,
            }
        )
    )
    with pytest.raises(ValueError, match="diagnostic input"):
        _export(run, tmp_path)


def test_preflight_refuses_missing_input_even_after_resigning_inventory(tmp_path, monkeypatch):
    """Deleting and re-signing an input cannot bypass diagnostic referential integrity."""
    _, run, _ = _run(tmp_path, monkeypatch)
    bundle = _export(run, tmp_path)
    rel = "diagnostic_inputs/output/calibration/anchors.json"
    path = bundle.bundle_dir / "payload" / rel
    path.unlink(missing_ok=True)
    manifest = json.loads(bundle.manifest_path.read_text())
    manifest["files"] = [e for e in manifest["files"] if e["path"] != rel]
    bundle.manifest_path.write_text(json.dumps(manifest))
    bundle.checksums_path.write_text(
        "".join(f"{e['sha256']}  payload/{e['path']}\n" for e in manifest["files"])
    )
    with pytest.raises(PublicationPreflightError, match="diagnostic input.*missing"):
        verify_publication_bundle_preflight(bundle.bundle_dir, require_release_reconciliation=False)


def test_cold_preflight_accepts_without_original_input_locations(tmp_path, monkeypatch):
    """Detached input custody remains verifiable after original source files disappear."""
    repo, run, provenance = _run(tmp_path, monkeypatch)
    bundle = _export(run, tmp_path)
    for key, rel in provenance.items():
        if key.endswith("_path"):
            (repo / rel).unlink()
    report = verify_publication_bundle_preflight(
        bundle.bundle_dir, require_release_reconciliation=False
    )
    assert report["status"] == "pass"


@pytest.mark.parametrize("mutation", ["changed", "unsigned"])
def test_cold_preflight_rejects_re_signed_input_substitution(tmp_path, monkeypatch, mutation):
    """A self-consistent checksum inventory cannot override diagnostic input provenance."""
    _, run, _ = _run(tmp_path, monkeypatch)
    bundle = _export(run, tmp_path)
    rel = "diagnostic_inputs/output/calibration/anchors.json"
    path = bundle.bundle_dir / "payload" / rel
    manifest = json.loads(bundle.manifest_path.read_text())
    if mutation == "changed":
        path.write_text("replacement anchors")
        for entry in manifest["files"]:
            if entry["path"] == rel:
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        manifest["files"] = [e for e in manifest["files"] if e["path"] != rel]
    bundle.manifest_path.write_text(json.dumps(manifest))
    bundle.checksums_path.write_text(
        "".join(f"{e['sha256']}  payload/{e['path']}\n" for e in manifest["files"])
    )
    with pytest.raises(PublicationPreflightError, match="diagnostic input.*(digest|signed)"):
        verify_publication_bundle_preflight(bundle.bundle_dir, require_release_reconciliation=False)


def test_export_rejects_digest_pinned_pending_calibration_stub(tmp_path, monkeypatch):
    """The real pending stub cannot become a publication input merely by signing its bytes."""
    repo, run, pins = _run(tmp_path, monkeypatch)
    anchors = repo / pins["snqi_v2_anchors_path"]
    anchors.write_bytes(
        (
            Path(__file__).resolve().parents[2] / "configs/benchmarks/snqi_v2/anchors.v2.0.json"
        ).read_bytes()
    )
    pins["snqi_v2_anchors_sha256"] = hashlib.sha256(anchors.read_bytes()).hexdigest()
    _set_diagnostics(run, pins)
    with pytest.raises(ValueError, match="calibration anchors are not frozen"):
        _export(run, tmp_path)


def test_preflight_rejects_digest_pinned_pending_calibration_stub(tmp_path, monkeypatch):
    """Canonical validity remains required after the stub and its pins are re-signed cold."""
    _, run, pins = _run(tmp_path, monkeypatch)
    bundle = _export(run, tmp_path)
    payload = bundle.bundle_dir / "payload"
    anchors = payload / "diagnostic_inputs" / pins["snqi_v2_anchors_path"]
    anchors.write_bytes(
        (
            Path(__file__).resolve().parents[2] / "configs/benchmarks/snqi_v2/anchors.v2.0.json"
        ).read_bytes()
    )
    pins["snqi_v2_anchors_sha256"] = hashlib.sha256(anchors.read_bytes()).hexdigest()
    _set_diagnostics(payload, pins)
    _resign(bundle)
    with pytest.raises(PublicationPreflightError, match="calibration anchors are not frozen"):
        verify_publication_bundle_preflight(bundle.bundle_dir, require_release_reconciliation=False)


def test_export_rejects_stripped_diagnostics_for_declared_v2_campaign(tmp_path, monkeypatch):
    """A v2 campaign cannot omit both its report and calibrated input."""
    repo, run, pins = _run(tmp_path, monkeypatch)
    _declare_campaign(repo, run, pins)
    (run / "reports/snqi_v2_diagnostics.json").unlink()
    (repo / pins["snqi_v2_anchors_path"]).unlink()
    with pytest.raises(ValueError, match="v2 campaign requires diagnostics"):
        _export(run, tmp_path)


def test_preflight_rejects_stripped_diagnostics_and_anchors(tmp_path, monkeypatch):
    """Deleting both dependencies and rebuilding the inventory cannot erase campaign obligations."""
    repo, run, pins = _run(tmp_path, monkeypatch)
    _declare_campaign(repo, run, pins)
    bundle = _export(run, tmp_path)
    payload = bundle.bundle_dir / "payload"
    (payload / "reports/snqi_v2_diagnostics.json").unlink()
    (payload / "diagnostic_inputs" / pins["snqi_v2_anchors_path"]).unlink()
    _resign(bundle)
    with pytest.raises(PublicationPreflightError, match="v2 campaign requires diagnostics"):
        verify_publication_bundle_preflight(bundle.bundle_dir, require_release_reconciliation=False)


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("conflict", ["path", "digest"])
def test_rejects_conflicting_campaign_and_diagnostic_anchor_pins(
    tmp_path, monkeypatch, cold, conflict
):
    """Campaign and diagnostic roles must bind the same anchor path and digest."""
    repo, run, pins = _run(tmp_path, monkeypatch)
    _declare_campaign(repo, run, pins)
    bundle = _export(run, tmp_path) if cold else None
    target = bundle.bundle_dir / "payload" if cold else run
    campaign_path = target / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    if conflict == "path":
        campaign["metrics"]["snqi_v2_anchors_path"] = "output/calibration/other-anchors.json"
        source = (target / "diagnostic_inputs" if cold else repo) / campaign["metrics"][
            "snqi_v2_anchors_path"
        ]
        source.write_bytes(
            (
                (target / "diagnostic_inputs" if cold else repo) / pins["snqi_v2_anchors_path"]
            ).read_bytes()
        )
    else:
        campaign["metrics"]["snqi_v2_anchors_sha256"] = "f" * 64
    campaign_path.write_text(json.dumps(campaign))
    if cold:
        _resign(bundle)
        with pytest.raises(PublicationPreflightError, match="conflicting v2 anchors pins"):
            verify_publication_bundle_preflight(
                bundle.bundle_dir, require_release_reconciliation=False
            )
    else:
        with pytest.raises(ValueError, match="conflicting v2 anchors pins"):
            _export(run, tmp_path)


def test_campaign_schedule_and_assets_validate_from_detached_inputs(tmp_path, monkeypatch):
    """A campaign's independently pinned schedule remains available to the canonical scorer cold."""
    repo, run, pins = _run(tmp_path, monkeypatch)
    _declare_campaign(repo, run, pins)
    campaign_path = run / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    schedule_path = repo / "configs/horizons.yaml"
    schedule_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "scenarios": {
                    name: {"recommended_horizon_steps": horizon, "status": "authored"}
                    for name, horizon in campaign["scenario_horizons"].items()
                },
            }
        )
    )
    campaign["scenario_horizons"] = {
        "path": "configs/horizons.yaml",
        "sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
    }
    campaign_path.write_text(json.dumps(campaign))
    bundle = _export(run, tmp_path)
    assert (
        bundle.bundle_dir / "payload/diagnostic_inputs/configs/horizons.yaml"
    ).read_bytes() == schedule_path.read_bytes()
    for key, rel in pins.items():
        if key.endswith("_path"):
            (repo / rel).unlink()
    schedule_path.unlink()
    report = verify_publication_bundle_preflight(
        bundle.bundle_dir, require_release_reconciliation=False
    )
    assert report["status"] == "pass"
