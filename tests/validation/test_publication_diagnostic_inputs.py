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
                "metric_list": ["success", "snqi_v2"],
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
        source.write_text(contents.get(name, json.dumps({"fixture": name})))
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
    with pytest.raises(
        PublicationPreflightError,
        match="diagnostic input.*(digest|signed|invalid structured content)",
    ):
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


def _round2_campaign(tmp_path, monkeypatch, cold):
    """Stage a valid campaign before applying a custody-erasing metadata mutation."""
    repo, run, pins = _run(tmp_path, monkeypatch)
    _declare_campaign(repo, run, pins)
    bundle = _export(run, tmp_path) if cold else None
    target = bundle.bundle_dir / "payload" if cold else run
    return repo, target, pins, bundle


def _strip_v2_custody(repo, target, pins, bundle):
    """Erase reports and anchors, then re-sign cold bytes to isolate activation logic."""
    (target / "reports/snqi_v2_diagnostics.json").unlink()
    ((target / "diagnostic_inputs" if bundle else repo) / pins["snqi_v2_anchors_path"]).unlink()
    if bundle:
        _resign(bundle)


def _assert_round2_rejected(target, tmp_path, bundle, reason="diagnostic input"):
    """Assert on the semantic refusal, rather than an incidental checksum failure."""
    if bundle:
        with pytest.raises(PublicationPreflightError, match=reason):
            verify_publication_bundle_preflight(
                bundle.bundle_dir, require_release_reconciliation=False
            )
    else:
        with pytest.raises(ValueError, match=reason):
            _export(target, tmp_path)


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize(
    "probe", ["remove_version", "alter_version", "metric_list_only", "duplicate_metrics"]
)
def test_round2_campaign_activation_bypass_probes(tmp_path, monkeypatch, cold, probe):
    """Version erasure, list declarations and duplicate-key replacement cannot erase custody."""
    repo, target, pins, bundle = _round2_campaign(tmp_path, monkeypatch, cold)
    campaign_path = target / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    if probe == "remove_version":
        del campaign["metrics"]["snqi_v2_version"]
    elif probe == "alter_version":
        campaign["metrics"]["snqi_v2_version"] = "SNQI-v1"
    elif probe == "metric_list_only":
        campaign["metrics"] = ["success", "snqi_v2"]
        del campaign["metric_list"]
    if probe == "duplicate_metrics":
        campaign_path.write_text(json.dumps(campaign)[:-1] + ', "metrics": {}}')
    else:
        campaign_path.write_text(json.dumps(campaign))
    _strip_v2_custody(repo, target, pins, bundle)
    _assert_round2_rejected(
        target,
        tmp_path,
        bundle,
        "duplicate JSON key" if probe == "duplicate_metrics" else "diagnostic input.*v2",
    )


@pytest.mark.parametrize("cold", [False, True])
def test_v2_weights_file_independently_requires_campaign_custody(tmp_path, monkeypatch, cold):
    """The actual weights format activates v2 even through an unversioned legacy path field."""
    repo, target, pins, bundle = _round2_campaign(tmp_path, monkeypatch, cold)
    campaign_path = target / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["metrics"] = {}
    del campaign["metric_list"]
    campaign["snqi_weights_path"] = pins["snqi_v2_weights_path"]
    campaign["snqi_weights_sha256"] = pins["snqi_v2_weights_sha256"]
    campaign_path.write_text(json.dumps(campaign))
    _strip_v2_custody(repo, target, pins, bundle)
    _assert_round2_rejected(target, tmp_path, bundle, "diagnostic input.*v2")


_ACTIVATION_SIGNALS = [
    "snqi_v2_version",
    "snqi_v2_weights_path",
    "snqi_v2_weights_sha256",
    "snqi_v2_anchors_path",
    "snqi_v2_anchors_sha256",
    "snqi_v2_family_path",
    "snqi_v2_family_sha256",
    "metric_list",
    "weights_file_version",
]


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("mutation", ["remove", "alter"])
@pytest.mark.parametrize("signal", _ACTIVATION_SIGNALS)
def test_removing_or_altering_any_one_v2_signal_cannot_erase_custody(
    tmp_path, monkeypatch, cold, mutation, signal
):
    """Every remaining signal independently preserves the report/input obligations."""
    repo, target, pins, bundle = _round2_campaign(tmp_path, monkeypatch, cold)
    campaign_path = target / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    if signal == "weights_file_version":
        weights = (target / "diagnostic_inputs" if cold else repo) / pins["snqi_v2_weights_path"]
        document = json.loads(weights.read_text())
        if mutation == "remove":
            del document["version"]
        else:
            document["version"] = "SNQI-v1"
        weights.write_text(json.dumps(document))
        campaign["metrics"]["snqi_v2_weights_sha256"] = hashlib.sha256(
            weights.read_bytes()
        ).hexdigest()
    else:
        declaration = campaign if signal == "metric_list" else campaign["metrics"]
        if mutation == "remove":
            del declaration[signal]
        else:
            declaration[signal] = (
                ["success", "snqi"]
                if signal == "metric_list"
                else "SNQI-v1"
                if signal == "snqi_v2_version"
                else "configs/wrong.json"
                if signal.endswith("_path")
                else "f" * 64
            )
    campaign_path.write_text(json.dumps(campaign))
    _strip_v2_custody(repo, target, pins, bundle)
    _assert_round2_rejected(target, tmp_path, bundle)


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("mutation", ["remove_version", "alter_version", "alter_metric_list"])
def test_v2_campaign_signals_must_agree_even_with_complete_custody(
    tmp_path, monkeypatch, cold, mutation
):
    """Existing calibrated inputs do not excuse a missing or contradictory campaign declaration."""
    _, target, _, bundle = _round2_campaign(tmp_path, monkeypatch, cold)
    campaign_path = target / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text())
    if mutation == "remove_version":
        del campaign["metrics"]["snqi_v2_version"]
    elif mutation == "alter_version":
        campaign["metrics"]["snqi_v2_version"] = "SNQI-v1"
    else:
        campaign["metric_list"] = ["success", "snqi"]
    campaign_path.write_text(json.dumps(campaign))
    if bundle:
        _resign(bundle)
    _assert_round2_rejected(target, tmp_path, bundle, "diagnostic input.*v2")


@pytest.mark.parametrize("cold", [False, True])
def test_nested_duplicate_campaign_version_is_refused_at_parse_time(tmp_path, monkeypatch, cold):
    """Duplicate keys below metrics cannot replace the positive v2 version with a non-v2 value."""
    repo, target, pins, bundle = _round2_campaign(tmp_path, monkeypatch, cold)
    campaign_path = target / "campaign_manifest.json"
    text = campaign_path.read_text().replace(
        '"snqi_v2_version": "SNQI-v2"', '"snqi_v2_version": "SNQI-v2", "snqi_v2_version": "SNQI-v1"'
    )
    campaign_path.write_text(text)
    _strip_v2_custody(repo, target, pins, bundle)
    _assert_round2_rejected(target, tmp_path, bundle, "duplicate JSON key")


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize(
    "probe", ["no_campaign", "renamed_stub", "version_conflict", "force_conflict", "nested_pending"]
)
def test_round3_review_bypasses(tmp_path, monkeypatch, cold, probe):
    """Reproduce all ten export/cold controls from the third independent review."""
    repo, run, pins = _run(tmp_path, monkeypatch)
    if probe != "renamed_stub":
        _declare_campaign(repo, run, pins)
    bundle = _export(run, tmp_path) if cold else None
    target = bundle.bundle_dir / "payload" if cold else run
    report = target / "reports/snqi_v2_diagnostics.json"
    if probe == "no_campaign":
        (target / "campaign_manifest.json").unlink()
        report.unlink()
        ((target / "diagnostic_inputs" if cold else repo) / pins["snqi_v2_anchors_path"]).unlink()
        if not cold:
            # Retained weights and episode metrics survive the deleted declarations.
            (target / "reports/weights.json").write_bytes(
                (repo / pins["snqi_v2_weights_path"]).read_bytes()
            )
    elif probe == "renamed_stub":
        anchors = (target / "diagnostic_inputs" if cold else repo) / pins["snqi_v2_anchors_path"]
        anchors.write_bytes(
            (
                Path(__file__).resolve().parents[2] / "configs/benchmarks/snqi_v2/anchors.v2.0.json"
            ).read_bytes()
        )
        pins["snqi_v2_anchors_sha256"] = hashlib.sha256(anchors.read_bytes()).hexdigest()
        _set_diagnostics(target, pins)
        report.rename(target / "reports/renamed_diagnostics.json")
    elif probe in ("version_conflict", "force_conflict"):
        document = json.loads(report.read_text())
        campaign_path = target / "campaign_manifest.json"
        campaign = json.loads(campaign_path.read_text())
        key = "snqi_v2_version" if probe == "version_conflict" else "snqi_v2_force_source"
        campaign["metrics"][key] = (
            "SNQI-v2" if probe == "version_conflict" else "robot_force_impulse_total"
        )
        document["provenance"][key] = (
            "SNQI-v1" if probe == "version_conflict" else "robot_pedestrian_force_impulse_total"
        )
        campaign_path.write_text(json.dumps(campaign))
        report.write_text(json.dumps(document))
    else:
        campaign_path = target / "campaign_manifest.json"
        campaign = json.loads(campaign_path.read_text())
        campaign["metrics"]["snqi_v2"] = "pending_calibration"
        campaign_path.write_text(json.dumps(campaign))
    if cold:
        _resign(bundle)
    _assert_round2_rejected(target, tmp_path, bundle)


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize(
    "proof",
    [
        "absent",
        "unrelated_word",
        "declared_v1",
        "v2_metric_list",
        "v2_later_row",
        "v2_weights",
        "v2_pins",
    ],
)
def test_only_positive_historical_v1_proof_exempts_custody(  # noqa: C901
    tmp_path, monkeypatch, cold, proof
):
    """Unscored or ambiguously labelled bundles require custody even with no v2 signals."""
    _, run, _ = _run(tmp_path, monkeypatch)
    bundle = _export(run, tmp_path) if cold else None
    target = bundle.bundle_dir / "payload" if cold else run
    (target / "reports/snqi_v2_diagnostics.json").unlink()
    if cold:
        import shutil

        shutil.rmtree(target / "diagnostic_inputs")
    (target / "runs/goal/episodes.jsonl").write_text(
        json.dumps({"event_ledger": {"software_commit": "abc123"}}) + "\n"
    )
    meta = {"repo": {"commit": "abc123"}}
    if proof == "declared_v1":
        meta["snqi_score_version"] = "SNQI-v1"
    elif proof == "unrelated_word":
        meta["note"] = "SNQI-v1"
    elif proof != "absent":
        meta["snqi_score_version"] = "SNQI-v1"
        if proof == "v2_metric_list":
            meta["metric_list"] = ["snqi_v2"]
        elif proof == "v2_later_row":
            with (target / "runs/goal/episodes.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event_ledger": {"software_commit": "abc123"},
                            "metrics": {"snqi_v2_terms": {"S": 1}},
                        }
                    )
                    + "\n"
                )
        elif proof == "v2_weights":
            (target / "reports/unversioned.json").write_text(_valid_asset_contents()["weights"])
        elif proof == "v2_pins":
            meta["snqi_v2_anchors_sha256"] = "a" * 64
    (target / "run_meta.json").write_text(json.dumps(meta))
    if cold:
        _resign(bundle)
    if proof == "declared_v1":
        if cold:
            assert (
                verify_publication_bundle_preflight(
                    bundle.bundle_dir, require_release_reconciliation=False
                )["status"]
                == "pass"
            )
        else:
            _export(target, tmp_path)
    else:
        _assert_round2_rejected(target, tmp_path, bundle)


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize(
    "candidate", ["renamed_anchors", "wrapped_pending", "generic_version_conflict"]
)
def test_complete_custody_validates_every_content_candidate(tmp_path, monkeypatch, cold, candidate):
    """Valid primary pins cannot hide a second invalid candidate in otherwise complete custody."""
    _, target, _, bundle = _round2_campaign(tmp_path, monkeypatch, cold)
    if candidate == "renamed_anchors":
        contents = (
            Path(__file__).resolve().parents[2] / "configs/benchmarks/snqi_v2/anchors.v2.0.json"
        ).read_bytes()
    elif candidate == "wrapped_pending":
        contents = json.dumps(
            {"arbitrary": [{"state": {"status": "pending_calibration"}}]}
        ).encode()
    else:
        contents = json.dumps(
            {"schema_version": "snqi-v2-diagnostics.v1", "provenance": {"version": "SNQI-v1"}}
        ).encode()
    (target / "reports/arbitrary.data").write_bytes(contents)
    if cold:
        _resign(bundle)
    _assert_round2_rejected(target, tmp_path, bundle)


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("input_state", ["pending", "v2_weights"])
def test_historical_v1_exemption_inspects_its_retained_inputs(
    tmp_path, monkeypatch, cold, input_state
):
    """The historical exemption also proves that external diagnostic inputs contain no v2 state."""
    repo, run, _ = _run(tmp_path, monkeypatch)
    (run / "reports/snqi_v2_diagnostics.json").unlink()
    (run / "runs/goal/episodes.jsonl").write_text(
        json.dumps({"event_ledger": {"software_commit": "abc123"}}) + "\n"
    )
    (run / "run_meta.json").write_text(
        json.dumps({"repo": {"commit": "abc123"}, "snqi_score_version": "SNQI-v1"})
    )
    relative = "configs/input.json"
    source = repo / relative
    source.write_text(json.dumps({"historical": True}))
    report = run / "reports/historical_diagnostics.json"
    report.write_text(
        json.dumps(
            {
                "input_path": relative,
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    )
    bundle = _export(run, tmp_path) if cold else None
    target = bundle.bundle_dir / "payload" if cold else run
    source = target / "diagnostic_inputs" / relative if cold else source
    source.write_text(
        json.dumps({"wrapped": [{"state": "pending_calibration"}]})
        if input_state == "pending"
        else _valid_asset_contents()["weights"]
    )
    (target / "reports/historical_diagnostics.json").write_text(
        json.dumps(
            {
                "input_path": relative,
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    )
    if cold:
        _resign(bundle)
    _assert_round2_rejected(target, tmp_path, bundle)
