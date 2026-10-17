"""Offline synthetic report projection controls; no simulation or release admission."""

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.snqi.v2_reports import score_episode, write_v2_reports
from scripts.tools import revalidate_benchmark_release as recovery
from tests.unit.benchmark.test_snqi_v2 import fixture_spec, records

SCORERS = ("robot_sf/benchmark/snqi/v2_reports.py", "robot_sf/benchmark/snqi/v2_spec.py")


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def projection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "tests@example.invalid")
    _git(source, "config", "user.name", "Synthetic fixture")
    helper = Path(recovery.__file__).resolve().parents[2]
    for name in SCORERS:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(helper / name, path)
    for key in ("a", "b"):
        (source / f"{key}.yaml").write_text(f"algo: {key}\n")
    _git(source, "add", *SCORERS, "a.yaml", "b.yaml")
    _git(source, "commit", "-qm", "Synthetic scoring source")
    manifest = SimpleNamespace(
        source_sha=_git(source, "rev-parse", "HEAD"),
        resolved_manifest_payload={
            "planners": {
                "config_identities": [
                    {"key": key, "algo": key, "path": f"{key}.yaml"} for key in ("a", "b")
                ]
            }
        },
    )
    assets = tmp_path / "custody"
    assets.mkdir()
    paths, hashes = {}, {}
    for key in ("weights", "anchors", "family"):
        path = assets / f"{key}.json"
        path.write_bytes(f"synthetic {key} bytes\n".encode())
        paths[key] = str(path)
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = replace(fixture_spec(), paths=paths, hashes=hashes, calibration_seeds=(1001, 1002))
    root = tmp_path / "producer"
    rows = records()
    for index, row in enumerate(rows):
        row["seed"] = 1001 + index % 2
        row["metrics"].pop("snqi")
        row.update(planner_key=row["algo"], kinematics="differential_drive")
    # Use producer order b,a, exercising manifest order rather than sorted paths.
    ordered = [row for key in ("b", "a") for row in rows if row["algo"] == key]
    for key in ("b", "a"):
        path = root / "runs" / f"{key}__differential_drive" / "episodes.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            "".join(
                json.dumps(score_episode(row, spec)) + "\n" for row in ordered if row["algo"] == key
            )
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"planner": {"key": key, "kinematics": "differential_drive"}}
                    for key in ("b", "a")
                ]
            }
        )
    )
    write_v2_reports(
        ordered,
        spec,
        root / "reports",
        bootstrap_samples=10,
        expected_algorithms={f"{key}__differential_drive": key for key in ("a", "b")},
    )
    return root, source, manifest, spec


def _verify(projection, spec=None):
    root, source, manifest, producer_spec = projection
    with recovery._source_repository_binding(source, scoring_v2=True):
        return recovery._verify_publication_v2_reports(
            root,
            SimpleNamespace(snqi_v2_spec=spec or producer_spec, bootstrap_samples=10),
            manifest,
            expected_row_count=4,
            expected_arm_count=2,
        )


def test_planner_bound_reports_accept_identical_anchors_at_new_custody_path(projection, tmp_path):
    """Planner/source-bound reports survive an anchor copy without producer rewrites."""
    root, _, _, spec = projection
    before = {p.name: p.read_bytes() for p in (root / "reports").iterdir()}
    copied = tmp_path / "copied-anchors.json"
    shutil.copyfile(spec.paths["anchors"], copied)
    moved = replace(spec, paths={**spec.paths, "anchors": str(copied)})
    assert _verify(projection, moved)["status"] == "verified_v2_reports_preserved"
    assert before == {p.name: p.read_bytes() for p in (root / "reports").iterdir()}
    assert not (root / "reports/snqi_diagnostics.json").exists()


def test_report_verification_rejects_different_anchor_digest(projection):
    """A path adjustment must never hide different anchor contents."""
    spec = projection[3]
    changed = replace(spec, hashes={**spec.hashes, "anchors": "0" * 64})
    with pytest.raises(recovery.DerivedReleaseError, match="scoring content hash differs.*anchors"):
        _verify(projection, changed)


def test_planner_report_manifest_without_runs_is_a_domain_refusal(projection):
    """Malformed producer run order cannot leak a raw KeyError through the CLI."""
    (projection[0] / "manifest.json").write_text("{}")
    with pytest.raises(recovery.DerivedReleaseError, match="manifest.*runs"):
        _verify(projection)


@pytest.mark.parametrize("name", SCORERS)
def test_report_verification_refuses_source_scoring_blob_drift(projection, name):
    """Matching rows cannot conceal helper/source scoring implementation drift."""
    _, source, manifest, _ = projection
    path = source / name
    path.write_bytes(path.read_bytes() + b"\n# Synthetic source drift control.\n")
    _git(source, "add", name)
    _git(source, "commit", "-qm", "Synthetic source drift")
    manifest.source_sha = _git(source, "rev-parse", "HEAD")
    with pytest.raises(recovery.DerivedReleaseError, match="scoring implementation differs"):
        _verify(projection)


def test_independent_validator_rehashes_identity_before_manifest_reconstruction(
    tmp_path, monkeypatch
):
    """A changed identity must fail in the launched validator, despite cached helper evidence."""
    source, validator = tmp_path / "source", tmp_path / "validator"
    (source / "maps").mkdir(parents=True)
    (source / "maps/registry.yaml").write_text("maps: {}\n")
    identity = source / "identity.json"
    identity.write_text('{"synthetic_identity": true}')
    evidence = {
        "manifest": {},
        "path_fields": [],
        "tuple_fields": [],
        "identity_sha256": hashlib.sha256(identity.read_bytes()).hexdigest(),
    }
    # Minimal reviewed-validator stand-in isolates transfer integrity, not source authenticity.
    modules = {
        "robot_sf/__init__.py": "",
        "robot_sf/benchmark/__init__.py": "",
        "robot_sf/benchmark/release_protocol.py": "from types import SimpleNamespace\nBenchmarkReleaseManifest=SimpleNamespace\ndef load_release_campaign_config(*a, **k): return {}\n",
        "robot_sf/benchmark/release_acceptance.py": "def validate_full_benchmark_release_acceptance(*a, **k): return {'status':'valid'}\n",
        "robot_sf/benchmark/camera_ready/__init__.py": "",
        "robot_sf/benchmark/camera_ready/_config.py": "",
        "robot_sf/benchmark/camera_ready/_run_state.py": "",
        "robot_sf/benchmark/camera_ready/_util.py": "",
        "robot_sf/benchmark/snqi/__init__.py": "",
        "robot_sf/benchmark/snqi/v2_spec.py": "",
    }
    for name, content in modules.items():
        path = validator / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(recovery, "_verified_frozen_identity", lambda *_a: evidence)
    kwargs = {
        "validator_root": validator,
        "source_root": source,
        "acceptance_root": source,
        "manifest_path": identity,
    }
    assert recovery._run_exact_validator(**kwargs)["status"] == "valid"
    identity.write_text('{"synthetic_identity": "changed after verification"}')
    with pytest.raises(recovery.DerivedReleaseError, match="validator execution failed"):
        recovery._run_exact_validator(**kwargs)


@pytest.mark.parametrize("missing", [True, False])
def test_malformed_report_provenance_is_a_domain_refusal(projection, missing):
    """Normalisation must not turn a corrupt report into a raw exception."""
    path = projection[0] / "reports/snqi_v2_family.json"
    payload = json.loads(path.read_bytes())
    if missing:
        del payload["provenance"]
    else:
        payload["provenance"] = None
    path.write_text(json.dumps(payload))
    with pytest.raises(recovery.DerivedReleaseError, match="report provenance.*object"):
        _verify(projection)


@pytest.mark.parametrize("name", ["family", "diagnostics"])
def test_non_object_report_is_a_domain_refusal(projection, name):
    """A corrupt top-level report must stay inside the CLI rejection contract."""
    path = projection[0] / f"reports/snqi_v2_{name}.json"
    path.write_text("null")
    with pytest.raises(recovery.DerivedReleaseError, match="report.*object"):
        _verify(projection)
