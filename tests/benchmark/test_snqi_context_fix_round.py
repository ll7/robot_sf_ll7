"""Regression witnesses for #10132 FIX: real bytes, dev-only data, no simulation.

Protect post-freeze receipt pins, learned-stack observation, resolved algorithms,
manifest roster and preserved-row revalidation. Existing context tests assume a
committed receipt and copy expected mappings; these exercise delivery/production
builders and downstream entrypoints. Only existing runtime seams are patched.
"""
# evidence-writer-exempt: deterministic JSONL grids carry shared review sidecars.

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.benchmark.camera_ready._util import _config_hash_payload
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.map_runner import map_runner_episode as episode
from robot_sf.benchmark.result_provenance import build_execution_context_provenance
from robot_sf.evidence.writers import write_json, write_review_sidecar
from tests.benchmark.test_mintorder_binding import CONFIG
from tests.benchmark.test_snqi_execution_context import (
    ENV,
    EVIDENCE,
    complete_context_asset_binding,
)
from tests.tools.test_run_benchmark_release import synthetic_execution_admission  # noqa: F401
from tests.unit.benchmark.test_snqi_v2 import spec_files as _spec_files

spec_files = _spec_files


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.usefixtures("synthetic_execution_admission")
def test_post_freeze_external_receipt_admits_same_source_through_release_cli(
    tmp_path, monkeypatch, capsys, spec_files
):
    """No git blob at the acquired source is needed for the full context/source admission."""
    from robot_sf.benchmark.snqi import execution_context
    from scripts.tools import run_benchmark_release as runner

    source = tmp_path / "source"
    source.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Fixture"),
        ("config", "user.email", "fixture@example.invalid"),
        ("commit", "--allow-empty", "-qm", "synthetic freeze"),
    ):
        subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True)
    freeze = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert subprocess.check_output(["git", "-C", str(source), "ls-tree", "-r", "HEAD"]) == b""
    # On cd9783b5 this makes git-show consult the synthetic freeze, not the tooling tree.
    monkeypatch.setattr(
        execution_context, "__file__", str(source / "robot_sf/benchmark/snqi/execution_context.py")
    )
    cfg = load_campaign_config(CONFIG)
    weights, anchors, family = spec_files
    document = json.loads(anchors.read_bytes())
    cal = document["calibration"]
    cal.update(source_commit=freeze, seeds=[1001, 1002])
    grid = sorted(product(cal["arms"], cal["scenarios"], cal["seeds"]))
    cal["grid_sha256"] = hashlib.sha256(
        json.dumps(grid, separators=(",", ":")).encode()
    ).hexdigest()
    cal["split_id"] = f"snqi-v2-dev1001-1002-{cal['grid_sha256'][:12]}"
    acquisition = load_campaign_config(cfg.snqi_v2_binding["acquisition_config_path"])
    cal["campaign_config_hash"] = hashlib.sha256(
        json.dumps(
            _config_hash_payload(acquisition), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    write_json(anchors, document)
    live = build_execution_context_provenance()
    receipt = json.loads((EVIDENCE / "determinism-receipt.json").read_bytes())
    receipt.update(source_commit=freeze, execution_contexts={"original": live, "repeat": live})
    receipt_path = tmp_path / "determinism-receipt.json"
    write_json(receipt_path, receipt)
    proof = json.loads((EVIDENCE / "acquisition-proof.json").read_bytes())
    proof.update(source_commit=freeze, anchors_sha256=digest(anchors))
    proof["determinism_receipt"] = {"path": receipt_path.name, "sha256": digest(receipt_path)}
    write_json(anchors.parent / "acquisition-proof.json", proof)
    binding = {
        **cfg.snqi_v2_binding,
        "weights_path": weights,
        "weights_sha256": digest(weights),
        "family_path": family,
        "family_sha256": digest(family),
        "determinism_receipt_path": receipt_path,
        "determinism_receipt_sha256": digest(receipt_path),
    }
    cfg = replace(cfg, snqi_v2_binding=binding)
    from tests.tools.test_run_benchmark_release import (
        _default_spawn_matrix_preflight_passes,
        _manifest_fixture,
    )

    manifest = _manifest_fixture()
    manifest.__dict__.update(canonical_campaign_config_path=CONFIG, source_sha=freeze)
    _default_spawn_matrix_preflight_passes.__wrapped__(monkeypatch)
    monkeypatch.setattr(runner, "load_release_manifest", lambda _p: manifest)
    monkeypatch.setattr(runner, "load_campaign_config", lambda _p: cfg)
    monkeypatch.setattr(runner, "_current_source_commit", lambda: freeze)

    admitted_specs = []

    def native_preflight(scored):
        admitted_specs.append(scored.snqi_v2_spec)

    monkeypatch.setattr(runner, "check_orca_rvo2_preflight", native_preflight)
    monkeypatch.setattr(runner, "validate_release_manifest", lambda *_a, **_k: {"status": "valid"})
    monkeypatch.setattr(runner, "gate_manifest", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "build_resolved_release_manifest", lambda *_a, **_k: {})
    monkeypatch.setattr(
        runner, "_preflight_checkpoint_admission", lambda *_a: {"status": "admitted"}
    )
    monkeypatch.setattr(
        runner,
        "prepare_campaign_preflight",
        lambda *_a, **_k: {
            "campaign_id": "dev-fixture",
            "campaign_root": tmp_path,
            "validate_config_path": tmp_path / "validate.json",
            "preview_scenarios_path": tmp_path / "preview.json",
            "matrix_summary_json_path": tmp_path / "matrix.json",
            "matrix_summary_csv_path": tmp_path / "matrix.csv",
        },
    )
    result = runner.main(
        ["--manifest", "dev.yaml", "--mode", "preflight", "--snqi-v2-anchors", str(anchors)]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0, payload
    assert admitted_specs[0].calibration_seeds == (1001, 1002)
    assert document["calibration"]["source_commit"] == manifest.source_sha == freeze


@pytest.mark.parametrize("loader", ["campaign", "manifest"])
def test_receipt_asset_binding_validates_real_delivered_bytes(tmp_path, loader):
    from robot_sf.benchmark import release_protocol
    from robot_sf.benchmark.snqi import v2_binding

    raw = yaml.safe_load(CONFIG.read_bytes())["snqi_v2_spec"]
    raw.update(
        determinism_receipt_path=str(EVIDENCE / "determinism-receipt.json"),
        determinism_receipt_sha256=digest(EVIDENCE / "determinism-receipt.json"),
    )

    def load():
        if loader == "campaign":
            return v2_binding.load_acquisition_binding(raw, CONFIG)
        assets = {
            key[:-5]: {
                "path": str((CONFIG.parents[2] / value).resolve())
                if not Path(value).is_absolute()
                else value,
                "sha256": raw[key[:-5] + "_sha256"],
            }
            for key, value in raw.items()
            if key.endswith("_path")
        }
        return release_protocol._load_manifest_v2_binding(
            CONFIG, {"metrics": {"snqi_v2_binding": assets}}, None
        )

    assert load() is not None
    raw["determinism_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="determinism_receipt.*digest mismatch"):
        load()


@pytest.mark.parametrize(
    "algo", [" PPO ", "sa_cadrl", "prediction_planner", "predictive_mppi", "socnav_sampling"]
)
def test_all_checkpoint_workers_refuse_mismatch_before_episode_construction(monkeypatch, algo):
    from robot_sf.benchmark.snqi.execution_context import admit_episode_context

    expected = build_execution_context_provenance()
    expected["cpu_model"] = "different CPU"
    monkeypatch.setenv(ENV, json.dumps(expected))
    with pytest.raises(ValueError, match="mismatch: cpu_model"):
        admit_episode_context(algo)


def test_gate_uses_resolved_algorithm_before_environment_or_policy(monkeypatch):
    expected = build_execution_context_provenance()
    expected["cpu_model"] = "different CPU"
    monkeypatch.setenv(ENV, json.dumps(expected))
    monkeypatch.setattr(
        episode,
        "_resolve_episode_run_context",
        lambda **_kw: SimpleNamespace(algo=" PPO ", scenario={}),
    )

    def forbidden(*_a, **_kw):
        pytest.fail("resolved learned algorithm skipped execution-context admission")

    monkeypatch.setattr(episode, "telemetry_from_scenario", forbidden)
    with pytest.raises(ValueError, match="mismatch: cpu_model"):
        episode.run_map_episode(
            {},
            1004,
            horizon=1,
            dt=0.1,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo="goal",
            scenario_path=Path("dev.yaml"),
            policy_builder=forbidden,
        )


@pytest.mark.parametrize(
    "missing_arm", ["guarded_ppo", "prediction_planner", "predictive_mppi", "socnav_sampling"]
)
@pytest.mark.usefixtures("synthetic_execution_admission")
def test_production_release_census_refuses_a_missing_manifest_learned_arm(
    tmp_path, monkeypatch, capsys, missing_arm
):
    from scripts.tools import run_benchmark_release as runner
    from tests.benchmark.test_snqi_execution_context import (
        test_recorded_context_gate_precedes_full_release_acceptance,
    )
    from tests.tools.test_run_benchmark_release import _manifest_fixture

    def manifest():
        result = _manifest_fixture()

        result.planner_keys = ("ppo", missing_arm)
        return result

    import tests.tools.test_run_benchmark_release as support

    monkeypatch.setattr(support, "_manifest_fixture", manifest)
    # Reuse the actual CLI integration fixture; its acceptance sentinel is invalid.
    # The context census must intercept earlier with a distinct refusal status.
    original_main = runner.main

    def census_main(args):
        assert original_main(args) == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "snqi_v2_episode_context_refused", (
            "missing learned arm passed context census"
        )
        assert missing_arm in payload["status_reason"]
        raise CensusRefused

    class CensusRefused(Exception):
        pass

    monkeypatch.setattr(runner, "main", census_main)
    with pytest.raises(CensusRefused):
        test_recorded_context_gate_precedes_full_release_acceptance(
            monkeypatch, capsys, tmp_path, "equal"
        )


def test_receipt_builder_retains_production_learned_stack_versions(tmp_path, monkeypatch):
    from robot_sf.benchmark import result_provenance
    from scripts.dev.build_snqi_v2_determinism_receipt import build_receipt

    versions = {"torch": "9.8.7+fixture", "stable-baselines3": "2.9.1+fixture"}
    monkeypatch.setattr(result_provenance, "_installed_version", versions.__getitem__)
    context = build_execution_context_provenance()
    row = {
        "metrics": {"robot_force_impulse_total": 2.0, "jerk_mean": 3.0, "curvature_mean": 4.0},
        "metric_values": {},
        "steps": 1,
        "status": "success",
    }
    roots = [tmp_path / name for name in ("original", "repeat", "rehearsal")]
    for root in roots:
        root.mkdir()
        write_json(root / "run_meta.json", {"execution_context": context})
        for arm in range(14):
            path = root / "runs" / f"arm{arm}" / "episodes.jsonl"
            path.parent.mkdir(parents=True)
            rows = [
                {**row, "scenario_id": f"scenario{s}", "seed": seed}
                for s in range(48)
                for seed in (1001, 1002)
            ]
            path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in rows))
            write_review_sidecar(path)
    receipt = build_receipt(*roots)
    recorded = receipt["execution_contexts"]["original"]
    assert recorded.get("torch_version") == versions["torch"], (
        "production receipt omitted torch_version"
    )
    assert recorded.get("stable_baselines3_version") == versions["stable-baselines3"], (
        "production receipt omitted stable_baselines3_version"
    )
    assert receipt["original_vs_repeat"]["identical_rows"] == 1344
    from robot_sf import _execution_context as primitive

    # The primitive serializes supplied observations; only provenance captures versions.
    without_optional_imports = primitive.build_execution_context()
    assert "torch_version" not in without_optional_imports
    assert "stable_baselines3_version" not in without_optional_imports


def test_revalidation_refuses_missing_worker_context_before_acceptance(tmp_path, monkeypatch):
    """The preserved-row production builder must repeat context admission before scoring."""
    from contextlib import nullcontext

    from scripts.tools import revalidate_benchmark_release as recovery

    source, validator, producer = (tmp_path / name for name in ("source", "validator", "producer"))
    for root in (source, validator, producer):
        root.mkdir()
    manifest_path = source / "manifest.yaml"
    write_json(manifest_path, {})
    row_path = producer / "runs/ppo__differential_drive/episodes.jsonl"
    row_path.parent.mkdir(parents=True)
    write_json(row_path, {"algo": " PPO ", "seed": 1004, "algorithm_metadata": {}}, indent=None)
    manifest = SimpleNamespace(planner_keys=("ppo",), source_sha="a" * 40)
    reference_root = tmp_path / "current-reference"
    cfg = SimpleNamespace(
        snqi_v2_binding=complete_context_asset_binding(reference_root),
        snqi_v2_spec=SimpleNamespace(
            paths={"anchors": str(reference_root / "anchors.v2.0.acquired.json")}
        ),
    )
    for name in (
        "_assert_distinct_validator_checkout",
        "_assert_frozen_source_repository",
        "_assert_manifest_paths_from_source",
    ):
        monkeypatch.setattr(recovery, name, lambda *_a, **_k: None)
    monkeypatch.setattr(recovery, "_source_repository_binding", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(recovery, "_validator_provenance", lambda *_a, **_k: {})
    monkeypatch.setattr(recovery, "verify_producer_artifacts", lambda *_a, **_k: {})
    monkeypatch.setattr(recovery, "_verify_acceptance_campaign_subset", lambda *_a, **_k: {})
    monkeypatch.setattr(recovery, "load_release_manifest", lambda *_a: manifest)
    monkeypatch.setattr(recovery, "load_release_campaign_config", lambda *_a, **_k: cfg)
    from robot_sf.benchmark.snqi import v2_binding

    # This fixture supplies a bound spec; acquisition is upstream of the context census.
    monkeypatch.setattr(v2_binding, "bind_acquired_anchors", lambda config, **_kw: config)
    monkeypatch.setattr(
        recovery, "validate_release_manifest", lambda *_a, **_k: {"status": "valid"}
    )

    def forbidden(**_kw):
        pytest.fail("preserved learned rows reached acceptance without execution-context admission")

    monkeypatch.setattr(recovery, "_run_exact_validator", forbidden)
    with pytest.raises(ValueError, match="execution context missing"):
        recovery.build_derived_release(
            producer_root=producer,
            acceptance_root=producer,
            source_repository_root=source,
            validator_repository_root=validator,
            expected_validator_commit="a" * 40,
            manifest_path=manifest_path,
            output_root=tmp_path / "derived",
            derived_name="context-check",
        )


def test_identity_resolver_pins_untracked_receipt_without_requiring_a_source_blob(
    tmp_path, monkeypatch
):
    """Exercise canonical generate/verify; a receipt is an ignored post-freeze input."""
    import shutil

    from robot_sf.benchmark import release_protocol as protocol
    from tests.benchmark.test_release_resolved_identity import (
        _git,
        _identity_inputs,
        _release_template_repository,
        _write_yaml,
    )
    from tests.benchmark.test_sealed_source_pins import bind_runtime_sources

    repo, template, _source = _release_template_repository(tmp_path)
    bind_runtime_sources(repo, monkeypatch)
    cfg_path = repo / CONFIG.relative_to(CONFIG.parents[2])
    raw = yaml.safe_load(CONFIG.read_bytes())["snqi_v2_spec"]
    for key, value in tuple(raw.items()):
        if not key.endswith("_path"):
            continue
        target = repo / value
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CONFIG.parents[2] / value, target)
    cfg = yaml.safe_load(cfg_path.read_bytes())
    cfg["snqi_v2_spec"] = raw
    _write_yaml(cfg_path, cfg)
    payload = yaml.safe_load(template.read_bytes())
    assets = {
        key[:-5]: {"path": value, "sha256": raw[key[:-5] + "_sha256"]}
        for key, value in raw.items()
        if key.endswith("_path")
    }
    # This pre-known digest fixture probes untracked input admission. The second
    # identity below uses the explicit post-acquisition resolver input as F2 will.
    receipt = repo / "output/calibration/determinism-receipt.json"
    receipt.parent.mkdir(parents=True)
    write_json(receipt, {"fixture": "post-freeze receipt"})
    assets["determinism_receipt"] = {"path": str(receipt), "sha256": digest(receipt)}
    payload["campaign_config_sha256"] = digest(cfg_path)
    payload["metrics"]["snqi_v2_binding"] = assets
    _write_yaml(template, payload)
    _git(repo, "add", "configs", "release.template.yaml")
    _git(repo, "commit", "-qm", "fixture: pending acquisition source")
    source = _git(repo, "rev-parse", "HEAD").strip()
    assert _git(repo, "ls-files", "output/calibration/determinism-receipt.json") == ""
    output = repo / "output/identity/release_identity.resolved.json"
    inputs = _identity_inputs(repo, template, source)
    protocol.write_resolved_release_identity(output_path=output, **inputs)
    verified = protocol.verify_resolved_release_identity(output, repository_root=repo)
    cfg = protocol.load_release_campaign_config(verified, repository_root=repo)
    assert cfg.snqi_v2_binding["determinism_receipt_sha256"] == digest(receipt)
    second = repo / "output/identity2/release_identity.resolved.json"
    protocol.write_resolved_release_identity(
        output_path=second, determinism_receipt=assets["determinism_receipt"], **inputs
    )
    protocol.verify_resolved_release_identity(second, repository_root=repo)
    # Mutated delivered bytes refuse canonical reconstruction, even if the
    # source tree stays clean. This is integrity, not scientific admission.
    write_json(receipt, {"fixture": "mutated receipt"})
    with pytest.raises(ValueError, match="digest mismatch"):
        protocol.verify_resolved_release_identity(second, repository_root=repo)


@pytest.mark.usefixtures("synthetic_execution_admission")
def test_release_census_normalizes_recorded_algorithm(tmp_path, monkeypatch, capsys):
    import tests.benchmark.test_snqi_execution_context as support

    writer = support.write_json

    def normalized_row(path, payload, **kwargs):
        if path.name == "episodes.jsonl" and payload.get("algo") == "ppo":
            payload = {**payload, "algo": " PPO "}
        return writer(path, payload, **kwargs)

    monkeypatch.setattr(support, "write_json", normalized_row)
    from scripts.tools import run_benchmark_release as runner

    original = runner.main

    def observed_main(args):
        result = original(args)
        output = capsys.readouterr().out
        assert json.loads(output).get("status") != "snqi_v2_episode_context_refused", (
            "normalized learned algorithm skipped context census"
        )
        print(output)
        return result

    monkeypatch.setattr(runner, "main", observed_main)
    support.test_recorded_context_gate_precedes_full_release_acceptance(
        monkeypatch, capsys, tmp_path, "equal"
    )


def test_fresh_gate_observation_keeps_inference_stack_unloaded():
    """Observe and enforce installed policy versions without importing either runtime."""
    probe = r"""
import importlib.abc
import importlib.machinery
import json
import os
import sys

class RefusePolicyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "stable_baselines3"}:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                spec.loader = RefusePolicyExecution()
            return spec

class RefusePolicyExecution(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise AssertionError("gate observation attempted inference import: " + module.__name__)

assert "torch" not in sys.modules and "stable_baselines3" not in sys.modules
sys.meta_path.insert(0, RefusePolicyImports())
from robot_sf._execution_context import LEARNED_POLICY_CONTEXT_FIELDS
from robot_sf._numerical_thread_env import pin_thread_env_for_determinism
pin_thread_env_for_determinism()
from robot_sf.benchmark.snqi.execution_context import (
    CONTEXT_ENV, admit_episode_context, build_execution_context_provenance,
)
reference = build_execution_context_provenance()
for field in LEARNED_POLICY_CONTEXT_FIELDS:
    assert isinstance(reference.get(field), str) and reference[field], field
os.environ[CONTEXT_ENV] = json.dumps(reference)
admitted = admit_episode_context(" PPO ")
assert all(admitted[field] == reference[field] for field in LEARNED_POLICY_CONTEXT_FIELDS)
for field in LEARNED_POLICY_CONTEXT_FIELDS:
    for value in (None, "deliberately-different"):
        changed = dict(reference)
        changed[field] = value
        os.environ[CONTEXT_ENV] = json.dumps(changed)
        try:
            admit_episode_context("sa_cadrl")
        except ValueError as error:
            assert field in str(error), str(error)
        else:
            raise AssertionError("learned version mismatch bypassed gate: " + field)
assert "torch" not in sys.modules and "stable_baselines3" not in sys.modules
print(json.dumps({"loaded_policy_modules": [], "versions": {
    field: reference[field] for field in LEARNED_POLICY_CONTEXT_FIELDS
}}))
"""
    observed = subprocess.run(
        [sys.executable, "-c", probe], check=False, capture_output=True, text=True
    )
    assert observed.returncode == 0, observed.stderr
    assert json.loads(observed.stdout)["loaded_policy_modules"] == []
