"""Focused contract tests for the checkpoint compatibility audit (#8896, #8991)."""

from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import json
import zipfile
from io import StringIO
from pathlib import Path

import pytest
import yaml

from scripts.models import audit_checkpoint_compatibility as tool

FIXTURES = Path(__file__).parent / "fixtures" / "checkpoint_audit"
ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "fixture_missing_v1": {"artifact_missing"},
    "fixture_wrong_digest_v1": {"digest_mismatch"},
    "fixture_wrong_normalizer_v1": {"normalizer_missing"},
    "fixture_normalizer_digest_v1": {"normalizer_digest_mismatch"},
    "fixture_wrong_observation_space_v1": {"observation_contract_mismatch"},
    "fixture_mutable_alias_v1": {"mutable_alias", "digest_not_pinned"},
    "fixture_unavailable_dependency_v1": {"dependency_unavailable"},
    "fixture_rights_blocked_v1": {"rights_blocked"},
    "fixture_nonfinite_v1": {"non_finite_parameters", "normalizer_missing"},
    "fixture_missing_custom_object_v1": {"missing_custom_object"},
    "fixture_lineage_unresolved_v1": {"lineage_unresolved"},
    "fixture_duplicate_v1": {"duplicate_model_id"},
}


def _run(argv):
    out = StringIO()
    with contextlib.redirect_stdout(out):
        code = tool.main(argv)
    return code, out.getvalue()


def _check(path, *, root=ROOT):
    argv = ["--input", str(path), "--check", "--format", "json", "--root", str(root)]
    code, stdout = _run(argv)
    return code, json.loads(stdout)


def _rows(payload):
    return {row["model_id"]: row for row in payload["models"]}


def _codes(payload):
    return {item["code"] for item in payload["findings"]}


def _document(models, consumers=()):
    return {"schema": tool.INPUT_SCHEMA, "models": list(models), "consumers": list(consumers)}


def _write(tmp_path, document, name="case.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_compatible_fixture_passes_and_records_public_state():
    code, payload = _check(FIXTURES / "compatible.json")
    row = _rows(payload)["fixture_compatible_v1"]
    assert (code, payload["status"]) == (0, "pass")
    assert row["state"] == "public" and row["load_status"] == "verified"
    assert row["reason_codes"] == [] and row["observation_contract"]["shape"] == [4]


@pytest.mark.parametrize(("model_id", "expected"), sorted(CASES.items()))
def test_failing_fixture_reports_each_required_case(model_id, expected):
    code, payload = _check(FIXTURES / "failing.json")
    codes = set(_rows(payload)[model_id]["reason_codes"])
    assert code == 1 and payload["status"] == "fail"
    assert expected <= codes and expected <= _codes(payload)


def test_consumer_gate_fails_unstaged_recoverable_model(tmp_path):
    models = [
        {
            "model_id": "unstaged_v1",
            "artifact_version": "v1",
            "artifact_sha256": "a" * 64,
            "locator_class": "cloud_durable",
            "algorithm_class": "unknown",
        }
    ]
    consumer = {"consumer_id": "configs/baselines/ppo.yaml", "required_model_ids": ["unstaged_v1"]}
    code, payload = _check(_write(tmp_path, _document(models, [consumer])))
    assert code == 1 and "consumer_model_load_unverified" in _codes(payload)


def test_output_is_deterministic_and_rows_are_sorted(tmp_path):
    first = _check(FIXTURES / "failing.json")[1]
    assert tool.render_json(first) == tool.render_json(_check(FIXTURES / "failing.json")[1])
    identifiers = [row["model_id"] for row in first["models"]]
    assert identifiers == sorted(identifiers)
    shuffled = json.loads((FIXTURES / "failing.json").read_text(encoding="utf-8"))
    shuffled["models"].reverse()
    reordered = _check(_write(tmp_path, shuffled, "r.json"))[1]
    assert tool.render_json(reordered) == tool.render_json(first)


def test_private_input_is_rejected_and_never_rendered(tmp_path):
    secret = "/home/private-host/model.zip"
    leaky = {"model_id": "leaky_v1", "artifact_path": secret, "locator_class": "unknown"}
    document = _document([leaky])
    code, stdout = _run(["--input", str(_write(tmp_path, document)), "--check", "--format", "json"])
    assert code == 2 and secret not in stdout
    assert "forbidden_value" in json.loads(stdout)["findings"][0]["code"]


def test_unknown_input_exits_two_and_markdown_is_concise(tmp_path):
    code, stdout = _run(["--input", str(tmp_path / "missing.json"), "--check", "--format", "json"])
    assert code == 2 and json.loads(stdout)["status"] == "unknown"
    markdown = [
        "--input",
        str(FIXTURES / "compatible.json"),
        "--format",
        "markdown",
        "--root",
        str(ROOT),
    ]
    code, stdout = _run(markdown)
    assert code == 0 and stdout.startswith("# Checkpoint Compatibility Audit")
    assert "fixture_compatible_v1" in stdout


def _check_args(argv):
    code, stdout = _run(argv)
    return code, json.loads(stdout)


def _audit_run(tmp_path, *extra):
    return _check_args(["--check", "--format", "json", "--root", str(tmp_path), *extra])


def _write_yaml(tmp_path, document, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _registry_file(tmp_path):
    artifact = tmp_path / "artifacts" / "model.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"policy_class": "fixture.PPO"}', encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return _write_yaml(
        tmp_path,
        {
            "version": 1,
            "models": [
                {
                    "model_id": "mapped_v1",
                    "local_path": "artifacts/model.json",
                    "github_release": {"tag": "artifact/test", "sha256": digest},
                },
                {"model_id": "private_v1", "local_path": "/home/x/y"},
                {"local_path": "artifacts/model.json"},
            ],
        },
        "registry.yaml",
    )


def _probe_document(tmp_path, models=None):
    models = models or [
        {
            "model_id": "probed_v1",
            "artifact_kind": "torch_pt",
            "artifact_path": "artifact.pt",
            "locator_class": "local_scratch",
        }
    ]
    return _write(
        tmp_path, {"schema": tool.INPUT_SCHEMA, "models": models, "consumers": []}, "probe.json"
    )


def _probe_run(tmp_path, document, *, timeout="60"):
    return _audit_run(tmp_path, "--input", str(document), "--probe", "--probe-timeout", timeout)


def test_registry_intake_maps_rows_and_excludes_unsanitizable_entries(tmp_path):
    registry = _registry_file(tmp_path)
    code, payload = _audit_run(tmp_path, "--registry", str(registry), "--probe")
    digest = hashlib.sha256((tmp_path / "artifacts/model.json").read_bytes()).hexdigest()
    row = _rows(payload)["mapped_v1"]
    assert code == 1 and payload["status"] == "fail"
    assert (row["state"], row["load_status"]) == ("public", "verified")
    assert row["artifact"] == {
        "kind": "json",
        "version": "artifact/test",
        "path": "artifacts/model.json",
        "sha256": digest,
        "locator_class": "public_release",
        "companion_files": [],
    }
    assert payload["intake"]["registry"] == "registry.yaml"
    assert payload["intake"]["exclusions"] == [
        {"source": "registry.yaml", "model_id": None, "reason": "registry_entry_skipped"},
        {"source": "registry.yaml", "model_id": "private_v1", "reason": "unsanitized_local_path"},
    ]
    assert payload["intake"]["probe"]["skipped"] == [
        {"model_id": "mapped_v1", "reason": "unsupported_kind"}
    ]
    assert "/home/x/y" not in json.dumps(payload)


def test_reference_configs_map_consumers_and_unknown_ids_fail_closed(tmp_path):
    registry = _registry_file(tmp_path)
    good = _write_yaml(tmp_path, {"model_id": "mapped_v1"}, "configs/good.yaml")
    bad = _write_yaml(tmp_path, {"model_id": "absent_v1"}, "configs/bad.yaml")
    empty = _write_yaml(tmp_path, {"learning_rate": 0.1}, "configs/empty.yaml")
    code, payload = _audit_run(
        tmp_path,
        "--registry",
        str(registry),
        "--config",
        str(good),
        "--config",
        str(bad),
        "--config",
        str(empty),
    )
    consumers = {item["consumer_id"]: item for item in payload["consumers"]}
    assert code == 1 and consumers["configs/good.yaml"]["outcome"] == "pass"
    assert consumers["configs/bad.yaml"]["failed_model_ids"] == ["absent_v1"]
    assert "consumer_model_unresolved" in _codes(payload)
    assert {
        "source": "configs/empty.yaml",
        "model_id": None,
        "reason": "no_model_references",
    } in payload["intake"]["exclusions"]


@pytest.mark.parametrize(
    ("worker_body", "expected", "timeout"),
    [
        ("import time\ntime.sleep(30)\n", {"loader_probe_timeout"}, "0.4"),
        (
            'import json, sys\nprint(json.dumps({"ok": False, "error": '
            '"missing_custom_object"}))\nsys.exit(1)\n',
            {"loader_probe_failed", "missing_custom_object"},
            "60",
        ),
        (
            'print(\'{"ok": true, "facts": {"loader": "torch", "observation_shape": "bad"}}\')\n',
            {"loader_probe_malformed"},
            "60",
        ),
        (
            'print(\'{"ok": true, "facts": {"loader": "torch"}}\')\n',
            {"loader_probe_malformed"},
            "60",
        ),
        (
            'print(\'{"ok": true, "facts": {"loader": "sb3", "policy_class": "fixture.PPO", '
            '"observation_shape": [4], "action_shape": [2], "parameters_finite": true}}\')\n',
            {"loader_probe_malformed"},
            "60",
        ),
    ],
    ids=[
        "timeout",
        "nonzero-custom-object",
        "malformed-facts",
        "incomplete-facts",
        "loader-kind-mismatch",
    ],
)
def test_probe_process_failures_fail_closed(tmp_path, monkeypatch, worker_body, expected, timeout):
    (tmp_path / "artifact.pt").write_bytes(b"fixture-checkpoint-bytes")
    worker = tmp_path / "worker.py"
    worker.write_text(worker_body, encoding="utf-8")
    monkeypatch.setattr(tool, "PROBE_WORKER", worker)
    code, payload = _probe_run(tmp_path, _probe_document(tmp_path), timeout=timeout)
    row = _rows(payload)["probed_v1"]
    assert code == 1 and expected <= set(row["reason_codes"])
    assert payload["intake"]["probe"]["failed"] == [
        {"model_id": "probed_v1", "reason_codes": sorted(expected)}
    ]
    assert row["load_status"] == "inspected"


def test_probe_records_sb3_facts_for_zip_artifact(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from gymnasium.spaces import Box
    from stable_baselines3.common.save_util import data_to_json

    buffer = io.BytesIO()
    torch.save({"weight": torch.tensor([1.0, 2.0])}, buffer)
    with zipfile.ZipFile(tmp_path / "artifact.zip", "w") as archive:
        archive.writestr(
            "data",
            data_to_json(
                {
                    "policy_class": collections.OrderedDict,
                    "observation_space": Box(low=-1.0, high=1.0, shape=(4,)),
                    "action_space": Box(low=-1.0, high=1.0, shape=(2,)),
                }
            ),
        )
        archive.writestr("policy.pth", buffer.getvalue())
    models = [
        {
            "model_id": "probed_v1",
            "artifact_kind": "sb3_zip",
            "artifact_path": "artifact.zip",
            "locator_class": "local_scratch",
        }
    ]
    code, payload = _probe_run(tmp_path, _probe_document(tmp_path, models))
    row = _rows(payload)["probed_v1"]
    assert code == 1 and row["load_status"] == "verified"
    assert row["probe"] == {
        "kind": "sb3_zip",
        "loader": "sb3",
        "policy_class": "collections.OrderedDict",
        "observation_shape": [4],
        "action_shape": [2],
        "parameters_finite": True,
        "custom_objects_missing": [],
        "modules_missing": [],
    }


def test_probe_real_corrupt_artifact_fails_closed(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "artifact.pt").write_bytes(b"not-a-checkpoint")
    code, payload = _probe_run(tmp_path, _probe_document(tmp_path))
    row = _rows(payload)["probed_v1"]
    assert code == 1 and row["load_status"] == "inspected"
    assert {"artifact_unreadable", "loader_probe_failed"} <= set(row["reason_codes"])
    assert payload["intake"]["probe"]["failed"] == [
        {"model_id": "probed_v1", "reason_codes": ["artifact_unreadable", "loader_probe_failed"]}
    ]


def test_probe_records_torch_facts_and_nonfinite_parameters(tmp_path):
    torch = pytest.importorskip("torch")
    torch.save(
        {
            "policy_class": "fixture.PPO",
            "observation_shape": [4],
            "action_shape": [2],
            "weights": torch.tensor([1.0, 2.0]),
        },
        tmp_path / "finite.pt",
    )
    torch.save(
        {
            "policy_class": "fixture.PPO",
            "observation_shape": [4],
            "action_shape": [2],
            "weights": torch.tensor([1.0, float("inf")]),
        },
        tmp_path / "nonfinite.pt",
    )
    models = [
        {
            "model_id": f"{name}_v1",
            "artifact_kind": "torch_pt",
            "artifact_path": f"{name}.pt",
            "locator_class": "local_scratch",
        }
        for name in ("finite", "nonfinite")
    ]
    code, payload = _probe_run(tmp_path, _probe_document(tmp_path, models))
    rows = _rows(payload)
    assert code == 1
    assert payload["intake"]["probe"]["verified"] == ["finite_v1", "nonfinite_v1"]
    assert rows["finite_v1"]["load_status"] == "verified"
    assert rows["finite_v1"]["probe"] == {
        "kind": "torch_pt",
        "loader": "torch",
        "policy_class": "fixture.PPO",
        "observation_shape": [4],
        "action_shape": [2],
        "parameters_finite": True,
        "custom_objects_missing": [],
        "modules_missing": [],
    }
    assert rows["nonfinite_v1"]["probe"]["parameters_finite"] is False
    assert "non_finite_parameters" in rows["nonfinite_v1"]["reason_codes"]
