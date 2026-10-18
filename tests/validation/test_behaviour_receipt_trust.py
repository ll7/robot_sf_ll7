"""Admission must derive source identity and runtime validity from sealed evidence."""

import hashlib
import json
import shutil
import subprocess
import tarfile

import pytest

from robot_sf._execution_context import execution_context_digest
from scripts.ci import behaviour_receipt as gate
from scripts.ci import receipt_provenance
from tests.validation.receipt_scheduler_fixture import install_scheduler
from tests.validation.test_behaviour_receipt_provenance import refresh_sweep_binding, seal_sweep
from tests.validation.test_behaviour_receipt_sbatch_affinity import _collection, _unchanged_outcomes


@pytest.fixture(autouse=True)
def _scheduler_authority(tmp_path, monkeypatch):
    install_scheduler(tmp_path, monkeypatch)


def _clean_fixture(tmp_path, monkeypatch):
    git, _, head, baseline, command = _collection(tmp_path, monkeypatch)
    _unchanged_outcomes(head)
    # Normal sealed producers contain no post-seal allocation stamping.
    for root in (head, baseline):
        for path in (*root.glob("episodes_*.jsonl"), *root.glob("campaigns/**/episodes.jsonl")):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for row in rows:
                row.pop("scheduler_job_id", None)
                row.pop("scheduler_node", None)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        refresh_sweep_binding(root)
    return git, head, baseline, command


def _collected(tmp_path, monkeypatch):
    git, head, baseline, command = _clean_fixture(tmp_path, monkeypatch)
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    header = json.loads(result.stdout.split("\n", 1)[1].rsplit("-->", 1)[0])
    payload_path = tmp_path / header["rows_artifact"]["path"]
    payload = json.loads(payload_path.read_text())
    return git, head, baseline, header, payload, payload_path


def _unpack(tmp_path, reference, name):
    root = tmp_path / name
    root.mkdir()
    with tarfile.open(tmp_path / reference["path"], "r:gz") as archive:
        for member in archive:
            path = root / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream:
                path.write_bytes(stream.read())
    return root


def _pack(tmp_path, root, reference, source, side, job):
    """Act as an attacker: refresh unsigned bindings and all transport digests."""
    path = root / receipt_provenance.BINDING_FILE
    binding = json.loads(path.read_text())
    binding.update(source_sha=source, arm_role=side, job_id=job, node="auxme-imech142")
    binding["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in receipt_provenance.source_files(root)
    }
    path.write_text(json.dumps(binding))
    bundle = tmp_path / reference["path"]
    with tarfile.open(bundle, "w:gz") as archive:
        for path in [
            *receipt_provenance.source_files(root),
            root / receipt_provenance.BINDING_FILE,
        ]:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    reference["sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    reference["binding"]["sha256"] = hashlib.sha256(
        (root / receipt_provenance.BINDING_FILE).read_bytes()
    ).hexdigest()


def _admit(tmp_path, monkeypatch, git, header, payload, payload_path):
    payload_path.write_text(json.dumps(payload))
    git("add", "receipts/behaviour")
    git("commit", "-m", "resealed source attack")
    head = git("rev-parse", "HEAD")
    header["rows_artifact"]["sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    header["head_sha"] = header["refute_review"]["head_sha"] = head
    header["refute_review"]["verdict"] = "accepted"
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    return gate.check_receipt(
        "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->",
        ["robot_sf/planner/synthetic.py"],
        "ll7/robot_sf_ll7",
    )


SOURCE_ATTACKS = (
    "baseline_replayed_as_head",
    "run_source",
    "row_source",
    "raw_source",
    "arm_role",
    "producer_node",
    "producer_cpu",
    "same_producer_source",
    "binding_arm_role",
    "producer_job",
    "row_context_digest",
)


@pytest.mark.parametrize("attack", SOURCE_ATTACKS)
def test_admission_refuses_contradictory_producer_identity(tmp_path, monkeypatch, attack):
    git, _, _, header, payload, payload_path = _collected(tmp_path, monkeypatch)
    reference = payload["source_artifacts"]["head"]["123"]
    root = _unpack(tmp_path, reference, "head-evidence")
    raw = next(root.glob("campaigns/**/episodes.jsonl"))
    provenance = raw.with_suffix(".jsonl.provenance.json")
    sidecar = json.loads(provenance.read_text())
    source = header["scheduler"]["source_sha"]
    other = header["baseline"]["source_sha"]
    if attack == "baseline_replayed_as_head":
        baseline = _unpack(
            tmp_path, payload["source_artifacts"]["baseline"]["456"], "baseline-evidence"
        )
        for path in baseline.glob("campaigns/**/episodes.jsonl*"):
            shutil.copyfile(path, root / path.relative_to(baseline))
    elif attack in {"raw_source", "row_context_digest"}:
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        changes = {
            "raw_source": {"git_hash": other},
            "row_context_digest": {
                "execution_context": {
                    "hostname": "auxme-imech142",
                    "cpu_model": "AMD EPYC 9115 16-Core Processor",
                    "execution_context_sha256": "0" * 64,
                }
            },
        }
        rows[0].update(changes[attack])
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        # Keep normal sidecar hashes valid, and make their run/row source claims
        # coherent with the transplanted producer rather than corrupting digests.
        request = (
            json.loads((root / "execution_request.json").read_text())
            if (root / "execution_request.json").exists()
            else {"source_sha": source}
        )
        (root / "execution_request.json").write_text(json.dumps(request))
        seal_sweep(root)
    elif attack == "same_producer_source":
        reference = payload["source_artifacts"]["baseline"]["456"]
        root = _unpack(tmp_path, reference, "baseline-evidence")
        raw = next(root.glob("campaigns/**/episodes.jsonl"))
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        for row in rows:
            row["git_hash"] = source
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (root / "execution_request.json").write_text(json.dumps({"source_sha": source}))
        seal_sweep(root)
        meta = json.loads((root / "execution_main.json").read_text())
        meta["head_sha"] = source
        (root / "execution_main.json").write_text(json.dumps(meta))
        entry = payload["hardware"]["baseline"]["execution_request.json"]
        request = json.loads(entry["content"])
        request["source_sha"] = source
        entry["content"] = json.dumps(request)
        entry["sha256"] = hashlib.sha256(entry["content"].encode()).hexdigest()
        header["baseline"]["source_sha"] = source
        git("tag", "-f", header["baseline"]["release"], source)
    elif attack != "binding_arm_role":
        if attack == "run_source":
            sidecar["run"]["repo_commit"] = other
        elif attack == "row_source":
            sidecar["rows"][0]["repo_commit"] = other
        elif attack == "arm_role":
            sidecar["run"]["receipt_arm"] = "baseline"
        else:
            context = sidecar["run"]["execution_context"]
            field = {
                "producer_node": "hostname",
                "producer_cpu": "cpu_model",
                "producer_job": "scheduler_job_id",
            }[attack]
            context[field] = {
                "producer_node": "auxme-imech172",
                "producer_cpu": "Invented CPU",
                "producer_job": "456",
            }[attack]
            context["execution_context_sha256"] = execution_context_digest(
                {
                    k: v
                    for k, v in context.items()
                    if k not in {"hostname", "execution_context_sha256"}
                }
            )
        provenance.write_text(json.dumps(sidecar))
    side = "baseline" if attack in {"same_producer_source", "binding_arm_role"} else "head"
    _pack(
        tmp_path,
        root,
        reference,
        source,
        side,
        "456" if attack == "same_producer_source" else "123",
    )
    blockers = _admit(tmp_path, monkeypatch, git, header, payload, payload_path)
    assert blockers, f"admission accepted {attack} with unchanged outcomes"
    assert any("producer" in item.lower() for item in blockers), blockers


def _invalid_row(root, attack):
    raw = next(root.glob("campaigns/**/episodes.jsonl"))
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    row = rows[0]
    if attack in {"degraded", "fallback"}:
        row[attack] = True
    elif attack == "no_controller":
        row["controller_executed"] = False
    elif attack == "wrong_algorithm":
        row["algo"] = "orca"
    elif attack == "missing_trace":
        row["algorithm_metadata"].pop("simulation_step_trace")
    else:
        row["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = "adapter"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("side", ["head", "baseline"])
@pytest.mark.parametrize(
    "attack",
    ["degraded", "no_controller", "wrong_algorithm", "missing_trace", "fallback", "wrong_mode"],
)
def test_admission_rederives_execution_validity(tmp_path, monkeypatch, side, attack):
    git, _, _, header, payload, payload_path = _collected(tmp_path, monkeypatch)
    job = "123" if side == "head" else "456"
    reference = payload["source_artifacts"][side][job]
    root = _unpack(tmp_path, reference, "evidence")
    source = (
        header["scheduler"]["source_sha"] if side == "head" else header["baseline"]["source_sha"]
    )
    (root / "execution_request.json").write_text(
        json.dumps({"source_sha": source, "arm_role": side})
    )
    _invalid_row(root, attack)
    seal_sweep(root)
    _pack(tmp_path, root, reference, source, side, job)
    blockers = _admit(tmp_path, monkeypatch, git, header, payload, payload_path)
    assert blockers, f"admission accepted resealed {side} {attack} rows"
    assert any("raw execution" in item.lower() for item in blockers), blockers


@pytest.mark.parametrize("side", ["head", "baseline"])
@pytest.mark.parametrize(
    "attack",
    ["degraded", "no_controller", "wrong_algorithm", "missing_trace", "fallback", "wrong_mode"],
)
def test_collection_uses_the_same_raw_validity_rule(tmp_path, monkeypatch, side, attack):
    _, head, baseline, command = _clean_fixture(tmp_path, monkeypatch)
    root = head if side == "head" else baseline
    _invalid_row(root, attack)
    # Deliberately reseal invalid execution through the normal fixture producer;
    # integrity and scheduler records remain valid, so only validity may reject it.
    seal_sweep(root)
    binding_path = root / receipt_provenance.BINDING_FILE
    binding = json.loads(binding_path.read_text())
    binding["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in receipt_provenance.source_files(root)
    }
    binding_path.write_text(json.dumps(binding))
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode != 0, f"collection accepted {side} {attack}"
    assert result.stdout == ""
    assert not (tmp_path / "receipts/behaviour/synthetic.json").exists()
