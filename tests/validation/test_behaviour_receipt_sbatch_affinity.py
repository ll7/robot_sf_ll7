"""Receipt pairs must not silently compare different scheduler hardware."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import make_behaviour_receipt as producer
from tests.validation.receipt_scheduler_fixture import install_scheduler
from tests.validation.test_make_behaviour_receipt import RELEASE_TAG, _install_cli_fixture


@pytest.fixture(autouse=True)
def _scheduler_authority(tmp_path, monkeypatch):
    install_scheduler(tmp_path, monkeypatch)


def _args(tmp_path, *extra):
    return producer._parse_args(
        [
            "--head-sha",
            "a" * 40,
            "--receipt-id",
            "affinity-test",
            "--mode",
            "sbatch",
            "--work-dir",
            str(tmp_path),
            *extra,
        ]
    )


def test_sbatch_pair_requires_node_before_any_submission(tmp_path, monkeypatch):
    """An omitted hardware choice must not leave a partially submitted pair."""
    submissions = []
    monkeypatch.setattr(producer, "_ensure_worktree", lambda *args: tmp_path)
    monkeypatch.setattr(
        producer.subprocess,
        "check_output",
        lambda *args, **kwargs: submissions.append(args) or "123\n",
    )

    with pytest.raises(ValueError, match="--sbatch-node"):
        producer._prepare_sweeps(_args(tmp_path), tmp_path, "b" * 40)

    assert not submissions


@pytest.mark.parametrize("node", ["node1,node2", "node[1-2]", "node1\n#SBATCH --nodes=2"])
def test_sbatch_rejects_node_sets_and_directive_injection(tmp_path, monkeypatch, node):
    """The affinity option accepts one literal hostname, never a Slurm expression."""
    submissions = []
    monkeypatch.setattr(
        producer.subprocess,
        "check_output",
        lambda *args, **kwargs: submissions.append(args) or "123\n",
    )
    with pytest.raises(ValueError, match="single.*node"):
        producer._prepare_sweeps(_args(tmp_path, "--sbatch-node", node), tmp_path, "b" * 40)
    assert not submissions


HARDWARE_FILES = (
    "execution_node.txt",
    "execution_cpu.json",
    "execution_node_class.txt",
    "execution_request.json",
)
REVIEWER_CASES = (
    "cpu_node_mismatch",
    "mismatch_unchanged_outcomes",
    "missing_files",
    "requested_observed_mismatch",
)


def _write_hardware(root, source, job):
    (root / "execution_node.txt").write_text("auxme-imech142\n")
    (root / "execution_node_class.txt").write_text("pro6000\n")
    (root / "execution_cpu.json").write_text(
        json.dumps(
            {
                "lscpu": [
                    {"field": "Model name:", "data": "AMD EPYC 9115 16-Core Processor"},
                    {"field": "Architecture:", "data": "x86_64"},
                ]
            }
        )
    )
    (root / "execution_request.json").write_text(
        json.dumps(
            {
                "schema_version": "behaviour-hardware-request.v1",
                "requested_node": "auxme-imech142",
                "source_sha": source,
                "job_id": job,
            }
        )
    )


def _collection(tmp_path, monkeypatch):
    git, source, head, baseline = _install_cli_fixture(tmp_path, monkeypatch)
    _write_hardware(head, source, "123")
    _write_hardware(baseline, git("rev-parse", RELEASE_TAG), "456")
    command = [
        sys.executable,
        str(tmp_path / "scripts/ci/make_behaviour_receipt.py"),
        "--head-sha",
        source,
        "--baseline",
        "latest",
        "--receipt-id",
        "synthetic",
        "--mode",
        "existing",
        "--head-sweep-dir",
        str(head),
        "--baseline-sweep-dir",
        str(baseline),
        "--job-id",
        "123",
        "--baseline-job-id",
        "456",
    ]
    return git, source, head, baseline, command


def _fault(files, fault):
    if fault in ("cpu_node_mismatch", "mismatch_unchanged_outcomes"):
        files["execution_node.txt"] = "auxme-imech172\n"
        cpu = json.loads(files["execution_cpu.json"])
        cpu["lscpu"][0]["data"] = "Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz"
        files["execution_cpu.json"] = json.dumps(cpu)
        submission = json.loads(files["execution_request.json"])
        submission["requested_node"] = "auxme-imech172"
        files["execution_request.json"] = json.dumps(submission)
    elif fault == "missing_files":
        files.pop("execution_cpu.json")
    elif fault == "requested_observed_mismatch":
        files["execution_node.txt"] = "auxme-imech172\n"
    elif fault == "node_class_mismatch":
        files["execution_node_class.txt"] = "a30\n"
    elif fault == "wrong_job":
        request = json.loads(files["execution_request.json"])
        request["job_id"] = "999"
        files["execution_request.json"] = json.dumps(request)
    elif fault == "malformed_cpu":
        files["execution_cpu.json"] = '{"lscpu": []}'
    return files


def _unchanged_outcomes(head):
    summary = head / "episodes_main.jsonl"
    records = [json.loads(line) for line in summary.read_text().splitlines()]
    for r in records:
        r["success"] = True
    summary.write_text("".join(json.dumps(r) + "\n" for r in records))
    raw = head / "campaigns/empty_world_main/runs/goal__differential_drive/episodes.jsonl"
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    for r in records:
        r["metrics"]["success"] = True
    raw.write_text("".join(json.dumps(r) + "\n" for r in records))
    from tests.validation.test_behaviour_receipt_provenance import refresh_sweep_binding

    refresh_sweep_binding(head)


@pytest.mark.parametrize("fault", REVIEWER_CASES)
def test_real_collection_refuses_invalid_hardware_before_outcomes(tmp_path, monkeypatch, fault):
    """The real CLI must refuse hardware faults even when every outcome is unchanged."""
    _, _, head, _, command = _collection(tmp_path, monkeypatch)
    if fault == "mismatch_unchanged_outcomes":
        _unchanged_outcomes(head)
    files = _fault({name: (head / name).read_text() for name in HARDWARE_FILES}, fault)
    for name in HARDWARE_FILES:
        if name in files:
            (head / name).write_text(files[name])
        else:
            (head / name).unlink()
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode != 0, result.stdout
    assert "hardware comparison invalid" in result.stderr.lower()
    assert result.stdout == ""
    assert not (tmp_path / "receipts/behaviour/synthetic.json").exists()
    assert not (tmp_path / "output/behaviour_receipt/real_row_audit.json").exists()


def _admission_fixture(tmp_path, monkeypatch, fault=None):
    git, _, head, _, command = _collection(tmp_path, monkeypatch)
    if fault == "mismatch_unchanged_outcomes":
        _unchanged_outcomes(head)
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    header = json.loads(result.stdout.split("\n", 1)[1].rsplit("-->", 1)[0])
    payload_path = tmp_path / header["rows_artifact"]["path"]
    payload = json.loads(payload_path.read_text())
    if fault:
        if "hardware" in payload:
            entries = payload["hardware"]["head"]
            files = _fault({name: entry["content"] for name, entry in entries.items()}, fault)
            payload["hardware"]["head"] = {
                name: {"content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}
                for name, content in files.items()
            }
            payload_path.write_text(json.dumps(payload))
        else:
            # Old admission ignores the recorded files entirely. Keep its valid payload/schema
            # so failing-first proves acceptance of bad hardware rather than another failure.
            files = _fault({name: (head / name).read_text() for name in HARDWARE_FILES}, fault)
            evidence = tmp_path / "receipts/behaviour/hardware"
            evidence.mkdir()
            for name, content in files.items():
                (evidence / name).write_text(content)
    git("add", "receipts/behaviour")
    git("commit", "-m", "synthetic receipt evidence")
    final_head = git("rev-parse", "HEAD")
    header["head_sha"] = final_head
    header["rows_artifact"]["sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    header["refute_review"] = {
        "head_sha": final_head,
        "verdict": "accepted",
        "uri": "https://example.org/independent-review",
    }
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", final_head)
    body = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
    return header, body


@pytest.mark.parametrize("fault", REVIEWER_CASES)
def test_real_admission_refuses_invalid_hardware(tmp_path, monkeypatch, fault):
    """A valid committed receipt cannot admit mismatched or missing hardware evidence."""
    _, body = _admission_fixture(tmp_path, monkeypatch, fault)
    blockers = gate.check_receipt(body, ["robot_sf/planner/synthetic.py"], "ll7/robot_sf_ll7")
    assert blockers, "admission accepted invalid hardware"
    assert any("hardware comparison invalid" in item.lower() for item in blockers), blockers


def test_real_collection_and_admission_accept_matched_hardware(tmp_path, monkeypatch):
    """Matched evidence survives collection and the actual gate, including nonzero regressions."""
    _, body = _admission_fixture(tmp_path, monkeypatch)
    assert gate.check_receipt(body, ["robot_sf/planner/synthetic.py"], "ll7/robot_sf_ll7") == []


@pytest.mark.parametrize("fault", ["node_class_mismatch", "malformed_cpu", "wrong_job"])
def test_collection_and_admission_validate_identity_fields(tmp_path, monkeypatch, fault):
    """Valid presence alone cannot substitute for hardware identity validation."""
    test_real_collection_refuses_invalid_hardware_before_outcomes(tmp_path, monkeypatch, fault)
    # Each exercise owns its Git history and output custody.
    admission = tmp_path / "admission"
    admission.mkdir()
    test_real_admission_refuses_invalid_hardware(admission, monkeypatch, fault)


@pytest.mark.parametrize("side", ["head", "baseline"])
def test_admission_refuses_identity_digest_tampering(tmp_path, monkeypatch, side):
    """Even a correctly bound payload must authenticate each identity file's bytes."""
    header, _ = _admission_fixture(tmp_path, monkeypatch)
    path = tmp_path / header["rows_artifact"]["path"]
    payload = json.loads(path.read_text())
    payload["hardware"][side]["execution_cpu.json"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    subprocess.run(["git", "add", header["rows_artifact"]["path"]], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "tampered identity digest"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    final_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    header["head_sha"] = header["refute_review"]["head_sha"] = final_head
    header["rows_artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", final_head)
    body = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
    blockers = gate.check_receipt(body, ["robot_sf/planner/synthetic.py"], "ll7/robot_sf_ll7")
    assert any(
        "hardware comparison invalid" in item and "digest mismatch" in item for item in blockers
    ), blockers


@pytest.mark.parametrize("observed_node", ["auxme-imech142", "auxme-imech172"])
def test_submitted_scripts_execute_hardware_checks(tmp_path, monkeypatch, observed_node):
    """Execute the generated jobs with a synthetic scheduler instead of inspecting text."""
    _write_hardware(tmp_path, "a" * 40, "123")
    tools = tmp_path / "bin"
    tools.mkdir()
    for name, script in {
        "hostname": f"#!/bin/sh\nprintf '%s\\n' {observed_node}\n",
        "lscpu": f"#!/bin/sh\ncat '{tmp_path / 'execution_cpu.json'}'\n",
        "sbatch": '#!/bin/sh\nset -eu\npartition=$(sed -n \'s/^#SBATCH --partition=//p\' "$2")\nSLURM_JOB_ID=123 SLURM_JOB_NODELIST=auxme-imech142 SLURM_JOB_PARTITION="$partition" bash "$2"\nprintf "123\\n"\n',
    }.items():
        executable = tools / name
        executable.write_text(script)
        executable.chmod(0o755)
    uv = tools / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport os,sys\nassert sys.argv[1:3] == ['run','python']\nos.execv(sys.executable, [sys.executable,*sys.argv[3:]])\n"
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    monkeypatch.setattr(producer, "_ensure_worktree", lambda *args: tmp_path)
    marker = tmp_path / "sweep_executed"
    monkeypatch.setattr(
        producer,
        "_sweep_command",
        lambda *args, **kwargs: [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
    )
    args = _args(tmp_path, "--sbatch-node", "auxme-imech142")
    if observed_node != "auxme-imech142":
        with pytest.raises(subprocess.CalledProcessError):
            producer._prepare_sweeps(args, tmp_path, "b" * 40)
        assert not marker.exists()
        return
    head, baseline = producer._prepare_sweeps(args, tmp_path, "b" * 40)
    identities = gate.validate_hardware_pair(
        {
            "head": producer._read_hardware(head.output_dir, head.source_sha, head.job_id),
            "baseline": producer._read_hardware(
                baseline.output_dir, baseline.source_sha, baseline.job_id
            ),
        }
    )
    assert marker.exists()
    assert identities["head"]["node"] == "auxme-imech142"
    assert identities["baseline"]["node_class"] == "pro6000"
    before = (head.output_dir / "execution_request.json").read_bytes()
    with pytest.raises(subprocess.CalledProcessError):
        producer._prepare_sweeps(args, tmp_path, "b" * 40)
    assert (head.output_dir / "execution_request.json").read_bytes() == before
