"""Scheduler authority and fresh output directories protect receipt admission."""

import hashlib
import json
import subprocess

import pytest

from scripts.ci import behaviour_receipt as gate
from tests.validation.receipt_scheduler_fixture import install_scheduler
from tests.validation.test_behaviour_receipt_sbatch_affinity import (
    _admission_fixture,
    _collection,
    _unchanged_outcomes,
)


@pytest.fixture(autouse=True)
def _scheduler_authority(tmp_path, monkeypatch):
    install_scheduler(tmp_path, monkeypatch)


CASES = ("failed_resubmission", "unsigned_identity", "invented_cpu", "mixed_node_rows")
NODE = "auxme-imech142"


def _scheduler(tmp_path, monkeypatch):
    from tests.validation.receipt_scheduler_fixture import install_scheduler

    return install_scheduler(tmp_path, monkeypatch)


def _edit_scheduler(path, fault):
    records = json.loads(path.read_text())
    if fault == "failed_resubmission":
        for r in records.values():
            r.update(state="FAILED", exit_code="7:0")
    elif fault == "unsigned_identity":
        records = {}
    path.write_text(json.dumps(records))


def _row_fault(root, fault):
    paths = [
        root / "episodes_main.jsonl",
        root / "campaigns/empty_world_main/runs/goal__differential_drive/episodes.jsonl",
    ]
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for index, row in enumerate(rows):
            row["scheduler_job_id"] = "123"
            row["scheduler_node"] = (
                "auxme-imech172" if fault == "mixed_node_rows" and index == 0 else NODE
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    from tests.validation.test_behaviour_receipt_provenance import refresh_sweep_binding

    refresh_sweep_binding(root, validate_identity=False)


def _invent_cpu(root):
    path = root / "execution_cpu.json"
    cpu = json.loads(path.read_text())
    cpu["lscpu"][0]["data"] = "Invented CPU"
    path.write_text(json.dumps(cpu))


@pytest.mark.parametrize("fault", CASES)
def test_collection_refuses_round2_unsafe_accepts(tmp_path, monkeypatch, fault):
    records = _scheduler(tmp_path, monkeypatch)
    _, _, head, baseline, command = _collection(tmp_path, monkeypatch)
    _unchanged_outcomes(head)
    _row_fault(head, fault)
    _edit_scheduler(records, fault)
    if fault == "invented_cpu":
        _invent_cpu(head)
        _invent_cpu(baseline)
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode != 0, "collection emitted a receipt from invalid scheduler evidence"
    assert "hardware comparison invalid" in result.stderr.lower(), result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "receipts/behaviour/synthetic.json").exists()


@pytest.mark.parametrize("fault", CASES)
def test_admission_refuses_round2_unsafe_accepts(tmp_path, monkeypatch, fault):
    records = _scheduler(tmp_path, monkeypatch)
    header, _ = _admission_fixture(tmp_path, monkeypatch)
    path = tmp_path / header["rows_artifact"]["path"]
    payload = json.loads(path.read_text())
    _edit_scheduler(records, fault)
    if fault == "invented_cpu":
        for side in ("head", "baseline"):
            entry = payload["hardware"][side]["execution_cpu.json"]
            cpu = json.loads(entry["content"])
            cpu["lscpu"][0]["data"] = "Invented CPU"
            entry["content"] = json.dumps(cpu)
            entry["sha256"] = hashlib.sha256(entry["content"].encode()).hexdigest()
    elif fault == "mixed_node_rows":
        if "scheduler_job_id" in payload["rows"][0]:
            payload["rows"][0]["scheduler_node"] = "auxme-imech172"
        else:
            # The reviewed transport discards raw row context; retain the discrepant
            # source evidence without introducing an unrelated old-schema error.
            raw = tmp_path / "receipts/behaviour/mixed_row_context.json"
            raw.write_text(
                json.dumps({"scheduler_job_id": "123", "scheduler_node": "auxme-imech172"})
            )
    path.write_text(json.dumps(payload))
    subprocess.run(["git", "add", "receipts/behaviour"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "round two unsafe evidence"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    header["head_sha"] = header["refute_review"]["head_sha"] = head
    header["rows_artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    body = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
    blockers = gate.check_receipt(body, ["robot_sf/planner/synthetic.py"], "ll7/robot_sf_ll7")
    assert blockers, "admission accepted invalid scheduler evidence"
    assert any("hardware comparison invalid" in item.lower() for item in blockers), blockers


def test_failed_resubmission_cannot_collect_prior_success(tmp_path, monkeypatch):
    """Both generated reruns fail; their IDs cannot certify an older successful directory."""
    import shutil
    import sys

    from scripts.ci import make_behaviour_receipt as producer
    from tests.validation.test_behaviour_receipt_sbatch_affinity import _args
    from tests.validation.test_make_behaviour_receipt import RELEASE_TAG

    records = _scheduler(tmp_path, monkeypatch)
    git, source, head, baseline, command = _collection(tmp_path, monkeypatch)
    _unchanged_outcomes(head)
    work = tmp_path / "resubmission"
    for name, path in (("head", head), ("baseline", baseline)):
        shutil.copytree(path, work / "sweeps" / name)
    tools = tmp_path / "scheduler-bin"
    uv = tools / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport os,sys\nassert sys.argv[1:3] == ['run','python']\nos.execv(sys.executable, [sys.executable,*sys.argv[3:]])\n"
    )
    uv.chmod(0o755)
    for name, script in {
        "hostname": f"#!/bin/sh\nprintf '%s\\n' {NODE}\n",
        "lscpu": '#!/bin/sh\nprintf \'%s\\n\' \'{"lscpu":[{"field":"Model name:","data":"AMD EPYC 9115 16-Core Processor"},{"field":"Architecture:","data":"x86_64"}]}\'\n',
    }.items():
        path = tools / name
        path.write_text(script)
        path.chmod(0o755)
    counter = tmp_path / "submission-counter"
    counter.write_text("789")
    sbatch = tools / "sbatch"
    sbatch.write_text(f"""#!{sys.executable}
import json,os,subprocess,sys
from pathlib import Path
counter=Path({str(counter)!r});job=counter.read_text();counter.write_text(str(int(job)+1))
result=subprocess.run(["bash",sys.argv[-1]],env={{**os.environ,"SLURM_JOB_ID":job,"SLURM_JOB_PARTITION":"pro6000","SLURM_JOB_NODELIST":{NODE!r},"SLURM_JOB_NODELIST":{NODE!r}}},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
path=Path({str(records)!r});rows=json.loads(path.read_text())
rows[job]={{"job_id":job,"state":"FAILED","exit_code":str(result.returncode)+":0","node":{NODE!r},"partition":"pro6000"}}
path.write_text(json.dumps(rows));print(job)
""")
    sbatch.chmod(0o755)
    monkeypatch.setattr(producer, "_ensure_worktree", lambda *args: tmp_path)
    monkeypatch.setattr(
        producer, "_sweep_command", lambda *args, **kwargs: ["bash", "-c", "exit 7"]
    )
    args = _args(work, "--sbatch-node", NODE)
    args.head_sha = source
    rerun_head, rerun_base = producer._prepare_sweeps(args, tmp_path, git("rev-parse", RELEASE_TAG))
    for option, value in (
        ("--head-sweep-dir", str(rerun_head.output_dir)),
        ("--baseline-sweep-dir", str(rerun_base.output_dir)),
        ("--job-id", rerun_head.job_id),
        ("--baseline-job-id", rerun_base.job_id),
    ):
        command[command.index(option) + 1] = value
    observed = json.loads(records.read_text())
    assert observed["789"]["exit_code"] == observed["790"]["exit_code"] == "7:0"
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode != 0, "failed reruns certified prior successful rows"
    assert "hardware comparison invalid" in result.stderr.lower(), result.stderr
    assert not (tmp_path / "receipts/behaviour/synthetic.json").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_node",
        "nonzero_completed",
        "running",
        "transport_unavailable",
        "wrong_row_job",
        "nested_cpu",
    ],
)
def test_collection_rejects_unverified_allocations_and_contexts(tmp_path, monkeypatch, fault):
    records = _scheduler(tmp_path, monkeypatch)
    _, _, head, _, command = _collection(tmp_path, monkeypatch)
    data = json.loads(records.read_text())
    if fault == "unknown_node":
        data["123"]["node"] = "unregistered-node"
    elif fault == "nonzero_completed":
        data["123"]["exit_code"] = "0:9"
    elif fault == "running":
        data["123"]["state"] = "RUNNING"
    elif fault == "transport_unavailable":
        ssh = tmp_path / "scheduler-bin/ssh"
        ssh.write_text("#!/bin/sh\nexit 255\n")
    else:
        raw = head / "campaigns/empty_world_main/runs/goal__differential_drive/episodes.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        if fault == "wrong_row_job":
            rows[0]["scheduler_job_id"] = "124"
        else:
            rows[0]["algorithm_metadata"]["execution_context"] = {"cpu_model": "Invented CPU"}
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if fault in {"wrong_row_job", "nested_cpu"}:
        from tests.validation.test_behaviour_receipt_provenance import refresh_sweep_binding

        refresh_sweep_binding(head, validate_identity=False)
    records.write_text(json.dumps(data))
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "hardware comparison invalid" in result.stderr.lower(), (
        result.stderr
    )
    assert not result.stdout


def _split_shards(tmp_path, source, head):
    from tests.validation.test_make_behaviour_receipt import _sweep_rows

    second = _sweep_rows(tmp_path, "head", source_sha=source, failing=True, job="124")
    for index, shard in enumerate((head, second)):
        for path in (
            shard / "episodes_main.jsonl",
            shard / "campaigns/empty_world_main/runs/goal__differential_drive/episodes.jsonl",
        ):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows = rows[:15] if index == 0 else rows[15:]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        path = shard / "execution_main.json"
        meta = json.loads(path.read_text())
        meta.update(
            seeds=[row["seed"] for row in rows],
            expected_slots=15,
            written_rows=15,
            total_episodes=15,
        )
        path.write_text(json.dumps(meta))
        from tests.validation.test_behaviour_receipt_provenance import refresh_sweep_binding

        refresh_sweep_binding(shard)
    return second


@pytest.mark.parametrize("mixed", [False, True])
def test_each_shard_is_bound_and_mixed_sides_are_refused(tmp_path, monkeypatch, mixed):
    records = _scheduler(tmp_path, monkeypatch)
    git, source, head, _, command = _collection(tmp_path, monkeypatch)
    _split_shards(tmp_path, source, head)
    command[command.index("--head-sweep-dir") + 1] = str(head.parent)
    command[command.index("--job-id") + 1] = "123,124"
    if mixed:
        data = json.loads(records.read_text())
        data["124"].update(node="auxme-imech172", partition="a30")
        records.write_text(json.dumps(data))
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    if mixed:
        assert result.returncode != 0 and "mixed scheduler nodes" in result.stderr, result.stderr
        assert not result.stdout
        return
    assert result.returncode == 0, result.stderr
    header = json.loads(result.stdout.split("\n", 1)[1].rsplit("-->", 1)[0])
    payload = json.loads((tmp_path / header["rows_artifact"]["path"]).read_text())
    assert {row["scheduler_job_id"] for row in payload["rows"]} == {"123", "124"}
    assert {row["scheduler_node"] for row in payload["rows"]} == {NODE}
    git("add", "receipts/behaviour")
    git("commit", "-m", "paired sharded receipt")
    final = git("rev-parse", "HEAD")
    header["head_sha"] = header["refute_review"]["head_sha"] = final
    header["refute_review"]["verdict"] = "accepted"
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", final)
    body = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
    assert gate.check_receipt(body, ["robot_sf/planner/synthetic.py"], "ll7/robot_sf_ll7") == []
