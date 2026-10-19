"""Exercise frozen train evidence through the gate and read-only promotion CLI."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.ci import behaviour_receipt as gate
from tests.validation.test_behaviour_receipt_gate import _receipt
from tests.validation.test_behaviour_receipt_transport import _commit_file, _git_repo


def test_missing_receipt_waits_without_admission():
    """An absent receipt is a scheduling blocker, never permission to merge."""
    blockers = gate.check_receipt("", ["robot_sf/planner/goal.py"], "ll7/robot_sf_ll7")
    assert len(blockers) == 1
    assert blockers[0].startswith("BLOCKER:") and "WAITING_FOR_TRAIN:" in blockers[0]
    assert "missing or duplicated" in blockers[0]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture
def candidate(monkeypatch, tmp_path):
    git = _git_repo(tmp_path)
    base = _commit_file(tmp_path, git, "code.txt", "base", "base")
    head = _commit_file(tmp_path, git, "robot_sf/planner/goal.py", "candidate", "candidate")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    scope = {
        "arms": ["goal", "orca"],
        "maps": ["open", "door"],
        "vehicle_id": "t60",
        "exceptions": [],
    }
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope))
    monkeypatch.setattr(gate, "SCOPE_PATH", scope_path)
    monkeypatch.setattr(gate, "latest_release", lambda repo: "0.0.7")
    monkeypatch.setattr(gate, "release_source", lambda release: "c" * 40)
    receipt = _receipt()
    receipt["head_sha"] = head
    receipt["scheduler"]["source_sha"] = head
    receipt["interaction_audit"]["source_sha"] = head
    receipt["refute_review"]["head_sha"] = head
    review = dict(receipt["refute_review"])
    config = {"resolved_config": "synthetic test only"}
    # The policy identity is computed independently from the documented file list.
    policy_root = Path(gate.__file__).parent
    policy = {
        name: hashlib.sha256((policy_root / name).read_bytes()).hexdigest()
        for name in ("behaviour_receipt.py", "behaviour_receipt.schema.json", "frozen_candidate.py")
        if (policy_root / name).exists()
    }
    manifest = {
        "schema_version": "behaviour-candidate.v1",
        "candidate_sha": head,
        "base_sha": base,
        "candidate_tree": git("rev-parse", f"{head}^{{tree}}"),
        "components": [{"pr": 123, "head_sha": head, "reviews": [review]}],
        "suite_sha256": digest(scope),
        "config_sha256": digest(config),
        "baseline_sha256": digest(receipt["baseline"]),
        "evaluation_policy_sha256": digest(policy),
    }
    executions = [
        {
            "arm": row["arm"],
            "map": row["map"],
            "seed": row["seed"],
            "source_sha": head,
            "execution_sha256": "e" * 64,
        }
        for row in receipt["rows"]
    ]
    envelope = {
        "schema_version": "behaviour-candidate-evidence.v1",
        "manifest": manifest,
        "receipt": receipt,
        "config": config,
        "episode_executions": executions,
        "receipt_sha256": digest(receipt),
        "episode_executions_sha256": digest(executions),
    }
    real_output = subprocess.check_output
    observed = []
    transport = {}

    def publish():
        raw = json.dumps(envelope).encode()
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("evidence.json", raw)
        blob = archive.getvalue()
        reference = {
            "artifact_id": 123,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "manifest_sha256": digest(manifest),
        }
        metadata = {
            "id": 123,
            "expired": False,
            "digest": "sha256:" + reference["sha256"],
            "size_in_bytes": len(blob),
            "workflow_run": {"head_sha": head},
        }

        def output(cmd, *args, **kwargs):
            if cmd[:2] == ["gh", "api"]:
                observed.append(cmd)
                if cmd[-1].endswith("/git/ref/heads/main"):
                    sha = transport.get("remote_main", git("rev-parse", "refs/remotes/origin/main"))
                    return json.dumps({"object": {"sha": sha}}).encode()
                if "metadata" in transport:
                    metadata.update(transport["metadata"])
                return blob if cmd[-1].endswith("/zip") else json.dumps(metadata).encode()
            return real_output(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "check_output", output)
        return "<!-- behaviour-change-receipt:v3\n" + json.dumps(reference) + "\n-->"

    return {
        "git": git,
        "base": base,
        "head": head,
        "manifest": manifest,
        "envelope": envelope,
        "publish": publish,
        "observed": observed,
        "transport": transport,
        "root": tmp_path,
    }


def test_external_evidence_does_not_change_candidate(candidate):
    receipt = candidate["envelope"]["receipt"]
    receipt["rows"][0]["success"] = False
    receipt["totals"]["new_failures"] = 1
    receipt["classifications"] = [
        {
            "arm": "goal",
            "map": "open",
            "seed": 1001,
            "kind": "success_to_failure",
            "class": "known_limitation",
            "count": 1,
            "evidence": "https://example.org/trace.json",
        }
    ]
    candidate["envelope"]["receipt_sha256"] = digest(receipt)
    body = candidate["publish"]()
    before = candidate["git"]("rev-parse", "HEAD")
    assert (
        gate.check_receipt(
            body, ["robot_sf/planner/goal.py"], "ll7/robot_sf_ll7", candidate["base"]
        )
        == []
    )
    receipt["classifications"][0]["evidence"] = "https://example.org/reviewed-trace.json"
    candidate["envelope"]["receipt_sha256"] = digest(receipt)
    assert (
        gate.check_receipt(
            candidate["publish"](),
            ["robot_sf/planner/goal.py"],
            "ll7/robot_sf_ll7",
            candidate["base"],
        )
        == []
    )
    assert candidate["git"]("rev-parse", "HEAD") == before
    assert candidate["git"]("status", "--porcelain", "--untracked-files=no") == ""
    assert candidate["observed"]


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("candidate", "candidate SHA mismatch"),
        ("base", "expected base mismatch"),
        ("tree", "candidate tree mismatch"),
        ("component", "component review head/verdict mismatch"),
        ("no_reviews", "component reviews missing"),
        ("suite", "suite digest mismatch"),
        ("config", "config digest mismatch"),
        ("baseline", "baseline digest mismatch"),
        ("policy", "evaluation policy digest mismatch"),
        ("receipt", "receipt digest mismatch"),
        ("executions_digest", "execution digest mismatch"),
        ("missing_episode", "episode execution coverage mismatch"),
        ("duplicate_episode", "episode execution coverage mismatch"),
        ("execution_source", "episode execution source/identity mismatch"),
        ("scheduler", "external evidence source must equal candidate"),
        ("stale_verdict", "exact-head refute verdict is missing"),
        ("coverage", "omitted or duplicate arm/map/development-seed coverage"),
        ("fallback", "fallback or degraded execution cannot satisfy the receipt"),
    ],
)
def test_external_refusals(candidate, fault, reason):  # noqa: C901 - independent refusal controls
    """A self-consistent archive cannot override frozen identities or scientific guards."""
    env, manifest = candidate["envelope"], candidate["manifest"]
    if fault in {"candidate", "base", "tree"}:
        manifest[
            {"candidate": "candidate_sha", "base": "base_sha", "tree": "candidate_tree"}[fault]
        ] = "b" * 40
    elif fault == "component":
        manifest["components"][0]["reviews"][0]["head_sha"] = "b" * 40
    elif fault == "no_reviews":
        manifest["components"][0]["reviews"] = []
    elif fault in {"suite", "config", "baseline", "policy"}:
        manifest[
            {
                "suite": "suite_sha256",
                "config": "config_sha256",
                "baseline": "baseline_sha256",
                "policy": "evaluation_policy_sha256",
            }[fault]
        ] = "0" * 64
    elif fault == "receipt":
        env["receipt_sha256"] = "0" * 64
    elif fault == "executions_digest":
        env["episode_executions_sha256"] = "0" * 64
    elif fault in {"missing_episode", "duplicate_episode", "execution_source"}:
        if fault == "missing_episode":
            env["episode_executions"].pop()
        elif fault == "duplicate_episode":
            env["episode_executions"].append(env["episode_executions"][0])
        else:
            env["episode_executions"][0]["source_sha"] = candidate["base"]
        env["episode_executions_sha256"] = digest(env["episode_executions"])
    else:
        if fault == "scheduler":
            env["receipt"]["scheduler"]["source_sha"] = candidate["base"]
        elif fault == "stale_verdict":
            env["receipt"]["refute_review"]["head_sha"] = candidate["base"]
        elif fault == "coverage":
            env["receipt"]["rows"].pop()
            env["episode_executions"].pop()
            env["episode_executions_sha256"] = digest(env["episode_executions"])
        else:
            env["receipt"]["rows"][0]["fallback"] = True
        env["receipt_sha256"] = digest(env["receipt"])
    blockers = gate.check_receipt(
        candidate["publish"](), ["robot_sf/planner/goal.py"], "ll7/robot_sf_ll7", candidate["base"]
    )
    assert len(blockers) == 1
    assert "blocked" in blockers[0] and reason in blockers[0]


def test_duplicate_receipts_are_blocked_not_waiting():
    block = "<!-- behaviour-change-receipt:v2\n{}\n-->"
    blockers = gate.check_receipt(block + block, ["robot_sf/planner/goal.py"], "ll7/robot_sf_ll7")
    assert "blocked" in blockers[0] and "WAITING_FOR_TRAIN" not in blockers[0]


@pytest.mark.parametrize(
    "phase", ["admitted", "WAITING_FOR_TRAIN", "candidate", "executing", "blocked", "ready"]
)
def test_train_state_is_explicit(candidate, capsys, phase):
    event = candidate["root"] / "event.json"
    event.write_text(
        json.dumps(
            {
                "pull_request": {"head": {"sha": candidate["head"]}, "body": ""},
                "repository": {"full_name": "ll7/robot_sf_ll7"},
            }
        )
    )
    # Make the production-path diff real, so the gate must remain nonzero.
    git = candidate["git"]
    head = _commit_file(candidate["root"], git, "robot_sf/planner/goal.py", "change", "planner")
    data = json.loads(event.read_text())
    data["pull_request"]["head"]["sha"] = head
    event.write_text(json.dumps(data))
    os.environ["BEHAVIOUR_PR_HEAD_SHA"] = head
    try:
        result = gate.main(
            [
                "--github-event-path",
                str(event),
                "--repo-root",
                str(candidate["root"]),
                "--base-ref",
                candidate["base"],
                "--train-state",
                phase,
            ]
        )
    except SystemExit as exc:
        result = exc.code
    output = capsys.readouterr().out
    assert result == 1 and '"merge_blocked": true' in output
    expected = phase if phase in {"candidate", "executing", "blocked"} else "WAITING_FOR_TRAIN"
    assert f'"train_state": "{expected}"' in output


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        (None, None),
        ("main", "main moved from expected base"),
        ("remote_main", "main moved from expected base"),
        ("tree", "landed tree mismatch"),
        ("receipt", "promotion receipt digest mismatch"),
        ("executions", "promotion execution digest mismatch"),
        ("head", "artifact candidate SHA mismatch"),
    ],
)
def test_promotion_refuses_identity_changes(candidate, capsys, fault, reason):
    env, git = candidate["envelope"], candidate["git"]
    body = candidate["publish"]()
    reference = json.loads(body.split("\n")[1])
    ref = candidate["root"] / "reference.json"
    ref.write_text(json.dumps(reference))
    git("update-ref", "refs/remotes/origin/main", candidate["base"])
    landed, head = candidate["head"], candidate["head"]
    receipt_digest, execution_digest = env["receipt_sha256"], env["episode_executions_sha256"]
    if fault == "main":
        git("update-ref", "refs/remotes/origin/main", head)
    elif fault == "remote_main":
        candidate["transport"]["remote_main"] = head
    elif fault == "tree":
        landed = candidate["base"]
    elif fault == "receipt":
        receipt_digest = "0" * 64
    elif fault == "executions":
        execution_digest = "0" * 64
    elif fault == "head":
        head = _commit_file(candidate["root"], git, "new.txt", "changed", "new integration")
    try:
        result = gate.main(
            [
                "promotion",
                "--reference",
                str(ref),
                "--repo",
                "ll7/robot_sf_ll7",
                "--candidate",
                head,
                "--expected-base",
                candidate["base"],
                "--landed",
                landed,
                "--receipt-sha256",
                receipt_digest,
                "--executions-sha256",
                execution_digest,
            ]
        )
    except SystemExit as exc:
        result = exc.code
    output = capsys.readouterr().out
    assert result == (1 if fault else 0), output
    assert '"merge_authorized": false' in output
    if fault:
        assert reason in output
    else:
        assert f'"candidate_sha": "{head}"' in output


def test_freeze_is_external_and_write_once(candidate, capsys):
    source = candidate["root"] / "manifest-input.json"
    source.write_text(json.dumps(candidate["manifest"]))
    target = candidate["root"].parent / (candidate["root"].name + "-frozen.json")
    args = [
        "freeze",
        "--input",
        str(source),
        "--output",
        str(target),
        "--candidate",
        candidate["head"],
        "--expected-base",
        candidate["base"],
    ]
    try:
        result = gate.main(args)
    except SystemExit as exc:
        result = exc.code
    assert result == 0, capsys.readouterr().out
    assert json.loads(target.read_text()) == candidate["manifest"]
    assert gate.main(args) == 1  # No rewriting an existing frozen manifest.
    assert "already exists" in capsys.readouterr().out


def test_artifact_read_permission_is_explicit():
    import yaml

    workflow = yaml.safe_load(
        (Path(__file__).parents[2] / ".github/workflows/pr-contract-check.yml").read_text()
    )
    assert workflow["permissions"].get("actions") == "read"


@pytest.mark.parametrize(
    ("metadata", "reason"),
    [
        ({"expired": True}, "artifact expired or wrong ID"),
        ({"id": 456}, "artifact expired or wrong ID"),
        ({"digest": "sha256:" + "0" * 64}, "artifact metadata digest mismatch"),
        ({"workflow_run": {"head_sha": "b" * 40}}, "artifact candidate SHA mismatch"),
        ({"size_in_bytes": 50 * 1024 * 1024}, "artifact archive size limit"),
    ],
)
def test_external_artifact_transport_refusals(candidate, metadata, reason):
    candidate["transport"]["metadata"] = metadata
    result = gate.check_receipt(
        candidate["publish"](), ["robot_sf/planner/goal.py"], "ll7/robot_sf_ll7", candidate["base"]
    )
    assert len(result) == 1 and "blocked" in result[0] and reason in result[0]


def test_missing_policy_inventory_is_blocked(candidate):
    gate.SCOPE_PATH.unlink()
    result = gate.check_receipt(
        "", ["robot_sf/planner/goal.py"], "ll7/robot_sf_ll7", candidate["base"]
    )
    assert len(result) == 1 and "WAITING_FOR_TRAIN" not in result[0]
    assert "scope inventory" in result[0]


def test_assemble_keeps_evidence_outside_source(candidate, capsys):
    env, root = candidate["envelope"], candidate["root"]
    paths = {}
    for name, value in (
        ("manifest", env["manifest"]),
        ("receipt", env["receipt"]),
        ("config", env["config"]),
        ("executions", env["episode_executions"]),
    ):
        paths[name] = root / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    target = root.parent / (root.name + "-evidence.json")
    args = ["assemble", "--repo", "ll7/robot_sf_ll7", "--output", str(target)]
    for name, path in paths.items():
        args.extend(["--" + name, str(path)])
    try:
        result = gate.main(args)
    except SystemExit as exc:
        result = exc.code
    assert result == 0, capsys.readouterr().out
    assert json.loads(target.read_text()) == env
    assert candidate["git"]("rev-parse", "HEAD") == candidate["head"]
    assert gate.main(args) == 1
    assert "already exists" in capsys.readouterr().out


def test_malformed_receipt_schema_has_a_blocked_verdict(candidate, capsys):
    env = candidate["envelope"]
    del env["receipt"]["totals"]
    env["receipt_sha256"] = digest(env["receipt"])
    body = candidate["publish"]()
    ref = candidate["root"] / "reference.json"
    ref.write_text(body.split("\n")[1])
    candidate["git"]("update-ref", "refs/remotes/origin/main", candidate["base"])
    try:
        result = gate.main(
            [
                "promotion",
                "--reference",
                str(ref),
                "--repo",
                "ll7/robot_sf_ll7",
                "--candidate",
                candidate["head"],
                "--expected-base",
                candidate["base"],
                "--landed",
                candidate["head"],
                "--receipt-sha256",
                env["receipt_sha256"],
                "--executions-sha256",
                env["episode_executions_sha256"],
            ]
        )
    except SystemExit as exc:
        result = exc.code
    assert result == 1
    assert "receipt schema mismatch" in capsys.readouterr().out


def test_base_owned_standalone_gate_reads_external_candidate(candidate):
    """The deployed script imports only its own frozen policy and checks source objects."""
    env, root = candidate["envelope"], candidate["root"]
    trusted = root.parent / (root.name + "-trusted")
    policy_root = Path(gate.__file__).parent
    for name in ("behaviour_receipt.py", "behaviour_receipt.schema.json", "frozen_candidate.py"):
        source = policy_root / name
        if source.exists():
            target = trusted / "scripts/ci" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    scope = trusted / gate.SCOPE_FILE
    scope.parent.mkdir(parents=True)
    shutil.copyfile(gate.SCOPE_PATH, scope)
    candidate["git"]("tag", "0.0.7", candidate["base"])
    env["receipt"]["baseline"]["source_sha"] = candidate["base"]
    env["receipt_sha256"] = digest(env["receipt"])
    env["manifest"]["baseline_sha256"] = digest(env["receipt"]["baseline"])
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("evidence.json", json.dumps(env))
    blob = archive.getvalue()
    reference = {
        "artifact_id": 123,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "manifest_sha256": digest(env["manifest"]),
    }
    (trusted / "archive.zip").write_bytes(blob)
    metadata = {
        "id": 123,
        "expired": False,
        "digest": "sha256:" + reference["sha256"],
        "size_in_bytes": len(blob),
        "workflow_run": {"head_sha": candidate["head"]},
    }
    (trusted / "metadata.json").write_text(json.dumps(metadata))
    tools = trusted / "bin"
    tools.mkdir()
    gh = tools / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        + """import json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
endpoint = sys.argv[2]
if endpoint.endswith("/releases"):
    print(json.dumps([[{"draft": False, "prerelease": False, "tag_name": "0.0.7", "published_at": "2026-01-01"}]]))
else:
    sys.stdout.buffer.write((root / ("archive.zip" if endpoint.endswith("/zip") else "metadata.json")).read_bytes())
"""
    )
    gh.chmod(0o755)
    event = trusted / "event.json"
    event.write_text(
        json.dumps(
            {
                "repository": {"full_name": "ll7/robot_sf_ll7"},
                "pull_request": {
                    "head": {"sha": candidate["head"]},
                    "body": "<!-- behaviour-change-receipt:v3\n" + json.dumps(reference) + "\n-->",
                },
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(trusted / "scripts/ci/behaviour_receipt.py"),
            "--github-event-path",
            str(event),
            "--repo-root",
            str(root),
            "--base-ref",
            candidate["base"],
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"]},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"merge_blocked": false' in result.stdout
    assert "accepted or out of scope" in result.stdout
