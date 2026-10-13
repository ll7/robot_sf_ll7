"""Exact-byte default-path admission, using positive controls before refusals.

Every negative case starts with an accepted committed receipt and changes only
its named guard, so an unrelated missing fixture cannot satisfy the assertion.
"""

import copy
import hashlib
import itertools
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import default_path_equivalence as eq
from tests.validation.receipt_scheduler_fixture import CPU, NODE, install_scheduler

PATH = "robot_sf/benchmark/camera_ready/_run_state.py"
BEFORE = '''def _episode_identity(record: Mapping[str, Any]) -> tuple[str, int | str | None]:
    """Return the stable logical identity used by aggregate-integrity checks."""
    return str(record.get("scenario_id", "")).strip(), _coerce_identifier(record.get("seed"))
'''
AFTER = '''def _episode_identity(record: Mapping[str, Any]) -> tuple[str, int | str | None]:
    """Return a logical identity only for exact integer saved seeds.

    Invalid seed types remain unavailable rather than aliasing an expected slot.
    """
    seed = record.get("seed")
    return str(record.get("scenario_id", "")).strip(), seed if type(seed) is int else None
'''


@pytest.fixture(autouse=True)
def scheduler_endpoint(tmp_path, monkeypatch):
    records_path = install_scheduler(tmp_path, monkeypatch)
    records = json.loads(records_path.read_text())
    for job in ("11", "12"):
        records[job] = {**records["123"], "job_id": job}
    records_path.write_text(json.dumps(records))
    return records_path


def _hardware(source, job, *, node=NODE, cpu=CPU, node_class="pro6000", role=None):
    contents = {
        "execution_node.txt": node + "\n",
        "execution_node_class.txt": node_class + "\n",
        "execution_scheduler.json": json.dumps(
            gate.scheduler_snapshot(gate.scheduler_job_ids(job))
        ),
        "execution_cpu.json": json.dumps(
            {
                "lscpu": [
                    {"field": "Model name:", "data": cpu},
                    {"field": "Architecture:", "data": "x86_64"},
                ]
            }
        ),
        "execution_request.json": json.dumps(
            {
                "schema_version": "behaviour-hardware-request.v1",
                "requested_node": node,
                "source_sha": source,
                "job_id": job,
                "arm_role": role or ("baseline" if job == "12" else "head"),
            }
        ),
    }
    return {
        name: {"content": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()}
        for name, raw in contents.items()
    }


def _fixture_execution_algorithm(scope, arm, scenario):
    from scripts.ci import receipt_arm_execution

    if arm in receipt_arm_execution.PROFILES and scenario == receipt_arm_execution.SCENARIO:
        return "orca"
    return scope["arm_algorithms"][arm]


def _fixture_execution_mode(scope, arm, scenario):
    from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata

    profile = enrich_algorithm_metadata(algo=_fixture_execution_algorithm(scope, arm, scenario))[
        "planner_kinematics"
    ]
    if _fixture_execution_algorithm(scope, arm, scenario) == "guarded_ppo":
        return "mixed"
    return "native" if profile["supports_native_commands"] else "adapter"


def _build_admitted(tmp_path, monkeypatch, *, scope_override=None):
    # This synthetic historical base declares no reviewed evidence trees.
    monkeypatch.setattr(eq, "REVIEWED_CLOSED_TREES", {})

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    module = tmp_path / PATH
    module.parent.mkdir(parents=True)
    module.write_text(BEFORE)
    tooling_path = tmp_path / "scripts/validation/run_empty_world_sweep.py"
    tooling_path.parent.mkdir(parents=True)
    tooling_path.write_text("# pinned acquisition tooling fixture\n")
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    monkeypatch.setattr(gate, "current_main_source", lambda _repo: base)
    module.write_text(AFTER)
    git("add", ".")
    git("commit", "-m", "saved-seed validator")
    source = git("rev-parse", "HEAD")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    scope = {
        "arms": ["ppo", "guarded_ppo", *[f"arm-{i}" for i in range(12)]],
        "maps": [f"map-{i}" for i in range(51)],
    }
    scope["arm_algorithms"] = {
        arm: arm if arm in {"ppo", "guarded_ppo"} else "goal" for arm in scope["arms"]
    }
    if scope_override is not None:
        scope = scope_override
    rows = [
        {
            "arm": arm,
            "map": scenario,
            "seed": seed,
            "scheduler_job_id": "11",
            "scheduler_node": NODE,
            "baseline_scheduler_job_id": "12",
            "baseline_scheduler_node": NODE,
            "baseline_sha256": "0" * 64,
            "head_sha256": "0" * 64,
            "head_execution": {
                "algorithm": _fixture_execution_algorithm(scope, arm, scenario),
                "execution_mode": _fixture_execution_mode(scope, arm, scenario),
                "controller_executed": True,
                "fallback": False,
                "degraded": False,
            },
            "baseline_execution": {
                "algorithm": _fixture_execution_algorithm(scope, arm, scenario),
                "execution_mode": _fixture_execution_mode(scope, arm, scenario),
                "controller_executed": True,
                "fallback": False,
                "degraded": False,
            },
            "head_path": f"campaigns/empty_world_main/runs/{arm}__differential_drive/episodes.jsonl",
            "baseline_path": f"campaigns/empty_world_main/runs/{arm}__differential_drive/episodes.jsonl",
        }
        for arm, scenario, seed in itertools.product(
            scope["arms"], scope["maps"], [1001, 1002, 1003]
        )
    ]
    payload = {
        "schema_version": "default-path-equivalence.v1",
        "admission_context": "pull_request",
        "declaration": "default_benchmark_behaviour_unchanged",
        "head_source_sha": source,
        "scheduler": {"source_sha": source, "job_id": "11_12"},
        "baseline": {
            "source_sha": base,
            "artifact": {"uri": "https://evidence.example/baseline.tar", "sha256": "b" * 64},
        },
        "artifact": {"uri": "https://evidence.example/head.tar", "sha256": "a" * 64},
        "tooling_sha": base,
        "tooling_path": "scripts/validation/run_empty_world_sweep.py",
        "tooling_file_sha256": hashlib.sha256(tooling_path.read_bytes()).hexdigest(),
        "harness_sha256": "c" * 64,
        "harness_path": "acquisition_harness.py",
        "scope_sha256": hashlib.sha256(
            json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "projection_sha256": hashlib.sha256(
            json.dumps(eq.EXCLUDED_LEAVES, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "scope_transitions": [PATH + ":_episode_identity"],
        "seeds": [1001, 1002, 1003],
        "rows": rows,
        "hardware": {"head": _hardware(source, "11"), "baseline": _hardware(base, "12")},
        "totals": {"episodes": 2142, "changed_episodes": 0},
    }
    from scripts.ci import make_behaviour_receipt as producer

    payload["source_artifacts"] = {}
    with tempfile.TemporaryDirectory() as temporary:
        for side, runtime_source, job in (("head", source, "11"), ("baseline", base, "12")):
            sweep = Path(temporary) / side / job
            sweep.parent.mkdir()
            _write_equivalence_sweep(
                sweep, runtime_source, job, base, payload["tooling_file_sha256"], scope
            )
            raw_rows = producer._load_sweep_rows(
                sweep, scheduler_job_id=job, scheduler_node=NODE, preserve_raw=True
            )
            for compact in rows:
                compact[f"{side}_sha256"] = hashlib.sha256(
                    eq.episode_bytes(raw_rows[gate._key(compact)])
                ).hexdigest()
            bundle = tmp_path / f"receipts/behaviour/fixture-sources/{side}/{job}.tar.gz"
            producer._archive_source_bundle(
                sweep,
                bundle,
                extra_files=(sweep / "acquisition_harness.py", sweep / "execution_request.json"),
            )
            payload["source_artifacts"][side] = {
                job: {
                    "path": bundle.relative_to(tmp_path).as_posix(),
                    "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                    "binding": {
                        "path": "execution_binding.json",
                        "sha256": hashlib.sha256(
                            (sweep / "execution_binding.json").read_bytes()
                        ).hexdigest(),
                    },
                }
            }
    payload["harness_sha256"] = hashlib.sha256(
        b"# pinned acquisition wrapper fixture\n"
    ).hexdigest()
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope))
    monkeypatch.setattr(gate, "SCOPE_PATH", scope_path)
    receipt_path = tmp_path / "receipts/behaviour/equivalence.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload))
    git("add", "receipts")
    git("commit", "-m", "receipt-only artifact")
    head = git("rev-parse", "HEAD")
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    monkeypatch.delenv("BEHAVIOUR_RECEIPT_CONTEXT", raising=False)
    header = {
        "schema_version": "default-path-equivalence-header.v1",
        "admission_context": "pull_request",
        "head_sha": head,
        "rows_artifact": {
            "path": "receipts/behaviour/equivalence.json",
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        },
        "refute_review": {
            "head_sha": head,
            "verdict": "accepted",
            "uri": "https://review.example/exact-head",
            "comparison_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    }
    body = "<!-- default-path-equivalence:v1\n" + json.dumps(header) + "\n-->"
    assert (
        gate.check_receipt(body, [PATH, "receipts/behaviour/equivalence.json"], "test/repo", base)
        == []
    )
    return payload, header, scope, head, base, body


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    return _build_admitted(tmp_path, monkeypatch)


def test_committed_valid_equivalence_route_admits(admitted):
    """A valid exact-source artifact plus exact-head review passes the real gate."""
    _, _, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []


@pytest.mark.parametrize(
    "path",
    [
        "robot_sf/planner/example.py",
        "robot_sf/benchmark/metrics.py",
        "robot_sf/gym_env/robot_env.py",
        "robot_sf/training/existing_default.py",
    ],
)
def test_default_and_unknown_code_paths_refuse(admitted, path):
    """Self-declaration cannot admit default physical/metric or unknown paths."""
    _, _, _, _, base, body = admitted
    blockers = gate.check_receipt(body, [PATH, path], "test/repo", base)
    assert len(blockers) == 1 and (
        "scope refused:" in blockers[0] or "scope uncertain:" in blockers[0]
    )


def test_default_change_inside_allowed_surface_refuses(admitted, tmp_path):
    """A modified saved-state module outside the precise predicate is unknown."""
    _, _, _, head, base, _ = admitted
    with pytest.raises(ValueError, match="unreviewed AST transition"):
        # Different base source bytes, not a missing fixture, trigger the guard.
        (tmp_path / PATH).write_text(BEFORE + "\ndef default_action():\n    return 2\n")
        subprocess.run(["git", "add", PATH], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "default change"], cwd=tmp_path, check=True, capture_output=True
        )
        changed = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
        ).strip()
        eq.classify_scope([PATH], base, changed)
    assert head != changed


@pytest.mark.parametrize(
    "kind",
    ["byte", "node", "learned_cpu", "release", "pending_review", "base_drift", "missing_row"],
)
def test_named_admission_guards_refuse(admitted, kind, scheduler_endpoint):
    """Hardware, coverage, byte equality, main identity and authority fail closed."""
    payload, header, scope, head, base, _ = admitted
    receipt = {
        **copy.deepcopy(payload),
        "head_sha": head,
        "refute_review": copy.deepcopy(header["refute_review"]),
    }
    reason = {
        "byte": "changed behavioral byte",
        "node": "claimed node differs from scheduler allocation",
        "learned_cpu": "head and baseline cpu_model differ",
        "release": "releases require full",
        "pending_review": "independent exact-head verdict",
        "base_drift": "baseline differs from current main",
        "missing_row": "incomplete/duplicate paired episode coverage",
    }[kind]
    if kind == "byte":
        receipt["rows"][0]["head_sha256"] = "0" + receipt["rows"][0]["head_sha256"][1:]
    elif kind == "node":
        receipt["hardware"]["head"] = _hardware(payload["head_source_sha"], "11", node="compute-2")
    elif kind == "learned_cpu":
        records = json.loads(scheduler_endpoint.read_text())
        records["11"].update(node="auxme-imech172", partition="a30")
        scheduler_endpoint.write_text(json.dumps(records))
        receipt["hardware"]["head"] = _hardware(
            payload["head_source_sha"],
            "11",
            node="auxme-imech172",
            cpu="Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz",
            node_class="a30",
        )
    elif kind == "pending_review":
        receipt["refute_review"]["verdict"] = "pending_independent_review"
    elif kind == "base_drift":
        base = "d" * 40
    elif kind == "missing_row":
        receipt["rows"].pop()
    with pytest.raises(ValueError, match=reason):
        eq.validate_receipt(receipt, scope, head, base, [PATH], is_release=kind == "release")


def _episode():
    return {
        "algo": "goal",
        "execution_status": "written",
        "steps": 1,
        "metrics": {"success": True, "collisions": 0, "total_collision_count": 0},
        "algorithm_metadata": {
            "status": "ok",
            "planner_kinematics": {"execution_mode": "native"},
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v2",
                "dt": 0.1,
                "reset": {"robot": {"position": [0.0, 0.0]}},
                "steps": [
                    {
                        "robot": {"position": [0.1, 0.0]},
                        "pedestrians": [],
                        "collision": {},
                        "rl": {"reward": 0.0},
                        "planner": {
                            "selected_action": {"linear_velocity": 1.0},
                            "applied_environment_action": {"linear_velocity": 1.0},
                        },
                    }
                ],
            },
        },
    }


@pytest.mark.parametrize(
    "mutation", ["action", "pose", "event", "metric", "reward", "unknown_provenance"]
)
@pytest.mark.parametrize("algorithm", ["goal", "ppo"])
def test_producer_compares_each_behavioral_byte(mutation, algorithm):
    """All behavior including learned reward and unknown metadata remains compared."""
    row = _episode()
    row["algo"] = algorithm
    changed = copy.deepcopy(row)
    if mutation == "action":
        changed["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["planner"][
            "selected_action"
        ]["linear_velocity"] = 1.1
    elif mutation == "pose":
        changed["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["robot"]["position"][
            0
        ] = 0.2
    elif mutation == "event":
        changed["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["collision"]["robot"] = (
            True
        )
    elif mutation == "metric":
        changed["metrics"]["success"] = False
    elif mutation == "reward":
        changed["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["rl"]["reward"] = 0.1
    else:
        changed["provenance"] = {"unknown_behavior": "changed"}
    scope = {"arms": [algorithm], "maps": ["open"]}
    base = {(algorithm, "open", seed): row for seed in (1001, 1002, 1003)}
    head = {**base, (algorithm, "open", 1001): changed}
    assert len(eq.compare_rows(base, base, scope)) == 3
    with pytest.raises(ValueError, match="changed behavioral byte"):
        eq.compare_rows(head, base, scope)


def test_only_named_transport_leaves_are_excluded():
    """Volatile wall-clock/source labels are excluded; configuration is retained."""
    base = _episode()
    changed = copy.deepcopy(base)
    changed.update(
        git_hash="different",
        timestamps={"start": "different", "end": "different"},
        wall_time_sec=23.0,
        timing={"steps_per_second": 1.0},
    )
    # Parent object presence remains compared. Both have identical transport shape.
    base.update(
        git_hash="base",
        timestamps={"start": "base", "end": "base"},
        wall_time_sec=1.0,
        timing={"steps_per_second": 2.0},
    )
    assert eq.episode_bytes(base) == eq.episode_bytes(changed)
    changed["timing"]["simulated_time"] = 10.0
    assert eq.episode_bytes(base) != eq.episode_bytes(changed)


def test_scope_cannot_hide_current_main_default_changes(admitted, tmp_path):
    """An incomplete caller diff cannot hide exact execution versus current-main drift."""
    _, _, _, head, base, _ = admitted
    protected = tmp_path / "robot_sf/planner/default.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("DEFAULT_SPEED = 2.0\n")
    subprocess.run(["git", "add", str(protected)], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "main default behavior"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    current_main = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    with pytest.raises(ValueError, match="scope refused: protected default path"):
        eq.classify_scope([PATH], current_main, head)
    assert base != current_main


def test_release_context_never_exempts_document_only_release(admitted, monkeypatch):
    """Release callers must supply full evidence even when their diff is only prose."""
    _, _, _, _, base, body = admitted
    monkeypatch.setenv("BEHAVIOUR_RECEIPT_CONTEXT", "release")
    assert gate.check_receipt(body, ["docs/release.md"], "test/repo", base) == [
        "BLOCKER: equivalence refused: releases require full behaviour receipt"
    ]


def test_pinned_tooling_byte_mismatch_refuses(admitted):
    """Claiming an immutable script source does not excuse a different archived digest."""
    payload, header, scope, head, base, _ = admitted
    receipt = {**copy.deepcopy(payload), "head_sha": head, "refute_review": header["refute_review"]}
    receipt["tooling_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tooling differs from pinned Git blob"):
        eq.validate_receipt(receipt, scope, head, base, [PATH])


def _write_equivalence_sweep(output, source, job, tooling_sha, tooling_digest, scope):
    output.mkdir()
    wrapper_bytes = b"# pinned acquisition wrapper fixture\n"
    (output / "acquisition_harness.py").write_bytes(wrapper_bytes)
    for name, entry in _hardware(source, job).items():
        (output / name).write_text(entry["content"])
    summaries = []
    for arm in scope["arms"]:
        relative = f"runs/{arm}__differential_drive/episodes.jsonl"
        raw_path = output / "campaigns/empty_world_main" / relative
        raw_path.parent.mkdir(parents=True)
        raw_rows = []
        for scenario, seed in itertools.product(scope["maps"], [1001, 1002, 1003]):
            raw = _episode()
            raw.update(
                scenario_id=scenario,
                seed=seed,
                scheduler_job_id=job,
                scheduler_node=NODE,
                git_hash=source,
                algo=scope.get("arm_algorithms", {}).get(arm, "goal"),
            )
            if raw["algo"] == "guarded_ppo":
                from tests.validation.receipt_guarded_fixture import guarded_episode

                raw = guarded_episode(raw)
            raw_rows.append(raw)
            summaries.append(
                {
                    "arm": arm,
                    "scenario": scenario,
                    "seed": seed,
                    "source_file": relative,
                    "success": True,
                    "collisions": 0,
                    "steps": 1,
                    "execution_status": "written",
                    "scheduler_job_id": job,
                    "scheduler_node": NODE,
                }
            )
        raw_path.write_text("\n".join(json.dumps(row) for row in raw_rows) + "\n")
    (output / "episodes_main.jsonl").write_text(
        "\n".join(json.dumps(row) for row in summaries) + "\n"
    )
    metadata = {
        "head_sha": source,
        "suite": "main",
        "complete": True,
        "trace_requested": True,
        "campaign_execution_status": "completed",
        "exit_code": 0,
        "unexpected_failed_runs": 0,
        "failed_slots": [],
        "missing_slots": [],
        "duplicate_slots": [],
        "unexpected_slots": [],
        "incomplete_trace_slots": [],
        "expected_slots": 2142,
        "written_rows": 2142,
        "total_episodes": 2142,
        "seeds": [1001, 1002, 1003],
        "tooling_sha": tooling_sha,
        "tooling_path": "scripts/validation/run_empty_world_sweep.py",
        "tooling_file_sha256": tooling_digest,
        "private_harness_sha256": hashlib.sha256(wrapper_bytes).hexdigest(),
    }
    (output / "execution_main.json").write_text(json.dumps(metadata))
    _refresh_equivalence_binding(output)


def _refresh_equivalence_binding(root, *, validate_identity=True):
    """Generate fresh fixture producer seals, with explicit synthetic runtime Git identity."""
    from robot_sf.benchmark import result_provenance
    from tests.validation.test_behaviour_receipt_provenance import refresh_sweep_binding

    source = json.loads((root / "execution_main.json").read_text())["head_sha"]
    with patch.object(result_provenance, "_git_hash_fallback", return_value=source):
        refresh_sweep_binding(root, validate_identity=validate_identity)


def rewrite_source_archive(
    repo, payload, side, mutator, *, synchronize=False, suffix="changed", post_binding_mutator=None
):
    """Produce another sealed fixture source; admission still verifies every seal and projection."""
    from scripts.ci import make_behaviour_receipt as producer

    job, reference = next(iter(payload["source_artifacts"][side].items()))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / job
        root.mkdir()
        with tarfile.open(repo / reference["path"], "r:gz") as archive:
            archive.extractall(root, filter="data")
        mutator(root)
        _refresh_equivalence_binding(root, validate_identity=False)
        if post_binding_mutator is not None:
            post_binding_mutator(root)
        destination = repo / reference["path"]
        destination = destination.with_name(
            destination.name.removesuffix(".tar.gz") + f"-{suffix}.tar.gz"
        )
        producer._archive_source_bundle(
            root,
            destination,
            extra_files=(
                root / "acquisition_harness.py",
                root / "execution_request.json",
                *root.glob("campaigns/**/fake.jsonl"),
            ),
        )
        reference.update(
            path=destination.relative_to(repo).as_posix(),
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        )
        reference["binding"]["sha256"] = hashlib.sha256(
            (root / "execution_binding.json").read_bytes()
        ).hexdigest()
        if synchronize:
            raw = producer._load_sweep_rows(
                root, scheduler_job_id=job, scheduler_node=NODE, preserve_raw=True
            )
            statuses = producer._sweep_execution_statuses(root)
            for row in payload["rows"]:
                key = gate._key(row)
                row[f"{side}_sha256"] = hashlib.sha256(eq.episode_bytes(raw[key])).hexdigest()
                row[f"{side}_execution"] = producer._execution_evidence(
                    {**raw[key], "execution_status": statuses[key]}
                )


def duplicate_source_json(root, fault):
    """Conflicting first fields are lost by ordinary json.loads despite fresh seals."""
    if fault.startswith("raw_"):
        path = next(root.glob("campaigns/**/episodes.jsonl"))
        lines = path.read_text().splitlines()
        if fault == "raw_top":
            lines[0] = '{"seed":1002,' + lines[0][1:]
        else:
            lines[0] = lines[0].replace('"rl": {', '"rl": {"reward":9,', 1)
    elif fault == "summary":
        path = root / "episodes_main.jsonl"
        lines = path.read_text().splitlines()
        lines[0] = '{"execution_status":"failed",' + lines[0][1:]
    else:
        path = root / ("execution_binding.json" if fault == "binding" else "execution_main.json")
        lines = path.read_text().splitlines()
        prefix = '"job_id":"999",' if fault == "binding" else '"complete":false,'
        lines[0] = "{" + prefix + lines[0][1:]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("fault", ["raw_top", "raw_nested", "summary", "metadata", "binding"])
def test_private_comparison_refuses_duplicate_source_fields(exact_source_sweeps, capsys, fault):
    """Every input kind starts as accepted; coherent seals cannot hide duplicate claims."""
    from scripts.ci import make_behaviour_receipt as producer

    args, _, head = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    if fault != "binding":
        duplicate_source_json(head, fault)
        _refresh_equivalence_binding(head)
    else:
        duplicate_source_json(head, fault)
    with pytest.raises(gate.HardwareIdentityError, match="duplicate object key refused"):
        producer.main(args)


@pytest.mark.parametrize("fault", ["header", "payload", "hardware"])
def test_actual_gate_refuses_duplicate_selected_claims(admitted, tmp_path, monkeypatch, fault):
    """Previously discarded selected claims refuse despite otherwise valid immutable evidence."""
    payload, header, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    payload, header = copy.deepcopy(payload), copy.deepcopy(header)
    if fault == "hardware":
        entry = payload["hardware"]["head"]["execution_request.json"]
        entry["content"] = '{"job_id":"999",' + entry["content"][1:]
        entry["sha256"] = hashlib.sha256(entry["content"].encode()).hexdigest()
        body = committed_control_body(tmp_path, payload, header, monkeypatch)
    elif fault == "header":
        raw = '{"head_sha":"' + "0" * 40 + '",' + json.dumps(header)[1:]
        body = "<!-- default-path-equivalence:v1\n" + raw + "\n-->"
    else:
        path = tmp_path / header["rows_artifact"]["path"]
        path.write_text('{"head_source_sha":"' + "0" * 40 + '",' + json.dumps(payload)[1:])
        subprocess.run(["git", "add", "receipts"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "duplicate compact claim"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
        ).strip()
        header["head_sha"] = header["refute_review"]["head_sha"] = head
        header["rows_artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
        body = "<!-- default-path-equivalence:v1\n" + json.dumps(header) + "\n-->"
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert len(blockers) == 1 and "duplicate object key refused" in blockers[0]


def stamp_runtime_aliases(root, *, wrong_source=None, wrong_leaf=None):
    """Actual-shaped row aliases identify the rollout source, not a model checkpoint."""
    source = json.loads((root / "execution_main.json").read_text())["head_sha"]
    for raw in root.glob("campaigns/**/episodes.jsonl"):
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        for row in rows:
            for path in eq.RUNTIME_SOURCE_LEAVES:
                current = row
                for key in path[:-1]:
                    current = current.setdefault(key, {})
                current[path[-1]] = wrong_source if path == wrong_leaf else source
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_different_exact_runtime_aliases_are_transport_only(admitted, tmp_path, monkeypatch):
    """A real source pair may differ in precisely validated repository commit leaves."""
    payload, header, _, _, base, _ = admitted
    payload, header = copy.deepcopy(payload), copy.deepcopy(header)
    for side in ("head", "baseline"):
        rewrite_source_archive(tmp_path, payload, side, stamp_runtime_aliases, synchronize=True)
    assert all(row["head_sha256"] == row["baseline_sha256"] for row in payload["rows"])
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []


@pytest.mark.parametrize("wrong_leaf", eq.RUNTIME_SOURCE_LEAVES)
def test_private_comparison_refuses_copied_runtime_alias(exact_source_sweeps, capsys, wrong_leaf):
    """Private equality cannot ignore a sealed row's contradictory source identity."""
    from scripts.ci import make_behaviour_receipt as producer

    args, baseline, head = exact_source_sweeps
    for root in (baseline, head):
        stamp_runtime_aliases(root)
        _refresh_equivalence_binding(root)
    assert producer.main(args) == 0
    capsys.readouterr()
    stamp_runtime_aliases(
        head,
        wrong_source=json.loads((baseline / "execution_main.json").read_text())["head_sha"],
        wrong_leaf=wrong_leaf,
    )
    _refresh_equivalence_binding(head, validate_identity=False)
    with pytest.raises(
        gate.HardwareIdentityError,
        match=(
            "original (row|execution) source differs|producer row source or link differs"
            "|provenance digest producer source contradicts claimed commit"
        ),
    ):
        producer.main(args)


def mutate_summary_source(root, fault, payload):
    """Produce coherent seals for an invalid source-to-summary join."""
    path = root / "episodes_main.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    if fault == "summary_missing":
        row["source_file"] = "runs/nonexistent/episodes.jsonl"
    elif fault == "summary_wrong_arm":
        row["source_file"] = f"runs/{payload['rows'][-1]['arm']}__differential_drive/episodes.jsonl"
    elif fault == "summary_extra":
        original = root / "campaigns/empty_world_main" / row["source_file"]
        fake = original.with_name("fake.jsonl")
        fake.write_bytes(original.read_bytes())
        row["source_file"] = row["source_file"].replace("episodes.jsonl", "fake.jsonl")
        payload["rows"][0]["head_path"] = "campaigns/empty_world_main/" + row["source_file"]
    else:
        lines[-1] = lines[0]
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n")


def committed_control_body(repo, payload, header, monkeypatch):
    """Commit a current selected fixture with immutable payload and archive identities."""
    receipt = repo / header["rows_artifact"]["path"]
    receipt.write_text(json.dumps(payload))
    subprocess.run(["git", "add", "receipts"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "source custody control"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    header["head_sha"] = header["refute_review"]["head_sha"] = head
    header["rows_artifact"]["sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    header["refute_review"]["comparison_sha256"] = eq.digest(payload["rows"])
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    return "<!-- default-path-equivalence:v1\n" + json.dumps(header) + "\n-->"


@pytest.mark.parametrize(
    "fault", ["fabricated_equal_hashes", "source_swap", "metadata_source", "metadata_count"]
)
def test_actual_admission_rederives_sealed_source_proof(admitted, tmp_path, monkeypatch, fault):
    """Real seals cannot authorize fabricated equality, runtime identity or completeness."""
    payload, header, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    payload, header = copy.deepcopy(payload), copy.deepcopy(header)

    def mutate(root):
        if fault in {"fabricated_equal_hashes"}:
            change_first_raw_reward(root)
        elif fault == "provenance_source":
            path = next(root.glob("campaigns/**/episodes.jsonl"))
            lines = path.read_text().splitlines()
            row = json.loads(lines[0])
            row.setdefault("provenance", {})["git_hash"] = base
            lines[0] = json.dumps(row)
            path.write_text("\n".join(lines) + "\n")
        elif fault == "source_swap":
            for raw in root.glob("campaigns/**/episodes.jsonl"):
                rows = [json.loads(line) for line in raw.read_text().splitlines()]
                for row in rows:
                    row["git_hash"] = base
                raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        else:
            path = root / "execution_main.json"
            metadata = json.loads(path.read_text())
            metadata["head_sha" if fault == "metadata_source" else "total_episodes"] = (
                base if fault == "metadata_source" else 1
            )
            path.write_text(json.dumps(metadata))

    rewrite_source_archive(tmp_path, payload, "head", mutate)
    reasons = {
        "fabricated_equal_hashes": "behavioral/execution source projection differs",
        "source_swap": "producer row source or link differs",
        "metadata_source": "recorded SHA differs",
        "metadata_count": "execution row counts differ",
        "provenance_source": "original row source differs",
    }
    changed_body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(changed_body, [PATH], "test/repo", base)
    assert len(blockers) == 1 and reasons[fault] in blockers[0]


def test_historical_planner_commit_is_not_runtime_source(admitted, tmp_path, monkeypatch):
    """A sealed upstream/model identity can predate both exact rollout sources."""
    payload, header, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    payload, header = copy.deepcopy(payload), copy.deepcopy(header)

    def historical(root):
        for raw in root.glob("campaigns/**/episodes.jsonl"):
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            for row in rows:
                row.setdefault("provenance", {})["planner_commit"] = "f" * 40
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))

    for side in ("head", "baseline"):
        rewrite_source_archive(tmp_path, payload, side, historical, synchronize=True)
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []


@pytest.mark.parametrize("kind", ["node", "learned_cpu"])
def test_actual_admitted_path_refuses_hardware_difference(
    admitted, tmp_path, monkeypatch, scheduler_endpoint, kind
):
    """A complete sealed/source-valid candidate reaches the named hardware guard."""
    payload, header, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    payload, header = copy.deepcopy(payload), copy.deepcopy(header)
    if kind == "node":
        payload["hardware"]["head"] = _hardware(payload["head_source_sha"], "11", node="compute-2")
        reason = "claimed node differs from scheduler allocation"
    else:
        records = json.loads(scheduler_endpoint.read_text())
        records["11"].update(node="auxme-imech172", partition="a30")
        scheduler_endpoint.write_text(json.dumps(records))
        payload["hardware"]["head"] = _hardware(
            payload["head_source_sha"],
            "11",
            node="auxme-imech172",
            cpu="Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz",
            node_class="a30",
        )
        reason = "head and baseline cpu_model differ"
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert len(blockers) == 1 and reason in blockers[0]


def test_unknown_source_commit_leaf_remains_behavioral():
    """An exact transport inventory never strips a new site or arbitrary commit key."""
    before, after = _episode(), _episode()
    for row, source in ((before, "a" * 40), (after, "b" * 40)):
        row["algorithm_metadata"]["obstacle_force_law"] = {
            "sites": {"unknown_site": {"source_commit": source}}
        }
    assert eq.episode_bytes(before) != eq.episode_bytes(after)


def test_unrelated_base_git_attribute_drift_refuses(admitted, tmp_path):
    """The finite receipt LFS exception cannot exempt model filters or arbitrary Git rules."""
    _, _, _, head, _, _ = admitted
    (tmp_path / PATH).write_text(BEFORE)
    (tmp_path / ".gitattributes").write_text(
        "models/*.pt filter=unreviewed diff=unreviewed -text\n"
    )
    subprocess.run(["git", "add", PATH, ".gitattributes"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "unrelated main attribute drift"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    main = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    with pytest.raises(ValueError, match="unreviewed base Git attribute drift"):
        eq.classify_scope([], main, head)


def split_head_source_archive(repo, payload):
    """Two real successful allocations partition the exact head episode matrix."""
    import shutil

    from scripts.ci import make_behaviour_receipt as producer

    original = payload["source_artifacts"]["head"]["11"]
    with tempfile.TemporaryDirectory() as temporary:
        first = Path(temporary) / "11"
        with tarfile.open(repo / original["path"], "r:gz") as archive:
            archive.extractall(first, filter="data")
        second = Path(temporary) / "123"
        shutil.copytree(first, second)
        arm = payload["rows"][0]["arm"]
        references = {}
        for root, keep in ((first, False), (second, True)):
            summaries = [
                json.loads(line) for line in (root / "episodes_main.jsonl").read_text().splitlines()
            ]
            summaries = [row for row in summaries if (row["arm"] == arm) == keep]
            for row in summaries:
                row["scheduler_job_id"] = root.name
            (root / "episodes_main.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in summaries)
            )
            for raw in root.glob("campaigns/**/episodes.jsonl"):
                if (raw.parent.name.removesuffix("__differential_drive") == arm) != keep:
                    shutil.rmtree(raw.parent)
                    continue
                rows = [json.loads(line) for line in raw.read_text().splitlines()]
                for row in rows:
                    row["scheduler_job_id"] = root.name
                raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            metadata = json.loads((root / "execution_main.json").read_text())
            for field in ("expected_slots", "written_rows", "total_episodes"):
                metadata[field] = len(summaries)
            (root / "execution_main.json").write_text(json.dumps(metadata))
            request_path = root / "execution_request.json"
            request = json.loads(request_path.read_text())
            request["job_id"] = root.name
            request_path.write_text(json.dumps(request))
            _refresh_equivalence_binding(root)
            bundle = repo / f"receipts/behaviour/fixture-sources/head/{root.name}-split.tar.gz"
            producer._archive_source_bundle(
                root,
                bundle,
                extra_files=(root / "acquisition_harness.py", root / "execution_request.json"),
            )
            references[root.name] = {
                "path": bundle.relative_to(repo).as_posix(),
                "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "binding": {
                    "path": "execution_binding.json",
                    "sha256": hashlib.sha256(
                        (root / "execution_binding.json").read_bytes()
                    ).hexdigest(),
                },
            }
        payload["source_artifacts"]["head"] = references
        for row in payload["rows"]:
            job = "123" if row["arm"] == arm else "11"
            row["scheduler_job_id"] = job
            row["head_path"] = job + "/" + row["head_path"]


def change_first_raw_reward(root):
    """A fresh fixture execution emits one different reward byte."""
    path = next((root / "campaigns").rglob("episodes.jsonl"))
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["rl"]["reward"] = 0.1
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n")


def test_existing_producer_emits_pending_review_receipt_and_byte_refusal(
    admitted, tmp_path, monkeypatch, capsys
):
    """Real file loading/metadata/hardware/byte comparison reaches an unpromoted header."""
    from scripts.ci import make_behaviour_receipt as producer

    payload, _, scope, head, base, _ = admitted
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    scope = {**scope, "arm_algorithms": dict.fromkeys(scope["arms"], "goal")}
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope))
    baseline_dir, head_dir = tmp_path / "12", tmp_path / "11"
    for output, source, job in (
        (baseline_dir, base, "12"),
        (head_dir, payload["head_source_sha"], "11"),
    ):
        _write_equivalence_sweep(output, source, job, base, payload["tooling_file_sha256"], scope)
    arguments = [
        "--receipt-route",
        "default-path-equivalence",
        "--head-sha",
        head,
        "--head-source-sha",
        payload["head_source_sha"],
        "--base-ref",
        base,
        "--tooling-sha",
        base,
        "--receipt-id",
        "cli-equivalence",
        "--seeds",
        "1001",
        "1002",
        "1003",
        "--scope-path",
        str(scope_path),
        "--head-sweep-dir",
        str(head_dir),
        "--baseline-sweep-dir",
        str(baseline_dir),
        "--job-id",
        "11",
        "--baseline-job-id",
        "12",
    ]
    assert producer.main([*arguments, "--comparison-only"]) == 0
    private_output = capsys.readouterr().out
    assert "<!--" not in private_output
    assert json.loads(private_output)["result"] == "equal"
    assert not (tmp_path / "receipts/behaviour/cli-equivalence.json").exists()
    assert producer.main(arguments) == 0
    output = capsys.readouterr().out
    header = json.loads(output.split("\n", 1)[1].rsplit("-->", 1)[0])
    assert header["refute_review"]["verdict"] == "pending_independent_review"
    comparison = json.loads((tmp_path / header["rows_artifact"]["path"]).read_text())
    assert comparison["totals"] == {"episodes": 2142, "changed_episodes": 0}
    assert (
        "refute_review" not in comparison
    )  # final review cannot be self-bound in the committed payload
    wrapper = head_dir / "acquisition_harness.py"
    wrapper_bytes = wrapper.read_bytes()
    wrapper.write_bytes(wrapper_bytes + b"# changed byte\n")
    with pytest.raises(ValueError, match="wrapper bytes differ from metadata"):
        producer.main([*arguments, "--comparison-only"])
    wrapper.write_bytes(wrapper_bytes)
    raw_path = next((head_dir / "campaigns").rglob("episodes.jsonl"))
    rows = raw_path.read_text().splitlines()
    row = json.loads(rows[0])
    row["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["rl"]["reward"] = 0.1
    rows[0] = json.dumps(row)
    raw_path.write_text("\n".join(rows) + "\n")
    _refresh_equivalence_binding(head_dir)
    assert producer.main([*arguments, "--comparison-only"]) == 1
    private_refusal = capsys.readouterr().out
    assert "<!--" not in private_refusal and '"result": "refused"' in private_refusal
    with pytest.raises(ValueError, match="changed behavioral byte"):
        producer.main(arguments)
    diagnostic = json.loads(
        (tmp_path / "output/behaviour_receipt/cli-equivalence.comparison.json").read_text()
    )
    assert diagnostic["result"] == "refused"
    assert diagnostic["changed_episodes"] == 1
    assert len(diagnostic["rows"]) == 2142
    with pytest.raises(ValueError, match="releases require full"):
        producer.main([*arguments, "--receipt-context", "release"])


def test_raw_episode_source_is_parsed_once_and_duplicate_identity_still_refuses(
    tmp_path, monkeypatch
):
    """Shared raw files are read once, while duplicate episode identities stay blockers."""
    from scripts.ci import make_behaviour_receipt as producer

    output = tmp_path / "11"
    scope = {"arms": ["goal"], "maps": ["open"]}
    _write_equivalence_sweep(output, "a" * 40, "11", "b" * 40, "c" * 64, scope)
    (output / "execution_binding.json").unlink()
    raw = next((output / "campaigns").rglob("episodes.jsonl"))
    original_open = Path.open
    reads = []

    def counted_open(path, *args, **kwargs):
        if path == raw:
            reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    assert set(producer._load_sweep_rows(output)) == {
        ("goal", "open", 1001),
        ("goal", "open", 1002),
        ("goal", "open", 1003),
    }
    assert reads == [raw]
    duplicate = raw.read_text().splitlines()[0]
    with original_open(raw, "a") as stream:
        stream.write(duplicate + "\n")
    with pytest.raises(ValueError, match="raw episode identity differs"):
        producer._load_sweep_rows(output)


@pytest.mark.parametrize("policy_change", ["modify", "delete"])
@pytest.mark.parametrize(
    "policy_path",
    [
        "scripts/ci/default_path_equivalence.py",
        "scripts/ci/behaviour_receipt_nodes.json",
        "scripts/ci/receipt_provenance.py",
        "scripts/ci/receipt_outcomes.py",
        ".gitattributes",
        "scripts/dev/merge_queue_gate.py",
        "scripts/validation/profile_receipt_loading.py",
        ".github/workflows/merge-queue-gate.yml",
        ".github/workflows/scripts-catalog.yml",
        "docs/dev/ci_job_value_audit_2026_10_12/observations.csv",
        "docs/dev/ci_job_value_audit_2026_10_12/summary.json",
    ],
)
def test_base_only_gate_promotion_drift_is_distinct_from_pr_policy_edit(
    admitted, tmp_path, policy_change, policy_path
):
    """Promoted base policy does not classify stale runtime as a policy-changing PR."""
    _, _, _, head, _base, _ = admitted
    module = tmp_path / PATH
    module.write_text(BEFORE)
    policy = tmp_path / policy_path
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(
        "receipts/behaviour/*-sources/**/*.tar.gz filter=lfs diff=lfs merge=lfs -text\n"
        if policy_path == ".gitattributes"
        else "# independently promoted base gate\n"
    )
    subprocess.run(["git", "add", PATH, str(policy)], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "promote gate on main"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    main = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    assert eq.classify_scope([], main, head) == [PATH + ":_episode_identity"]
    # An actual PR branch editing policy is not merely missing the promoted base version.
    module.write_text(AFTER)
    if policy_change == "modify":
        policy.write_text("# unreviewed policy edit introduced by PR\n")
    else:
        policy.unlink()
    subprocess.run(["git", "add", PATH, str(policy)], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "PR edits policy"], cwd=tmp_path, check=True, capture_output=True
    )
    altered = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    with pytest.raises(ValueError, match="unreviewed executable path " + policy_path):
        eq.classify_scope([], main, altered)


def test_identical_wrong_adapter_mode_cannot_admit_native_controller():
    """Equality cannot substitute for the trusted algorithm's command-space contract."""
    row = _episode()
    scope = {"arms": ["goal"], "maps": ["open"]}
    rows = {("goal", "open", seed): row for seed in (1001, 1002, 1003)}
    assert len(eq.compare_rows(rows, rows, scope)) == 3
    row["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = "adapter"
    with pytest.raises(ValueError, match="wrong arm/command execution"):
        eq.compare_rows(rows, rows, scope)


@pytest.mark.parametrize("side", ["head", "baseline"])
@pytest.mark.parametrize(
    "fault", ["wrong_mode", "wrong_algorithm", "fallback", "degraded", "unexecuted"]
)
def test_compact_receipt_retains_independent_execution_admission(admitted, side, fault):
    """Equal digest pairs cannot excuse wrong command-space or execution fault claims."""
    payload, header, scope, head, base, _ = admitted
    receipt = {**copy.deepcopy(payload), "head_sha": head, "refute_review": header["refute_review"]}
    execution = receipt["rows"][0][f"{side}_execution"]
    if fault == "wrong_mode":
        execution["execution_mode"] = "adapter"
    elif fault == "wrong_algorithm":
        execution["algorithm"] = "not_the_reviewed_arm"
    elif fault == "unexecuted":
        execution["controller_executed"] = False
    else:
        execution[fault] = True
    reason = (
        "wrong arm/command execution"
        if fault in {"wrong_mode", "wrong_algorithm"}
        else "fallback/degraded/unexecuted"
    )
    with pytest.raises(ValueError, match=reason):
        eq.validate_receipt(receipt, scope, head, base, [PATH])


def test_actual_gate_refuses_old_event_and_receipt_when_remote_main_moves(
    admitted, tmp_path, monkeypatch
):
    """A valid receipt/event pair cannot use their old base as current-main proof."""
    _, _, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    protected = tmp_path / "robot_sf/planner/main_only_default.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("DEFAULT_SPEED = 2.0\n")
    subprocess.run(["git", "add", str(protected)], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "remote main default change"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    remote_main = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    monkeypatch.setattr(gate, "current_main_source", lambda _repo: remote_main)
    assert gate.check_receipt(body, [PATH], "test/repo", base) == [
        "BLOCKER: equivalence freshness refused: event base and receipt baseline must match current remote main"
    ]


def test_remote_main_helper_uses_live_api_and_in_process_token(monkeypatch):
    """Local tracking refs and gh credential-file fallback never substitute for the API."""
    monkeypatch.delenv("GH_TOKEN", raising=False)
    calls = []

    def github_output(command, **kwargs):
        calls.append(command)
        if command == ["gh", "auth", "token"]:
            return "test-fixture-token\n"
        assert command == ["gh", "api", "repos/test/repo/git/ref/heads/main"]
        assert kwargs["env"]["GH_TOKEN"] == "test-fixture-token"
        return json.dumps({"ref": "refs/heads/main", "object": {"type": "commit", "sha": "a" * 40}})

    monkeypatch.setattr(gate.subprocess, "check_output", github_output)
    assert gate.current_main_source("test/repo") == "a" * 40
    assert calls == [["gh", "auth", "token"], ["gh", "api", "repos/test/repo/git/ref/heads/main"]]


@pytest.fixture
def exact_source_sweeps(admitted, tmp_path, monkeypatch):
    from scripts.ci import make_behaviour_receipt as producer

    payload, _, scope, head, base, _ = admitted
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    scope = {**scope, "arm_algorithms": dict.fromkeys(scope["arms"], "goal")}
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope))
    baseline_dir, head_dir = tmp_path / "12", tmp_path / "11"
    for output, source, job in (
        (baseline_dir, base, "12"),
        (head_dir, payload["head_source_sha"], "11"),
    ):
        _write_equivalence_sweep(output, source, job, base, payload["tooling_file_sha256"], scope)
    args = [
        "--receipt-route",
        "default-path-equivalence",
        "--comparison-only",
        "--head-sha",
        head,
        "--head-source-sha",
        payload["head_source_sha"],
        "--base-ref",
        base,
        "--tooling-sha",
        base,
        "--receipt-id",
        "exact-source-control",
        "--seeds",
        "1001",
        "1002",
        "1003",
        "--scope-path",
        str(scope_path),
        "--head-sweep-dir",
        str(head_dir),
        "--baseline-sweep-dir",
        str(baseline_dir),
        "--job-id",
        "11",
        "--baseline-job-id",
        "12",
    ]
    return args, baseline_dir, head_dir


@pytest.mark.parametrize("status_mutation", ["added_written", "changed_byte"])
def test_actual_producer_refuses_unmasked_raw_execution_status_bytes(
    exact_source_sweeps, tmp_path, status_mutation, capsys
):
    """Summary 'written' cannot hide an added or changed raw episode status field."""
    from scripts.ci import make_behaviour_receipt as producer

    args, baseline_dir, head_dir = exact_source_sweeps
    raw = next((head_dir / "campaigns").rglob("episodes.jsonl"))
    baseline_raw = baseline_dir / raw.relative_to(head_dir)
    for path in (raw, baseline_raw):
        lines = path.read_text().splitlines()
        row = json.loads(lines[0])
        row.pop("execution_status")
        if status_mutation == "changed_byte":
            row["execution_status"] = "written"
        lines[0] = json.dumps(row)
        path.write_text("\n".join(lines) + "\n")
        _refresh_equivalence_binding(path.parents[4])
    assert producer.main(args) == 0
    capsys.readouterr()
    lines = raw.read_text().splitlines()
    row = json.loads(lines[0])
    row["execution_status"] = "written" if status_mutation == "added_written" else "writteX"
    lines[0] = json.dumps(row)
    raw.write_text("\n".join(lines) + "\n")
    _refresh_equivalence_binding(head_dir, validate_identity=False)
    if status_mutation == "changed_byte":
        with pytest.raises(
            gate.HardwareIdentityError, match="raw execution status contradicts summary"
        ):
            producer.main(args)
        return
    assert producer.main(args) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["result"] == "refused" and result["changed_episodes"] == 1
    diagnostic = json.loads(
        (tmp_path / "output/behaviour_receipt/exact-source-control.comparison.json").read_text()
    )
    assert diagnostic["changed_episodes"] == 1 and len(diagnostic["rows"]) == 2142


def test_actual_producer_refuses_stale_baseline_even_in_private_mode(
    exact_source_sweeps, monkeypatch, capsys
):
    """Private measurement freezes live main, not an arbitrary recorded/local baseline."""
    from scripts.ci import make_behaviour_receipt as producer

    args, _, _ = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    monkeypatch.setattr(gate, "current_main_source", lambda _repo: "d" * 40)
    with pytest.raises(
        ValueError, match="freshness refused: recorded baseline must match current remote main"
    ):
        producer.main(args)


def test_release_event_cannot_call_the_pr_only_gate(tmp_path, capsys, monkeypatch):
    """The actual CLI rejects release payloads without a caller-supplied context flag."""
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    event = tmp_path / "release-event.json"
    event.write_text(
        json.dumps(
            {"release": {"tag_name": "release-candidate"}, "repository": {"full_name": "test/repo"}}
        )
    )
    assert (
        gate.main(
            ["--github-event-path", str(event), "--repo-root", str(tmp_path), "--base-ref", "HEAD"]
        )
        == 1
    )
    assert "PR-only equivalence cannot admit a release/non-PR event" in capsys.readouterr().out


def test_explicit_new_route_cannot_skip_classifier_outside_legacy_triggers(admitted):
    """An explicit equivalence claim must classify every path, even outside old triggers."""
    _, _, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    assert gate.check_receipt(body, ["third_party/new_planner.py"], "test/repo", base) == [
        "BLOCKER: scope uncertain: unreviewed executable path third_party/new_planner.py"
    ]
    assert (
        gate.check_receipt("Ordinary documentation change", ["docs/example.md"], "test/repo", base)
        == []
    )


@pytest.mark.parametrize("side", ["head", "baseline"])
@pytest.mark.parametrize("field", ["scheduler_job_id", "scheduler_node"])
def test_compact_pair_requires_each_authoritative_allocation(admitted, side, field):
    """Equal hashes cannot admit an episode attributed to another job or node."""
    payload, header, scope, head, base, _ = admitted
    receipt = {**copy.deepcopy(payload), "head_sha": head, "refute_review": header["refute_review"]}
    prefix = "" if side == "head" else "baseline_"
    receipt["rows"][0][prefix + field] = "123" if field.endswith("id") else "auxme-imech172"
    reason = "row job differs" if field.endswith("id") else "row node differs"
    with pytest.raises(gate.HardwareIdentityError, match=reason):
        eq.validate_receipt(receipt, scope, head, base, [PATH])


@pytest.mark.parametrize("row_source", ["raw", "summary"])
@pytest.mark.parametrize("field", ["scheduler_job_id", "scheduler_node"])
def test_actual_producer_validates_allocation_before_byte_projection(
    exact_source_sweeps, tmp_path, capsys, row_source, field
):
    """A valid paired control passes; altered source allocation never produces an outcome."""
    from scripts.ci import make_behaviour_receipt as producer

    args, _, head_dir = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    diagnostic = tmp_path / "output/behaviour_receipt/exact-source-control.comparison.json"
    diagnostic.unlink()
    path = (
        next((head_dir / "campaigns").rglob("episodes.jsonl"))
        if row_source == "raw"
        else head_dir / "episodes_main.jsonl"
    )
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row[field] = "123" if field.endswith("id") else "auxme-imech172"
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n")
    _refresh_equivalence_binding(head_dir, validate_identity=False)
    reason = (
        "source row job differs|producer execution context disagrees"
        if field.endswith("id")
        else "source row node differs|producer execution context disagrees"
    )
    with pytest.raises(gate.HardwareIdentityError, match=reason):
        producer.main(args)
    assert not diagnostic.exists()


def test_actual_producer_refuses_unsuccessful_authoritative_job(
    exact_source_sweeps, tmp_path, capsys, scheduler_endpoint
):
    """Matching self-written claims cannot authorize a scheduler-failed allocation."""
    from scripts.ci import make_behaviour_receipt as producer

    args, _, _ = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    diagnostic = tmp_path / "output/behaviour_receipt/exact-source-control.comparison.json"
    diagnostic.unlink()
    records = json.loads(scheduler_endpoint.read_text())
    records["11"]["state"] = "FAILED"
    scheduler_endpoint.write_text(json.dumps(records))
    with pytest.raises(gate.HardwareIdentityError, match="did not complete successfully"):
        producer.main(args)
    assert not diagnostic.exists()


@pytest.mark.parametrize("multi_job", [False, True])
@pytest.mark.parametrize("wrong_path", [False, True])
def test_actual_admission_binds_raw_archive_path_to_verified_job(
    admitted, tmp_path, monkeypatch, multi_job, wrong_path
):
    """Hash-bound committed raw references cannot point into another job directory."""
    payload, header, _scope, _head, base, _body = admitted
    payload, header = copy.deepcopy(payload), copy.deepcopy(header)
    if multi_job:
        payload["hardware"]["head"] = _hardware(payload["head_source_sha"], "11,123")
        payload["scheduler"]["job_id"] = "11_123_12"
        split_head_source_archive(tmp_path, payload)
    if wrong_path:
        payload["rows"][0]["head_path"] = "999/" + payload["rows"][0]["head_path"].removeprefix(
            payload["rows"][0]["scheduler_job_id"] + "/"
        )
    receipt_path = tmp_path / header["rows_artifact"]["path"]
    receipt_path.write_text(json.dumps(payload))
    subprocess.run(["git", "add", "receipts"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "paired raw custody control"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    header["head_sha"] = head
    header["refute_review"]["head_sha"] = head
    header["rows_artifact"]["sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    header["refute_review"]["comparison_sha256"] = eq.digest(payload["rows"])
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    body = "<!-- default-path-equivalence:v1\n" + json.dumps(header) + "\n-->"
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    if wrong_path:
        assert (
            len(blockers) == 1 and "behavioral/execution source projection differs" in blockers[0]
        )
    else:
        assert blockers == []


def test_multi_job_raw_archive_requires_explicit_row_job_prefix(admitted):
    """The accepted direct single-job path becomes ambiguous for a multi-job source."""
    payload, header, scope, head, base, _ = admitted
    receipt = {**copy.deepcopy(payload), "head_sha": head, "refute_review": header["refute_review"]}
    receipt["hardware"]["head"] = _hardware(payload["head_source_sha"], "11,123")
    receipt["scheduler"]["job_id"] = "11_123_12"
    with pytest.raises(ValueError, match="archive path differs from head row job"):
        eq.validate_receipt(receipt, scope, head, base, [PATH])


@pytest.mark.parametrize(
    "body",
    [
        '<!-- default-path-equivalence:v1 {"head_sha":"ignored"} -->',
        "<!-- default-path-equivalence:v1\n{}",
        "<!-- default-path-equivalence:v1\n{}--><!-- default-path-equivalence:v1\n{}-->",
        "<!-- default-path-equivalence:v1\n{}--><!-- behaviour-change-receipt:v2\n{}-->",
    ],
)
def test_explicit_malformed_marker_cannot_skip_gate_outside_legacy_paths(admitted, body):
    """No-newline and malformed comments refuse before the neutral-path shortcut."""
    _, _, _, _, base, _ = admitted
    blockers = gate.check_receipt(body, ["third_party/unknown.py"], "test/repo", base)
    assert (
        len(blockers) == 1
        and "malformed, duplicate or mixed equivalence receipt markers" in blockers[0]
    )


def test_pr_gate_rechecks_current_main_after_complete_admission(admitted, monkeypatch):
    """Current main must remain the receipt baseline through the full PR validation."""
    _, _, _, _, base, body = admitted
    reads = iter([base, "d" * 40])
    monkeypatch.setattr(gate, "current_main_source", lambda _repo: next(reads))
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert blockers == [
        "BLOCKER: equivalence freshness refused: current remote main moved during admission"
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "binding_source",
        "binding_role",
        "request_source",
        "request_role",
        "request_job",
        "request_node",
        "request_schema",
        "request_extra",
    ],
)
def test_admission_binds_archived_producer_source_and_role(admitted, tmp_path, monkeypatch, fault):
    """Actual accepted source custody cannot be relabelled by refreshing unsigned seals."""
    payload, header, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    payload, header = copy.deepcopy(payload), copy.deepcopy(header)
    mutate_archived_identity(tmp_path, payload, fault, base)
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    reason = (
        "producer source or arm role differs"
        if fault.startswith("binding_")
        else "archived request source/role differs"
        if fault in {"request_source", "request_role"}
        else "archived request differs from verified allocation/hardware"
    )
    assert len(blockers) == 1 and reason in blockers[0]


def mutate_archived_identity(repo, payload, fault, baseline):
    """Build an adversarial source archive without bypassing any admission checks."""

    def mutate(root):
        name = (
            "execution_binding.json" if fault.startswith("binding_") else "execution_request.json"
        )
        path = root / name
        record = json.loads(path.read_text())
        changes = {
            "source": {"source_sha": baseline},
            "role": {"arm_role": "baseline"},
            "job": {"job_id": "999"},
            "node": {"requested_node": "auxme-imech172"},
            "schema": {"schema_version": "untrusted-request"},
            "extra": {"unbound_option": True},
        }
        record.update(changes[fault.rsplit("_", 1)[1]])
        path.write_text(json.dumps(record))

    rewrite_source_archive(
        repo, payload, "head", lambda _root: None, suffix="identity", post_binding_mutator=mutate
    )


@pytest.mark.parametrize("fault", ["binding_source", "binding_role", "same_source"])
def test_private_comparison_requires_disjoint_source_and_correct_role(
    exact_source_sweeps, capsys, fault
):
    """Private success also requires the source/role admission that commits would require."""
    from scripts.ci import make_behaviour_receipt as producer

    args, baseline, head = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    source = json.loads((baseline / "execution_main.json").read_text())["head_sha"]
    if fault == "same_source":
        args = args.copy()
        args[args.index("--head-sha") + 1] = source
        args[args.index("--head-source-sha") + 1] = source
        reason = "share a producer commit"
    else:
        path = head / "execution_binding.json"
        binding = json.loads(path.read_text())
        binding["source_sha" if fault.endswith("source") else "arm_role"] = (
            source if fault.endswith("source") else "baseline"
        )
        path.write_text(json.dumps(binding))
        reason = "producer source or arm role differs"
    with pytest.raises(gate.HardwareIdentityError, match=reason):
        producer.main(args)


NONFINITE_LITERALS = ("NaN", "Infinity", "-Infinity", "1e999")


def nonfinite_source_json(root, literal, nested):
    """Keep the literal bytes, including overflow syntax, in an excluded raw leaf."""
    path = next(root.glob("campaigns/**/episodes.jsonl"))
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    target = row.setdefault("timing", {}) if nested else row
    target["steps_per_second" if nested else "wall_time_sec"] = "__numeric_fixture__"
    lines[0] = json.dumps(row).replace('"__numeric_fixture__"', literal)
    path.write_text("\n".join(lines) + "\n")


def test_strict_source_json_preserves_finite_numbers_and_constant_strings():
    result = eq.strict_json_loads('{"a":1.25,"b":1e308,"c":"NaN","d":"Infinity","e":"-Infinity"}')
    assert result == {"a": 1.25, "b": 1e308, "c": "NaN", "d": "Infinity", "e": "-Infinity"}


@pytest.mark.parametrize("literal", NONFINITE_LITERALS)
@pytest.mark.parametrize("nested", [False, True])
def test_private_comparison_refuses_nonfinite_source_json(
    exact_source_sweeps, capsys, literal, nested
):
    from scripts.ci import make_behaviour_receipt as producer

    args, _, head = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    nonfinite_source_json(head, literal, nested)
    _refresh_equivalence_binding(head, validate_identity=False)
    with pytest.raises(gate.HardwareIdentityError, match="non-finite number refused"):
        producer.main(args)


@pytest.mark.parametrize("literal", NONFINITE_LITERALS)
@pytest.mark.parametrize("nested", [False, True])
def test_actual_pr_gate_refuses_nonfinite_source_json(
    admitted, tmp_path, monkeypatch, literal, nested
):
    payload, header, _, _, base, body = admitted
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    payload, header = copy.deepcopy(payload), copy.deepcopy(header)
    rewrite_source_archive(
        tmp_path,
        payload,
        "head",
        lambda root: nonfinite_source_json(root, literal, nested),
        suffix="nonfinite",
    )
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert len(blockers) == 1 and "non-finite number refused" in blockers[0]


@pytest.mark.parametrize("fault", ["request_job", "request_node", "request_schema"])
def test_private_comparison_binds_request_allocation(exact_source_sweeps, capsys, fault):
    from scripts.ci import make_behaviour_receipt as producer

    args, _, head = exact_source_sweeps
    assert producer.main(args) == 0
    capsys.readouterr()
    path = head / "execution_request.json"
    request = json.loads(path.read_text())
    request.update(
        {
            "request_job": {"job_id": "123"},
            "request_node": {"requested_node": "auxme-imech172"},
            "request_schema": {"schema_version": "untrusted-request"},
        }[fault]
    )
    path.write_text(json.dumps(request))
    reason = {
        "request_job": "stored scheduler evidence differs from authoritative records",
        "request_node": "requested and observed nodes differ",
        "request_schema": "invalid hardware request schema",
    }[fault]
    with pytest.raises(gate.HardwareIdentityError, match=reason):
        producer.main(args)


LEGACY_MUTATIONS = ("null", "string", "positive_infinity", "negative_infinity", "finite", "missing")


def legacy_metrics(root, mutation=None):
    """Mirror all four actual producer undefined/no-actor diagnostic metric leaves."""
    for raw in root.glob("campaigns/**/episodes.jsonl"):
        lines = raw.read_text().splitlines()
        rows = [json.loads(line) for line in lines]
        for row in rows:
            row.setdefault("algorithm_metadata", {}).setdefault("tracking_precision", {})[
                "min_separation_corrupted_m"
            ] = float("inf")
            row.setdefault("metrics", {})["min_separation_corrupted_m"] = float("inf")
            row.setdefault("metric_values", {})["min_predicted_separation_m"] = float("nan")
            row["metrics"].setdefault("metric_values", {})["min_predicted_separation_m"] = float(
                "nan"
            )
        if mutation is not None:
            field = rows[0]["metrics"]["metric_values"]
            if mutation == "missing":
                field.pop("min_predicted_separation_m")
            else:
                field["min_predicted_separation_m"] = {
                    "null": None,
                    "string": "NaN",
                    "positive_infinity": float("inf"),
                    "negative_infinity": -float("inf"),
                    "finite": 0.0,
                }[mutation]
            mutation = None
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))


def prepare_legacy_receipt(repo, payload):
    for side in ("head", "baseline"):
        rewrite_source_archive(
            repo, payload, side, legacy_metrics, synchronize=True, suffix="legacy"
        )


@pytest.mark.parametrize("mutation", LEGACY_MUTATIONS)
def test_private_comparison_retains_legacy_behavior_tokens(exact_source_sweeps, capsys, mutation):
    from scripts.ci import make_behaviour_receipt as producer

    args, base, head = exact_source_sweeps
    for root in (base, head):
        legacy_metrics(root)
        _refresh_equivalence_binding(root)
    assert producer.main(args) == 0
    capsys.readouterr()
    legacy_metrics(head, mutation)
    _refresh_equivalence_binding(head)
    assert producer.main(args) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["changed_episodes"] == 1 and result["result"] == "refused"


@pytest.mark.parametrize("mutation", LEGACY_MUTATIONS)
def test_actual_pr_gate_retains_legacy_behavior_tokens(admitted, tmp_path, monkeypatch, mutation):
    payload, header, _, _, base, _ = admitted
    payload, header = copy.deepcopy(payload), copy.deepcopy(header)
    prepare_legacy_receipt(tmp_path, payload)
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    assert gate.check_receipt(body, [PATH], "test/repo", base) == []
    rewrite_source_archive(
        tmp_path,
        payload,
        "head",
        lambda root: legacy_metrics(root, mutation),
        synchronize=True,
        suffix="legacy-changed",
    )
    body = committed_control_body(tmp_path, payload, header, monkeypatch)
    blockers = gate.check_receipt(body, [PATH], "test/repo", base)
    assert len(blockers) == 1 and "changed behavioral byte" in blockers[0]


def test_raw_legacy_tokens_are_distinct_and_general_receipt_numbers_remain_finite():
    tokens = [
        eq.raw_token_bytes({"value": value})
        for value in (float("nan"), float("inf"), -float("inf"), None, "NaN", 0.0)
    ]
    assert len(set(tokens)) == 6
    with pytest.raises(ValueError, match="Out of range float"):
        eq.canonical_bytes({"value": float("nan")})
