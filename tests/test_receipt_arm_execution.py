"""Only source-bound frozen scenario overrides may change a receipt arm's execution."""

import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from robot_sf._execution_context import execution_context_digest
from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.manifest import save_manifest
from robot_sf.benchmark.result_provenance import (
    build_result_provenance_manifest,
    manifest_path_for_result_jsonl,
    write_result_provenance_manifest,
)
from scripts.ci import behaviour_receipt as gate
from scripts.ci import receipt_provenance
from tests.validation.test_behaviour_receipt_provenance import native_orca_identity
from tests.validation.test_make_behaviour_receipt import _raw_row

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4",
    "scenario_adaptive_hybrid_orca_v2_collision_guard_v4",
)
MAP = "francis2023_leave_group"
NODE = "auxme-imech091"
JOB = "31624"


def profile_path(arm):
    return f"configs/policy_search/candidates/{arm}_s30_h600_release_0_0_8_frozen.yaml"


@pytest.fixture
def source_repo(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Receipt test")
    git("config", "user.email", "receipt@example.invalid")
    for arm in ARMS:
        path = repo / profile_path(arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / profile_path(arm)).read_bytes())
    git("add", ".")
    git("commit", "-m", "unchanged published profiles")
    monkeypatch.setattr(gate, "ROOT", repo)
    return repo, git, git("rev-parse", "HEAD")


@pytest.fixture
def sealed_override(tmp_path, source_repo):
    repo, _git, source = source_repo

    def make(arm):
        root = tmp_path / arm
        raw = root / "campaigns/empty_world_main/runs" / f"{arm}__differential_drive/episodes.jsonl"
        raw.parent.mkdir(parents=True)
        rows = []
        for seed in (1001, 1002, 1003):
            row = _raw_row(seed)
            row.update(
                algo="orca",
                scenario_id=MAP,
                git_hash=source,
                scheduler_job_id=JOB,
                scheduler_node=NODE,
                episode_id=f"{arm}-{seed}",
                config_hash="fixture",
            )
            row["algorithm_metadata"]["algorithm"] = "orca"
            row["algorithm_metadata"]["planner_kinematics"] = enrich_algorithm_metadata(
                algo="orca"
            )["planner_kinematics"]
            step = row["algorithm_metadata"]["simulation_step_trace"]["steps"][0]
            step.update(collision={"obstacle": False}, rl={"reward": 0.0})
            step["planner"]["applied_environment_action"] = {
                "linear_velocity": 1.0,
                "angular_velocity": 0.0,
            }
            native_orca_identity(row)
            rows.append(row)
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        schema = root / "schema.json"
        schema.write_text('{"type":"object"}\n')
        scenario = root / "scenario.yaml"
        scenario.write_text(f"- name: {MAP}\n")
        context = {
            "schema_version": "benchmark_execution_context.v1",
            "cpu_model": "AMD EPYC 9354P 32-Core Processor",
            "platform": "fixture",
            "python_version": "fixture",
            "thread_env": {},
        }
        context = {
            "hostname": NODE,
            **context,
            "execution_context_sha256": execution_context_digest(context),
        }
        with (
            patch("robot_sf.benchmark.result_provenance._git_hash_fallback", return_value=source),
            patch(
                "robot_sf.benchmark.result_provenance.build_execution_context_provenance",
                return_value=context,
            ),
        ):
            sidecar = build_result_provenance_manifest(
                out_path=raw,
                episode_records=rows,
                schema_path=schema,
                scenario_path=scenario,
                scenarios=[{"name": MAP}],
                algo="hybrid_rule_local_planner",
                algo_config_path=repo / profile_path(arm),
                benchmark_profile="baseline-safe",
                suite_key="main",
                total_jobs=3,
                written=3,
                horizon=1,
                dt=0.1,
                record_forces=False,
                active_observation_mode=None,
                active_observation_level=None,
            )
        sidecar["inputs"]["algo_config"]["path"] = "/unmounted-producer/" + profile_path(arm)
        _rewrite_primary(raw, sidecar)
        save_manifest(raw, [row["episode_id"] for row in rows])
        return root, raw, sidecar, source

    return make


@pytest.mark.parametrize("arm", ARMS)
def test_normal_producer_seals_accept_only_declared_leave_group_dispatch(sealed_override, arm):
    """The normal producer's two seals bind each published hybrid's three ORCA rows."""
    root, _raw, _sidecar, source = sealed_override(arm)
    receipt_provenance.verify_producer_seals(
        root, source_sha=source, arm_role="baseline", job=JOB, node=NODE
    )


def _sealed_rows(root, source):
    return receipt_provenance.verify_producer_seals(
        root, source_sha=source, arm_role="baseline", job=JOB, node=NODE
    )


def _rewrite_primary(raw, sidecar):
    from robot_sf.benchmark.result_provenance import _canonical_input_bundle_sha256

    campaign = sidecar["campaign_identity"]
    campaign["input_bundle_sha256"] = _canonical_input_bundle_sha256(
        inputs=sidecar["inputs"],
        algo=campaign["algorithm"],
        protocol_version=sidecar["run"]["protocol_version"],
        suite_key=campaign["suite_key"],
    )
    write_result_provenance_manifest(manifest_path_for_result_jsonl(raw), sidecar)


@pytest.mark.parametrize("attack", ["sha", "profile_mix", "path_suffix", "unavailable"])
def test_resealed_config_claim_cannot_override_frozen_profile(sealed_override, attack):
    """Even a schema-valid, internally rehashed producer input must match its source blob."""
    root, raw, sidecar, source = sealed_override(ARMS[0])
    config = sidecar["inputs"]["algo_config"]
    if attack == "sha":
        config["sha256"] = "0" * 64
    elif attack == "profile_mix":
        config.update(
            path="/unmounted-producer/" + profile_path(ARMS[1]),
            sha256=hashlib.sha256((ROOT / profile_path(ARMS[1])).read_bytes()).hexdigest(),
        )
    elif attack == "path_suffix":
        config["path"] += ".backup"
    else:
        config.update(artifact_status="not_provided", path=None, sha256=None)
    _rewrite_primary(raw, sidecar)
    with pytest.raises(
        gate.HardwareIdentityError, match="receipt arm execution proof invalid: producer profile"
    ):
        _sealed_rows(root, source)


@pytest.mark.parametrize(
    "attack", ["changed_git_blob", "executable_git_blob", "symlink_git_blob", "unknown_source"]
)
def test_source_commit_and_regular_blob_are_required(sealed_override, source_repo, attack):
    """The claimed producer commit must contain the exact unchanged regular profile."""
    repo, git, _source = source_repo
    root, raw, sidecar, _ = sealed_override(ARMS[0])
    path = repo / profile_path(ARMS[0])
    if attack == "changed_git_blob":
        path.write_text(path.read_text().replace("algo: orca", "algo: goal"))
        git("add", ".")
        git("commit", "-m", "changed published profile")
        source = git("rev-parse", "HEAD")
    elif attack == "executable_git_blob":
        path.chmod(0o755)
        git("add", ".")
        git("commit", "-m", "changed profile mode")
        source = git("rev-parse", "HEAD")
    elif attack == "symlink_git_blob":
        path.unlink()
        path.symlink_to(Path(profile_path(ARMS[1])).name)
        git("add", ".")
        git("commit", "-m", "symlink published profile")
        source = git("rev-parse", "HEAD")
    else:
        source = "0" * 40
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    for row in rows:
        row["git_hash"] = source
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sidecar["run"]["repo_commit"] = source
    for link in sidecar["rows"]:
        link["repo_commit"] = source
    for artifact in sidecar["raw_artifacts"]:
        if Path(artifact["path"]).name == raw.name:
            artifact["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    _rewrite_primary(raw, sidecar)
    save_manifest(raw, [row["episode_id"] for row in rows])
    with pytest.raises(
        gate.HardwareIdentityError,
        match="receipt arm execution proof invalid: (published profile Git bytes changed|profile is not the exact regular|source Git blob unavailable)",
    ):
        _sealed_rows(root, source)


@pytest.mark.parametrize("arm", ARMS)
def test_verified_proof_survives_execution_projection_but_not_json(
    sealed_override, source_repo, arm
):
    """Actual serialized evidence needs archive re-attestation before accepting dispatch."""
    from scripts.ci import default_path_equivalence as eq
    from scripts.ci import make_behaviour_receipt as producer
    from scripts.ci import receipt_arm_execution as execution

    repo, _git, _ = source_repo
    root, raw_path, _sidecar, source = sealed_override(arm)
    row = _sealed_rows(root, source)[raw_path.resolve()][0]
    plain = dict(row)
    assert eq.episode_bytes(row) == eq.episode_bytes(plain)
    assert json.dumps(row, sort_keys=True) == json.dumps(plain, sort_keys=True)
    audited = execution.carry_execution_proof(row, {**row, "execution_status": "written"})
    evidence = producer._execution_evidence(audited)
    algorithm = execution.expected_algorithm(
        arm, MAP, "hybrid_rule_local_planner", row=evidence, source_sha=source, repo_root=repo
    )
    assert algorithm == "orca"
    producer.require_valid_execution(
        evidence, algorithm, enrich_algorithm_metadata(algo=algorithm)["planner_kinematics"]
    )
    decoded = json.loads(json.dumps(evidence))
    decoded["_receipt_execution"] = {"source": source, "arm": arm, "scenario": MAP}
    with pytest.raises(gate.HardwareIdentityError, match="JSON claims cannot attest dispatch"):
        execution.expected_algorithm(
            arm, MAP, "hybrid_rule_local_planner", row=decoded, source_sha=source, repo_root=repo
        )
    with pytest.raises(gate.HardwareIdentityError, match="header cannot redefine"):
        execution.expected_algorithm(
            arm, "another_map", "orca", row=decoded, source_sha=source, repo_root=repo
        )


@pytest.mark.parametrize("change", ["map", "arm", "source"])
def test_attestation_cannot_move_to_another_execution(sealed_override, source_repo, change):
    """A genuine sealed allowance is finite to its original arm, map, and source."""
    from scripts.ci import receipt_arm_execution as execution

    repo, _git, _ = source_repo
    root, path, _sidecar, source = sealed_override(ARMS[0])
    row = _sealed_rows(root, source)[path.resolve()][0]
    arm, scenario = ARMS[0], MAP
    if change == "map":
        scenario = "another_map"
        assert (
            execution.expected_algorithm(
                arm,
                scenario,
                "hybrid_rule_local_planner",
                row=row,
                source_sha=source,
                repo_root=repo,
            )
            == "hybrid_rule_local_planner"
        )
    else:
        if change == "arm":
            arm = ARMS[1]
        else:
            source = "0" * 40
        with pytest.raises(gate.HardwareIdentityError, match="attested arm/map/source differs"):
            execution.expected_algorithm(
                arm,
                scenario,
                "hybrid_rule_local_planner",
                row=row,
                source_sha=source,
                repo_root=repo,
            )


@pytest.mark.parametrize(
    "arm, algorithm", [("orca", "orca"), (ARMS[0], "hybrid_rule_local_planner")]
)
def test_default_execution_requires_no_override_attestation(source_repo, arm, algorithm):
    """Normal ORCA and default-map hybrid executions retain their original algorithm."""
    from scripts.ci import receipt_arm_execution as execution

    repo, _git, source = source_repo
    assert (
        execution.expected_algorithm(
            arm, "ordinary_map", algorithm, row={}, source_sha=source, repo_root=repo
        )
        == algorithm
    )


@pytest.mark.parametrize("fault", ["execution", "digest", "path", "allocation"])
def test_compact_projection_must_match_before_proof_restoration(
    sealed_override, source_repo, fault
):
    """Resealed JSON/header claims cannot gain attributes through a changed projection."""
    from scripts.ci import default_path_equivalence as equivalence
    from scripts.ci import make_behaviour_receipt as producer
    from scripts.ci import receipt_arm_execution as execution

    repo, _git, _ = source_repo
    root, path, _sidecar, source = sealed_override(ARMS[0])
    raw = _sealed_rows(root, source)[path.resolve()][0]
    key = (ARMS[0], MAP, raw["seed"])
    audited = execution.carry_execution_proof(raw, {**raw, "execution_status": "written"})
    evidence = producer._execution_evidence(audited)
    digest = hashlib.sha256(equivalence.episode_bytes(raw)).hexdigest()
    row = {
        "arm": key[0],
        "map": key[1],
        "seed": key[2],
        "head_execution": json.loads(json.dumps(evidence)),
        "head_path": "campaigns/raw.jsonl",
        "head_sha256": digest,
        "scheduler_job_id": JOB,
        "scheduler_node": NODE,
    }
    projections = {key: {"execution": evidence, "path": row["head_path"], "sha256": digest}}
    if fault == "execution":
        row["head_execution"]["fallback_used"] = True
    elif fault == "digest":
        row["head_sha256"] = "0" * 64
    elif fault == "path":
        row["head_path"] = "another/file.jsonl"
    else:
        row["scheduler_job_id"] = "99999"
    with pytest.raises(gate.HardwareIdentityError, match="source projection differs"):
        gate._verify_equivalence_projection(
            {"rows": [row]}, "", "head", {key: raw}, projections, {JOB}
        )
    with pytest.raises(gate.HardwareIdentityError, match="JSON claims cannot attest dispatch"):
        execution.expected_algorithm(
            key[0],
            key[1],
            "hybrid_rule_local_planner",
            row=row["head_execution"],
            source_sha=source,
            repo_root=repo,
        )


def test_full_committed_archive_rederives_hybrid_dispatch_before_pr_and_queue_validation(
    tmp_path, monkeypatch
):
    """All 14 slots admit only after committed archives independently restore six allowances."""
    from scripts.ci import default_path_equivalence as equivalence
    from scripts.ci import receipt_arm_execution as execution
    from tests.validation import test_default_path_equivalence as fixture
    from tests.validation.receipt_scheduler_fixture import install_scheduler

    records_path = install_scheduler(tmp_path, monkeypatch)
    records = json.loads(records_path.read_text())
    for job in ("11", "12"):
        records[job] = {**records["123"], "job_id": job}
    records_path.write_text(json.dumps(records))
    for arm in ARMS:
        target = tmp_path / profile_path(arm)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / profile_path(arm)).read_bytes())
    scope = {
        "arms": ["ppo", "guarded_ppo", *ARMS, *[f"arm-{i}" for i in range(10)]],
        "maps": [MAP, *[f"map-{i}" for i in range(50)]],
    }
    scope["arm_algorithms"] = {
        arm: "hybrid_rule_local_planner"
        if arm in ARMS
        else arm
        if arm in {"ppo", "guarded_ppo"}
        else "goal"
        for arm in scope["arms"]
    }

    monkeypatch.setattr(
        fixture,
        "_refresh_equivalence_binding",
        lambda root: _seal_full_dispatch(root, tmp_path, scope),
    )
    payload, header, _scope, head, base, _body = fixture._build_admitted(
        tmp_path, monkeypatch, scope_override=scope
    )
    # The fixture's initial check above exercised the real PR gate, including archives.
    decoded = gate.decode_committed_payload(
        header, gate.read_committed_payload_bytes(header, head), head
    )
    assert len(decoded["rows"]) == 2142
    special = [row for row in decoded["rows"] if row["arm"] in ARMS and row["map"] == MAP]
    assert len(special) == 6
    for row in special:
        for side in ("head", "baseline"):
            assert (
                execution.expected_algorithm(
                    row["arm"],
                    MAP,
                    "hybrid_rule_local_planner",
                    row=row[f"{side}_execution"],
                    source_sha=json.loads(
                        payload["hardware"][side]["execution_request.json"]["content"]
                    )["source_sha"],
                    repo_root=tmp_path,
                )
                == "orca"
            )
    # Queue consumes the same base-owned decoder/validator; no candidate/header attributes.
    equivalence.validate_receipt(decoded, scope, head, base, [fixture.PATH])
    plain = json.loads(json.dumps(decoded))
    with pytest.raises(gate.HardwareIdentityError, match="JSON claims cannot attest dispatch"):
        equivalence.validate_receipt(plain, scope, head, base, [fixture.PATH])

    # A fresh committed archive with both normal seals and every transport hash
    # recomputed must refuse a contradictory executed-controller identity.
    _reseal_controller_identity_archive(tmp_path, payload)
    changed_body = fixture.committed_control_body(tmp_path, payload, header, monkeypatch)
    refused = gate.check_receipt(changed_body, [fixture.PATH], "test/repo", base)
    assert len(refused) == 1 and "explicit ORCA execution identity differs" in refused[0]


def _seal_full_dispatch(root, repo, scope):
    from tests.validation import test_default_path_equivalence as fixture
    from tests.validation.receipt_scheduler_fixture import CPU
    from tests.validation.receipt_scheduler_fixture import NODE as fixture_node

    source = json.loads((root / "execution_main.json").read_text())["head_sha"]
    schema = root / "fixture_schema.json"
    scenario = root / "fixture_scenario.yaml"
    schema.write_text('{"type":"object"}\n')
    scenario.write_text("- name: fixture\n")
    context = {
        "schema_version": "benchmark_execution_context.v1",
        "cpu_model": CPU,
        "platform": "fixture",
        "python_version": "fixture",
        "thread_env": {},
    }
    context = {
        "hostname": fixture_node,
        **context,
        "execution_context_sha256": execution_context_digest(context),
    }
    for raw in root.glob("campaigns/**/episodes.jsonl"):
        arm = raw.parent.name.split("__")[0]
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        for row in rows:
            algo = fixture._fixture_execution_algorithm(scope, arm, row["scenario_id"])
            row["algo"] = algo
            # Guarded PPO's source-emitted controller identity is PPO inside
            # the declared guarded composite; retain the complete fixture.
            if algo != "guarded_ppo":
                row["algorithm_metadata"]["algorithm"] = algo
            if algo == "orca":
                native_orca_identity(row)
            if arm in ARMS:
                row["algorithm_metadata"]["planner_kinematics"] = enrich_algorithm_metadata(
                    algo=algo
                )["planner_kinematics"]
                row["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = (
                    fixture._fixture_execution_mode(scope, arm, row["scenario_id"])
                )
                for step in row["algorithm_metadata"]["simulation_step_trace"]["steps"]:
                    step["planner"]["applied_environment_action"] = {
                        "linear_velocity": 1.0,
                        "angular_velocity": 0.0,
                    }
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with (
            patch("robot_sf.benchmark.result_provenance._git_hash_fallback", return_value=source),
            patch(
                "robot_sf.benchmark.result_provenance.build_execution_context_provenance",
                return_value=context,
            ),
        ):
            manifest = build_result_provenance_manifest(
                out_path=raw,
                episode_records=[
                    {
                        **row,
                        "episode_id": f"{row['scenario_id']}-{row['seed']}",
                        "config_hash": "fixture",
                    }
                    for row in rows
                ],
                schema_path=schema,
                scenario_path=scenario,
                scenarios=[{"name": item} for item in scope["maps"]],
                algo=scope["arm_algorithms"][arm],
                algo_config_path=repo / profile_path(arm) if arm in ARMS else None,
                benchmark_profile="baseline-safe",
                suite_key="main",
                total_jobs=len(rows),
                written=len(rows),
                horizon=1,
                dt=0.1,
                record_forces=False,
                active_observation_mode=None,
                active_observation_level=None,
            )
        write_result_provenance_manifest(manifest_path_for_result_jsonl(raw), manifest)
        save_manifest(raw, [f"{row['scenario_id']}-{row['seed']}" for row in rows])
    receipt_provenance.write_binding(root, root.name, fixture_node)


def _reseal_controller_identity_archive(tmp_path, payload):
    from scripts.ci import make_behaviour_receipt as producer

    reference = payload["source_artifacts"]["head"]["11"]
    with tempfile.TemporaryDirectory() as temporary:
        archive_root = Path(temporary) / "11"
        archive_root.mkdir()
        with tarfile.open(tmp_path / reference["path"], "r:gz") as archive:
            archive.extractall(archive_root, filter="data")
        raw = next(archive_root.glob(f"campaigns/**/{ARMS[0]}__differential_drive/episodes.jsonl"))
        raw_rows = [json.loads(line) for line in raw.read_text().splitlines()]
        special_raw = next(row for row in raw_rows if row["scenario_id"] == MAP)
        special_raw["algorithm_metadata"]["algorithm"] = "hybrid_rule_local_planner"
        raw.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
        sidecar = json.loads(manifest_path_for_result_jsonl(raw).read_text())
        for entry in sidecar["raw_artifacts"]:
            if Path(entry["path"]).name == raw.name:
                entry["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        _rewrite_primary(raw, sidecar)
        save_manifest(raw, [f"{row['scenario_id']}-{row['seed']}" for row in raw_rows])
        binding_path = archive_root / receipt_provenance.BINDING_FILE
        binding = json.loads(binding_path.read_text())
        binding["files"] = {
            path.relative_to(archive_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in receipt_provenance.source_files(archive_root)
        }
        binding_path.write_text(json.dumps(binding))
        destination = (tmp_path / reference["path"]).with_name("contradictory-controller.tar.gz")
        reference["path"] = destination.relative_to(tmp_path).as_posix()
        producer._archive_source_bundle(
            archive_root,
            destination,
            extra_files=(
                archive_root / "acquisition_harness.py",
                archive_root / "execution_request.json",
            ),
        )
        reference["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        reference["binding"]["sha256"] = hashlib.sha256(binding_path.read_bytes()).hexdigest()


IDENTITY_PATHS = (
    ("algorithm_metadata", "algorithm"),
    ("algorithm_metadata", "canonical_algorithm"),
    ("algorithm_metadata", "planner_contract", "planner_id"),
    ("algorithm_metadata", "planner_kinematics", "adapter_name"),
    ("algorithm_metadata", "planner_runtime", "planner_type"),
    ("algorithm_metadata", "policy_semantics"),
    ("algorithm_metadata", "paired_effect_metric_producer", "pairing", "key", "planner"),
    ("event_ledger", "planner"),
    ("scenario_params", "algo"),
    ("algorithm",),
)


@pytest.mark.parametrize("identity_path", IDENTITY_PATHS)
def test_resealed_contradictory_executed_identity_cannot_mint_proof(sealed_override, identity_path):
    """Both normal raw seals can be consistent while explicit controller identity is false."""
    root, raw, sidecar, source = sealed_override(ARMS[0])
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    target = rows[0]
    for component in identity_path[:-1]:
        target = target.setdefault(component, {})
    target[identity_path[-1]] = "hybrid_rule_local_planner"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for entry in sidecar["raw_artifacts"]:
        if Path(entry["path"]).name == raw.name:
            entry["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    _rewrite_primary(raw, sidecar)
    save_manifest(raw, [row["episode_id"] for row in rows])
    with pytest.raises(
        gate.HardwareIdentityError, match="explicit ORCA execution identity differs"
    ):
        _sealed_rows(root, source)


@pytest.mark.parametrize("where", ["raw", "primary", "secondary"])
def test_resealed_duplicate_json_identity_cannot_mint_proof(sealed_override, where):
    """Original sealed bytes must contain one unambiguous identity per JSON object."""
    root, raw, sidecar, source = sealed_override(ARMS[0])
    if where == "raw":
        raw.write_text(
            raw.read_text().replace(
                '"algorithm": "orca"',
                '"algorithm": "hybrid_rule_local_planner", "algorithm": "orca"',
                1,
            )
        )
        for entry in sidecar["raw_artifacts"]:
            if Path(entry["path"]).name == raw.name:
                entry["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        _rewrite_primary(raw, sidecar)
        save_manifest(
            raw, [row["episode_id"] for row in map(json.loads, raw.read_text().splitlines())]
        )
    elif where == "primary":
        path = manifest_path_for_result_jsonl(raw)
        path.write_text(
            path.read_text().replace(
                '"algorithm": "hybrid_rule_local_planner"',
                '"algorithm": "orca", "algorithm": "hybrid_rule_local_planner"',
                1,
            )
        )
    else:
        path = raw.with_suffix(".jsonl.manifest.json")
        text = path.read_text()
        path.write_text('{"schema_version":"forged",' + text.lstrip()[1:])
    with pytest.raises(gate.HardwareIdentityError, match="duplicate JSON key in sealed source"):
        _sealed_rows(root, source)


@pytest.mark.parametrize("identity_path", IDENTITY_PATHS[:-1])
def test_resealed_missing_native_identity_cannot_mint_proof(sealed_override, identity_path):
    """Deleting actual native ORCA identity evidence cannot manufacture an allowance."""
    root, raw, sidecar, source = sealed_override(ARMS[0])
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    target = rows[0]
    for component in identity_path[:-1]:
        target = target[component]
    del target[identity_path[-1]]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for entry in sidecar["raw_artifacts"]:
        if Path(entry["path"]).name == raw.name:
            entry["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    _rewrite_primary(raw, sidecar)
    save_manifest(raw, [row["episode_id"] for row in rows])
    with pytest.raises(
        gate.HardwareIdentityError, match="explicit ORCA execution identity differs"
    ):
        _sealed_rows(root, source)
