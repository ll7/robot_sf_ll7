"""Sealed producer bytes survive scheduler binding and are mandatory at admission."""

import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.manifest import save_manifest
from robot_sf.benchmark.result_provenance import (
    build_result_provenance_manifest,
    manifest_path_for_result_jsonl,
    write_result_provenance_manifest,
)
from scripts.ci import behaviour_receipt as gate
from scripts.ci import make_behaviour_receipt as producer
from scripts.ci import receipt_provenance
from tests.validation.receipt_scheduler_fixture import install_scheduler
from tests.validation.test_behaviour_receipt_sbatch_affinity import _collection, _unchanged_outcomes


@pytest.fixture(autouse=True)
def _scheduler_authority(tmp_path, monkeypatch):
    install_scheduler(tmp_path, monkeypatch)


def native_orca_identity(row):
    """Fixture shape copied from the normal native31624 ORCA producer's explicit fields."""
    metadata = row["algorithm_metadata"]
    metadata.update(algorithm="orca", canonical_algorithm="orca", policy_semantics="orca_adapter")
    metadata.setdefault("planner_contract", {})["planner_id"] = "orca"
    metadata["planner_kinematics"]["adapter_name"] = "ORCAPlannerAdapter"
    metadata.setdefault("planner_runtime", {})["planner_type"] = "ORCAPlannerAdapter"
    metadata.setdefault("paired_effect_metric_producer", {}).setdefault("pairing", {}).setdefault(
        "key", {}
    )["planner"] = "orca"
    row.setdefault("event_ledger", {})["planner"] = "orca"
    row.setdefault("scenario_params", {})["algo"] = "orca"


def seal_sweep(root, *, algo_configs=None, arm_algorithms=None):
    """Seal synthetic completed rows using the normal producer's two sidecars."""
    for raw in root.glob("campaigns/**/episodes.jsonl"):
        arm = raw.parent.name.split("__")[0]
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        schema = root / "fixture_schema.json"
        schema.write_text('{"type": "object"}\n')
        scenario = root / "fixture_scenario.yaml"
        scenario.write_text("- name: open\n")
        records = [
            {**row, "episode_id": f"fixture-{row['seed']}", "config_hash": "fixture"}
            for row in rows
        ]
        request = json.loads((root / "execution_request.json").read_text())
        source = request["source_sha"]
        from robot_sf._execution_context import execution_context_digest

        context = {
            "schema_version": "benchmark_execution_context.v1",
            "cpu_model": "AMD EPYC 9115 16-Core Processor",
            "platform": "fixture",
            "python_version": "fixture",
            "thread_env": {},
        }
        context = {
            "hostname": "auxme-imech142",
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
            manifest = build_result_provenance_manifest(
                out_path=raw,
                episode_records=records,
                schema_path=schema,
                scenario_path=scenario,
                scenarios=[{"name": "open"}],
                algo=(arm_algorithms or {}).get(arm, rows[0]["algo"]),
                algo_config_path=(algo_configs or {}).get(arm),
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
        save_manifest(raw, [rec["episode_id"] for rec in records])


def refresh_sweep_binding(root, *, validate_identity=True, algo_configs=None, arm_algorithms=None):
    """Reproduce a fresh fixture execution after a deliberate fixture edit."""
    binding_path = root / receipt_provenance.BINDING_FILE
    previous = json.loads(binding_path.read_text()) if binding_path.exists() else None
    binding_path.unlink(missing_ok=True)
    seal_sweep(root, algo_configs=algo_configs, arm_algorithms=arm_algorithms)
    if validate_identity:
        receipt_provenance.write_binding(root, root.name, "auxme-imech142")
    else:
        # The negative controls emulate an attacker refreshing unsigned hashes,
        # without going through the producer's identity admission.
        assert previous is not None
        previous["files"] = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in receipt_provenance.source_files(root)
        }
        binding_path.write_text(json.dumps(previous))


def _fixture_execution_trace(raw, row, algo_configs):
    if row["arm"] in (algo_configs or {}):
        raw["algorithm_metadata"]["algorithm"] = raw["algo"]
        if raw["algo"] == "orca":
            native_orca_identity(raw)
    if row["execution_mode"] == "adapter":
        for step in raw["algorithm_metadata"]["simulation_step_trace"]["steps"]:
            step["planner"]["applied_environment_action"] = dict(step["planner"]["selected_action"])


def source_artifacts_for_receipt(repo, receipt, *, algo_configs=None, arm_algorithms=None):
    """Provide actual sealed sources for hand-built admission fixtures."""
    from tests.validation.test_make_behaviour_receipt import _raw_row

    result = {}
    for side, prefix in (("head", ""), ("baseline", "baseline_")):
        refs = {}
        jobs = {row[prefix + "scheduler_job_id"] for row in receipt["rows"]}
        for job in jobs:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / job
                root.mkdir()
                from tests.validation.test_behaviour_receipt_gate import _hardware

                source = json.loads(receipt["hardware"][side]["execution_request.json"]["content"])[
                    "source_sha"
                ]
                for name, entry in _hardware(source, job).items():
                    (root / name).write_text(entry["content"])
                request_path = root / "execution_request.json"
                request = json.loads(request_path.read_text())
                request["arm_role"] = side
                request_path.write_text(json.dumps(request))
                by_arm = {}
                summaries = []
                for row in receipt["rows"]:
                    if row[prefix + "scheduler_job_id"] != job:
                        continue
                    raw = _raw_row(row["seed"])
                    raw.update(
                        algo=row["algorithm"],
                        scenario_id=row["map"],
                        scheduler_job_id=job,
                        git_hash=source,
                    )
                    raw["metrics"].update(
                        success=row[prefix + "success"],
                        collisions=row[prefix + "collisions"],
                        total_collision_count=row[prefix + "collisions"],
                    )
                    raw["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = row[
                        "execution_mode"
                    ]
                    _fixture_execution_trace(raw, row, algo_configs)
                    by_arm.setdefault(row["arm"], []).append(raw)
                    summaries.append(
                        {
                            "arm": row["arm"],
                            "scenario": row["map"],
                            "seed": row["seed"],
                            "success": row[prefix + "success"],
                            "collisions": row[prefix + "collisions"],
                            "steps": raw["steps"],
                            "execution_status": "written",
                            "trace_complete": True,
                            "source_file": f"runs/{row['arm']}__differential_drive/episodes.jsonl",
                        }
                    )
                for arm, rows in by_arm.items():
                    path = (
                        root
                        / f"campaigns/empty_world_main/runs/{arm}__differential_drive/episodes.jsonl"
                    )
                    path.parent.mkdir(parents=True)
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                (root / "episodes_main.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in summaries)
                )
                (root / "execution_main.json").write_text(
                    json.dumps(
                        {
                            "suite": "main",
                            "head_sha": source,
                            "seeds": sorted({row["seed"] for row in summaries}),
                            "complete": True,
                            "trace_requested": True,
                            "campaign_execution_status": "completed",
                            "exit_code": 0,
                            "unexpected_failed_runs": 0,
                            "expected_slots": len(summaries),
                            "written_rows": len(summaries),
                            "total_episodes": len(summaries),
                            **{
                                field: []
                                for field in (
                                    "failed_slots",
                                    "missing_slots",
                                    "duplicate_slots",
                                    "unexpected_slots",
                                    "incomplete_trace_slots",
                                )
                            },
                        }
                    )
                )
                with patch.object(gate, "ROOT", repo):
                    refresh_sweep_binding(
                        root, algo_configs=algo_configs, arm_algorithms=arm_algorithms
                    )
                bundle = repo / f"receipts/behaviour/fixture-sources/{side}/{job}.tar.gz"
                bundle.parent.mkdir(parents=True, exist_ok=True)
                with tarfile.open(bundle, "w:gz") as archive:
                    for path in [
                        *receipt_provenance.source_files(root),
                        root / receipt_provenance.BINDING_FILE,
                    ]:
                        archive.add(
                            path, arcname=path.relative_to(root).as_posix(), recursive=False
                        )
                refs[job] = {
                    "path": bundle.relative_to(repo).as_posix(),
                    "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                    "binding": {
                        "path": receipt_provenance.BINDING_FILE,
                        "sha256": sha256_file(root / receipt_provenance.BINDING_FILE),
                    },
                }
        result[side] = refs
    return result


def _assert_sealed(root):
    for raw in root.glob("campaigns/**/episodes.jsonl"):
        sidecar = json.loads(manifest_path_for_result_jsonl(raw).read_text())
        assert sidecar["raw_artifacts"][0]["sha256"] == sha256_file(raw)
        manifest = json.loads(raw.with_suffix(".jsonl.manifest.json").read_text())
        assert manifest["simulation_run_provenance"]["outputs"][0]["sha256"] == sha256_file(raw)
    receipt_provenance.verify_producer_seals(root)


def test_successful_sweep_preserves_sealed_bytes_through_collection(tmp_path, monkeypatch):
    """A complete sweep can be bound, collected and admitted without invalidating sidecars."""
    git, _, head, baseline, command = _collection(tmp_path, monkeypatch)
    _unchanged_outcomes(head)
    originals = {}
    for root in (head, baseline):
        for path in (*root.glob("episodes_*.jsonl"), *root.glob("campaigns/**/episodes.jsonl")):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row.pop("scheduler_job_id", None)
                row.pop("scheduler_node", None)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            originals[path] = path.read_bytes()
        seal_sweep(root)
        (root / receipt_provenance.BINDING_FILE).unlink(missing_ok=True)
        for path in root.glob("campaigns/**/episodes.jsonl.*.json"):
            originals[path] = path.read_bytes()
        monkeypatch.setenv("SLURM_JOB_ID", root.name)
        monkeypatch.setenv("SLURM_JOB_NODELIST", "auxme-imech142")
        assert producer._stamp_scheduler_rows(root) == 0
        _assert_sealed(root)
    assert all(path.read_bytes() == content for path, content in originals.items())
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    header = json.loads(result.stdout.split("\n", 1)[1].rsplit("-->", 1)[0])
    assert header["totals"] == {"episodes": 30, "new_failures": 0, "new_collisions": 0}
    assert _admit_collected(tmp_path, monkeypatch, git, result.stdout) == []


def _admit_collected(tmp_path, monkeypatch, git, stdout):
    header = json.loads(stdout.split("\n", 1)[1].rsplit("-->", 1)[0])
    git("add", "receipts/behaviour")
    git("commit", "-m", "sealed receipt evidence")
    head = git("rev-parse", "HEAD")
    header["head_sha"] = header["refute_review"]["head_sha"] = head
    header["refute_review"]["verdict"] = "accepted"
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    body = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
    return gate.check_receipt(body, ["robot_sf/planner/synthetic.py"], "ll7/robot_sf_ll7")


@pytest.mark.parametrize("boundary", ["collection", "admission"])
def test_tampered_raw_is_refused_before_outcomes(tmp_path, monkeypatch, boundary):
    """Valid scheduler records cannot admit raw bytes whose producer digest is stale."""
    git, _, head, baseline, command = _collection(tmp_path, monkeypatch)
    _unchanged_outcomes(head)
    for root in (head, baseline):
        refresh_sweep_binding(root)
    raw = next(head.glob("campaigns/**/episodes.jsonl"))
    # Whitespace changes leave parsed outcomes and row identity unchanged.
    raw.write_bytes(raw.read_bytes() + b"\n")
    binding_path = head / receipt_provenance.BINDING_FILE
    binding = json.loads(binding_path.read_text())
    binding["files"][raw.relative_to(head).as_posix()] = sha256_file(raw)
    binding_path.write_text(json.dumps(binding))
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    if boundary == "collection":
        assert result.returncode != 0, "collection accepted a stale producer digest"
        assert "provenance digest mismatch in sealed artifacts" in result.stderr.lower(), (
            result.stderr
        )
        assert result.stdout == ""
        assert not (tmp_path / "receipts/behaviour/synthetic.json").exists()
    else:
        # Pre-fix collection emits the unsafe receipt; the admission boundary
        # must independently reject it even after its envelope is rehashed.
        if result.returncode:
            raw.write_bytes(raw.read_bytes().removesuffix(b"\n"))
            refresh_sweep_binding(head)
            result = subprocess.run(
                command, cwd=tmp_path, capture_output=True, text=True, check=True
            )
            payload_path = tmp_path / "receipts/behaviour/synthetic.json"
            payload = json.loads(payload_path.read_text())
            # The fixed transport stores exact source bytes in separate bundles.
            import io
            import tarfile

            reference = payload["source_artifacts"]["head"]["123"]
            bundle = tmp_path / reference["path"]
            with tarfile.open(bundle, "r:gz") as archive:
                members = [(m, archive.extractfile(m).read()) for m in archive.getmembers()]
            changed_hashes = {
                member.name: hashlib.sha256(data + b"\n").hexdigest()
                for member, data in members
                if member.name.endswith("/episodes.jsonl")
            }
            with tarfile.open(bundle, "w:gz") as archive:
                for member, data in members:
                    if member.name.endswith("/episodes.jsonl"):
                        data += b"\n"
                    elif member.name == receipt_provenance.BINDING_FILE:
                        binding = json.loads(data)
                        binding["files"].update(changed_hashes)
                        data = json.dumps(binding).encode()
                        reference["binding"]["sha256"] = hashlib.sha256(data).hexdigest()
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            reference["sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
            payload_path.write_text(json.dumps(payload))
            header = json.loads(result.stdout.split("\n", 1)[1].rsplit("-->", 1)[0])
            header["rows_artifact"]["sha256"] = hashlib.sha256(
                payload_path.read_bytes()
            ).hexdigest()
            result.stdout = "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
        blockers = _admit_collected(tmp_path, monkeypatch, git, result.stdout)
        assert blockers, "admission accepted a stale producer digest"
        assert any(
            "provenance digest mismatch in sealed artifacts" in item.lower() for item in blockers
        ), blockers
