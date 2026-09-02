"""Additional branch coverage for release-doctor admission checks."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.benchmark import release_doctor


def _result(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    """Build a deterministic subprocess result fixture."""
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _private_ops_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], str]:
    """Create a packet and object-addressed private ledger fixture."""
    repository = tmp_path / "private-ops"
    (repository / "ops/jobs").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "doctor@example.invalid"], cwd=repository, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "release-doctor-test"], cwd=repository, check=True
    )
    source_sha = "a" * 40
    result_sha = "b" * 64
    preservation_sha = "c" * 64
    queue_id = "runtime-queue-14884"
    campaign = "runtime-campaign-14884"
    result_path = "output/runtime/release_result.json"
    queue = {
        "queue_id": queue_id,
        "campaign": campaign,
        "state": "complete",
        "expected_public_commit": source_sha,
        "preservation_state": "preserved",
        "preservation_artifact": "wandb://example/runtime:v1",
        "preservation_digest": f"sha256:{preservation_sha}",
        "submit_args": (f"--sbatch-arg --export=ALL,SMOKE_RELEASE_RESULT_PATH={result_path}"),
        "execution_status": "passed",
        "artifact_status": "verified",
        "evaluation_status": "canary_passed",
        "completion_status": "complete",
    }
    job = {
        "job_id": "14884",
        "public_commit": source_sha,
        "campaign": campaign,
        "state": "retrieved",
        "slurm_state": "COMPLETED",
        "exit_code": "0:0",
        "derived_exit_code": "0:0",
        "startup_status": "started",
        "execution_status": "passed",
        "artifact_status": "verified",
        "evaluation_status": "canary_passed",
        "completion_status": "complete",
        "evaluation_receipt_digest": f"sha256:{result_sha}",
        "submitted_at": "2026-08-25T08:00:00+00:00",
    }
    (repository / "ops/jobs/jobs.yaml").write_text(yaml.safe_dump([job]), encoding="utf-8")
    (repository / "ops/jobs/queue.yaml").write_text(yaml.safe_dump([queue]), encoding="utf-8")
    subprocess.run(
        ["git", "add", "ops/jobs/jobs.yaml", "ops/jobs/queue.yaml"], cwd=repository, check=True
    )
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    packet = {
        "execution_contract": {"private_ops_reviewed_base_commit": commit},
        "identity": {
            "public_source_commit": source_sha,
            "runtime_smoke_queue_id": queue_id,
            "runtime_smoke_campaign_id": campaign,
            "runtime_smoke_receipt_path": result_path,
            "runtime_smoke_receipt_sha256": result_sha,
        },
        "accepted_runtime_smoke": {
            "status": "accepted_preserved_verified",
            "job_id": "14884",
            "queue_id": queue_id,
            "campaign_id": campaign,
            "public_source_commit": source_sha,
            "release_result_path": result_path,
            "release_result_sha256": result_sha,
            "preservation_artifact": "wandb://example/runtime:v1",
            "preservation_manifest_digest": f"sha256:{preservation_sha}",
            "fallback_or_degraded_rows": 0,
        },
    }
    return repository, packet, commit


@pytest.mark.parametrize(
    ("head", "status", "expected", "match"),
    [
        ("a" * 40, "", "a" * 40, "pass"),
        ("b" * 40, "", "a" * 40, "HEAD differs"),
        ("a" * 40, " M file.py", "a" * 40, "dirty"),
    ],
)
def test_git_check_requires_exact_clean_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    head: str,
    status: str,
    expected: str,
    match: str,
) -> None:
    """Doctor Git admission reports both source drift and dirty state."""
    calls = iter(
        [
            _result(["git", "rev-parse", "HEAD"], stdout=head),
            _result(["git", "status", "--porcelain"], stdout=status),
        ]
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: next(calls))
    check = release_doctor._git_check(tmp_path, expected)
    assert (check.status == "pass") is (match == "pass")
    if match != "pass":
        assert match in check.summary


def test_git_check_fails_when_git_commands_are_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A diagnostic command failure is not silently treated as clean state."""
    monkeypatch.setattr(
        release_doctor,
        "_run",
        lambda *args: _result([], returncode=1),
    )
    check = release_doctor._git_check(tmp_path, "a" * 40)
    assert check.status == "fail"
    assert "could not be inspected" in check.summary


def test_private_ops_evidence_reads_packet_pinned_git_blobs_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Working-tree edits cannot replace exact job/queue blobs used for freshness."""
    repository, packet, commit = _private_ops_fixture(tmp_path)
    monkeypatch.setattr(
        release_doctor,
        "_utc_now",
        lambda: datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )
    queue_path = repository / "ops/jobs/queue.yaml"
    queue_path.write_text("- queue_id: tampered\n", encoding="utf-8")
    evidence, problems = release_doctor._private_ops_evidence(packet, "a" * 40, repository)
    assert not problems
    assert evidence is not None
    assert evidence.reviewed_commit == commit
    assert evidence.runtime_smoke_job["job_id"] == "14884"
    assert evidence.runtime_smoke_queue["state"] == "complete"


@pytest.mark.parametrize(
    ("field", "value", "summary"),
    [
        ("runtime_smoke_queue_id", "other-queue", "queue identity"),
        ("runtime_smoke_campaign_id", "other-campaign", "campaign identity"),
    ],
)
def test_private_ops_evidence_rejects_packet_identity_alias_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: str,
    summary: str,
) -> None:
    """Packet identity aliases must match accepted smoke and pinned ledger rows."""
    repository, packet, _ = _private_ops_fixture(tmp_path)
    monkeypatch.setattr(
        release_doctor,
        "_utc_now",
        lambda: datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )
    packet["identity"][field] = value

    evidence, problems = release_doctor._private_ops_evidence(packet, "a" * 40, repository)

    assert evidence is None
    assert any(summary in problem for problem in problems)


@pytest.mark.parametrize(
    "mutation",
    [
        "stale",
        "future",
        "result_digest",
        "job_source",
        "queue_state",
        "duplicate_job",
    ],
)
def test_private_ops_evidence_rejects_stale_or_inconsistent_terminal_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    """Freshness and every packet/ledger identity binding fail closed."""
    repository, packet, _ = _private_ops_fixture(tmp_path)
    monkeypatch.setattr(
        release_doctor,
        "_utc_now",
        lambda: datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )
    if mutation in {"stale", "future"}:
        submitted_at = (
            "2026-08-24T11:59:59+00:00" if mutation == "stale" else "2026-08-25T12:00:01+00:00"
        )
        jobs = yaml.safe_load((repository / "ops/jobs/jobs.yaml").read_text(encoding="utf-8"))
        jobs[0]["submitted_at"] = submitted_at
        (repository / "ops/jobs/jobs.yaml").write_text(yaml.safe_dump(jobs), encoding="utf-8")
    elif mutation == "result_digest":
        jobs = yaml.safe_load((repository / "ops/jobs/jobs.yaml").read_text(encoding="utf-8"))
        jobs[0]["evaluation_receipt_digest"] = "sha256:" + "d" * 64
        (repository / "ops/jobs/jobs.yaml").write_text(yaml.safe_dump(jobs), encoding="utf-8")
    elif mutation == "job_source":
        jobs = yaml.safe_load((repository / "ops/jobs/jobs.yaml").read_text(encoding="utf-8"))
        jobs[0]["public_commit"] = "e" * 40
        (repository / "ops/jobs/jobs.yaml").write_text(yaml.safe_dump(jobs), encoding="utf-8")
    elif mutation == "queue_state":
        queues = yaml.safe_load((repository / "ops/jobs/queue.yaml").read_text(encoding="utf-8"))
        queues[0]["state"] = "failed"
        (repository / "ops/jobs/queue.yaml").write_text(yaml.safe_dump(queues), encoding="utf-8")
    elif mutation == "duplicate_job":
        jobs = yaml.safe_load((repository / "ops/jobs/jobs.yaml").read_text(encoding="utf-8"))
        jobs.append(dict(jobs[0]))
        (repository / "ops/jobs/jobs.yaml").write_text(yaml.safe_dump(jobs), encoding="utf-8")
    subprocess.run(
        ["git", "add", "ops/jobs/jobs.yaml", "ops/jobs/queue.yaml"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", f"mutation-{mutation}"], cwd=repository, check=True)
    packet["execution_contract"]["private_ops_reviewed_base_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    evidence, problems = release_doctor._private_ops_evidence(packet, "a" * 40, repository)
    assert evidence is None
    assert problems


def test_legacy_packet_aliases_normalize_only_after_strict_equality(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Legacy inputs/startup fields are synthesized only from agreeing aliases."""
    repository, packet, _ = _private_ops_fixture(tmp_path)
    monkeypatch.setattr(
        release_doctor,
        "_utc_now",
        lambda: datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )
    evidence, problems = release_doctor._private_ops_evidence(packet, "a" * 40, repository)
    assert evidence is not None, problems
    wrapper_sha = "f" * 64
    helper_sha = "e" * 64
    sentinel_sha = "d" * 64
    packet["execution_contract"].update(
        {
            "canonical_entrypoint": "/private/ops/submit_and_record.sh",
            "private_script": "/private/ops/submit_release.sh",
            "private_ops_reviewed_base_commit": evidence.reviewed_commit,
        }
    )
    packet["identity"].update(
        {
            "private_wrapper_sha256": wrapper_sha,
            "admission_helper_sha256": helper_sha,
            "startup_sentinel_sha256": sentinel_sha,
        }
    )
    queue_exports = {
        "RELEASE_RUNTIME_SMOKE_RECEIPT_PATH": ["output/runtime/release_result.json"],
        "RELEASE_RUNTIME_SMOKE_RECEIPT_SHA256": ["b" * 64],
        "RELEASE_WRAPPER_SHA256": [wrapper_sha],
        "RELEASE_STARTUP_HELPER_SHA256": [helper_sha],
        "RELEASE_STARTUP_SENTINEL_SHA256": [sentinel_sha],
        "RELEASE_EXPECTED_PRIVATE_OPS_COMMIT": [evidence.reviewed_commit],
    }
    normalized, problems = release_doctor._normalize_legacy_packet_aliases(
        packet,
        expected_sha="a" * 40,
        private_evidence=evidence,
        packet_queue_exports=queue_exports,
    )
    assert not problems
    assert normalized["inputs"]["source"]["public_commit"] == "a" * 40
    assert normalized["inputs"]["runtime_smoke_receipt"]["sha256"] == "b" * 64
    assert normalized["inputs"]["private_wrapper"]["sha256"] == wrapper_sha
    assert normalized["execution_contract"]["startup_sentinel_required"] is True
    assert normalized["sentinel_traceability"]["required"] is True

    queue_exports["RELEASE_WRAPPER_SHA256"] = ["0" * 64]
    _, problems = release_doctor._normalize_legacy_packet_aliases(
        packet,
        expected_sha="a" * 40,
        private_evidence=evidence,
        packet_queue_exports=queue_exports,
    )
    assert any("private wrapper hash" in problem for problem in problems)


@pytest.mark.parametrize(
    ("result", "expected_status", "summary"),
    [
        (_result([], returncode=1), "fail", "unavailable"),
        (_result([], stdout="not-json"), "fail", "missing"),
        (
            _result(
                [],
                stdout=json.dumps(
                    [
                        {
                            "headSha": "b" * 40,
                            "status": "completed",
                            "conclusion": "success",
                            "workflowName": "CI",
                        },
                        {
                            "headSha": "b" * 40,
                            "status": "completed",
                            "conclusion": "success",
                            "workflowName": "CodeQL",
                        },
                    ]
                ),
            ),
            "fail",
            "missing",
        ),
        (
            _result(
                [],
                stdout=json.dumps(
                    [
                        {
                            "headSha": "a" * 40,
                            "status": "completed",
                            "conclusion": "success",
                            "workflowName": "CI",
                        },
                        {
                            "headSha": "a" * 40,
                            "status": "completed",
                            "conclusion": "success",
                            "workflowName": "CodeQL",
                        },
                    ]
                ),
            ),
            "pass",
            "green",
        ),
    ],
)
def test_ci_check_requires_completed_success_for_exact_sha(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result: subprocess.CompletedProcess[str],
    expected_status: str,
    summary: str,
) -> None:
    """Only a completed successful CI run for the requested SHA admits release."""
    monkeypatch.setattr(release_doctor, "_run", lambda *args: result)
    check = release_doctor._ci_check(tmp_path, "a" * 40)
    assert check.status == expected_status
    assert summary in check.summary


@pytest.mark.parametrize(
    ("local_code", "remote_ref_code", "release_code", "expected"),
    [(0, 2, 1, "fail"), (1, 0, 1, "fail"), (1, 2, 0, "fail"), (1, 2, 1, "fail")],
)
def test_tag_check_rejects_local_or_remote_collisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    local_code: int,
    remote_ref_code: int,
    release_code: int,
    expected: str,
) -> None:
    """A planned release tag must be unused in both Git and GitHub."""
    calls = iter(
        [
            _result(["git"], returncode=local_code),
            _result(["git", "ls-remote"], returncode=remote_ref_code),
            _result(["gh"], returncode=release_code),
        ]
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: next(calls))
    check = release_doctor._tag_check(tmp_path, "tag")
    assert check.status == expected


def test_tag_check_accepts_explicit_github_release_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only an explicit GitHub not-found response means no release collision."""
    calls = iter(
        [
            _result(["git"], returncode=1),
            _result(["git", "ls-remote"], returncode=2),
            _result(["gh"], returncode=1, stderr="release not found"),
        ]
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: next(calls))
    check = release_doctor._tag_check(tmp_path, "tag")
    assert check.status == "pass"


@pytest.mark.parametrize(
    ("command_index", "returncode", "stderr", "summary"),
    [
        (0, 1, "fatal: not a git repository", "local tag state is unavailable"),
        (1, 2, "fatal: could not read from remote", "remote tag state is unavailable"),
        (2, 1, "authentication failed", "GitHub release state is unavailable"),
        (2, 1, "HTTP 404: Not Found", "GitHub release state is unavailable"),
    ],
)
def test_tag_check_fails_closed_on_ambiguous_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command_index: int,
    returncode: int,
    stderr: str,
    summary: str,
) -> None:
    """An error that resembles absence must not be treated as an unused tag."""
    results = [
        _result(["git"], returncode=1),
        _result(["git", "ls-remote"], returncode=2),
        _result(["gh"], returncode=1, stderr="release not found"),
    ]
    results[command_index] = _result(
        results[command_index].args, returncode=returncode, stderr=stderr
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: results.pop(0))
    check = release_doctor._tag_check(tmp_path, "tag")
    assert check.status == "fail"
    assert summary in check.summary


def test_manifest_check_reports_bad_path_and_wrong_cell_count(tmp_path: Path) -> None:
    """Manifest diagnostics fail safely for malformed paths and cardinality drift."""
    missing, manifest, cfg = release_doctor._manifest_check(tmp_path / "missing.yaml", 1)
    assert missing.status == "fail"
    assert manifest is None
    assert cfg is None

    check, manifest, cfg = release_doctor._manifest_check(
        Path(
            "configs/benchmarks/releases/"
            "paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml"
        ),
        1,
    )
    assert check.status == "fail"
    assert "20160" in check.summary
    assert manifest is not None
    assert cfg is not None


def test_checkpoint_check_reports_missing_and_validator_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Checkpoint admission remains blocked until a valid receipt is present."""
    missing = release_doctor._checkpoint_check(None, None, None)
    assert missing.status == "fail"
    assert "missing" in missing.summary

    manifest = SimpleNamespace(canonical_campaign_config_path=tmp_path / "campaign.yaml")
    monkeypatch.setattr(
        release_doctor,
        "validate_checkpoint_staging_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            release_doctor.CheckpointStagingReceiptError("receipt mismatch")
        ),
    )
    rejected = release_doctor._checkpoint_check(object(), manifest, tmp_path / "receipt.json")
    assert rejected.status == "fail"
    assert rejected.summary == "receipt mismatch"

    monkeypatch.setattr(
        release_doctor,
        "validate_checkpoint_staging_receipt",
        lambda *args, **kwargs: {"arms": [{"planner_key": "ppo"}]},
    )
    admitted = release_doctor._checkpoint_check(object(), manifest, tmp_path / "receipt.json")
    assert admitted.status == "pass"
    assert "1 checkpoint" in admitted.summary


def test_checkpoint_check_forwards_remap_and_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Doctor passes the explicit map and containment root to receipt validation."""
    captured: dict[str, object] = {}

    def fake_validate(*args, **kwargs):
        captured.update(kwargs)
        return {"arms": [{"planner_key": "ppo"}]}

    monkeypatch.setattr(release_doctor, "validate_checkpoint_staging_receipt", fake_validate)
    manifest = SimpleNamespace(canonical_campaign_config_path=tmp_path / "campaign.yaml")
    mapping = ["/hpc/source.zip=checkpoints/model.zip"]
    check = release_doctor._checkpoint_check(
        object(),
        manifest,
        tmp_path / "receipt.json",
        repo_root=tmp_path,
        checkpoint_path_map=mapping,
    )
    assert check.status == "pass"
    assert captured["checkpoint_path_map"] == mapping
    assert captured["repo_root"] == tmp_path


def test_release_identity_check_reports_each_mismatch() -> None:
    """The final manifest must match schema, latest-main base, and tag."""
    rejected = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.1",
            latest_main_base_commit="b" * 40,
            release_tag="other",
        ),
        "a" * 40,
        "expected",
    )
    assert rejected.status == "fail"
    assert "v0.2" in rejected.summary
    assert "base commit" in rejected.summary
    assert "release tag" in rejected.summary

    admitted = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.2",
            latest_main_base_commit="a" * 40,
            release_tag="expected",
        ),
        "a" * 40,
        "expected",
    )
    assert admitted.status == "pass"


def test_release_identity_check_rejects_planning_sha_when_final_source_differs() -> None:
    """Doctor compares a SHA-bearing tag with final source, not planning/base SHA."""
    source_sha = "b" * 40
    planning_sha = "a" * 40
    tag = f"paper-matrix-future-{planning_sha}"
    check = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.2",
            latest_main_base_commit=planning_sha,
            release_tag=tag,
            release_kind="benchmark-data",
        ),
        planning_sha,
        tag,
        source_sha,
    )

    assert check.status == "fail"
    assert "disagrees with" in check.summary


def test_release_identity_check_rejects_invalid_source_identity_fields() -> None:
    """Final source identity fields must be exact Git SHAs and agree."""
    invalid_expected = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.2",
            latest_main_base_commit="a" * 40,
            release_tag="paper-matrix-future-2026-09",
        ),
        "a" * 40,
        "paper-matrix-future-2026-09",
        "not-a-sha",
    )
    assert invalid_expected.status == "fail"
    assert "expected final source SHA" in invalid_expected.summary

    invalid_declared = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.2",
            latest_main_base_commit="a" * 40,
            release_tag="paper-matrix-future-2026-09",
            source_sha="not-a-sha",
        ),
        "a" * 40,
        "paper-matrix-future-2026-09",
        "b" * 40,
    )
    assert invalid_declared.status == "fail"
    assert "manifest source_sha is not an exact 40-character Git SHA" in invalid_declared.summary


def test_release_identity_check_rejects_source_sha_mismatch() -> None:
    check = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.2",
            latest_main_base_commit="a" * 40,
            release_tag="paper-matrix-future-2026-09",
            source_sha="b" * 40,
        ),
        "a" * 40,
        "paper-matrix-future-2026-09",
        "c" * 40,
    )

    assert check.status == "fail"
    assert "does not match expected final source SHA" in check.summary


def test_release_identity_check_rejects_mismatched_historical_source() -> None:
    check = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.2",
            latest_main_base_commit="c" * 40,
            release_tag=release_doctor.HISTORICAL_RELEASE_TAG,
            release_kind="benchmark-data",
            source_sha="a" * 40,
        ),
        "c" * 40,
        release_doctor.HISTORICAL_RELEASE_TAG,
        "b" * 40,
    )

    assert check.status == "fail"
    assert "historical release source SHA" in check.summary
    assert "historical manifest source SHA" in check.summary


def test_release_identity_check_requires_source_for_future_benchmark_data() -> None:
    check = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.2",
            latest_main_base_commit="a" * 40,
            release_tag="paper-matrix-future-2026-09",
            release_kind="benchmark-data",
        ),
        "a" * 40,
        "paper-matrix-future-2026-09",
    )

    assert check.status == "fail"
    assert "requires manifest source_sha" in check.summary


def test_release_identity_check_rejects_planning_base_mismatch() -> None:
    check = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.2",
            latest_main_base_commit="a" * 40,
            release_tag="paper-matrix-future-2026-09",
            planning_base_sha="b" * 40,
        ),
        "a" * 40,
        "paper-matrix-future-2026-09",
    )

    assert check.status == "fail"
    assert "planning_base_sha" in check.summary


def test_release_identity_check_accepts_immutable_historical_exception() -> None:
    """The already-published stale-suffix tag remains readable but immutable."""
    check = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.2",
            latest_main_base_commit="c" * 40,
            release_tag=release_doctor.HISTORICAL_RELEASE_TAG,
            release_kind="benchmark-data",
        ),
        "c" * 40,
        release_doctor.HISTORICAL_RELEASE_TAG,
        release_doctor.HISTORICAL_RELEASE_SOURCE_SHA,
    )

    assert check.status == "pass"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_load_mapping_supports_json_and_yaml(tmp_path: Path, suffix: str) -> None:
    """Private launch packets may use either supported serialization."""
    path = tmp_path / f"packet{suffix}"
    if suffix == ".json":
        path.write_text('{"ok": true}\n', encoding="utf-8")
    else:
        path.write_text("ok: true\n", encoding="utf-8")
    assert release_doctor._load_mapping(path) == {"ok": True}


def test_load_mapping_rejects_non_mapping(tmp_path: Path) -> None:
    """Launch packets must deserialize to mappings."""
    path = tmp_path / "packet.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected mapping"):
        release_doctor._load_mapping(path)


@pytest.mark.parametrize(
    ("payload", "expected_status", "summary"),
    [
        ({"admission": {"status": "admitted"}, "dispatchable": True}, "fail", "source SHA"),
        ({"admission": {"status": "pending"}, "dispatchable": True}, "fail", "not admitted"),
    ],
)
def test_cluster_check_rejects_invalid_admission_or_identity(
    tmp_path: Path,
    payload: dict[str, object],
    expected_status: str,
    summary: str,
) -> None:
    """A packet without admission and source binding cannot dispatch."""
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    check = release_doctor._cluster_check(path, "a" * 40)
    assert check.status == expected_status
    assert summary in check.summary


def test_cluster_check_reports_missing_and_invalid_packet(tmp_path: Path) -> None:
    """Missing and malformed private packets fail without exposing contents."""
    missing = release_doctor._cluster_check(tmp_path / "missing.json", "a" * 40)
    assert missing.status == "fail"
    assert "missing" in missing.summary
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not-json", encoding="utf-8")
    invalid = release_doctor._cluster_check(invalid_path, "a" * 40)
    assert invalid.status == "fail"
    assert "invalid" in invalid.summary


@pytest.mark.parametrize("minimum", [0.0, 10_000_000.0])
def test_disk_check_applies_free_space_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, minimum: float
) -> None:
    """Artifact capacity admission compares free bytes with the configured threshold."""
    monkeypatch.setattr(
        release_doctor.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=2 * 1024**3),
    )
    check = release_doctor._disk_check(tmp_path, minimum)
    assert check.status == ("pass" if minimum <= 2.0 else "fail")
    assert "GiB free" in check.summary


class _AuthResponse:
    """Minimal successful auth response."""

    def raise_for_status(self) -> None:
        """Accept the request."""


class _AuthSession:
    """Minimal authenticated Zenodo session."""

    def get(self, *args, **kwargs) -> _AuthResponse:
        """Return a successful response."""
        return _AuthResponse()


@pytest.mark.parametrize(
    ("hook_result", "require_disabled", "expected_status", "summary"),
    [
        (_result([], returncode=1), False, "fail", "unavailable"),
        (_result([], stdout="not-json"), False, "fail", "invalid"),
        (_result([], stdout="[]"), False, "pass", "absent"),
        (_result([], stdout="[]"), True, "pass", "absent"),
        (
            _result([], stdout=json.dumps([{"active": False, "config": {"url": "zenodo"}}])),
            True,
            "pass",
            "disabled",
        ),
    ],
)
def test_zenodo_check_sanitizes_auth_and_hook_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hook_result: subprocess.CompletedProcess[str],
    require_disabled: bool,
    expected_status: str,
    summary: str,
) -> None:
    """Doctor reports hook/auth state without returning private hook configuration."""
    monkeypatch.setattr(release_doctor, "read_token_file", lambda path: "secret")
    monkeypatch.setattr(release_doctor, "build_session", lambda path: _AuthSession())
    monkeypatch.setattr(release_doctor, "_run", lambda *args: hook_result)
    checks = release_doctor._zenodo_check(
        tmp_path,
        tmp_path / "token",
        require_hook_disabled=require_disabled,
    )
    hook_check = checks[-1]
    assert hook_check.status == expected_status
    assert summary in hook_check.summary
    assert "secret" not in json.dumps([check.summary for check in checks])


def test_zenodo_check_reports_auth_failure_without_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unavailable credentials are summarized, never echoed."""
    monkeypatch.setattr(
        release_doctor,
        "read_token_file",
        lambda path: (_ for _ in ()).throw(RuntimeError("private token")),
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: _result([], stdout="[]"))
    checks = release_doctor._zenodo_check(tmp_path, tmp_path / "token", require_hook_disabled=False)
    assert checks[0].status == "fail"
    assert "private token" not in json.dumps([check.summary for check in checks])


def _make_dissertation(path: Path, *, stale: bool = False) -> None:
    """Create the minimum dissertation release path fixture."""
    for relative in (
        "diss/robot_sf_release.tex",
        "docs/context/evidence_pins.yaml",
        "spine/evidence_release.yaml",
    ):
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "/Users/lennart/git/robot_sf_ll7\n" if stale else "healthy\n",
            encoding="utf-8",
        )


def test_dissertation_check_reports_missing_healthy_and_stale_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dissertation health checks cover required files and configurable paths."""
    missing = release_doctor._dissertation_check(None)
    assert missing.status == "fail"

    healthy = tmp_path / "healthy"
    _make_dissertation(healthy)
    monkeypatch.setattr(release_doctor, "_run", lambda *args: _result([], returncode=1))
    assert release_doctor._dissertation_check(healthy).status == "pass"

    stale = tmp_path / "stale"
    _make_dissertation(stale, stale=True)
    monkeypatch.setattr(
        release_doctor,
        "_run",
        lambda *args: _result(["rg"], stdout="./diss/robot_sf_release.tex"),
    )
    rejected = release_doctor._dissertation_check(stale)
    assert rejected.status == "fail"
    assert "hard-coded" in rejected.summary


def test_dissertation_check_allows_repository_urls_and_relative_paths(tmp_path: Path) -> None:
    """The path health check rejects local checkouts, not public URLs or names."""
    healthy = tmp_path / "healthy"
    _make_dissertation(healthy)
    (healthy / "docs" / "links.md").write_text(
        "See https://github.com/ll7/robot_sf_ll7 and robot_sf_ll7/configs.\n",
        encoding="utf-8",
    )
    assert release_doctor._dissertation_check(healthy).status == "pass"

    stale = tmp_path / "stale"
    _make_dissertation(stale)
    (stale / "docs" / "local.md").write_text(
        "\n\t/scratch/luttkule/projects/robot_sf_ll7/configs\n",
        encoding="utf-8",
    )
    assert release_doctor._dissertation_check(stale).status == "fail"


def test_ci_check_rejects_pending_codeql_even_when_ci_is_green(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pending required workflow cannot be hidden by a green aggregate CI run."""
    result = _result(
        [],
        stdout=json.dumps(
            [
                {
                    "databaseId": 1001,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "workflowName": "CI",
                },
                {
                    "databaseId": 1002,
                    "headSha": "a" * 40,
                    "status": "in_progress",
                    "conclusion": "",
                    "workflowName": "CodeQL",
                },
            ]
        ),
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: result)
    check = release_doctor._ci_check(tmp_path, "a" * 40)
    assert check.status == "fail"
    assert "CodeQL" in check.summary


def test_ci_check_accepts_successful_run_despite_later_concurrency_cancellations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A prior complete successful run is not invalidated by later concurrency cancellations."""
    result = _result(
        [],
        stdout=json.dumps(
            [
                {
                    "databaseId": 32807916917,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "workflowName": "CI",
                },
                {
                    "databaseId": 32813075613,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "cancelled",
                    "workflowName": "CI",
                },
                {
                    "databaseId": 32813075614,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "cancelled",
                    "workflowName": "CI",
                },
                {
                    "databaseId": 32807916918,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "workflowName": "CodeQL",
                },
            ]
        ),
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: result)
    check = release_doctor._ci_check(tmp_path, "a" * 40)
    assert check.status == "pass"
    assert "green" in check.summary
    assert "32807916917" in check.summary
    assert "32807916918" in check.summary


def test_ci_check_blocks_nonterminal_cancellation_even_with_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only completed cancellations are tolerated after independent success."""
    expected_sha = "a" * 40
    result = _result(
        [],
        stdout=json.dumps(
            [
                {
                    "databaseId": 60001,
                    "headSha": expected_sha,
                    "status": "queued",
                    "conclusion": "cancelled",
                    "workflowName": "CI",
                },
                {
                    "databaseId": 60002,
                    "headSha": expected_sha,
                    "status": "completed",
                    "conclusion": "success",
                    "workflowName": "CI",
                },
                {
                    "databaseId": 60003,
                    "headSha": expected_sha,
                    "status": "completed",
                    "conclusion": "success",
                    "workflowName": "CodeQL",
                },
            ]
        ),
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: result)

    check = release_doctor._ci_check(tmp_path, expected_sha)

    assert check.status == "fail"
    assert "CI pending" in check.summary
    assert "blocking exact-source run IDs: CI=60001" in check.summary


def test_ci_check_blocks_genuine_failure_alongside_success_for_same_workflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A later genuine failure cannot be hidden by an earlier same-workflow success."""
    expected_sha = "a" * 40
    result = _result(
        [],
        stdout=json.dumps(
            [
                {
                    "databaseId": 70001,
                    "headSha": expected_sha,
                    "status": "completed",
                    "conclusion": "success",
                    "workflowName": "CI",
                },
                {
                    "databaseId": 70002,
                    "headSha": expected_sha,
                    "status": "completed",
                    "conclusion": "failure",
                    "workflowName": "CI",
                },
                {
                    "databaseId": 70003,
                    "headSha": expected_sha,
                    "status": "completed",
                    "conclusion": "success",
                    "workflowName": "CodeQL",
                },
            ]
        ),
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: result)

    check = release_doctor._ci_check(tmp_path, expected_sha)

    assert check.status == "fail"
    assert "CI failed" in check.summary
    assert "supporting exact-source run IDs: CI=70001, CodeQL=70003" in check.summary
    assert "blocking exact-source run IDs: CI=70002" in check.summary


def test_ci_check_fails_closed_when_only_cancelled_runs_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If no successful run exists for a workflow, cancellation fails closed and records run IDs."""
    result = _result(
        [],
        stdout=json.dumps(
            [
                {
                    "databaseId": 32813075613,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "cancelled",
                    "workflowName": "CI",
                },
                {
                    "databaseId": 32807916918,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "workflowName": "CodeQL",
                },
            ]
        ),
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: result)
    check = release_doctor._ci_check(tmp_path, "a" * 40)
    assert check.status == "fail"
    assert "CI cancelled" in check.summary
    assert "32813075613" in check.summary


def test_ci_check_fails_closed_on_genuine_failure_without_successful_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A genuine failure blocks admission and reports the failing run ID."""
    result = _result(
        [],
        stdout=json.dumps(
            [
                {
                    "databaseId": 32800000001,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "failure",
                    "workflowName": "CI",
                },
                {
                    "databaseId": 32807916918,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "workflowName": "CodeQL",
                },
            ]
        ),
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: result)
    check = release_doctor._ci_check(tmp_path, "a" * 40)
    assert check.status == "fail"
    assert "CI failed" in check.summary
    assert "32800000001" in check.summary
