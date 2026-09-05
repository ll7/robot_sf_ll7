"""Tests for fail-closed research answerability and yield reporting."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

import robot_sf.benchmark.research_answerability as answerability_module
from robot_sf.benchmark.research_answerability import (
    DECISION_REQUIRED_PROOF_SURFACES,
    PROOF_BINDING_SCHEMA,
    PROOF_SURFACES,
    answerability_from_manifest,
    compute_proof_digest,
    evaluate_answerability,
)
from scripts.analysis.report_research_yield import (
    ResearchYieldError,
    build_research_yield_report,
    load_snapshot,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST = REPO_ROOT / "configs/benchmarks/research_campaign_manifest.example.yaml"
ISSUE_6474_FIXTURE = REPO_ROOT / "tests/fixtures/research_answerability/issue_6474_bounded.json"
YIELD_FIXTURE = REPO_ROOT / "tests/fixtures/research_yield_snapshot.v1.json"


def _example_contract() -> dict[str, object]:
    payload = yaml.safe_load(EXAMPLE_MANIFEST.read_text(encoding="utf-8"))
    return copy.deepcopy(payload["answerability"])


def _proof_contract() -> dict[str, object]:
    contract = json.loads(ISSUE_6474_FIXTURE.read_text(encoding="utf-8"))
    contract["proof_surfaces"] = {
        name: {"status": "passed", "required": True} for name in PROOF_SURFACES
    }
    return contract


def _proof_binding() -> dict[str, str]:
    """Return a deterministic synthetic identity for strict evaluator tests."""
    return {
        "schema_version": PROOF_BINDING_SCHEMA,
        "campaign_id": "issue_6474_fixture",
        "question": "Which bounded result is being checked?",
        "estimand": "The bounded fixture estimand",
        "source_manifest": "tests/fixtures/research_answerability/issue_6474_bounded.json",
        "campaign_config": "configs/benchmarks/issue_3425_empirical_vertical_slice_smoke.yaml",
        "manifest_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "proof_digest": "c" * 64,
    }


def _strict_bound_contract() -> dict[str, object]:
    """Build a strict contract with a runner-shaped, internally consistent binding."""
    contract = _proof_contract()
    binding = _proof_binding()
    source_manifest = ISSUE_6474_FIXTURE.relative_to(REPO_ROOT).as_posix()
    campaign_config = "configs/benchmarks/issue_3425_empirical_vertical_slice_smoke.yaml"
    binding.update(
        {
            "question": contract["question"]["research_question"],
            "estimand": contract["estimand"]["primary"],
            "source_manifest": source_manifest,
            "campaign_config": campaign_config,
            "manifest_sha256": hashlib.sha256(ISSUE_6474_FIXTURE.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((REPO_ROOT / campaign_config).read_bytes()).hexdigest(),
            "head_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "manifest_blob": subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{source_manifest}"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_blob": subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{campaign_config}"], cwd=REPO_ROOT, text=True
            ).strip(),
        }
    )
    proof_results = {name: {"status": "passed", "required": True} for name in PROOF_SURFACES}
    binding["proof_results"] = proof_results
    binding["proof_digest"] = compute_proof_digest(binding, proof_results)
    contract["proof_binding"] = binding
    return contract


def _tracked_strict_bound_contract() -> dict[str, object]:
    """Build a strict binding whose inputs are committed non-fixture files."""
    contract = _proof_contract()
    source_manifest = "configs/benchmarks/research_campaign_manifest.example.yaml"
    campaign_config = "configs/benchmarks/issue_3425_empirical_vertical_slice_smoke.yaml"
    source_path = REPO_ROOT / source_manifest
    config_path = REPO_ROOT / campaign_config
    binding = _proof_binding()
    binding.update(
        {
            "question": contract["question"]["research_question"],
            "estimand": contract["estimand"]["primary"],
            "source_manifest": source_manifest,
            "campaign_config": campaign_config,
            "manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "head_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "manifest_blob": subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{source_manifest}"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_blob": subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{campaign_config}"], cwd=REPO_ROOT, text=True
            ).strip(),
        }
    )
    proof_results = {name: {"status": "passed", "required": True} for name in PROOF_SURFACES}
    binding["proof_results"] = proof_results
    binding["proof_digest"] = compute_proof_digest(binding, proof_results)
    contract["proof_binding"] = binding
    return contract


def test_example_contract_is_diagnostic_only() -> None:
    """The canonical example is executable only as a bounded diagnostic packet."""
    result = evaluate_answerability(_example_contract())

    assert result.state == "diagnostic_only"
    assert result.as_dict()["decision_capable"] is False


def test_optional_unavailable_metric_is_preserved_without_blocking() -> None:
    """A bounded fixture may keep optional unavailable metrics explicit."""
    contract = json.loads(ISSUE_6474_FIXTURE.read_text(encoding="utf-8"))

    result = evaluate_answerability(contract)

    assert result.state == "answerable"
    assert result.warnings
    assert "secondary_realism_metric" in result.warnings[0]


def test_missing_proof_surface_is_invalid() -> None:
    """A declared proof set must name every admission surface explicitly."""
    contract = _proof_contract()
    del contract["proof_surfaces"]["result_packet"]

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert "result_packet" in result.reasons[0]


@pytest.mark.parametrize(
    ("update", "reason_fragment"),
    [
        ({"status": "unknown"}, "status"),
        ({"required": "yes"}, "required"),
        ({"status": "unavailable"}, "unavailable_reason"),
        ({"status": "passed", "unavailable_reason": " "}, "unavailable_reason"),
    ],
)
def test_proof_surface_shape_defects_are_invalid(
    update: dict[str, object], reason_fragment: str
) -> None:
    """Malformed proof-surface declarations cannot be treated as admission evidence."""
    contract = _proof_contract()
    contract["proof_surfaces"]["analysis"].update(update)

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert reason_fragment in result.reasons[0]


def test_unsupported_proof_surface_is_invalid() -> None:
    """A proof set cannot smuggle in an unsupported admission surface."""
    contract = _proof_contract()
    contract["proof_surfaces"]["unsupported"] = {"status": "passed", "required": True}

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert "unsupported" in result.reasons[0]


def test_strict_admission_requires_declared_proof_surfaces() -> None:
    """Strict admission requires the canonical proof-surface declarations."""
    contract = _example_contract()
    contract["proof_surfaces"] = None
    contract["design"].update({"mode": "decision_capable", "power_status": "adequate"})
    contract["artifacts"]["durability_status"] = "ready"

    result = evaluate_answerability(contract, enforce_admission_proof=True)

    assert result.state == "blocked_missing_proof"
    assert any(surface in result.reasons[0] for surface in DECISION_REQUIRED_PROOF_SURFACES)


@pytest.mark.parametrize("status", ["not_run", "unavailable", "failed"])
def test_required_proof_status_blocks_decision_capable_answerability(status: str) -> None:
    """Required proof cannot be silently promoted to a decision-capable result."""
    contract = _proof_contract()
    contract["proof_surfaces"]["analysis"] = {
        "status": status,
        "required": True,
        **(
            {"unavailable_reason": "analysis proof was not produced"}
            if status == "unavailable"
            else {}
        ),
    }

    result = evaluate_answerability(contract)

    assert result.state == "blocked_missing_proof"
    assert "analysis" in result.reasons[0]


@pytest.mark.parametrize("status", ["unavailable", "failed", "not_run"])
def test_optional_nonpassed_proof_is_a_warning(status: str) -> None:
    """Optional non-passed proof remains visible without blocking admission."""
    contract = _proof_contract()
    contract["proof_surfaces"]["result_packet"] = {
        "status": status,
        "required": False,
        **(
            {"unavailable_reason": "packet export is not available for this local comparison"}
            if status == "unavailable"
            else {}
        ),
    }

    result = evaluate_answerability(contract)

    assert result.state == "answerable"
    assert any("result_packet" in warning and status in warning for warning in result.warnings)


def test_all_required_proof_surfaces_pass() -> None:
    """A complete passed proof set preserves decision-capable answerability."""
    result = evaluate_answerability(_proof_contract())

    assert result.state == "answerable"
    assert not any("proof surfaces" in warning for warning in result.warnings)


def test_strict_admission_requires_claim_specific_proof_floor() -> None:
    """Production admission cannot make every claim-critical surface optional."""
    contract = _proof_contract()
    binding = _proof_binding()
    binding.update(
        {
            "question": contract["question"]["research_question"],
            "estimand": contract["estimand"]["primary"],
        }
    )
    contract["proof_binding"] = binding
    for surface in DECISION_REQUIRED_PROOF_SURFACES:
        contract["proof_surfaces"][surface] = {
            "status": "unavailable",
            "required": False,
            "unavailable_reason": "declared optional in a malicious manifest",
        }

    result = evaluate_answerability(contract, enforce_admission_proof=True)

    assert result.state == "blocked_missing_proof"
    assert any(surface in result.reasons[0] for surface in DECISION_REQUIRED_PROOF_SURFACES)


def test_strict_admission_requires_verified_proof_binding() -> None:
    """A passed declarative proof set cannot authorize without exact input identity."""
    contract = _proof_contract()

    result = evaluate_answerability(contract, enforce_admission_proof=True)

    assert result.state == "blocked_missing_proof"
    assert "proof_binding" in result.reasons[0]


def test_strict_admission_requires_repository_root_for_bound_proof() -> None:
    """A bound strict proof cannot crash when repository provenance is unavailable."""
    result = evaluate_answerability(
        _strict_bound_contract(),
        enforce_admission_proof=True,
        campaign_id="issue_6474_fixture",
    )

    assert result.state == "blocked_missing_proof"
    assert "repository root" in result.reasons[0]


def test_strict_admission_rejects_proof_surface_mutation_after_binding() -> None:
    """A status mutation cannot be evaluated as the proof that was bound."""
    contract = _strict_bound_contract()
    contract["proof_surfaces"]["analysis"]["status"] = "failed"

    result = evaluate_answerability(
        contract,
        enforce_admission_proof=True,
        campaign_id="issue_6474_fixture",
        repo_root=REPO_ROOT,
    )

    assert result.state == "blocked_missing_proof"
    assert "does not match proof results" in result.reasons[0]


def test_strict_admission_rejects_fixture_proof_binding() -> None:
    """A fixture source manifest cannot authorize strict admission."""
    result = evaluate_answerability(
        _strict_bound_contract(),
        enforce_admission_proof=True,
        campaign_id="issue_6474_fixture",
        repo_root=REPO_ROOT,
    )

    assert result.state == "blocked_missing_proof"
    assert "tests/fixtures provenance" in result.reasons[0]


def test_provenance_helpers_handle_outside_paths_and_git_failures(
    tmp_path: Path, monkeypatch
) -> None:
    """Repository provenance helpers fail closed on paths and Git failures."""
    assert answerability_module._repository_path_candidates(Path("/outside"), tmp_path) == ()
    assert not answerability_module._git_path_is_tracked(Path("/outside"), tmp_path)

    def _raise_os_error(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(answerability_module.subprocess, "run", _raise_os_error)
    assert not answerability_module._git_path_is_tracked(REPO_ROOT / "README.md", REPO_ROOT)


def test_strict_provenance_handles_resolution_and_repository_root_failures(
    tmp_path: Path, monkeypatch
) -> None:
    """Strict provenance returns an explicit safe result for resolution edge cases."""

    def _raise_resolution(*args, **kwargs):
        raise RuntimeError("resolution failed")

    monkeypatch.setattr(answerability_module, "_repository_path_candidates", _raise_resolution)
    assert "resolved safely" in answerability_module.strict_proof_input_provenance_error(
        tmp_path / "proof.json", repo_root=tmp_path, field="proof"
    )

    git_marker = tmp_path / ".git"
    git_marker.write_text("not a git directory", encoding="utf-8")
    monkeypatch.undo()
    assert (
        answerability_module.strict_proof_input_provenance_error(
            tmp_path / "proof.json", repo_root=tmp_path, field="proof"
        )
        is None
    )

    with monkeypatch.context() as patch:
        patch.setattr(
            answerability_module.subprocess, "check_output", lambda *args, **kwargs: "/other"
        )
        assert (
            answerability_module.strict_proof_input_provenance_error(
                REPO_ROOT / "README.md", repo_root=REPO_ROOT, field="proof"
            )
            is None
        )

    def _raise_is_file(self):
        raise OSError("stat failed")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "is_file", _raise_is_file)
        assert (
            answerability_module.strict_proof_input_provenance_error(
                REPO_ROOT / "README.md", repo_root=REPO_ROOT, field="proof"
            )
            is None
        )


@pytest.mark.parametrize(
    ("binding", "expected"),
    [
        ({"source_manifest": ""}, "non-empty path"),
        ({"source_manifest": str(REPO_ROOT / "README.md")}, "repository-relative"),
        ({"source_manifest": "../README.md"}, "repository-relative"),
        ({"source_manifest": "."}, "within the repository"),
        ({"source_manifest": "configs"}, "existing file"),
    ],
)
def test_proof_binding_file_paths_are_validated(binding: dict[str, str], expected: str) -> None:
    """Proof binding paths cannot escape, alias, or name a repository directory."""
    assert expected in answerability_module._proof_binding_file_error(
        binding,
        field="source_manifest",
        digest_field="manifest_sha256",
        repo_root=REPO_ROOT,
    )


def test_proof_binding_file_digest_accepts_tracked_bytes_and_rejects_drift() -> None:
    """Tracked proof bytes must match their declared digest exactly."""
    path = REPO_ROOT / "configs/benchmarks/research_campaign_manifest.example.yaml"
    relative = path.relative_to(REPO_ROOT).as_posix()
    binding = {
        "source_manifest": relative,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

    assert (
        answerability_module._proof_binding_file_error(
            binding,
            field="source_manifest",
            digest_field="manifest_sha256",
            repo_root=REPO_ROOT,
        )
        is None
    )
    binding["manifest_sha256"] = "0" * 64
    assert "does not match" in answerability_module._proof_binding_file_error(
        binding,
        field="source_manifest",
        digest_field="manifest_sha256",
        repo_root=REPO_ROOT,
    )


@pytest.mark.parametrize("failure", ["read", "changed"])
def test_proof_binding_file_read_is_stable_and_bounded(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    """Proof verification reports read failures and byte changes explicitly."""
    path = tmp_path / "proof.json"
    path.write_bytes(b"first")
    original = Path.read_bytes
    calls = 0

    def _read_bytes(current: Path) -> bytes:
        nonlocal calls
        if current != path:
            return original(current)
        calls += 1
        if failure == "read":
            raise OSError("read failed")
        return b"first" if calls == 1 else b"second"

    monkeypatch.setattr(Path, "read_bytes", _read_bytes)
    error = answerability_module._proof_binding_file_error(
        {"source_manifest": path.name, "manifest_sha256": hashlib.sha256(b"first").hexdigest()},
        field="source_manifest",
        digest_field="manifest_sha256",
        repo_root=tmp_path,
    )

    assert error is not None
    assert ("could not read" in error) if failure == "read" else ("changed while" in error)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("schema_version", "wrong", "schema_version"),
        ("manifest_sha256", "wrong", "64-hex SHA-256"),
        ("head_commit", "wrong", "40-hex Git identity"),
    ],
)
def test_strict_binding_rejects_malformed_identity_fields(
    field: str, value: str, expected: str
) -> None:
    """Strict binding identities use explicit checksum and Git formats."""
    contract = _strict_bound_contract()
    contract["proof_binding"][field] = value

    result = evaluate_answerability(
        contract,
        enforce_admission_proof=True,
        campaign_id="issue_6474_fixture",
        repo_root=REPO_ROOT,
    )

    assert result.state == "blocked_missing_proof"
    assert expected in result.reasons[0]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("head_commit", "0" * 40, "head_commit does not match"),
        ("campaign_id", "other_campaign", "campaign_id does not match"),
        ("question", "other question", "question does not match"),
        ("estimand", "other estimand", "estimand does not match"),
    ],
)
def test_strict_binding_rejects_claim_identity_mismatches(
    field: str, value: str, expected: str
) -> None:
    """Strict binding cannot be reused for another commit, campaign, or claim."""
    contract = _strict_bound_contract()
    contract["proof_binding"][field] = value

    result = evaluate_answerability(
        contract,
        enforce_admission_proof=True,
        campaign_id="issue_6474_fixture",
        repo_root=REPO_ROOT,
    )

    assert result.state == "blocked_missing_proof"
    assert expected in result.reasons[0]


@pytest.mark.parametrize("mutation", ["missing", "extra", "non_mapping", "required"])
def test_strict_binding_requires_matching_proof_result_shapes(mutation: str) -> None:
    """Bound proof results must name and match every declared surface."""
    contract = _strict_bound_contract()
    proof_results = contract["proof_binding"]["proof_results"]
    if mutation == "missing":
        proof_results.pop("analysis")
        expected = "exactly the six"
    elif mutation == "extra":
        proof_results["unsupported"] = {"status": "passed", "required": True}
        expected = "exactly the six"
    elif mutation == "non_mapping":
        proof_results["analysis"] = "passed"
        expected = "proof result must be a mapping"
    else:
        proof_results["analysis"]["required"] = False
        expected = "required flag does not match"

    result = evaluate_answerability(
        contract,
        enforce_admission_proof=True,
        campaign_id="issue_6474_fixture",
        repo_root=REPO_ROOT,
    )

    assert result.state == "blocked_missing_proof"
    assert expected in result.reasons[0]


def test_strict_binding_rejects_noncanonical_and_stale_digests(monkeypatch) -> None:
    """Strict proof rejects non-canonical proof JSON and stale proof digests."""
    contract = _strict_bound_contract()

    def _raise_digest(*args, **kwargs):
        raise TypeError("not canonical")

    monkeypatch.setattr(answerability_module, "compute_proof_digest", _raise_digest)
    result = evaluate_answerability(
        contract,
        enforce_admission_proof=True,
        campaign_id="issue_6474_fixture",
        repo_root=REPO_ROOT,
    )
    assert "not canonical JSON" in result.reasons[0]

    monkeypatch.undo()
    contract = _strict_bound_contract()
    contract["proof_binding"]["proof_digest"] = "0" * 64
    result = evaluate_answerability(
        contract,
        enforce_admission_proof=True,
        campaign_id="issue_6474_fixture",
        repo_root=REPO_ROOT,
    )
    assert "proof_digest does not match" in result.reasons[0]


def test_strict_binding_accepts_complete_tracked_inputs() -> None:
    """A complete binding over committed non-fixture inputs can pass strict checks."""
    result = evaluate_answerability(
        _tracked_strict_bound_contract(),
        enforce_admission_proof=True,
        campaign_id="issue_6474_fixture",
        repo_root=REPO_ROOT,
    )

    assert result.state == "answerable"


@pytest.mark.parametrize("blob_failure", ["missing", "mismatch"])
def test_strict_binding_validates_committed_blob_identity(blob_failure: str, monkeypatch) -> None:
    """Strict binding also verifies the Git blob identity of each input."""
    contract = _tracked_strict_bound_contract()
    original = answerability_module.subprocess.check_output
    blob_calls: dict[str, int] = {}

    def _check_output(command, **kwargs):
        if len(command) >= 3 and command[:2] == ["git", "rev-parse"]:
            identity = str(command[2])
            if identity.startswith("HEAD:"):
                blob_calls[identity] = blob_calls.get(identity, 0) + 1
                if blob_calls[identity] >= 2:
                    if blob_failure == "missing":
                        raise subprocess.CalledProcessError(128, command)
                    return "0" * 40
        return original(command, **kwargs)

    monkeypatch.setattr(answerability_module.subprocess, "check_output", _check_output)
    error = answerability_module._proof_binding_error(
        contract,
        campaign_id="issue_6474_fixture",
        repo_root=REPO_ROOT,
    )

    assert error is not None
    assert (
        "source_manifest is not present" in error
        if blob_failure == "missing"
        else "manifest_blob does not match" in error
    )


@pytest.mark.parametrize(
    ("section", "field", "value", "expected_state", "reason_fragment"),
    [
        (
            "analysis",
            "dry_run_status",
            "not_required",
            "blocked_analysis_contract",
            "requires analysis dry-run status 'passed'",
        ),
        (
            "design",
            "power_status",
            "not_required",
            "blocked_underpowered",
            "requires power status 'adequate'",
        ),
    ],
)
def test_strict_decision_capable_admission_rejects_waived_dry_run_or_power(
    section: str,
    field: str,
    value: str,
    expected_state: str,
    reason_fragment: str,
) -> None:
    """Decision-capable admission cannot waive the dry-run or power proof floor."""
    contract = _proof_contract()
    contract["proof_binding"] = _proof_binding()
    contract["design"]["mode"] = "decision_capable"
    contract["artifacts"]["durability_status"] = "ready"
    contract["analysis"]["dry_run_status"] = "passed"
    contract["design"]["power_status"] = "adequate"
    contract[section][field] = value

    result = evaluate_answerability(contract, enforce_admission_proof=True)

    assert result.state == expected_state
    assert reason_fragment in result.reasons[0]


def test_optional_fallback_producer_remains_visible_as_warning() -> None:
    """Optional fallback/degraded producers cannot disappear from the answerability report."""
    contract = json.loads(ISSUE_6474_FIXTURE.read_text(encoding="utf-8"))
    contract["producers"][1].update({"status": "blocked", "execution_mode": "fallback"})

    result = evaluate_answerability(contract)

    assert result.state == "answerable"
    assert any("secondary_realism_metric" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("section", "field", "value", "expected"),
    [
        ("producers", "status", "missing", "blocked_missing_producer"),
        ("producers", "execution_mode", "fallback", "blocked_missing_producer"),
        ("design", "power_status", "underpowered", "blocked_underpowered"),
        ("analysis", "dry_run_status", "failed", "blocked_analysis_contract"),
        ("analysis", "comparability_status", "mismatched", "blocked_noncomparable_rows"),
        ("artifacts", "durability_status", "blocked", "blocked_artifact_plan"),
    ],
)
def test_known_answerability_blockers_are_fail_closed(
    section: str, field: str, value: str, expected: str
) -> None:
    """Known campaign failure classes map to explicit non-answerable states."""
    contract = _example_contract()
    if section == "producers":
        contract[section][0][field] = value
    else:
        contract[section][field] = value

    assert evaluate_answerability(contract).state == expected


def test_malformed_contract_is_invalid() -> None:
    """Missing schema fields cannot be mistaken for an underpowered campaign."""
    contract = _example_contract()
    del contract["estimand"]["primary"]

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert "primary" in result.reasons[0]


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda contract: contract.update({"producers": []}), "producers"),
        (
            lambda contract: contract["question"].update({"decision_vocabulary": [1]}),
            "decision_vocabulary",
        ),
        (
            lambda contract: contract["question"].update({"decision_vocabulary": ["unsupported"]}),
            "unsupported values",
        ),
        (lambda contract: contract["producers"][0].update({"status": "invalid"}), "status"),
        (
            lambda contract: contract["producers"][0].update({"execution_mode": "invalid"}),
            "execution_mode",
        ),
        (lambda contract: contract["producers"][0].update({"required": "yes"}), "required"),
        (
            lambda contract: contract["analysis"].update({"dry_run_status": "invalid"}),
            "dry_run_status",
        ),
        (
            lambda contract: contract["analysis"].update({"comparability_status": "invalid"}),
            "comparability_status",
        ),
        (lambda contract: contract["design"].update({"mode": "invalid"}), "design.mode"),
        (lambda contract: contract["design"].update({"power_status": "invalid"}), "power_status"),
        (
            lambda contract: contract["artifacts"].update({"durability_status": "invalid"}),
            "durability_status",
        ),
    ],
)
def test_contract_validation_rejects_invalid_enum_and_shape_values(mutator, expected: str) -> None:
    """Structural answerability fields do not accept unrecognized values."""
    contract = _example_contract()
    mutator(contract)

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert expected in result.reasons[0]


def test_answerability_rejects_non_mapping_and_unknown_schema_contracts() -> None:
    """The public evaluator reports malformed top-level answerability values."""
    assert evaluate_answerability(None).state == "invalid_contract"
    assert "schema_version" in evaluate_answerability({"schema_version": "wrong"}).reasons[0]

    result = answerability_from_manifest({"answerability": []})
    assert result["state"] == "invalid_contract"
    assert result["reasons"] == ["answerability must be a mapping"]


def test_unknown_power_is_an_explicit_blocker() -> None:
    """Unknown power classification remains blocked rather than being treated as adequate."""
    contract = _example_contract()
    contract["design"]["power_status"] = "unknown"

    result = evaluate_answerability(contract)

    assert result.state == "blocked_underpowered"


@pytest.mark.parametrize("checksums", [None, [], ["  "], ["summary.json", 1]])
def test_invalid_checksum_declarations_are_fail_closed(checksums: object) -> None:
    """Artifact provenance requires a non-empty list of non-empty checksum names."""
    contract = _example_contract()
    if checksums is None:
        del contract["artifacts"]["checksums"]
    else:
        contract["artifacts"]["checksums"] = checksums

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert "checksums" in result.reasons[0]


@pytest.mark.parametrize(
    ("case_id", "mutator", "expected"),
    [
        (
            "6970_missing_normalized_producer",
            lambda contract: contract["producers"][0].update(
                {"status": "missing", "field": "normalized_reference_value"}
            ),
            "blocked_missing_producer",
        ),
        (
            "6849_underpowered_held_out_design",
            lambda contract: contract["design"].update({"power_status": "underpowered"}),
            "blocked_underpowered",
        ),
        (
            "6980_missing_reference_exposure",
            lambda contract: contract["analysis"].update({"comparability_status": "mismatched"}),
            "blocked_noncomparable_rows",
        ),
        (
            "6814_missing_durable_provenance",
            lambda contract: contract["artifacts"].update({"durability_status": "blocked"}),
            "blocked_artifact_plan",
        ),
    ],
)
def test_known_failure_cases_have_explicit_states(case_id: str, mutator, expected: str) -> None:
    """Known issue failure classes cannot be silently promoted to answerable."""
    contract = _example_contract()
    mutator(contract)

    result = evaluate_answerability(contract)

    assert case_id
    assert result.state == expected


def test_manifest_without_answerability_is_not_declared() -> None:
    """Existing manifests remain loadable but can be gated explicitly."""
    manifest = {"campaign": {}}

    result = answerability_from_manifest(manifest)

    assert result["state"] == "not_declared"
    assert result["decision_capable"] is False


def test_research_yield_report_separates_empirical_and_infrastructure() -> None:
    """Yield dimensions remain separate and carry the frozen source digest."""
    snapshot = load_snapshot(YIELD_FIXTURE)
    report = build_research_yield_report(snapshot, source_path=YIELD_FIXTURE)

    assert report["records_total"] == 5
    assert report["empirical_answers"] == {
        "records": 3,
        "statuses": {"completed": 1, "inconclusive": 2},
    }
    assert report["infrastructure_throughput"]["records"] == 2
    assert report["lag_days"]["approval_to_first_result"]["median_days"] == 2.0
    assert report["source_snapshot"]["sha256"]
    assert "closure" in report["definitions"]["empirical_answers"]
    assert "## Empirical Answers" in render_markdown(report)


def test_research_yield_report_renders_query_defined_dimensions() -> None:
    """Issue #7090 dimensions are copied from explicit snapshot queries, not inferred."""
    snapshot = load_snapshot(YIELD_FIXTURE)
    report = build_research_yield_report(snapshot, source_path=YIELD_FIXTURE)
    markdown = render_markdown(report)

    duplicate_dimension = report["dimensions"]["duplicate_competing_prs"]
    assert duplicate_dimension["denominator"] == 5
    assert duplicate_dimension["buckets"] == {
        "competing_pr": 1,
        "duplicate_and_competing": 0,
        "duplicate_pr": 1,
        "no_duplicate_or_competing": 3,
    }
    assert report["dimensions"]["post_merge_repairs"]["buckets"]["post_merge_repair"] == 1
    assert report["dimensions"]["admitted_result_packets"]["buckets"]["admitted_packet"] == 1
    assert report["dimensions"]["blocked_age_categories"]["buckets"] == {
        "blocked_0_7_days": 1,
        "blocked_8_30_days": 1,
        "blocked_over_30_days": 0,
        "not_blocked": 3,
    }
    assert "## Query-Defined Dimensions" in markdown
    assert "duplicate_competing_prs" in markdown
    assert "duplicate_or_competing_pr classification" in duplicate_dimension["query"]


def test_research_yield_report_rejects_unknown_kind(tmp_path: Path) -> None:
    """The report must not silently classify an unknown workflow record."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["records"][0]["kind"] = "merged_issue"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="kind is unsupported"):
        load_snapshot(path)


def test_research_yield_report_rejects_duplicate_record_ids(tmp_path: Path) -> None:
    """A frozen snapshot cannot count one work item twice under different rows."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["records"][1]["id"] = payload["records"][0]["id"]
    path = tmp_path / "duplicate_record.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="id is duplicated"):
        load_snapshot(path)


def test_research_yield_report_rejects_non_finite_lag(tmp_path: Path) -> None:
    """NaN lag values cannot enter a reproducible JSON report."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["records"][0]["approval_to_first_result_days"] = float("nan")
    path = tmp_path / "non_finite_lag.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="non-negative numeric values"):
        load_snapshot(path)


def test_research_yield_report_validates_in_memory_snapshots() -> None:
    """The public report builder must retain fail-closed validation without file loading."""
    snapshot = load_snapshot(YIELD_FIXTURE)
    del snapshot["dimensions"]["blocked_age_categories"]

    with pytest.raises(ResearchYieldError, match="missing required names"):
        build_research_yield_report(snapshot)


def test_research_yield_report_rejects_unknown_dimension(tmp_path: Path) -> None:
    """Snapshot dimensions are explicit reporting queries, not an open-ended tag bag."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["merged_without_review"] = {
        "query": "unsupported query",
        "denominator": 0,
        "buckets": {},
    }
    path = tmp_path / "invalid_dimension.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="unsupported names"):
        load_snapshot(path)


def test_research_yield_report_rejects_missing_required_dimension(tmp_path: Path) -> None:
    """Every supported dimension must remain explicit in the frozen snapshot."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    del payload["dimensions"]["blocked_age_categories"]
    path = tmp_path / "missing_dimension.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="missing required names"):
        load_snapshot(path)


def test_research_yield_report_rejects_unknown_dimension_bucket(tmp_path: Path) -> None:
    """Known dimensions cannot accept inferred or ad-hoc bucket names."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["post_merge_repairs"]["buckets"]["repair_inferred_from_ci"] = 1
    path = tmp_path / "unknown_bucket.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="buckets contain unsupported names"):
        load_snapshot(path)


def test_research_yield_report_rejects_unknown_dimension_field(tmp_path: Path) -> None:
    """A dimension cannot silently preserve fields outside its versioned contract."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["post_merge_repairs"]["review_source"] = "live-state"
    path = tmp_path / "unknown_dimension_field.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="contains unsupported fields"):
        load_snapshot(path)


def test_research_yield_report_rejects_dimension_denominator_mismatch(tmp_path: Path) -> None:
    """Dimension denominators must match the explicit bucket total."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["admitted_result_packets"]["denominator"] = 6
    path = tmp_path / "denominator_mismatch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="buckets sum to 5, expected denominator 6"):
        load_snapshot(path)


@pytest.mark.parametrize("bad_count", [-1, 1.5, True])
def test_research_yield_report_rejects_non_integer_dimension_counts(
    tmp_path: Path, bad_count: object
) -> None:
    """Dimension counts are non-negative integers; bool is rejected despite being an int subtype."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["duplicate_competing_prs"]["buckets"]["duplicate_pr"] = bad_count
    path = tmp_path / "bad_bucket_count.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="must be a non-negative integer"):
        load_snapshot(path)


@pytest.mark.parametrize("bad_denominator", [-1, 5.0, False])
def test_research_yield_report_rejects_non_integer_dimension_denominator(
    tmp_path: Path, bad_denominator: object
) -> None:
    """Dimension denominators use the same non-negative integer contract as bucket counts."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["blocked_age_categories"]["denominator"] = bad_denominator
    path = tmp_path / "bad_denominator.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="must be a non-negative integer"):
        load_snapshot(path)
