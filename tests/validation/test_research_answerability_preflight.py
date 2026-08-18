"""Tests for executable research-answerability proof collection."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.research_answerability import PROOF_SURFACES, answerability_from_manifest
from scripts.validation import research_answerability_preflight as preflight_module
from scripts.validation.research_answerability_preflight import (
    AnswerabilityProofError,
    apply_proof_results,
    collect_answerability_proof,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST = REPO_ROOT / "configs/benchmarks/research_campaign_manifest.example.yaml"
RUNNER = REPO_ROOT / "scripts/validation/run_research_campaign_manifest.py"
VALID_CATALOG = REPO_ROOT / "tests/fixtures/artifact_catalog/v1/valid_catalog.yaml"
VALID_PREREGISTRATION = (
    REPO_ROOT / "configs/benchmarks/issue_6942_orca_adapter_hedge_preregistration.yaml"
)


def _manifest() -> dict[str, object]:
    payload = yaml.safe_load(EXAMPLE_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_unexecuted_proof_is_not_promoted() -> None:
    """Configured checks remain not_run until the caller explicitly executes them."""
    manifest = _manifest()

    report = collect_answerability_proof(
        manifest,
        repo_root=REPO_ROOT,
        execute=False,
        build_rows=None,
    )

    assert report["status"] == "not_run"
    assert report["surfaces"]["producer"]["status"] == "not_run"
    assert report["surfaces"]["result_packet"]["status"] == "unavailable"


def test_manifest_row_and_command_checks_execute_without_shell() -> None:
    """The local producer and argv command adapters produce passed results."""
    manifest = _manifest()
    manifest["validation"]["answerability_proof"]["analysis"]["command"] = [
        sys.executable,
        "-m",
        "pytest",
        "tests/benchmark/test_research_campaign_manifest_contract.py",
        "-q",
    ]

    report = collect_answerability_proof(
        manifest,
        repo_root=REPO_ROOT,
        execute=True,
        build_rows=lambda value: [{"campaign_id": value["campaign"]["id"]}],
    )

    assert report["surfaces"]["producer"]["status"] == "passed"
    assert report["surfaces"]["analysis"]["status"] == "passed"
    assert report["surfaces"]["analysis"]["returncode"] == 0


def test_command_proof_rejects_unregistered_shell_validator() -> None:
    """Proof admission cannot execute an arbitrary shell or campaign command."""
    manifest = _manifest()
    manifest["validation"]["answerability_proof"]["analysis"]["command"] = [
        "/bin/sh",
        "-c",
        "echo unsafe",
    ]

    with pytest.raises(AnswerabilityProofError, match="registered pytest validator"):
        collect_answerability_proof(
            manifest,
            repo_root=REPO_ROOT,
            execute=True,
            build_rows=lambda value: [{"campaign_id": value["campaign"]["id"]}],
        )


def test_command_proof_timeout_is_bounded() -> None:
    """Registered validators cannot opt into an unbounded timeout."""
    manifest = _manifest()
    manifest["validation"]["answerability_proof"]["analysis"]["timeout_seconds"] = 121

    with pytest.raises(AnswerabilityProofError, match="timeout_seconds"):
        collect_answerability_proof(
            manifest,
            repo_root=REPO_ROOT,
            execute=True,
            build_rows=lambda value: [{"campaign_id": value["campaign"]["id"]}],
        )


def test_public_preregistration_and_artifact_validators_are_composed() -> None:
    """Typed checks invoke the existing public validators and preserve summaries."""
    manifest = _manifest()
    manifest["validation"]["answerability_proof"].update(
        {
            "preregistration": {
                "kind": "preregistration",
                "path": str(VALID_PREREGISTRATION.relative_to(REPO_ROOT)),
            },
            "artifact": {
                "kind": "artifact_catalog",
                "path": str(VALID_CATALOG.relative_to(REPO_ROOT)),
            },
        }
    )

    report = collect_answerability_proof(
        manifest,
        repo_root=REPO_ROOT,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    assert report["surfaces"]["preregistration"]["status"] == "passed"
    assert report["surfaces"]["preregistration"]["summary"]["status"] == "ok"
    assert report["surfaces"]["artifact"]["status"] == "passed"
    assert report["surfaces"]["artifact"]["issues"] == []


def test_decision_proof_cannot_reuse_one_generic_pytest_for_multiple_surfaces() -> None:
    """A generic test exit code cannot substitute for surface-specific proof."""
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    for surface in PROOF_SURFACES:
        manifest["answerability"]["proof_surfaces"][surface] = {
            "status": "unavailable",
            "required": True,
            "unavailable_reason": "not configured",
        }
    generic_spec = {
        "kind": "command",
        "validator_id": "pytest_contract",
        "proof_class": "decision_capable",
        "command": [
            sys.executable,
            "-m",
            "pytest",
            "tests/benchmark/test_research_campaign_manifest_contract.py",
            "-q",
        ],
    }
    manifest["validation"]["answerability_proof"] = {
        surface: copy.deepcopy(generic_spec) for surface in PROOF_SURFACES
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=REPO_ROOT,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    assert report["surfaces"]["producer"]["status"] == "failed"
    assert "canonical kind" in report["surfaces"]["producer"]["reason"]
    assert report["surfaces"]["analysis"]["status"] == "failed"
    assert "canonical kind" in report["surfaces"]["analysis"]["reason"]


def test_decision_proof_rejects_fixture_result_packet_even_with_matching_identity() -> None:
    """A controlled diagnostic packet cannot satisfy a decision-capable result surface."""
    manifest = _manifest()
    packet_path = (
        REPO_ROOT
        / "tests/fixtures/result_interpretation_packet/v1/issue_6962_lane_formation_diagnostic.json"
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    answerability = manifest["answerability"]
    answerability["design"]["mode"] = "decision_capable"
    answerability["artifacts"]["durability_status"] = "ready"
    answerability["proof_surfaces"]["result_packet"]["required"] = True
    answerability["question"]["research_question"] = packet["question"]["text"]
    answerability["estimand"]["primary"] = packet["estimand"]["description"]
    manifest["validation"]["answerability_proof"] = {
        "result_packet": {
            "kind": "result_packet",
            "proof_class": "decision_capable",
            "path": str(packet_path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            "identity": {
                "campaign_id": manifest["campaign"]["id"],
                "question": packet["question"]["text"],
                "estimand": packet["estimand"]["description"],
                "packet_id": packet["packet_id"],
                "evidence_id": packet["evidence"]["evidence_id"],
                "evidence_tier": packet["evidence"]["tier"],
                "admission_state": packet["evidence"]["admission_state"],
                "question_id": packet["question"]["question_id"],
                "estimand_id": packet["estimand"]["estimand_id"],
                "source_digests": {
                    source["source_id"]: source["sha256"] for source in packet["sources"]
                },
            },
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=REPO_ROOT,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["result_packet"]
    assert result["status"] == "failed"
    assert "diagnostic" in result["reason"] or "fixtures" in result["reason"]


def test_decision_proof_rejects_fixture_artifact_catalog_without_free_text_heuristics() -> None:
    """Fixture source identity blocks artifact proof even when wording is neutral."""
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    manifest["answerability"]["proof_surfaces"]["artifact"]["required"] = True
    catalog = yaml.safe_load(VALID_CATALOG.read_text(encoding="utf-8"))
    artifact_digests = {}
    for artifact in catalog["artifacts"]:
        digests = [ref["sha256"] for ref in artifact["source_files"]]
        digests.extend(ref["sha256"] for ref in artifact["outputs"].values())
        if artifact.get("caption_file"):
            digests.append(artifact["caption_file"]["sha256"])
        artifact_digests[artifact["artifact_id"]] = sorted(digests)
    manifest["validation"]["answerability_proof"] = {
        "artifact": {
            "kind": "artifact_catalog",
            "proof_class": "decision_capable",
            "path": str(VALID_CATALOG.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(VALID_CATALOG.read_bytes()).hexdigest(),
            "identity": {
                "campaign_id": manifest["campaign"]["id"],
                "question": manifest["answerability"]["question"]["research_question"],
                "estimand": manifest["answerability"]["estimand"]["primary"],
                "catalog_id": catalog["catalog_id"],
                "artifact_ids": sorted(artifact_digests),
                "artifact_digests": artifact_digests,
            },
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=REPO_ROOT,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["artifact"]
    assert result["status"] == "failed"
    assert "fixtures" in result["reason"]


def test_checksum_bound_receipt_fails_when_input_drifts_during_validation(
    tmp_path: Path, monkeypatch
) -> None:
    """A receipt cannot be accepted when its bytes change between reads."""
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    producer_fields = sorted(
        producer["field"]
        for producer in manifest["answerability"]["producers"]
        if producer.get("required", True)
    )
    receipt = {
        "schema_version": "research_answerability_producer_receipt.v1",
        "campaign_id": manifest["campaign"]["id"],
        "question": manifest["answerability"]["question"]["research_question"],
        "estimand": manifest["answerability"]["estimand"]["primary"],
        "producer_fields": producer_fields,
        "status": "passed",
    }
    receipt_path = tmp_path / "producer_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest["validation"]["answerability_proof"] = {
        "producer": {
            "kind": "producer_receipt",
            "proof_class": "decision_capable",
            "path": "producer_receipt.json",
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "identity": {
                "campaign_id": receipt["campaign_id"],
                "question": receipt["question"],
                "estimand": receipt["estimand"],
                "producer_fields": producer_fields,
            },
        }
    }
    original = preflight_module._stable_file_bytes
    calls = {"count": 0}

    def _drift(path: Path) -> bytes:
        data = original(path)
        calls["count"] += 1
        if calls["count"] == 2:
            path.write_text(json.dumps({**receipt, "status": "changed"}), encoding="utf-8")
        return data

    monkeypatch.setattr(preflight_module, "_stable_file_bytes", _drift)
    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["producer"]
    assert result["status"] == "failed"
    assert "changed during validation" in result["reason"]


def test_durable_path_cannot_promote_required_artifact_proof() -> None:
    """A path-existence check cannot satisfy a required durable artifact surface."""
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    manifest["answerability"]["proof_surfaces"]["artifact"]["required"] = True
    manifest["validation"]["answerability_proof"].update(
        {
            "preregistration": {
                "kind": "preregistration",
                "path": str(VALID_PREREGISTRATION.relative_to(REPO_ROOT)),
            },
            "evidence_contract": {
                "kind": "evidence_contract",
                "contract_id": "orca_residual_smoke",
            },
            "artifact": {"kind": "durable_path", "path": "README.md"},
        }
    )

    report = collect_answerability_proof(
        manifest,
        repo_root=REPO_ROOT,
        execute=True,
        build_rows=lambda value: [{"campaign_id": value["campaign"]["id"]}],
    )
    evaluated = copy.deepcopy(manifest)
    evaluated["answerability"] = apply_proof_results(manifest["answerability"], report)

    assert report["surfaces"]["artifact"]["status"] == "unavailable"
    assert "checksum identity" in report["surfaces"]["artifact"]["reason"]
    assert answerability_from_manifest(evaluated)["state"] == "blocked_missing_proof"


def test_result_packet_validator_is_explicitly_unavailable_when_unmerged() -> None:
    """No issue-specific packet checker is inferred when the generic owner is absent."""
    manifest = _manifest()
    manifest["validation"]["answerability_proof"].pop("result_packet", None)

    report = collect_answerability_proof(
        manifest,
        repo_root=REPO_ROOT,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["result_packet"]
    assert result["status"] == "unavailable"
    assert "generic result-interpretation" in result["reason"]


def test_apply_proof_results_replaces_declarative_statuses() -> None:
    """Runner output, not input declarations, becomes the evaluated proof state."""
    manifest = _manifest()
    contract = copy.deepcopy(manifest["answerability"])
    report = {
        "surfaces": {
            surface: {
                "status": "passed",
                "required": True,
            }
            for surface in PROOF_SURFACES
        }
    }

    updated = apply_proof_results(contract, report)

    assert all(
        updated["proof_surfaces"][surface]["status"] == "passed" for surface in PROOF_SURFACES
    )


def test_apply_proof_results_fails_closed_when_collector_omits_surfaces() -> None:
    """A missing collector report cannot preserve stale passed declarations."""
    manifest = _manifest()
    contract = copy.deepcopy(manifest["answerability"])
    contract["design"]["mode"] = "decision_capable"
    contract["artifacts"]["durability_status"] = "ready"
    for surface in PROOF_SURFACES:
        contract["proof_surfaces"][surface] = {"status": "passed", "required": True}

    updated = apply_proof_results(contract, {"surfaces": {}})

    assert all(
        updated["proof_surfaces"][surface]["status"] == "not_run" for surface in PROOF_SURFACES
    )


def test_require_answerable_runs_proof_and_fails_closed_without_decision_design(
    tmp_path: Path,
) -> None:
    """The canonical runner still rejects the diagnostic example in launch mode."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(EXAMPLE_MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "packet"),
            "--require-answerable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "answerability gate requires state=answerable" in completed.stderr
    assert not (tmp_path / "packet" / "summary.json").exists()


def test_require_answerable_rejects_unbound_diagnostic_proof_chain(tmp_path: Path) -> None:
    """A fixture and dry-run chain cannot satisfy strict decision admission."""
    manifest = _manifest()
    answerability = manifest["answerability"]
    answerability["design"]["mode"] = "decision_capable"
    answerability["artifacts"]["durability_status"] = "ready"
    manifest["scenario_suite"]["campaign_config"] = (
        "configs/benchmarks/issue_3425_empirical_vertical_slice_smoke.yaml"
    )
    for surface in PROOF_SURFACES:
        answerability["proof_surfaces"][surface] = {
            "status": "unavailable",
            "required": surface != "result_packet",
            "unavailable_reason": "not configured",
        }
    manifest["validation"]["answerability_proof"] = {
        "producer": {"kind": "manifest_rows"},
        "preregistration": {
            "kind": "preregistration",
            "path": str(VALID_PREREGISTRATION.relative_to(REPO_ROOT)),
        },
        "evidence_contract": {
            "kind": "evidence_contract",
            "contract_id": "orca_residual_smoke",
        },
        "analysis": {
            "kind": "command",
            "validator_id": "pytest_contract",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/benchmark/test_research_campaign_manifest_contract.py",
                "-q",
            ],
            "proof_class": "decision_capable",
        },
        "artifact": {
            "kind": "artifact_catalog",
            "path": str(VALID_CATALOG.relative_to(REPO_ROOT)),
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "packet"),
            "--require-answerable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "answerability gate requires state=answerable" in completed.stderr
    assert "blocked_missing_proof" in completed.stderr or "diagnostic_only" in completed.stderr
    assert not (tmp_path / "packet" / "summary.json").exists()
