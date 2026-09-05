"""Tests for executable research-answerability proof collection."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_proof_path_cannot_escape_repo_through_symlink(tmp_path: Path) -> None:
    """Proof inputs must remain inside the repository after symlink resolution."""
    outside = tmp_path.parent / "answerability-proof-outside.json"
    outside.write_text("{}", encoding="utf-8")
    link = tmp_path / "proof.json"
    link.symlink_to(outside)

    with pytest.raises(AnswerabilityProofError, match="resolve within the repository"):
        preflight_module._repo_relative_path(tmp_path, "proof.json", "proof.path")


def test_malformed_receipt_identity_returns_structured_failure(tmp_path: Path) -> None:
    """A receipt without identity is blocked instead of raising during collection."""
    manifest = _manifest()
    (tmp_path / "receipt.json").write_text("{}", encoding="utf-8")
    manifest["validation"]["answerability_proof"] = {
        "producer": {"kind": "producer_receipt", "path": "receipt.json"}
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["producer"]
    assert result["status"] == "failed"
    assert "identity must be a mapping" in result["reason"]


def test_required_receipt_cannot_be_self_attested(tmp_path: Path) -> None:
    """A passing receipt must name an independent executable or owner check."""
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
            "path": receipt_path.name,
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "identity": {
                "campaign_id": receipt["campaign_id"],
                "question": receipt["question"],
                "estimand": receipt["estimand"],
                "producer_fields": producer_fields,
            },
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["producer"]
    assert result["status"] == "failed"
    assert "self-attested" in result["reason"]


def test_required_receipt_canonical_owner_verification_fails_closed(tmp_path: Path) -> None:
    """An AST-only owner declaration cannot authorize a strict receipt."""
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    owner_path = tmp_path / "producer_owner.py"
    owner_path.write_text(
        "def build_rows(manifest):\n    return manifest\n",
        encoding="utf-8",
    )
    for producer in manifest["answerability"]["producers"]:
        producer["source"] = owner_path.name
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
            "path": receipt_path.name,
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "identity": {
                "campaign_id": receipt["campaign_id"],
                "question": receipt["question"],
                "estimand": receipt["estimand"],
                "producer_fields": producer_fields,
            },
            "verification": {
                "kind": "canonical_owner",
                "source": owner_path.name,
                "symbol": "build_rows",
                "sha256": hashlib.sha256(owner_path.read_bytes()).hexdigest(),
            },
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["producer"]
    assert result["status"] == "failed"
    assert "receipt-aware validator" in result["reason"]


def test_strict_proof_rejects_matching_untracked_output_file(tmp_path: Path) -> None:
    """A matching digest cannot make an untracked output file strict proof."""
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    output_path = tmp_path / "output" / "producer_receipt.json"
    output_path.parent.mkdir()
    output_path.write_text("{}\n", encoding="utf-8")
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    manifest["validation"]["answerability_proof"] = {
        "producer": {
            "kind": "producer_receipt",
            "proof_class": "decision_capable",
            "path": "./output/producer_receipt.json",
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "identity": {
                "campaign_id": manifest["campaign"]["id"],
                "question": manifest["answerability"]["question"]["research_question"],
                "estimand": manifest["answerability"]["estimand"]["primary"],
                "producer_fields": sorted(
                    producer["field"]
                    for producer in manifest["answerability"]["producers"]
                    if producer.get("required", True)
                ),
            },
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["producer"]
    assert result["status"] == "failed"
    assert "untracked files under output/" in result["reason"]


def test_required_analysis_receipt_canonical_owner_verification_fails_closed(
    tmp_path: Path,
) -> None:
    """An AST-only analysis owner declaration cannot authorize a strict receipt."""
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    owner_path = tmp_path / "analysis_owner.py"
    owner_path.write_text(
        "def run_analysis(manifest):\n    return manifest\n",
        encoding="utf-8",
    )
    manifest["answerability"]["analysis"].update(
        {"analysis_id": "analysis_fixture", "source": owner_path.name}
    )
    receipt = {
        "schema_version": "research_answerability_analysis_receipt.v1",
        "campaign_id": manifest["campaign"]["id"],
        "question": manifest["answerability"]["question"]["research_question"],
        "estimand": manifest["answerability"]["estimand"]["primary"],
        "analysis_id": "analysis_fixture",
        "command": manifest["answerability"]["analysis"]["command"],
        "dry_run_status": "passed",
        "comparability_status": "passed",
        "status": "passed",
    }
    receipt_path = tmp_path / "analysis_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest["validation"]["answerability_proof"] = {
        "analysis": {
            "kind": "analysis_receipt",
            "proof_class": "decision_capable",
            "path": receipt_path.name,
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "identity": {
                "campaign_id": receipt["campaign_id"],
                "question": receipt["question"],
                "estimand": receipt["estimand"],
                "analysis_id": receipt["analysis_id"],
            },
            "verification": {
                "kind": "canonical_owner",
                "source": owner_path.name,
                "symbol": "run_analysis",
                "sha256": hashlib.sha256(owner_path.read_bytes()).hexdigest(),
            },
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["analysis"]
    assert result["status"] == "failed"
    assert "receipt-aware validator" in result["reason"]


def test_required_receipt_command_verification_fails_closed_without_canonical_validator(
    tmp_path: Path, monkeypatch
) -> None:
    """A green unrelated pytest file cannot authorize a strict producer receipt."""
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
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/validation/test_research_answerability_preflight.py",
        "-q",
    ]
    manifest["validation"]["answerability_proof"] = {
        "producer": {
            "kind": "producer_receipt",
            "proof_class": "decision_capable",
            "path": receipt_path.name,
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "identity": {
                "campaign_id": receipt["campaign_id"],
                "question": receipt["question"],
                "estimand": receipt["estimand"],
                "producer_fields": producer_fields,
            },
            "verification": {
                "kind": "command",
                "validator_id": "producer_contract",
                "command": command,
            },
        }
    }
    calls = []

    def _run(command_value, **kwargs):
        calls.append((command_value, kwargs))
        return subprocess.CompletedProcess(command_value, 0, "", "")

    monkeypatch.setattr(preflight_module.subprocess, "run", _run)

    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["producer"]
    assert result["status"] == "failed"
    assert "receipt-aware validator" in result["reason"]
    assert calls == []


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


def test_evidence_contract_requires_canonical_claim_identity() -> None:
    """A decision proof cannot pass when the evidence contract has no bound row."""
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    manifest["answerability"]["proof_surfaces"]["evidence_contract"]["required"] = True
    identity = {
        "campaign_id": manifest["campaign"]["id"],
        "question": manifest["answerability"]["question"]["research_question"],
        "estimand": manifest["answerability"]["estimand"]["primary"],
        "contract_id": "orca_residual_smoke",
    }
    manifest["validation"]["answerability_proof"] = {
        "evidence_contract": {
            "kind": "evidence_contract",
            "proof_class": "decision_capable",
            "contract_id": identity["contract_id"],
            "identity": identity,
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=REPO_ROOT,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["evidence_contract"]
    assert result["status"] == "failed"
    assert "canonical evidence row" in result["reason"]


def test_artifact_catalog_rejects_substituted_claim_identity(tmp_path: Path) -> None:
    """A catalog with valid files but another claim identity cannot pass strict proof."""
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    manifest["answerability"]["proof_surfaces"]["artifact"]["required"] = True
    source = tmp_path / "source.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")
    output = tmp_path / "result.json"
    output.write_text('{"value": 1}\n', encoding="utf-8")
    claim_identity = {
        "campaign_id": "another_campaign",
        "question": manifest["answerability"]["question"]["research_question"],
        "estimand": manifest["answerability"]["estimand"]["primary"],
    }
    catalog = {
        "schema_version": "artifact_catalog.v1",
        "catalog_id": "strict_catalog",
        "claim_identity": claim_identity,
        "artifacts": [
            {
                "artifact_id": "fig_strict",
                "artifact_kind": "table",
                "source_kind": "benchmark_campaign",
                "source_files": [
                    {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                ],
                "outputs": {
                    "json": {
                        "path": output.name,
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    }
                },
                "generation_command": "bounded fixture construction",
                "generation_commit": "44f4f364",
                "claim_boundary": "bounded test artifact",
            }
        ],
    }
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    artifact_identity = {
        "campaign_id": manifest["campaign"]["id"],
        "question": manifest["answerability"]["question"]["research_question"],
        "estimand": manifest["answerability"]["estimand"]["primary"],
        "catalog_id": catalog["catalog_id"],
        "artifact_ids": ["fig_strict"],
        "artifact_digests": {
            "fig_strict": sorted(
                [
                    catalog["artifacts"][0]["source_files"][0]["sha256"],
                    catalog["artifacts"][0]["outputs"]["json"]["sha256"],
                ]
            )
        },
    }
    manifest["validation"]["answerability_proof"] = {
        "artifact": {
            "kind": "artifact_catalog",
            "proof_class": "decision_capable",
            "path": catalog_path.name,
            "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "identity": artifact_identity,
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["artifact"]
    assert result["status"] == "failed"
    assert "claim_identity" in result["reason"]


def test_artifact_catalog_rejects_fixture_file_refs_after_source_kind_relabel(
    tmp_path: Path,
) -> None:
    """Fixture file refs remain diagnostic after a catalog source-kind relabel."""
    fixture_dir = tmp_path / "tests" / "fixtures"
    fixture_dir.mkdir(parents=True)
    source = fixture_dir / "source.json"
    source.write_text('{"source": "fixture"}\n', encoding="utf-8")
    output = fixture_dir / "rendered.json"
    output.write_text('{"rendered": true}\n', encoding="utf-8")
    source_ref = {
        "path": "./tests/fixtures/source.json",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    output_ref = {
        "path": "./tests/fixtures/rendered.json",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    manifest["answerability"]["proof_surfaces"]["artifact"]["required"] = True
    catalog = {
        "schema_version": "artifact_catalog.v1",
        "catalog_id": "relabelled_fixture_catalog",
        "claim_identity": {
            "campaign_id": manifest["campaign"]["id"],
            "question": manifest["answerability"]["question"]["research_question"],
            "estimand": manifest["answerability"]["estimand"]["primary"],
        },
        "artifacts": [
            {
                "artifact_id": "tab_relabelled_fixture",
                "artifact_kind": "table",
                "source_kind": "benchmark_campaign",
                "source_files": [source_ref],
                "outputs": {"json": output_ref},
                "generation_command": "benchmark-campaign: generated artifact",
                "generation_commit": "44f4f364",
                "claim_boundary": "bounded artifact",
            }
        ],
    }
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    manifest["validation"]["answerability_proof"] = {
        "artifact": {
            "kind": "artifact_catalog",
            "proof_class": "decision_capable",
            "path": catalog_path.name,
            "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "identity": {
                "campaign_id": manifest["campaign"]["id"],
                "question": manifest["answerability"]["question"]["research_question"],
                "estimand": manifest["answerability"]["estimand"]["primary"],
                "catalog_id": catalog["catalog_id"],
                "artifact_ids": ["tab_relabelled_fixture"],
                "artifact_digests": {
                    "tab_relabelled_fixture": sorted([source_ref["sha256"], output_ref["sha256"]])
                },
            },
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["artifact"]
    assert result["status"] == "failed"
    assert "tests/fixtures provenance" in result["reason"]


def test_result_packet_rejects_substituted_claim_identity(tmp_path: Path, monkeypatch) -> None:
    """A packet validator that exposes another campaign/question/estimand is rejected."""
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    manifest["answerability"]["proof_surfaces"]["result_packet"]["required"] = True
    identity = {
        "campaign_id": manifest["campaign"]["id"],
        "question": manifest["answerability"]["question"]["research_question"],
        "estimand": manifest["answerability"]["estimand"]["primary"],
        "packet_id": "packet_strict",
        "evidence_id": "evidence_strict",
        "evidence_tier": "nominal_benchmark",
        "admission_state": "bounded_simulator_defined",
        "question_id": "question_strict",
        "estimand_id": "estimand_strict",
        "source_digests": {},
    }
    packet_path = tmp_path / "packet.json"
    packet_path.write_text("{}", encoding="utf-8")
    packet = SimpleNamespace(
        packet_id=identity["packet_id"],
        evidence=SimpleNamespace(
            evidence_id=identity["evidence_id"],
            tier=identity["evidence_tier"],
            admission_state=identity["admission_state"],
        ),
        question=SimpleNamespace(
            question_id=identity["question_id"],
            text=identity["question"],
        ),
        estimand=SimpleNamespace(
            estimand_id=identity["estimand_id"],
            description=identity["estimand"],
        ),
        sources=[],
        claim_identity=SimpleNamespace(
            campaign_id=identity["campaign_id"],
            question="substituted question",
            estimand=identity["estimand"],
        ),
    )
    monkeypatch.setattr(
        preflight_module.importlib,
        "import_module",
        lambda _: SimpleNamespace(load_result_interpretation_packet=lambda _: packet),
    )
    manifest["validation"]["answerability_proof"] = {
        "result_packet": {
            "kind": "result_packet",
            "proof_class": "decision_capable",
            "path": packet_path.name,
            "sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            "identity": identity,
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["result_packet"]
    assert result["status"] == "failed"
    assert "identity" in result["reason"]


def test_result_packet_rejects_dot_prefixed_fixture_source(tmp_path: Path, monkeypatch) -> None:
    """A ``./tests/fixtures`` source path cannot evade strict provenance checks."""
    manifest = _manifest()
    manifest["answerability"]["design"]["mode"] = "decision_capable"
    manifest["answerability"]["artifacts"]["durability_status"] = "ready"
    manifest["answerability"]["proof_surfaces"]["result_packet"]["required"] = True
    fixture_path = tmp_path / "tests" / "fixtures" / "source.json"
    fixture_path.parent.mkdir(parents=True)
    fixture_path.write_text('{"source": "fixture"}\n', encoding="utf-8")
    source = SimpleNamespace(
        source_id="fixture_source",
        path="./tests/fixtures/source.json",
        sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
    )
    identity = {
        "campaign_id": manifest["campaign"]["id"],
        "question": manifest["answerability"]["question"]["research_question"],
        "estimand": manifest["answerability"]["estimand"]["primary"],
        "packet_id": "packet_fixture_source",
        "evidence_id": "evidence_fixture_source",
        "evidence_tier": "nominal_benchmark",
        "admission_state": "bounded_simulator_defined",
        "question_id": "question_fixture_source",
        "estimand_id": "estimand_fixture_source",
        "source_digests": {source.source_id: source.sha256},
    }
    packet_path = tmp_path / "packet.json"
    packet_path.write_text("{}", encoding="utf-8")
    packet = SimpleNamespace(
        packet_id=identity["packet_id"],
        evidence=SimpleNamespace(
            evidence_id=identity["evidence_id"],
            tier=identity["evidence_tier"],
            admission_state=identity["admission_state"],
        ),
        question=SimpleNamespace(
            question_id=identity["question_id"],
            text=identity["question"],
        ),
        estimand=SimpleNamespace(
            estimand_id=identity["estimand_id"],
            description=identity["estimand"],
        ),
        sources=[source],
        claim_identity=SimpleNamespace(
            campaign_id=identity["campaign_id"],
            question=identity["question"],
            estimand=identity["estimand"],
        ),
    )
    monkeypatch.setattr(
        preflight_module.importlib,
        "import_module",
        lambda _: SimpleNamespace(load_result_interpretation_packet=lambda _: packet),
    )
    manifest["validation"]["answerability_proof"] = {
        "result_packet": {
            "kind": "result_packet",
            "proof_class": "decision_capable",
            "path": packet_path.name,
            "sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            "identity": identity,
        }
    }

    report = collect_answerability_proof(
        manifest,
        repo_root=tmp_path,
        execute=True,
        build_rows=lambda _: [{"row": 1}],
    )

    result = report["surfaces"]["result_packet"]
    assert result["status"] == "failed"
    assert "tests/fixtures provenance" in result["reason"]


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


def test_checksum_bound_receipt_fails_when_verifier_mutates_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    """Receipt bytes must remain stable through the final verifier call."""
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
            "path": receipt_path.name,
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "identity": {
                "campaign_id": receipt["campaign_id"],
                "question": receipt["question"],
                "estimand": receipt["estimand"],
                "producer_fields": producer_fields,
            },
            "verification": {"kind": "canonical_owner"},
        }
    }

    def _mutating_verifier(*args, **kwargs):
        receipt_path.write_text(json.dumps({**receipt, "status": "changed"}), encoding="utf-8")
        return {"status": "passed", "required": True, "kind": "producer_receipt"}

    monkeypatch.setattr(preflight_module, "_run_receipt_verification", _mutating_verifier)
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


def test_require_answerable_rejects_external_source_manifest(tmp_path: Path) -> None:
    """Strict admission cannot bind an external manifest, even with proof declarations."""
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
    assert "source research manifest must resolve within the repository root" in completed.stderr
    assert not (tmp_path / "packet" / "summary.json").exists()
