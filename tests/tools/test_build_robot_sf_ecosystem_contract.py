"""Unit tests for the deterministic ecosystem contract builder."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from scripts.tools import build_robot_sf_ecosystem_contract as builder

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/ecosystem_contract/v1"
SCHEMA_PATH = ROOT / builder.DEFAULT_CONTRACT_SCHEMA_PATH


def _schema() -> dict[str, object]:
    """Load the contract schema used by unit-level validation."""
    return builder._load_schema(SCHEMA_PATH, "test contract schema")


def _seal_contract(contract: dict[str, object]) -> dict[str, object]:
    """Recompute a scoped digest after a test-only mutation."""
    sealed = copy.deepcopy(contract)
    sealed.pop("contract_digest", None)
    sealed["contract_digest"] = {
        "algorithm": "sha256",
        "scope": "canonical_document_without_contract_digest",
        "value": builder.sha256_bytes(builder.canonical_bytes(sealed)),
    }
    return sealed


def test_shared_rfc8785_vectors_match_exact_bytes() -> None:
    """All implementations can reuse fixed vectors to detect JCS drift."""
    document = builder._mapping(
        builder.load_json(FIXTURES / "canonicalization_vectors.json"),
        "canonicalization vectors",
    )

    assert document["profile"] == "RFC8785"
    for vector in document["vectors"]:
        assert builder.canonical_bytes(vector["value"]) == vector["canonical"].encode()


def test_rfc8785_rejects_values_outside_the_i_json_domain() -> None:
    """Unsupported integers and non-finite numbers must not get fallback bytes."""
    with pytest.raises(builder.ContractError, match="RFC 8785"):
        builder.canonical_bytes({"unsafe_integer": 2**60})
    with pytest.raises(builder.ContractError, match="RFC 8785"):
        builder.canonical_bytes({"not_a_number": float("nan")})


def test_generation_is_byte_identical_and_matches_committed_outputs(tmp_path: Path) -> None:
    """Repeated generation must produce the exact reviewed artifact and sidecar."""
    first_contract = tmp_path / "first.json"
    first_digest = tmp_path / "first.sha256"
    second_contract = tmp_path / "second.json"
    second_digest = tmp_path / "second.sha256"

    builder.write_contract(
        root=ROOT,
        registry_path=builder.DEFAULT_REGISTRY_PATH,
        schema_path=builder.DEFAULT_CONTRACT_SCHEMA_PATH,
        output_path=first_contract,
        digest_path=first_digest,
    )
    builder.write_contract(
        root=ROOT,
        registry_path=builder.DEFAULT_REGISTRY_PATH,
        schema_path=builder.DEFAULT_CONTRACT_SCHEMA_PATH,
        output_path=second_contract,
        digest_path=second_digest,
    )

    assert first_contract.read_bytes() == second_contract.read_bytes()
    assert first_contract.read_bytes() == (ROOT / builder.DEFAULT_CONTRACT_PATH).read_bytes()
    assert first_digest.read_text().split()[0] == second_digest.read_text().split()[0]


def test_check_detects_tampered_contract_bytes(tmp_path: Path) -> None:
    """Check mode must report drift when reviewed output bytes are modified."""
    output = tmp_path / "contract.json"
    digest = tmp_path / "contract.sha256"
    builder.write_contract(
        root=ROOT,
        registry_path=builder.DEFAULT_REGISTRY_PATH,
        schema_path=builder.DEFAULT_CONTRACT_SCHEMA_PATH,
        output_path=output,
        digest_path=digest,
    )
    output.write_bytes(output.read_bytes() + b"\n")

    with pytest.raises(builder.ContractDriftError, match="not canonical RFC 8785 bytes"):
        builder.check_contract(
            root=ROOT,
            registry_path=builder.DEFAULT_REGISTRY_PATH,
            schema_path=builder.DEFAULT_CONTRACT_SCHEMA_PATH,
            output_path=output,
            digest_path=digest,
        )


def test_source_verification_detects_stale_selector_digest() -> None:
    """A valid internal digest cannot hide a stale authoritative source digest."""
    contract = builder._mapping(
        builder.load_json(ROOT / builder.DEFAULT_CONTRACT_PATH), "committed contract"
    )
    contract["authoritative_inputs"][0]["sha256"] = "0" * 64
    stale = _seal_contract(contract)

    with pytest.raises(builder.ContractError, match="stale authoritative input digest"):
        builder.validate_contract_document(stale, schema=_schema(), root=ROOT, verify_sources=True)


def test_registry_rejects_unresolved_authoritative_input() -> None:
    """Every capability authority reference must resolve inside the registry."""
    registry = builder._mapping(
        builder.load_json(ROOT / builder.DEFAULT_REGISTRY_PATH), "capability registry"
    )
    registry["capabilities"][0]["authority"].append("missing.input.v1")

    with pytest.raises(builder.ContractError, match=r"unknown input\(s\)"):
        builder.validate_registry(registry)


def test_strict_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Duplicate object keys must not depend on parser-specific last-value rules."""
    path = tmp_path / "duplicate.json"
    path.write_text('{"value":1,"value":2}', encoding="utf-8")

    with pytest.raises(builder.ContractError, match="duplicate JSON object key"):
        builder.load_json(path)


def test_python_projection_includes_decorators_and_node_type(tmp_path: Path) -> None:
    """A decorator can change behavior and must be part of the source digest."""
    path = tmp_path / "test_contract.py"
    source = "@marker('contract')\ndef test_contract():\n    return True\n"
    path.write_text(source, encoding="utf-8")

    projection = builder._python_symbol_projection(source, path, "test_contract", test=True)

    assert projection == {
        "node_type": "FunctionDef",
        "selector": "test_contract",
        "source": "@marker('contract')\ndef test_contract():\n    return True",
    }


def test_contract_rejects_non_normalized_repository_path() -> None:
    """Contract paths must not escape or depend on platform path normalization."""
    contract = builder._mapping(builder.load_json(FIXTURES / "valid_initial.json"), "valid fixture")
    contract["authoritative_inputs"][0]["path"] = "../outside.json"
    escaped = _seal_contract(contract)

    with pytest.raises(builder.ContractError, match="failed schema validation"):
        builder.validate_contract_document(escaped, schema=_schema())


def test_repository_path_error_is_typed_for_missing_file() -> None:
    """A normalized missing path must report ContractError instead of NameError."""
    with pytest.raises(builder.ContractError, match="does not resolve"):
        builder._repo_relative_path(ROOT, "missing/ecosystem-contract.json", "test path")


def test_absolute_repository_path_resolves_to_portable_relative_path() -> None:
    """CLI users can pass an absolute in-repository path without leaking it."""
    relative, resolved = builder._root_relative_existing_path(
        ROOT, ROOT / builder.DEFAULT_REGISTRY_PATH, "registry"
    )

    assert relative == builder.DEFAULT_REGISTRY_PATH.as_posix()
    assert resolved == (ROOT / builder.DEFAULT_REGISTRY_PATH).resolve()


def test_source_digest_change_is_breaking_even_with_same_major_version_bump() -> None:
    """V1 must not assume that an opaque source change only adds optional fields."""
    baseline = builder._mapping(builder.load_json(FIXTURES / "valid_initial.json"), "valid fixture")
    candidate = copy.deepcopy(baseline)
    candidate["contract_version"] = "1.1.0"
    candidate["change"] = {
        "declared_type": "additive",
        "based_on_contract_digest": baseline["contract_digest"]["value"],
    }
    candidate["capabilities"][0]["interface_version"] = "1.1.0"
    candidate["authoritative_inputs"][0]["sha256"] = "b" * 64
    candidate = _seal_contract(candidate)

    report = builder.check_declared_change(baseline, candidate, schema=_schema())

    assert report["valid"] is False
    assert report["detected_change"] == "breaking"
    assert any("treats source changes as breaking" in reason for reason in report["reasons"])


def test_non_deprecation_status_transition_is_breaking() -> None:
    """A status promotion can still violate an exact downstream status match."""
    baseline = builder._mapping(builder.load_json(FIXTURES / "valid_initial.json"), "valid fixture")
    candidate = copy.deepcopy(baseline)
    candidate["contract_version"] = "1.1.0"
    candidate["change"] = {
        "declared_type": "additive",
        "based_on_contract_digest": baseline["contract_digest"]["value"],
    }
    candidate["capabilities"][0]["status"] = "stable"
    candidate = _seal_contract(candidate)

    report = builder.check_declared_change(baseline, candidate, schema=_schema())

    assert report["valid"] is False
    assert report["detected_change"] == "breaking"
    assert any("changed status" in reason for reason in report["reasons"])


def test_schema_rejects_fields_from_another_capability_kind() -> None:
    """A schema capability must not carry an unrelated CLI command contract."""
    contract = builder._mapping(builder.load_json(FIXTURES / "valid_initial.json"), "valid fixture")
    contract["capabilities"][0]["command"] = {
        "entrypoint": "robot-sf",
        "entrypoint_target": "robot_sf.cli:main",
        "argv": ["envs"],
        "exit_codes": [{"code": 0, "meaning": "success"}],
    }
    mixed = _seal_contract(contract)

    with pytest.raises(builder.ContractError, match="failed schema validation"):
        builder.validate_contract_document(mixed, schema=_schema())


def test_revision_envelope_binds_commit_contract_and_lock(tmp_path: Path) -> None:
    """Revision provenance must be separate and detect later file mutations."""
    contract_path = Path("contract.json")
    lock_path = Path("uv.lock")
    contract_schema_path = Path("contract.schema.json")
    revision_schema_path = Path("revision.schema.json")
    shutil.copyfile(ROOT / builder.DEFAULT_CONTRACT_PATH, tmp_path / contract_path)
    shutil.copyfile(SCHEMA_PATH, tmp_path / contract_schema_path)
    shutil.copyfile(ROOT / builder.DEFAULT_REVISION_SCHEMA_PATH, tmp_path / revision_schema_path)
    (tmp_path / lock_path).write_text("version = 1\n", encoding="utf-8")

    envelope = builder.build_revision_envelope(
        root=tmp_path,
        contract_path=tmp_path / contract_path,
        contract_schema_path=contract_schema_path,
        revision_schema_path=revision_schema_path,
        lock_path=tmp_path / lock_path,
        source_commit="1" * 40,
        release_status="unreleased",
        release_tag=None,
    )
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_bytes(builder.canonical_bytes(envelope))

    assert (
        builder.validate_revision_envelope_path(
            envelope_path,
            root=tmp_path,
            contract_schema_path=contract_schema_path,
            revision_schema_path=revision_schema_path,
        )
        == envelope
    )

    (tmp_path / lock_path).write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(builder.ContractError, match="stale revision envelope lock digest"):
        builder.validate_revision_envelope_path(
            envelope_path,
            root=tmp_path,
            contract_schema_path=contract_schema_path,
            revision_schema_path=revision_schema_path,
        )


def test_cli_check_returns_success_for_current_committed_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The documented check command must provide a stable CI exit contract."""
    exit_code = builder.main(["--check", "--root", str(ROOT)])

    assert exit_code == 0
    assert "ecosystem contract is current" in capsys.readouterr().out


def test_cli_compatibility_check_has_success_and_incompatible_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirements mode must return 0 for a match and 3 for incompatibility."""
    requirements = {
        "schema_version": builder.REQUIREMENTS_SCHEMA_VERSION,
        "supported_contract_schema_majors": [1],
        "supported_consumer_features": sorted(builder.MINIMUM_CONSUMER_FEATURES),
        "required_capabilities": [
            {
                "capability_id": "robot_sf.schema.episode.v1",
                "interface_major": 1,
                "accepted_statuses": ["beta"],
                "semantics_id": "robot_sf.schema.episode.record_semantics.v1",
            }
        ],
    }
    requirements_path = tmp_path / "requirements.json"
    requirements_path.write_text(json.dumps(requirements), encoding="utf-8")
    args = [
        "--root",
        str(ROOT),
        "--validate",
        str(builder.DEFAULT_CONTRACT_PATH),
        "--requirements",
        str(requirements_path),
    ]

    assert builder.main(args) == 0
    assert json.loads(capsys.readouterr().out)["compatible"] is True

    requirements["supported_contract_schema_majors"] = [2]
    requirements_path.write_text(json.dumps(requirements), encoding="utf-8")
    assert builder.main(args) == 3
    captured = capsys.readouterr()
    assert json.loads(captured.out)["compatible"] is False
    assert "does not meet the supplied requirements" in captured.err


def test_cli_rejects_requirements_without_validation(capsys: pytest.CaptureFixture[str]) -> None:
    """Action-specific options must fail closed instead of being ignored."""
    exit_code = builder.main(["--requirements", "requirements.json"])

    assert exit_code == 2
    assert "--requirements requires --validate" in capsys.readouterr().err


def test_cli_rejects_revision_status_without_envelope_action(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Revision-only status must not be ignored during normal generation."""
    exit_code = builder.main(["--release-status", "released"])

    assert exit_code == 2
    assert "revision metadata options require" in capsys.readouterr().err


def test_requirements_shape_rejects_duplicate_capability_ids() -> None:
    """Ambiguous downstream requirements must fail before compatibility matching."""
    requirement = {
        "capability_id": "robot_sf.schema.example.v1",
        "interface_major": 1,
        "accepted_statuses": ["beta"],
        "semantics_id": "robot_sf.schema.example.semantics.v1",
    }
    requirements = {
        "schema_version": builder.REQUIREMENTS_SCHEMA_VERSION,
        "supported_contract_schema_majors": [1],
        "supported_consumer_features": sorted(builder.MINIMUM_CONSUMER_FEATURES),
        "required_capabilities": [requirement, copy.deepcopy(requirement)],
    }

    with pytest.raises(builder.ContractError, match="must be unique and sorted"):
        builder.validate_requirements(requirements)


def test_machine_report_is_deterministic_json() -> None:
    """Compatibility reports must have stable JSON ordering for downstream logs."""
    report = {"z": 1, "a": [2]}

    assert (
        builder._json_report(report)
        == json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
