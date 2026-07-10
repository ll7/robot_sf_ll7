"""Contract-equivalence regression gates for real #4929 helper migrations."""

from __future__ import annotations

import inspect
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.identity import hash_utils
from scripts.dev import preflight_launch_packet
from tests.tooling.shared_helper_contracts import (
    ErrorExpectation,
    SharedHelperContract,
    assert_shared_helper_contract,
)

if TYPE_CHECKING:
    from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(name: str, relative_path: str) -> ModuleType:
    """Load an executable script whose directory is shadowed by ``scripts/*.py``."""

    spec = spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fidelity_report = _load_script_module(
    "issue_4999_fidelity_report",
    "scripts/benchmark/build_fidelity_sensitivity_smoke_report.py",
)
codesign_runner = _load_script_module(
    "issue_4999_codesign_runner",
    "scripts/benchmark/run_issue_4205_codesign_loop_campaign.py",
)


def test_migrated_json_sha_and_jsonl_call_sites_keep_their_contracts(
    tmp_path: Path, capsys
) -> None:
    """Exercise the #4929 JSON, SHA-256, and JSONL migration call sites.

    This is the reusable-template's first real use.  The JSON call site also
    checks the runner's public exit-code translation; JSONL pins source order
    and ``path:line`` context; SHA-256 pins its streaming implementation.
    """

    json_path = tmp_path / "manifest.json"
    json_path.write_text('{"second": 2, "first": 1}\n', encoding="utf-8")
    missing_json = tmp_path / "missing-manifest.json"
    malformed_json = tmp_path / "malformed.json"
    malformed_json.write_text("{not-json}\n", encoding="utf-8")

    json_contract = SharedHelperContract(
        call_site="scripts/benchmark/run_issue_4205_codesign_loop_campaign.py",
        helper_name="hash_utils.load_json",
        valid_call=lambda: codesign_runner._load_json(json_path),
        expected_return_type=dict,
        validate_result=lambda result: assert_json_schema(result, {"second": 2, "first": 1}),
        missing_call=lambda: codesign_runner._load_json(missing_json),
        missing_error=ErrorExpectation(FileNotFoundError),
        malformed_call=lambda: codesign_runner._load_json(malformed_json),
        malformed_error=ErrorExpectation(json.JSONDecodeError, ("Expecting property name",)),
        not_applicable=("import: the benchmark runner intentionally requires robot_sf.",),
        validate_read_strategy=assert_json_is_eager,
        validate_output_ordering=lambda: assert_json_key_order(
            codesign_runner._load_json(json_path)
        ),
    )
    assert_shared_helper_contract(json_contract)

    # The production runner turns the canonical helper's OSError into the
    # documented fail-closed ContractError and CLI exit code rather than leaking
    # FileNotFoundError (#4961).
    assert codesign_runner.main(["--hydration-manifest", str(missing_json)]) == 2
    assert "missing or unreadable hydration manifest" in capsys.readouterr().err

    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"shared-helper-contract")
    missing_payload = tmp_path / "missing-payload.bin"
    sha_contract = SharedHelperContract(
        call_site="scripts/dev/preflight_launch_packet.py",
        helper_name="hash_utils.sha256_file",
        valid_call=lambda: preflight_launch_packet._sha256(payload_path),
        expected_return_type=str,
        validate_result=assert_sha256,
        missing_call=lambda: preflight_launch_packet._sha256(missing_payload),
        missing_error=ErrorExpectation(FileNotFoundError),
        validate_read_strategy=assert_sha256_is_streaming,
        not_applicable=(
            "malformed: SHA-256 accepts arbitrary byte streams.",
            "import: the launch-packet preflight intentionally requires robot_sf.",
            "ordering: SHA-256 returns one scalar digest.",
        ),
    )
    assert_shared_helper_contract(sha_contract)

    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text('{"sequence": 2}\n\n{"sequence": 1}\n', encoding="utf-8")
    missing_jsonl = tmp_path / "missing-records.jsonl"
    non_object_jsonl = tmp_path / "non-object.jsonl"
    non_object_jsonl.write_text('{"sequence": 1}\n[]\n', encoding="utf-8")
    jsonl_contract = SharedHelperContract(
        call_site="scripts/benchmark/build_fidelity_sensitivity_smoke_report.py",
        helper_name="hash_utils.read_jsonl",
        valid_call=lambda: fidelity_report._load_jsonl(jsonl_path),
        expected_return_type=list,
        validate_result=assert_jsonl_schema,
        missing_call=lambda: fidelity_report._load_jsonl(missing_jsonl),
        missing_error=ErrorExpectation(FileNotFoundError),
        malformed_call=lambda: fidelity_report._load_jsonl(non_object_jsonl),
        malformed_error=ErrorExpectation(
            ValueError,
            (f"{non_object_jsonl}:2", "not a JSON object"),
        ),
        validate_read_strategy=assert_jsonl_is_streaming,
        validate_output_ordering=lambda: assert_source_order(
            fidelity_report._load_jsonl(jsonl_path)
        ),
        not_applicable=("import: the benchmark report intentionally requires robot_sf.",),
    )
    assert_shared_helper_contract(jsonl_contract)


def assert_json_schema(result: object, expected: dict[str, int]) -> None:
    """Assert that a JSON helper still returns the expected mapping schema."""

    assert result == expected


def assert_json_key_order(result: object) -> None:
    """Assert JSON object key order remains the source order, not a sorted rewrite."""

    assert list(result) == ["second", "first"]


def assert_json_is_eager() -> None:
    """Pin the object JSON helper's deliberate one-document eager read strategy."""

    source = inspect.getsource(hash_utils.load_json)
    assert "read_text" in source


def assert_sha256(result: object) -> None:
    """Assert SHA-256 output remains a lower-case 64-character hexadecimal string."""

    assert isinstance(result, str)
    assert len(result) == 64
    assert result == result.lower()
    assert int(result, 16) >= 0


def assert_sha256_is_streaming() -> None:
    """Pin the bounded-memory chunked read strategy promised by the canonical helper."""

    source = inspect.getsource(hash_utils.sha256_file)
    assert "path.open" in source
    assert "handle.read(1024 * 1024)" in source
    assert "read_bytes" not in source


def assert_jsonl_schema(result: object) -> None:
    """Assert the JSONL helper still yields a list of mapping records."""

    assert result == [{"sequence": 2}, {"sequence": 1}]


def assert_jsonl_is_streaming() -> None:
    """Pin line-by-line JSONL reading instead of an eager whole-file rewrite."""

    source = inspect.getsource(hash_utils.read_jsonl)
    assert "for line_no, raw in enumerate(handle, start=1)" in source
    assert "read_text" not in source


def assert_source_order(result: object) -> None:
    """Assert records preserve source order rather than sorting by a data field."""

    assert result == [{"sequence": 2}, {"sequence": 1}]
