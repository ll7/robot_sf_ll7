"""Tests for host-independent campaign analysis capsule packaging and verification."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tools.package_campaign_analysis_capsule import (
    build_capsule,
    main,
    unpack_bundle_if_needed,
    verify_capsule,
)

FIXTURES_DIR = Path("tests/tools/fixtures/analysis_capsule")


def test_portable_capsule_passes_verification(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "portable_capsule.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path)
    res = verify_capsule(capsule_root)
    assert res["verdict"] == "pass"
    assert res["status"] == "pass"
    assert len(res["discrepancies"]) == 0


def test_stale_code_fails_verification(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "stale_code.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path)
    res = verify_capsule(capsule_root)
    assert res["verdict"] == "fail"
    assert any("unbound_analysis_code" in d for d in res["discrepancies"])


def test_missing_row_file_fails_verification(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "missing_row_file.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path)
    res = verify_capsule(capsule_root)
    assert res["verdict"] == "fail"
    assert any("missing_row_file" in d for d in res["discrepancies"])


def test_private_path_fails_verification(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "private_path.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path)
    res = verify_capsule(capsule_root)
    assert res["verdict"] == "fail"
    assert any("hidden_absolute_path" in d for d in res["discrepancies"])


def test_mutable_uri_fails_verification(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "mutable_uri.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path)
    res = verify_capsule(capsule_root)
    assert res["verdict"] == "fail"
    assert any("mutable_artifact_alias" in d for d in res["discrepancies"])


def test_dependency_mismatch_fails_verification(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "dependency_mismatch.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path)
    res = verify_capsule(capsule_root)
    assert res["verdict"] == "fail"
    assert any("dependency_mismatch" in d for d in res["discrepancies"])


def test_deterministic_report_regeneration(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "deterministic_report.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path / "capsule")
    out_dir = tmp_path / "report_out"
    res = verify_capsule(capsule_root, regenerate=True, output_dir=out_dir)
    assert res["verdict"] == "pass"
    assert "report.json" in res["regenerated_outputs"]
    assert (out_dir / "report.json").is_file()


def test_output_overwrite_rejected(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "deterministic_report.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path / "capsule")
    out_dir = tmp_path / "report_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text("pre-existing content", encoding="utf-8")
    res = verify_capsule(capsule_root, regenerate=True, output_dir=out_dir)
    assert res["verdict"] == "fail"
    assert any("output_overwrite" in d for d in res["discrepancies"])


def test_unsupported_analysis_marked_explicitly(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "unsupported_analysis.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path / "capsule")
    out_dir = tmp_path / "report_out"
    res = verify_capsule(capsule_root, regenerate=True, output_dir=out_dir)
    assert res["verdict"] == "pass"


def test_unsupported_analysis_missing_reason_rejected(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "unsupported_analysis.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path / "capsule")
    manifest = json.loads((capsule_root / "manifest.json").read_text(encoding="utf-8"))
    for cmd in manifest["report_commands"]:
        if cmd["status"] == "unsupported":
            cmd["reason"] = None
    (capsule_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    res = verify_capsule(capsule_root)
    assert res["verdict"] == "fail"
    assert any("unsupported_analysis_missing_reason" in d for d in res["discrepancies"])


def test_duplicate_and_missing_rows_rejected(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "portable_capsule.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path / "capsule")
    dup_rows = '{"row_id": "r1"}\n{"row_id": "r1"}\n'
    (capsule_root / "data/harvested_rows.jsonl").write_text(dup_rows, encoding="utf-8")
    res = verify_capsule(capsule_root)
    assert res["verdict"] == "fail"
    assert any("duplicate_rows" in d for d in res["discrepancies"])


def test_editable_sibling_import_rejected(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "portable_capsule.json"
    capsule_root = unpack_bundle_if_needed(bundle_path, tmp_path / "capsule")
    bad_code = "import sys\nsys.path.insert(0, '/home/user/lib')\n"
    (capsule_root / "code/generate_report.py").write_text(bad_code, encoding="utf-8")
    res = verify_capsule(capsule_root)
    assert res["verdict"] == "fail"
    assert any("editable_sibling_import" in d for d in res["discrepancies"])


def test_build_capsule_and_verify(tmp_path: Path) -> None:
    data_file = tmp_path / "test_rows.jsonl"
    data_file.write_text('{"row_id": "r1"}\n{"row_id": "r2"}\n', encoding="utf-8")
    code_file = tmp_path / "script.py"
    code_file.write_text("print('hello')\n", encoding="utf-8")
    schema_file = Path("docs/contracts/campaign_expected_row_ledger.v1.schema.json")
    lock_file = Path("uv.lock")

    spec = {
        "capsule_id": "built-test-01",
        "campaign_id": "camp-test-01",
        "source_commit": "1234567890abcdef",
        "claim_boundary": "Testing build in tmp_path.",
        "data_sources": [{"source": str(data_file), "row_count": 2}],
        "code_sources": [str(code_file)],
        "schema_sources": [str(schema_file)],
        "dependency_lock": str(lock_file),
        "expected_row_count": 2,
        "report_commands": [
            {
                "name": "noop",
                "command_tokens": ["python3", "code/script.py"],
                "deterministic": True,
                "status": "available",
            }
        ],
        "output_expectations": [],
    }

    out_capsule = tmp_path / "capsule_out"
    res = build_capsule(spec, out_capsule)
    assert res["verdict"] == "pass"
    assert (out_capsule / "manifest.json").is_file()
    assert (out_capsule / "SHA256SUMS").is_file()
    assert (out_capsule / "data/test_rows.jsonl").is_file()
    assert (out_capsule / "code/script.py").is_file()


def test_cli_check_success(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "portable_capsule.json"
    out_file = tmp_path / "receipt.json"
    rc = main(
        ["--check", "--capsule", str(bundle_path), "--output", str(out_file), "--format", "json"]
    )
    assert rc == 0
    receipt = json.loads(out_file.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "pass"


def test_cli_check_failure(tmp_path: Path) -> None:
    bundle_path = FIXTURES_DIR / "stale_code.json"
    out_file = tmp_path / "receipt.json"
    rc = main(
        ["--check", "--capsule", str(bundle_path), "--output", str(out_file), "--format", "json"]
    )
    assert rc == 2
    receipt = json.loads(out_file.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "fail"
