"""Restore and validate one preserved campaign capsule without running simulation.

The packaged fixture is synthetic and diagnostic-only. This example restores it into a fresh
marker-owned directory, then reuses the capsule, chunk-manifest, episode-row, and lineage readers
to produce deterministic JSON or a concise text summary. It performs no simulation, training,
benchmark run, scientific aggregation, evidence promotion, or network access.

Run from the repository root with ``uv run python examples/advanced/40_restore_campaign_capsule.py
--json``. ``--case`` selects one of the focused fail-closed negative fixtures.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from scripts.tools import chunk_manifest, lineage_index
from scripts.validation import cross_host_conformance_capsule

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "examples/fixtures/campaign_capsule_v1.json"
FIXTURE_SCHEMA = "campaign_capsule_fixture.v1"
CONFIG_SCHEMA = "synthetic_config.v1"
CAMPAIGN_SCHEMA = "synthetic_campaign_manifest.v1"
MISSINGNESS_SCHEMA = "synthetic_missingness_ledger.v1"
REPORT_SCHEMA = "synthetic_campaign_report.v1"
SUMMARY_SCHEMA = "campaign_capsule_summary.v1"
OWNERSHIP_MARKER = ".robot_sf_offline_capsule_owned"
OWNERSHIP_SCHEMA = "offline_capsule_cleanup_marker.v1"
CLAIM_BOUNDARY = (
    "Synthetic offline capsule plumbing only; diagnostic-only, not simulation, benchmark evidence, "
    "or a scientific claim."
)
NEGATIVE_CASES = ("stale_manifest", "missing_row", "duplicate_row", "wrong_source_identity", "wrong_config_identity", "checksum_mismatch", "path_escape", "unsupported_schema", "incomplete_copy")  # fmt: skip


class CapsuleError(ValueError):
    """Fail-closed capsule error with a stable example-local reason code."""

    def __init__(self, code: str) -> None:
        """Store the stable reason code used in the JSON summary."""
        super().__init__(code)
        self.code = code


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapsuleError("malformed_json") from exc
    if not isinstance(value, dict):
        raise CapsuleError("invalid_schema")
    return value


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise CapsuleError(code)


def _safe(root: Path, name: str) -> Path:
    try:
        relative = chunk_manifest.normalize_relative_path(name)
    except chunk_manifest.ChunkManifestError as exc:
        raise CapsuleError(exc.code) from exc
    path = (root / relative).resolve(strict=False)
    if not path.is_relative_to(root.resolve()):
        raise CapsuleError("path_escape")
    return path


def _member(root: Path, name: str) -> Path:
    path = _safe(root / "artifact", name)
    if not path.is_file() or path.is_symlink():
        raise CapsuleError("missing_member")
    return path


def _validate_fixture(fixture: Mapping[str, Any]) -> None:
    _require(fixture.get("schema") == FIXTURE_SCHEMA, "unsupported_fixture_schema")
    evidence = fixture.get("evidence")
    _require(
        isinstance(evidence, Mapping)
        and evidence.get("synthetic") is True
        and evidence.get("execution_status") == "not_run"
        and evidence.get("evidence_status") == "diagnostic_only"
        and evidence.get("promotable") is False
        and evidence.get("claim_boundary") == CLAIM_BOUNDARY,
        "fixture_boundary_invalid",
    )
    expected, files, manifest = (fixture.get(key) for key in ("expected", "files", "artifact_manifest"))  # fmt: skip
    _require(isinstance(expected, Mapping), "missing_expected_identity")
    _require(isinstance(files, Mapping) and files, "missing_fixture_files")
    _require(isinstance(manifest, Mapping), "missing_artifact_manifest")
    issues = chunk_manifest.validate_manifest(manifest)
    _require(not issues, issues[0]["code"] if issues else "invalid_artifact_manifest")
    _require(fixture.get("negative_cases") == list(NEGATIVE_CASES), "negative_case_roster_invalid")


def _update_rows(text: str, case: str) -> str:
    try:
        rows = [json.loads(line) for line in text.splitlines() if line]
    except json.JSONDecodeError as exc:
        raise CapsuleError("malformed_rows") from exc
    _require(rows and all(isinstance(row, dict) for row in rows), "malformed_rows")
    if case == "missing_row":
        rows.pop()
    elif case == "duplicate_row":
        rows.append(copy.deepcopy(rows[0]))
    elif case == "wrong_source_identity":
        rows[0]["source_commit"] = "b" * 40
    elif case == "wrong_config_identity":
        rows[0]["config_id"] = "other-config"
    return "".join(f"{_canonical(row)}\n" for row in rows)


def _apply_case(
    fixture: dict[str, Any], case: str
) -> tuple[dict[str, Any], str | None, bool, bool]:
    if case != "base" and case not in NEGATIVE_CASES:
        raise CapsuleError("unknown_case")
    changed = copy.deepcopy(fixture)
    skip, refresh, mutate = None, False, False
    if case in {"missing_row", "duplicate_row", "wrong_source_identity", "wrong_config_identity"}:
        changed["files"]["episodes.jsonl"] = _update_rows(changed["files"]["episodes.jsonl"], case)
        refresh = True
    elif case == "unsupported_schema":
        changed["files"]["capsule.json"]["schema_version"] = "cross_host_conformance_capsule.v2"
        refresh = True
    elif case == "stale_manifest":
        changed["artifact_manifest"]["files"][0]["content_sha256"] = "0" * 64
    elif case == "checksum_mismatch":
        mutate = True
    elif case == "path_escape":
        changed["files"]["../outside.json"] = "must not be written"
    elif case == "incomplete_copy":
        skip = "report.json"
    return changed, skip, refresh, mutate


def _owned_root(output_dir: Path) -> Path:
    root = output_dir.expanduser().resolve()
    if root.exists():
        raise CapsuleError("output_dir_exists")
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
        marker = _marker(root)
        (root / OWNERSHIP_MARKER).write_text(_canonical(marker) + "\n", encoding="utf-8")
    except OSError as exc:
        raise CapsuleError("output_dir_unusable") from exc
    return root


def _cleanup(root: Path) -> None:
    try:
        marker = _load(root / OWNERSHIP_MARKER)
    except CapsuleError as exc:
        raise CapsuleError("cleanup_ownership") from exc
    if marker != _marker(root) or root == Path("/") or root.is_symlink() or not root.is_dir():
        raise CapsuleError("cleanup_ownership")
    try:
        shutil.rmtree(root)
    except OSError as exc:
        raise CapsuleError("cleanup_failed") from exc


def _marker(root: Path) -> dict[str, str]:
    return {"schema": OWNERSHIP_SCHEMA, "owner": "restore_campaign_capsule_example", "root": str(root.resolve())}  # fmt: skip


def _restore(fixture: Mapping[str, Any], root: Path, skip: str | None) -> None:
    artifact = root / "artifact"
    artifact.mkdir()
    for raw, value in sorted(fixture["files"].items(), key=lambda item: str(item[0])):
        target = _safe(artifact, str(raw))
        if target.relative_to(artifact).as_posix() == skip:
            continue
        if not isinstance(value, (str, Mapping)):
            raise CapsuleError("invalid_fixture_file")
        target.parent.mkdir(parents=True, exist_ok=True)
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, indent=2) + "\n"  # fmt: skip
        target.write_text(text, encoding="utf-8")


def _manifest(root: Path, original: Mapping[str, Any]) -> dict[str, Any]:
    artifact = chunk_manifest.ArtifactIdentity(**dict(original["artifact"]))
    chunking = original["chunking"]
    policy = chunk_manifest.ChunkingPolicy(int(chunking["chunk_size_bytes"]), int(chunking["full_digest_threshold_bytes"]), tuple(chunking.get("exclude_patterns", ())))  # fmt: skip
    return chunk_manifest.build_manifest(root / "artifact", artifact=artifact, policy=policy)


def _write_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    path = root / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")  # fmt: skip
    return path


def _inventory(root: Path, manifest: Mapping[str, Any]) -> None:
    expected = sorted(str(entry["path"]) for entry in manifest["files"])
    actual = sorted(path.relative_to(root / "artifact").as_posix() for path in (root / "artifact").rglob("*") if path.is_file())  # fmt: skip
    if actual != expected:
        raise CapsuleError("missing_member" if set(expected) - set(actual) else "unexpected_member")


def _check(document: Mapping[str, Any], code: str, **fields: Any) -> None:
    if any(document.get(key) != value for key, value in fields.items()):
        raise CapsuleError(code)


def _checked(root: Path, name: str, code: str, **fields: Any) -> dict[str, Any]:
    document = _load(_member(root, name))
    _check(document, code, **fields)
    return document


def _validate_rows(root: Path, expected: Mapping[str, Any], spec: Mapping[str, Any], config_sha: str) -> tuple[dict[str, Any], list[str]]:  # fmt: skip
    path = _member(root, "episodes.jsonl")
    schema_path = (REPO_ROOT / str(spec["scenario"]["schema_path"])).resolve()
    _require(
        schema_path == (REPO_ROOT / expected["episode_schema_path"]).resolve(),
        "schema_path_mismatch",
    )
    try:
        rows = read_jsonl(path, strict=True)
        schema = load_schema(schema_path)
        for row in rows:
            _require(isinstance(row, dict), "invalid_row")
            validate_episode(row, schema)
    except CapsuleError:
        raise
    except Exception as exc:
        raise CapsuleError("row_schema_invalid") from exc
    ids = [str(row.get("episode_id")) for row in rows]
    _require(len(ids) == len(set(ids)), "duplicate_row")
    _require(len(rows) == expected["expected_row_count"], "row_count_mismatch")
    _require(ids == expected["row_ids"], "row_identity_mismatch")
    statuses = [str(row.get("row_status", "")) for row in rows]
    _require(set(statuses) <= set(spec["expected_structure"]["allowed_row_statuses"]), "row_status_invalid")  # fmt: skip
    for row in rows:
        for key, code in {
            "campaign_id": "campaign_identity_mismatch",
            "config_id": "config_identity_mismatch",
            "source_commit": "source_identity_mismatch",
            "seed": "source_identity_mismatch",
        }.items():
            _check(row, code, **{key: expected[key]})
        _check(row, "row_boundary_or_digest_invalid", config_sha256=config_sha, evidence_status="diagnostic_only", evidence_admissible=False)  # fmt: skip
    digests = [hashlib.sha256(f"{_canonical(row)}\n".encode()).hexdigest() for row in rows]
    return {"expected_count": expected["expected_row_count"], "observed_count": len(rows), "unique_count": len(set(ids)), "row_ids": ids, "row_status_counts": dict(sorted(Counter(statuses).items()))}, digests  # fmt: skip


def _validate_lineage(root: Path, expected: Mapping[str, Any], config_sha: str, report_sha: str, row_sha: Sequence[str]) -> dict[str, Any]:  # fmt: skip
    path = _member(root, "lineage.json")
    try:
        payload = lineage_index.build_index([path])
    except Exception as exc:
        raise CapsuleError("lineage_unreadable") from exc
    _require(payload["ok"], str(payload["findings"][0]["code"]) if payload["findings"] else "lineage_invalid")  # fmt: skip
    matches = [row for row in payload["rows"] if expected["campaign_id"] in row["records"]["campaign_ids"]]  # fmt: skip
    _require(len(matches) == 1, "lineage_campaign_mismatch")
    joined = matches[0]["records"]["artifact_ids"]
    row_ids = sorted(item for item in joined if item.startswith("row-"))
    report_ids = sorted(item for item in joined if item.startswith("report-"))
    _require(row_ids == sorted(expected["lineage_row_artifact_ids"]), "lineage_row_inventory_mismatch")  # fmt: skip
    _require(report_ids == [expected["lineage_report_artifact_id"]], "lineage_report_inventory_mismatch")  # fmt: skip
    records = _load(path).get("records", [])
    by_key = {f"{item['kind']}:{item['id']}": item for item in records}
    _check(by_key.get(f"config:{expected['config_id']}", {}), "lineage_config_digest_mismatch", digest=config_sha)  # fmt: skip
    for artifact_id, digest in zip(row_ids, row_sha, strict=True):
        _check(by_key.get(f"artifact:{artifact_id}", {}), "lineage_row_digest_mismatch", digest=digest)  # fmt: skip
    _check(by_key.get(f"artifact:{report_ids[0]}", {}), "lineage_report_digest_mismatch", digest=report_sha)  # fmt: skip
    return {"status": "ok", "campaign_match_count": 1, "row_artifact_count": len(row_ids), "report_artifact_count": 1}  # fmt: skip


def _base(case: str, status: str) -> dict[str, Any]:
    return {"schema": SUMMARY_SCHEMA, "case": case, "status": status, "execution_status": "not_run", "evidence_status": "diagnostic_only", "synthetic": True, "promotable": False, "claim_boundary": CLAIM_BOUNDARY}  # fmt: skip


def _failure(case: str, error: CapsuleError, cleanup: Mapping[str, Any]) -> dict[str, Any]:
    return {**_base(case, "failed"), "error": {"code": error.code}, "cleanup": dict(cleanup)}


def summarize_capsule(capsule_path: str | Path, output_dir: str | Path, *, case: str = "base") -> dict[str, Any]:  # fmt: skip
    """Restore one fixture capsule and return a deterministic validation summary."""
    try:
        fixture = _load(Path(capsule_path))
        _validate_fixture(fixture)
        fixture, skip, refresh, mutate = _apply_case(fixture, case)
        root = _owned_root(Path(output_dir))
    except CapsuleError as exc:
        return _failure(case, exc, {"status": "not_created", "ownership_verified": False})
    try:
        _restore(fixture, root, skip)
        manifest = _manifest(root, fixture["artifact_manifest"]) if refresh else dict(fixture["artifact_manifest"])  # fmt: skip
        manifest_path = _write_manifest(root, manifest)
        if mutate:
            path = root / "artifact/config.json"
            data = path.read_bytes()
            path.write_bytes(data[:-1] + b" " if data.endswith(b"\n") else data + b" ")
        _inventory(root, manifest)
        try:
            verification = chunk_manifest.verify_manifest(root / "artifact", manifest=chunk_manifest.load_manifest_file(manifest_path))  # fmt: skip
        except chunk_manifest.ChunkManifestError as exc:
            raise CapsuleError(exc.code) from exc
        if verification["status"] != "ok":
            raise CapsuleError(str(verification["failures"][0]["code"]))
        expected = fixture["expected"]
        capsule_path = _member(root, "capsule.json")
        capsule = _load(capsule_path)
        try:
            spec = cross_host_conformance_capsule.load_capsule_spec(capsule_path)
        except cross_host_conformance_capsule.CapsuleContractError as exc:
            raise CapsuleError("unsupported_schema") from exc
        _check(spec, "capsule_identity_mismatch", capsule_id=expected["capsule_id"])
        digest = cross_host_conformance_capsule.capsule_digest(cross_host_conformance_capsule.resolve_capsule(spec, expected["source_commit"]))  # fmt: skip
        _check(capsule, "source_identity_mismatch", source_commit=expected["source_commit"])
        _check(expected, "source_identity_mismatch", capsule_digest=digest)
        config_path = _member(root, "config.json")
        config = _load(config_path)
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        _check(config, "config_identity_mismatch", schema=CONFIG_SCHEMA, config_id=expected["config_id"], source_commit=expected["source_commit"], seed=expected["seed"])  # fmt: skip
        campaign = _checked(root, "campaign.json", "campaign_identity_mismatch", schema=CAMPAIGN_SCHEMA, campaign_id=expected["campaign_id"], capsule_digest=digest, config_sha256=config_sha, row_ids=expected["row_ids"])  # fmt: skip
        _check(campaign, "campaign_identity_mismatch", **{key: expected[key] for key in ("capsule_id", "config_id", "source_commit", "seed")})  # fmt: skip
        rows, row_sha = _validate_rows(root, expected, spec, config_sha)
        report = _checked(root, "report.json", "report_identity_mismatch", schema=REPORT_SCHEMA, campaign_id=expected["campaign_id"], config_sha256=config_sha, status="complete", row_count=expected["expected_row_count"], row_ids=expected["row_ids"], claim_boundary=CLAIM_BOUNDARY)  # fmt: skip
        _checked(root, "missingness.json", "missingness_mismatch", schema=MISSINGNESS_SCHEMA, campaign_id=expected["campaign_id"], expected_row_count=rows["expected_count"], observed_row_count=rows["observed_count"], missing_row_ids=[], excluded_row_ids=[], claim_boundary=CLAIM_BOUNDARY)  # fmt: skip
        _check(report, "report_identity_mismatch", **{key: expected[key] for key in ("capsule_id", "source_commit", "config_id", "seed")})  # fmt: skip
        report_sha = hashlib.sha256(_member(root, "report.json").read_bytes()).hexdigest()
        lineage = _validate_lineage(root, expected, config_sha, report_sha, row_sha)
        result = {**_base(case, "verified"), "campaign": {"campaign_id": expected["campaign_id"], "capsule_id": expected["capsule_id"], "capsule_digest": digest, "source_commit": expected["source_commit"], "config_id": expected["config_id"], "config_sha256": config_sha, "seed": expected["seed"]}, "restore": {"status": "verified", "member_count": len(manifest["files"]), "manifest_id": verification["manifest_id"], "tree_sha256": verification["tree_sha256"]}, "validation": {"schema_versions": {"capsule": spec["schema_version"], "config": CONFIG_SCHEMA, "episode": "v1", "lineage_input": lineage_index.INPUT_SCHEMA, "lineage_index": lineage_index.INDEX_SCHEMA, "missingness": MISSINGNESS_SCHEMA, "report": report["schema"]}, "checksums": {"members": "passed", "semantic": "passed"}, "row_ids": rows["row_ids"], "report": {"status": report["status"], "row_count": report["row_count"]}, "lineage": lineage}, "summary": {"members": len(manifest["files"]), "rows": rows["observed_count"], "missing_rows": 0, "row_status_counts": rows["row_status_counts"]}}  # fmt: skip
        _cleanup(root)
        result["cleanup"] = {
            "status": "removed",
            "ownership_verified": True,
            "marker": OWNERSHIP_MARKER,
        }
        return result
    except (CapsuleError, chunk_manifest.ChunkManifestError, OSError) as exc:
        error = (
            exc
            if isinstance(exc, CapsuleError)
            else CapsuleError(getattr(exc, "code", "restore_io"))
        )
        try:
            _cleanup(root)
            cleanup = {"status": "removed", "ownership_verified": True, "marker": OWNERSHIP_MARKER}
        except CapsuleError:
            cleanup = {
                "status": "preserved",
                "ownership_verified": False,
                "marker": OWNERSHIP_MARKER,
            }
        return _failure(case, error, cleanup)


def _text(summary: Mapping[str, Any]) -> str:
    if summary["status"] == "failed":
        return (
            f"capsule: failed ({summary['error']['code']})\ncleanup: {summary['cleanup']['status']}"
        )
    return "\n".join(
        (
            "capsule: verified (synthetic diagnostic-only)",
            f"campaign: {summary['campaign']['campaign_id']}",
            f"members: {summary['restore']['member_count']}",
            f"rows: {summary['summary']['rows']} observed / {summary['summary']['missing_rows']} missing",
            f"lineage: {summary['validation']['lineage']['row_artifact_count']} rows + 1 report",
            f"execution: {summary['execution_status']}",
            f"cleanup: {summary['cleanup']['status']} (marker verified)",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline campaign-capsule example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Fresh task-owned restore root."
    )
    parser.add_argument("--case", default="base", choices=("base", *NEGATIVE_CASES))
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON.")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="robot-sf-offline-capsule-") as temp:
            result = summarize_capsule(args.capsule, Path(temp) / "restore", case=args.case)
    else:
        result = summarize_capsule(args.capsule, args.output_dir, case=args.case)
    print(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
        if args.json
        else _text(result)
    )
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
