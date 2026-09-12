#!/usr/bin/env python3
"""Package and verify host-independent campaign analysis capsules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

DEFAULT_SCHEMA_PATH = Path("docs/contracts/campaign_analysis_capsule.v1.schema.json")
MUTABLE_URI_RE = re.compile(
    r":latest\b|/latest(/|$)|@latest\b|\bmaster\b|\bmain\b|\bhead\b", re.IGNORECASE
)
PRIVATE_PATH_RE = re.compile(
    r"(?:^|[\s\"'=])(/(?:home|tmp|var|opt|private|Users|root)/|[a-zA-Z]:[/\\])"
)
SCHEDULER_RE = re.compile(
    r"\bSLURM_[A-Z_]+\b|\bslurm-[0-9]+\.out\b|\b[\w-]+\.(?:cluster|local|internal)\b"
)
SIBLING_IMPORT_RE = re.compile(
    r"sys\.path\.(?:insert|append)|\bsite-packages\b|\bpip\s+install\s+-e\b|\.\./"
)
TEXT_EXTS = frozenset({".py", ".json", ".jsonl", ".yaml", ".yml", ".sh", ".md", ".txt", ".lock"})


def sha256_file(path: Path) -> str:
    """Compute sha256 hex digest of file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_code_digest(capsule_root: Path) -> str:
    """Compute collective sha256 of code files in capsule."""
    code_dir = capsule_root / "code"
    if not code_dir.is_dir():
        return ""
    files = sorted(p for p in code_dir.rglob("*") if p.is_file() and not p.name.startswith("."))
    if not files:
        return ""
    h = hashlib.sha256()
    for f in files:
        rel = f.relative_to(capsule_root).as_posix()
        h.update(f"{rel}:{sha256_file(f)}\n".encode())
    return h.hexdigest()


def _read_json_or_yaml(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unpack_bundle_if_needed(target: Path, temp_dir: Path) -> Path:
    """If target is a JSON bundle file, unpack into temp_dir."""
    if target.is_file() and target.suffix == ".json":
        data = _read_json_or_yaml(target)
        if data.get("schema") == "campaign_analysis_capsule_bundle.v1":
            temp_dir.mkdir(parents=True, exist_ok=True)
            manifest = data["manifest"]
            (temp_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            for rel_path, content in data.get("files", {}).items():
                dest = temp_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            return temp_dir
    if target.is_file() and target.name == "manifest.json":
        return target.parent
    return target


def write_sha256sums(root: Path) -> None:
    """Generate SHA256SUMS for all files in root except SHA256SUMS itself."""
    sums: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            rel = p.relative_to(root).as_posix()
            sums.append(f"{sha256_file(p)}  {rel}\n")
    (root / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")


def _check_manifest_portability(manifest: dict[str, Any]) -> list[str]:
    """Check manifest fields for private paths, scheduler state, and mutable URIs."""
    discrepancies: list[str] = []
    m_text = json.dumps(manifest)
    if match := PRIVATE_PATH_RE.search(m_text):
        discrepancies.append(
            f"hidden_absolute_path: private path detected in manifest: {match.group(1)}"
        )
    if match := SCHEDULER_RE.search(m_text):
        discrepancies.append(
            f"source_host_dependency: scheduler/host artifact in manifest: {match.group(0)}"
        )
    for df in manifest.get("data_files", []):
        uri = df.get("durable_uri")
        if uri and MUTABLE_URI_RE.search(uri):
            discrepancies.append(
                f"mutable_artifact_alias: durable URI '{uri}' contains mutable tag/alias"
            )
    return discrepancies


def _check_file_portability(root: Path) -> list[str]:
    """Scan text files in capsule for absolute paths, scheduler state, and sibling imports."""
    discrepancies: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT_EXTS or p.name == "SHA256SUMS":
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        if match := PRIVATE_PATH_RE.search(content):
            discrepancies.append(
                f"hidden_absolute_path: {rel} contains absolute path: {match.group(1)}"
            )
        if match := SCHEDULER_RE.search(content):
            discrepancies.append(
                f"source_host_dependency: {rel} contains scheduler artifact: {match.group(0)}"
            )
        if p.suffix == ".py" and (match := SIBLING_IMPORT_RE.search(content)):
            discrepancies.append(
                f"editable_sibling_import: {rel} contains non-portable import: {match.group(0)}"
            )
    return discrepancies


def _parse_row_records(path: Path) -> list[dict[str, Any]]:
    """Parse rows from json or jsonl."""
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "rows" in data:
            return data["rows"]
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def check_rows(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Check data files for missing/duplicate rows."""
    discrepancies: list[str] = []
    seen_ids: set[str] = set()
    total_rows = 0

    for df in manifest.get("data_files", []):
        rel = df["path"]
        p = root / rel
        if not p.is_file():
            discrepancies.append(f"missing_row_file: data file '{rel}' does not exist")
            continue
        if df.get("sha256") and sha256_file(p) != df["sha256"]:
            discrepancies.append(f"sha256_mismatch: data file '{rel}' hash mismatch")
        if p.suffix in (".json", ".jsonl"):
            try:
                rows = _parse_row_records(p)
                for row in rows:
                    total_rows += 1
                    rid = str(row.get("row_id", ""))
                    if rid:
                        if rid in seen_ids:
                            discrepancies.append(
                                f"duplicate_rows: duplicate row_id '{rid}' in {rel}"
                            )
                        seen_ids.add(rid)
            except (ValueError, KeyError, TypeError, OSError) as e:
                discrepancies.append(f"row_parse_error: failed parsing {rel}: {e}")

    exp_count = manifest.get("expected_row_count")
    if exp_count is not None and total_rows < exp_count:
        discrepancies.append(f"missing_rows: expected {exp_count} rows, found {total_rows}")
    return discrepancies


def _check_manifest_schema(manifest: dict[str, Any], schema_path: Path) -> list[str]:
    """Validate manifest against Draft 2020-12 schema."""
    discrepancies: list[str] = []
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        for err in validator.iter_errors(manifest):
            discrepancies.append(
                f"schema_validation: {'/'.join(map(str, err.path))}: {err.message}"
            )
    else:
        discrepancies.append(f"missing_schema: validation schema file not found: {schema_path}")
    return discrepancies


def _check_sha256sums(capsule_root: Path) -> list[str]:
    """Verify SHA256SUMS file existence and entry hashes."""
    sums_file = capsule_root / "SHA256SUMS"
    if not sums_file.is_file():
        return ["missing_sha256sums: SHA256SUMS file missing from capsule"]

    discrepancies: list[str] = []
    sum_entries: dict[str, str] = {}
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            expected_sha = parts[0]
            target_rel = parts[1].lstrip("*").strip()
            if target_rel == "SHA256SUMS":
                continue
            sum_entries[target_rel] = expected_sha
            tf = capsule_root / target_rel
            if not tf.is_file():
                discrepancies.append(
                    f"missing_file_in_sums: {target_rel} listed in SHA256SUMS but missing"
                )
            elif sha256_file(tf) != expected_sha:
                discrepancies.append(
                    f"sha256_mismatch: {target_rel} sha256 does not match SHA256SUMS"
                )
    return discrepancies


def _check_code_and_deps(capsule_root: Path, manifest: dict[str, Any]) -> list[str]:
    """Verify code digest and dependency lock digest."""
    discrepancies: list[str] = []
    actual_code_digest = compute_code_digest(capsule_root)
    exp_code_digest = manifest.get("analysis_code_digest", "")
    if exp_code_digest and actual_code_digest != exp_code_digest:
        discrepancies.append(
            f"unbound_analysis_code: analysis code digest mismatch (stale code: expected {exp_code_digest}, got {actual_code_digest})"
        )

    lock_file = capsule_root / "dependencies" / "uv.lock"
    if not lock_file.is_file():
        lock_file = capsule_root / "uv.lock"
    if lock_file.is_file():
        actual_dep_digest = sha256_file(lock_file)
        exp_dep_digest = manifest.get("dependency_digest", "")
        if exp_dep_digest and actual_dep_digest != exp_dep_digest:
            discrepancies.append(
                f"dependency_mismatch: dependency digest mismatch (expected {exp_dep_digest}, got {actual_dep_digest})"
            )
    return discrepancies


def _regenerate_reports(
    capsule_root: Path, manifest: dict[str, Any], output_dir: Path
) -> tuple[list[str], list[str]]:
    """Regenerate deterministic reports into output_dir and verify outputs."""
    discrepancies: list[str] = []
    regenerated_files: list[str] = []
    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    for exp in manifest.get("output_expectations", []):
        if (resolved_output / exp["path"]).exists():
            discrepancies.append(
                f"output_overwrite: output file '{exp['path']}' already exists in output root"
            )

    for cmd in manifest.get("report_commands", []):
        if cmd.get("status") != "available":
            continue
        raw_tokens = cmd.get("command_tokens", [])
        tokens = [t.replace("{OUTPUT_DIR}", resolved_output.as_posix()) for t in raw_tokens]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": capsule_root.as_posix(),
            "OUTPUT_DIR": resolved_output.as_posix(),
        }
        res = subprocess.run(
            tokens, cwd=capsule_root, env=env, capture_output=True, text=True, check=False
        )
        if res.returncode != 0:
            discrepancies.append(
                f"report_command_failed: command '{cmd.get('name')}' exited {res.returncode}: {res.stderr[:200]}"
            )

    for exp in manifest.get("output_expectations", []):
        out_path = resolved_output / exp["path"]
        if not out_path.is_file():
            discrepancies.append(
                f"missing_expected_output: output '{exp['path']}' was not generated"
            )
        else:
            regenerated_files.append(exp["path"])
            exp_sha = exp.get("expected_sha256")
            if exp_sha and sha256_file(out_path) != exp_sha:
                discrepancies.append(
                    f"deterministic_output_mismatch: {exp['path']} expected sha {exp_sha}, got {sha256_file(out_path)}"
                )
    return discrepancies, regenerated_files


def verify_capsule(
    capsule_root: Path,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    regenerate: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full verification on unpacked capsule directory."""
    manifest_file = capsule_root / "manifest.json"
    if not manifest_file.is_file():
        return {
            "schema": "campaign_analysis_capsule_verification.v1",
            "status": "fail",
            "verdict": "fail",
            "discrepancies": ["missing_manifest: manifest.json not found"],
            "verified_at_utc": datetime.now(UTC).isoformat(),
        }

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    discrepancies: list[str] = []
    discrepancies.extend(_check_manifest_schema(manifest, schema_path))

    inp_schema = manifest.get("input_schema_version")
    if inp_schema:
        schema_file = capsule_root / "schemas" / f"{inp_schema}.schema.json"
        if not schema_file.is_file():
            schema_file = Path("docs/contracts") / f"{inp_schema}.schema.json"
        if not schema_file.is_file():
            discrepancies.append(f"missing_schema: referenced schema '{inp_schema}' not found")

    discrepancies.extend(_check_sha256sums(capsule_root))
    discrepancies.extend(_check_code_and_deps(capsule_root, manifest))
    discrepancies.extend(_check_manifest_portability(manifest))
    discrepancies.extend(_check_file_portability(capsule_root))
    discrepancies.extend(check_rows(capsule_root, manifest))

    for cmd in manifest.get("report_commands", []):
        if cmd.get("status") in ("unavailable", "unsupported") and not cmd.get("reason"):
            discrepancies.append(
                f"unsupported_analysis_missing_reason: command '{cmd.get('name', 'unnamed')}' status is {cmd.get('status')} but reason is missing"
            )

    regenerated_files: list[str] = []
    if regenerate:
        if not output_dir:
            discrepancies.append(
                "report_regeneration: --output-dir required for report regeneration"
            )
        else:
            regen_disc, regenerated_files = _regenerate_reports(capsule_root, manifest, output_dir)
            discrepancies.extend(regen_disc)

    verdict = "pass" if not discrepancies else "fail"
    return {
        "schema": "campaign_analysis_capsule_verification.v1",
        "capsule_id": manifest.get("capsule_id"),
        "campaign_id": manifest.get("campaign_id"),
        "status": verdict,
        "verdict": verdict,
        "discrepancies": discrepancies,
        "regenerated_outputs": regenerated_files,
        "verified_at_utc": datetime.now(UTC).isoformat(),
    }


def build_capsule(spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Build a complete capsule into output_dir from spec."""
    resolved_out = output_dir.resolve()
    resolved_out.mkdir(parents=True, exist_ok=True)
    data_dir = resolved_out / "data"
    code_dir = resolved_out / "code"
    schemas_dir = resolved_out / "schemas"
    dep_dir = resolved_out / "dependencies"

    for d in (data_dir, code_dir, schemas_dir, dep_dir):
        d.mkdir(parents=True, exist_ok=True)

    data_files_meta: list[dict[str, Any]] = []
    for src in spec.get("data_sources", []):
        src_path = Path(src["source"])
        dest_rel = f"data/{src_path.name}"
        dest_path = resolved_out / dest_rel
        shutil.copy2(src_path, dest_path)
        data_files_meta.append(
            {
                "path": dest_rel,
                "sha256": sha256_file(dest_path),
                "byte_size": dest_path.stat().st_size,
                "row_count": src.get("row_count"),
                "durable_uri": src.get("durable_uri"),
            }
        )

    for src in spec.get("code_sources", []):
        src_path = Path(src)
        shutil.copy2(src_path, code_dir / src_path.name)

    for src in spec.get("schema_sources", []):
        src_path = Path(src)
        shutil.copy2(src_path, schemas_dir / src_path.name)

    lock_src = spec.get("dependency_lock")
    if lock_src:
        shutil.copy2(Path(lock_src), dep_dir / "uv.lock")
        dep_digest = sha256_file(dep_dir / "uv.lock")
    else:
        dep_digest = hashlib.sha256(b"").hexdigest()

    code_digest = compute_code_digest(resolved_out)

    manifest: dict[str, Any] = {
        "schema": "campaign_analysis_capsule.v1",
        "capsule_id": spec["capsule_id"],
        "campaign_id": spec["campaign_id"],
        "source_commit": spec["source_commit"],
        "analysis_code_digest": code_digest,
        "dependency_digest": dep_digest,
        "input_schema_version": spec.get("input_schema_version"),
        "expected_row_digest": spec.get("expected_row_digest", hashlib.sha256(b"").hexdigest()),
        "expected_row_count": spec.get("expected_row_count"),
        "deterministic_seeds": spec.get("deterministic_seeds", []),
        "claim_boundary": spec.get(
            "claim_boundary", "Host-independent analysis verification; no simulation rerun implied."
        ),
        "data_files": data_files_meta,
        "report_commands": spec.get("report_commands", []),
        "output_expectations": spec.get("output_expectations", []),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    manifest_path = resolved_out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_sha256sums(resolved_out)

    return verify_capsule(resolved_out)


def format_summary(res: dict[str, Any]) -> str:
    """Format human-readable verification summary."""
    lines = [
        f"Capsule: {res.get('capsule_id')} (campaign: {res.get('campaign_id')})",
        f"Status: {res.get('status', '').upper()} | Verdict: {res.get('verdict', '').upper()}",
    ]
    for d in res.get("discrepancies", []):
        lines.append(f"  Discrepancy: {d}")
    for out in res.get("regenerated_outputs", []):
        lines.append(f"  Regenerated: {out}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for campaign analysis capsule packager and verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, help="Path to capsule directory or bundle JSON")
    parser.add_argument(
        "--check", action="store_true", help="Fail closed on verification discrepancy"
    )
    parser.add_argument("--build", action="store_true", help="Build capsule from spec")
    parser.add_argument("--spec", type=Path, help="Capsule build specification JSON")
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory for build or report regeneration"
    )
    parser.add_argument(
        "--regenerate-reports",
        action="store_true",
        help="Execute report commands and verify outputs",
    )
    parser.add_argument("--format", choices=["json", "summary"], default="json")
    parser.add_argument("--output", type=Path, help="Write verification receipt to file")
    args = parser.parse_args(argv)

    if args.build:
        if not args.spec or not args.output_dir:
            print("ERROR: --build requires --spec and --output-dir", file=sys.stderr)
            return 1
        spec_data = _read_json_or_yaml(args.spec)
        res = build_capsule(spec_data, args.output_dir)
    elif args.capsule:
        with tempfile.TemporaryDirectory() as td:
            temp_path = Path(td)
            capsule_root = unpack_bundle_if_needed(args.capsule, temp_path)
            res = verify_capsule(
                capsule_root,
                regenerate=args.regenerate_reports,
                output_dir=args.output_dir,
            )
    else:
        parser.print_help()
        return 1

    text = (
        json.dumps(res, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else format_summary(res) + "\n"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    return 2 if (args.check and res.get("verdict") != "pass") else 0


if __name__ == "__main__":
    sys.exit(main())
