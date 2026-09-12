#!/usr/bin/env python3
"""Fail-closed source-host artifact prune eligibility guard (#8846).

Consumes a source artifact/harvest manifest and a destination transfer custody receipt,
verifying durable destination custody, checksums, consumer coverage, and retention
dispositions before producing a deterministic deletion plan.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.identity.hash_utils import sha256_file  # noqa: E402
from scripts.tools.chunk_manifest import ChunkManifestError, normalize_relative_path  # noqa: E402

REPORT_SCHEMA = "robot_sf.prune_eligibility_report.v1"
EXIT_ELIGIBLE, EXIT_BLOCKED, EXIT_MALFORMED = 0, 2, 3

CLASSIFICATIONS = (
    "retain_required",
    "eligible_after_review",
    "eligible_verified",
    "blocked_missing_destination",
    "blocked_checksum",
    "blocked_active_consumer",
    "blocked_unknown_owner",
    "not_managed",
)
APPROVED_DURABLE = frozenset(
    {"cloud_durable", "public_release", "personal_durable", "durable", "verified_durable"}
)
DISPOSABLE = frozenset(
    {"disposable", "ignored-cache", "durable-required", "durable_required", "eligible_verified"}
)
REVIEW = frozenset({"eligible_after_review", "handoff-needed", "conditional"})
RETAIN = frozenset(
    {"retain_required", "tracked-manifest", "historical", "release-facing", "release_facing"}
)
ALLOWED_RETENTION = DISPOSABLE | REVIEW | RETAIN
ACTIVE_STATES = frozenset({"active", "running", "open", "queued"})
MUTABLE_SUFFIXES = (":latest", "/latest", ":head", "/head", ":main", "/main")


def _is_expired(val: Any, uri: str | None = None) -> bool:
    now = datetime.now(UTC).timestamp()
    if isinstance(val, (int, float)):
        return val < now
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp() < now
        except ValueError:
            pass
    return bool(uri and (m := re.search(r"[?&]Expires=([0-9]+)", uri)) and float(m.group(1)) < now)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON object safely or raise ValueError."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Root JSON element must be an object")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Failed to load JSON from {path}: {exc}") from exc


def extract_members(m: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract owner and declared member records from source manifest."""
    owner = m.get("owner") or m.get("ownership", {}).get("owner")
    raw = m.get("inventory") or m.get("members") or m.get("files") or m.get("artifacts") or []
    members = []
    for it in raw:
        if isinstance(it, Mapping) and (p := it.get("relative_path") or it.get("path")):
            ret = str(it.get("retention_class") or it.get("disposition") or "unspecified")
            ow = str(it.get("owner") or owner) if (it.get("owner") or owner) else None
            members.append(
                {
                    "relative_path": str(p),
                    "sha256": str(it["sha256"]) if it.get("sha256") else None,
                    "byte_size": int(it.get("byte_size", it.get("size", 0))),
                    "retention_class": ret,
                    "owner": ow,
                    "regenerable": bool(it.get("regenerable", False)),
                    "regeneration_verified": bool(it.get("regeneration_verified", False)),
                }
            )
    return owner, members


def scan_source_disk(source_root: Path) -> tuple[dict[str, Path], list[str]]:
    """Scan source root directory for existing files."""
    found, problems = {}, []
    if not source_root.exists() or not source_root.is_dir():
        return found, ["source_root is not an existing directory"]
    for root_dir, _dirs, files in os.walk(source_root, followlinks=False):
        for name in files:
            p = Path(root_dir) / name
            try:
                found[normalize_relative_path(p.relative_to(source_root).as_posix())] = p
            except ChunkManifestError as exc:
                problems.append(f"{p.name}: {exc}")
    return found, problems


def _extract_inputs(
    src_path: Path, dst_path: Path, c_path: Path | None, d_path: Path | None, case_name: str | None
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    src = load_json(src_path)
    dst = load_json(dst_path) if dst_path.is_file() else {}
    c_data = load_json(c_path) if c_path else None
    d_data = load_json(d_path) if d_path else None
    if src.get("schema") == "prune_eligibility_cases.v1" or "cases" in src:
        raw_c = src.get("cases", {})
        cases = (
            {c.get("case_id", f"case_{i}"): c for i, c in enumerate(raw_c)}
            if isinstance(raw_c, list)
            else raw_c
        )
        case = cases.get(case_name) if case_name else next(iter(cases.values()), None)
        if case:
            src = case.get("source_manifest", case)
            dst = case.get("destination_receipt", dst)
            c_data = case.get("consumers", c_data)
            d_data = case.get("dispositions", d_data)

    dispositions = {}
    if isinstance(d_data, Mapping):
        raw = d_data.get("dispositions", d_data)
        if isinstance(raw, Mapping):
            dispositions = {
                k: (v if isinstance(v, str) else v.get("disposition", "unknown"))
                for k, v in raw.items()
            }
        elif isinstance(raw, list):
            dispositions = {
                it["path"]: it.get("disposition", "unknown")
                for it in raw
                if isinstance(it, Mapping) and "path" in it
            }

    consumers = []
    if isinstance(c_data, list):
        consumers.extend(c_data)
    elif isinstance(c_data, Mapping):
        for g in (
            "consumers",
            "active_tasks",
            "configs",
            "scripts",
            "reports",
            "releases",
            "papers",
        ):
            v = c_data.get(g, [])
            consumers.extend(
                v if isinstance(v, list) else (v.values() if isinstance(v, Mapping) else [])
            )
    if isinstance(src.get("consumers"), list):
        consumers.extend(src["consumers"])
    return src, dst, consumers, dispositions


def _err(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _rec(p: str, c: str, s: str | None, b: int, r: str, st: str, rs: list[str]) -> dict[str, Any]:
    d = {"path": p, "classification": c, "sha256": s, "byte_size": b}
    d.update({"retention_class": r, "source_state": st, "reasons": rs})
    return d


def check_prune_eligibility(  # noqa: C901, PLR0912, PLR0915
    source_manifest_path: Path,
    destination_receipt_path: Path,
    *,
    source_root: Path | None = None,
    consumers_path: Path | None = None,
    dispositions_path: Path | None = None,
    case_name: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Check prune eligibility for manifest-owned artifacts."""
    src, dst, consumers, dispositions = _extract_inputs(
        source_manifest_path, destination_receipt_path, consumers_path, dispositions_path, case_name
    )
    src_digest, (owner, members) = sha256_file(source_manifest_path), extract_members(src)
    m_id = src.get("manifest_id") or src.get("receipt_id")

    rejections: list[dict[str, str]] = []
    if dst.get("status") != "verified":
        rejections.append(_err("destination_not_verified", f"status '{dst.get('status')}'"))
    durability = dst.get("durability_class", "unspecified")
    if durability not in APPROVED_DURABLE:
        rejections.append(
            _err("unapproved_durability_class", f"durability '{durability}' unapproved")
        )
    dest_obj = dst.get("destination", {})
    indep = dst.get("independent_verification", False) or dest_obj.get(
        "independent_verification", False
    )
    vbasis = dst.get("verification_basis") or dest_obj.get("verification_basis")
    if not indep and not (
        dst.get("status") == "verified" and vbasis in ("checksum_receipt", "manifest_verification")
    ):
        rejections.append(
            _err("missing_independent_verification", "lacks independent verification")
        )
    loc = (
        dst.get("destination_locator") or dst.get("locator") or dest_obj.get("destination_locator")
    )
    if loc and any(loc.endswith(sfx) for sfx in MUTABLE_SUFFIXES):
        rejections.append(_err("mutable_destination_rejected", f"locator '{loc}' mutable"))
    if dst.get("expired") is True or _is_expired(
        dst.get("expires_at") or dest_obj.get("expires_at"), loc
    ):
        rejections.append(_err("expired_destination_uri", "destination access window expired"))
    if dst.get("partial_transfer") is True or (
        dest_obj.get("partial_members_cleaned", 0) > 0 and dst.get("status") != "verified"
    ):
        rejections.append(_err("partial_transfer_detected", "interrupted partial transfer"))

    dest_files = {}
    for df in dst.get("files", []):
        if isinstance(df, Mapping) and (p := df.get("relative_path") or df.get("path")):
            try:
                norm = normalize_relative_path(str(p))
                dest_files[norm] = df
                if norm.endswith(".transfer-partial"):
                    rejections.append(_err("partial_transfer_detected", f"partial file '{norm}'"))
            except ChunkManifestError:
                pass

    dm = dst.get("manifest", {})
    if dm.get("manifest_digest") not in (None, "any", src_digest, src.get("receipt_id")):
        rejections.append(_err("manifest_identity_mismatch", "digest mismatch"))
    if dm.get("member_count") not in (None, len(members)):
        rejections.append(_err("member_count_mismatch", "member count mismatch"))
    if dm.get("total_bytes") not in (None, sum(m["byte_size"] for m in members)):
        rejections.append(_err("total_bytes_mismatch", "total bytes mismatch"))

    writers = src.get("writers", [])
    active_writer = src.get("active_job") is True or bool(src.get("active_writers"))
    active_writer = active_writer or any(
        isinstance(w, Mapping) and w.get("state") in ("active", "running") for w in writers
    )
    active_writer = active_writer or bool(
        source_root and (source_root / ".gate_lease.json").exists()
    )

    disk_files, disk_errs = scan_source_disk(source_root) if source_root else ({}, [])
    rejections.extend(_err("source_disk_problem", de) for de in disk_errs)

    classifications, manifest_paths = [], set()
    for m in members:
        raw_p = m["relative_path"]
        try:
            rel = normalize_relative_path(raw_p)
        except ChunkManifestError as exc:
            classifications.append(
                _rec(
                    raw_p,
                    "blocked_checksum",
                    m.get("sha256"),
                    0,
                    "unspecified",
                    "invalid_path",
                    [str(exc)],
                )
            )
            continue

        manifest_paths.add(rel)
        cls, reasons = None, []

        m_owner = m.get("owner") or owner
        if not m_owner or m_owner in ("unknown", "unspecified"):
            cls, reasons = "blocked_unknown_owner", ["unspecified owner"]
        elif active_writer:
            cls, reasons = "blocked_active_consumer", ["active writer or lease"]
        else:
            for c in consumers:
                if c.get("state", c.get("status")) in ACTIVE_STATES:
                    refs = c.get("refs", c.get("references", []))
                    ref_ids = {
                        r if isinstance(r, str) else r.get("logical_id", r.get("path"))
                        for r in (refs if isinstance(refs, list) else refs.keys())
                    }
                    if {rel, m.get("sha256"), m_id} & ref_ids and not (
                        c.get("points_to_destination")
                        or c.get("destination_verified")
                        or (m.get("regenerable") and m.get("regeneration_verified"))
                    ):
                        cid = c.get("id", c.get("consumer_id", "unnamed"))
                        cls, reasons = (
                            "blocked_active_consumer",
                            [f"active consumer '{cid}' points to source"],
                        )
                        break

        if not cls and rejections:
            has_sum = any(
                r["code"] in ("manifest_identity_mismatch", "total_bytes_mismatch")
                for r in rejections
            )
            cls = "blocked_checksum" if has_sum else "blocked_missing_destination"
            reasons.extend(f"[{r['code']}] {r['message']}" for r in rejections)

        if not cls:
            df = dest_files.get(rel)
            if not df:
                cls, reasons = "blocked_missing_destination", [f"missing: {rel}"]
            elif df.get("state") not in ("already_verified", "copied", "verified"):
                cls, reasons = "blocked_missing_destination", [f"unverified: {df.get('state')}"]
            elif df.get("sha256") != m["sha256"] or df.get("byte_size") != m["byte_size"]:
                cls, reasons = "blocked_checksum", ["destination digest or size mismatch"]

        state = "unverified_no_root"
        if source_root:
            d_file = disk_files.get(rel)
            if d_file and d_file.exists():
                state = "present"
                if d_file.is_symlink() or not d_file.is_file():
                    cls, reasons = "blocked_checksum", ["source is symlink or not regular file"]
                elif d_file.stat().st_size != m["byte_size"] or sha256_file(d_file) != m["sha256"]:
                    cls, reasons = "blocked_checksum", ["source disk digest or size mismatch"]
            else:
                state = "already_pruned"

        eff = dispositions.get(rel, m.get("retention_class"))
        if not cls:
            if eff not in ALLOWED_RETENTION or eff in RETAIN:
                cls, reasons = (
                    "retain_required",
                    [f"disposition '{eff}' requires retaining source copy"],
                )
            elif eff in REVIEW:
                cls, reasons = "eligible_after_review", [f"disposition '{eff}' requires review"]
            elif eff in DISPOSABLE:
                cls = "eligible_verified"

        classifications.append(
            _rec(rel, cls, m.get("sha256"), m.get("byte_size", 0), eff, state, reasons)
        )

    if source_root:
        for rel, d_file in sorted(disk_files.items()):
            if rel not in manifest_paths:
                classifications.append(
                    _rec(
                        rel,
                        "not_managed",
                        sha256_file(d_file) if d_file.is_file() else None,
                        d_file.stat().st_size if d_file.is_file() else 0,
                        "untracked",
                        "untracked",
                        ["untracked file"],
                    )
                )

    plan = [
        {"path": it["path"], "sha256": it["sha256"], "byte_size": it["byte_size"]}
        for it in sorted(classifications, key=lambda x: x["path"])
        if it["classification"] == "eligible_verified" and it["source_state"] != "already_pruned"
    ]

    apply_res, cas_failed = [], False
    if apply:
        if any(c["classification"].startswith("blocked_") for c in classifications):
            cas_failed = True
        else:
            for e in plan:
                tgt = (source_root / e["path"]) if source_root else None
                bad = not tgt or tgt.is_symlink() or not tgt.is_file()
                if bad or tgt.stat().st_size != e["byte_size"] or sha256_file(tgt) != e["sha256"]:
                    cas_failed = True
                    break
                tgt.unlink()
                apply_res.append({"path": e["path"], "status": "deleted"})

    counts = Counter(c["classification"] for c in classifications)
    if sum(v for k, v in counts.items() if k.startswith("blocked_")) > 0 or cas_failed:
        status = "blocked"
    elif counts.get("eligible_after_review", 0) > 0:
        status = "review_required"
    elif (
        counts.get("retain_required", 0) > 0
        and len(plan) == 0
        and not any(c["source_state"] == "already_pruned" for c in classifications)
    ):
        status = "retain_required"
    else:
        status = "eligible"

    summary = {k: counts.get(k, 0) for k in CLASSIFICATIONS}
    summary["total_paths"], summary["deletion_plan_items"] = len(classifications), len(plan)

    res = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "mode": "apply" if apply else "check",
        "manifest_id": m_id,
        "durability_class": durability,
        "independent_verification": indep,
        "summary": summary,
        "classifications": classifications,
        "deletion_plan": plan,
        "rejections": rejections,
    }
    if apply:
        res["apply"] = {
            "success": not cas_failed,
            "deleted_count": len(apply_res),
            "results": apply_res,
        }
    return res


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="check_prune_eligibility",
        description="Gate source-host artifact pruning on verified durable custody.",
    )
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--destination-receipt", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=None)
    p.add_argument("--consumers", type=Path, default=None)
    p.add_argument("--dispositions", type=Path, default=None)
    p.add_argument("--case", type=str, default=None)
    p.add_argument("--check", action="store_true", default=False)
    p.add_argument("--apply", action="store_true", default=False)
    p.add_argument("--format", choices=("json", "summary", "text"), default="json")
    p.add_argument("--output", type=Path, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run prune eligibility CLI and return status code."""
    args = _parser().parse_args(argv)
    if args.check and args.apply:
        print("error: pass at most one of --check or --apply", file=sys.stderr)
        return EXIT_MALFORMED

    try:
        rep = check_prune_eligibility(
            args.source_manifest,
            args.destination_receipt,
            source_root=args.source_root,
            consumers_path=args.consumers,
            dispositions_path=args.dispositions,
            case_name=args.case,
            apply=bool(args.apply),
        )
    except (OSError, ValueError, RuntimeError, ChunkManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MALFORMED

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.format == "json":
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        s = rep["summary"]
        print(
            f"prune eligibility: {rep['status']}\n  paths={s['total_paths']} verified={s['eligible_verified']} plan={s['deletion_plan_items']}"
        )
        print(
            f"  blocked: dest={s['blocked_missing_destination']} sum={s['blocked_checksum']} consumer={s['blocked_active_consumer']}"
        )

    return EXIT_ELIGIBLE if rep["status"] == "eligible" else EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
