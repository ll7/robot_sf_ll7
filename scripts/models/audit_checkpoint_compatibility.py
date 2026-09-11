#!/usr/bin/env python3
"""Read-only checkpoint/model compatibility audit (issue #8896).

Inventories checkpoints referenced by a sanitized overlay document
(``robot_sf.checkpoint_compatibility_input.v1``). Each flat row records the stable model id,
artifact version/digest/locator class, companion files, algorithm/policy class,
observation/action contract, normalizer, source/training lineage, consumer ownership, and
environment class, and terminates in exactly one availability state. Contract bytes are
inspected read-only; no framework is imported, no training and no benchmark execution is
performed, and loader compatibility is represented by sanitized probe facts
(``policy_class``, observation/action schema, normalizer, parameters, custom objects,
dependency modules) in the artifact record.

Report schema: ``robot_sf.checkpoint_compatibility_audit.v1``. Exit codes: ``0`` pass or
report-only, ``1`` ``--check`` failure, ``2`` unknown input. The audit writes no state, starts
no download, and emits no private path, host, or credential.

    uv run python scripts/models/audit_checkpoint_compatibility.py \
        --input tests/models/fixtures/checkpoint_audit/compatible.json --check --format json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_sf.models.registry import sha256_of_file

INPUT_SCHEMA = "robot_sf.checkpoint_compatibility_input.v1"
REPORT_SCHEMA = "robot_sf.checkpoint_compatibility_audit.v1"
CLAIM_BOUNDARY = (
    "Read-only checkpoint/model provenance projection: rows classify declared availability and "
    "sanitized-set load compatibility from overlay records. A row is not a scientific-quality, "
    "redistribution-rights, or benchmark verdict."
)
RECOVERABLE_STATES = frozenset(("public", "private_durable"))
DURABLE_LOCATORS = frozenset(("public_release", "cloud_durable", "personal_durable"))
LOCATOR_CLASSES = (
    "public_release cloud_durable personal_durable institutional_durable institutional_cache "
    "local_scratch mutable_alias rights_blocked missing unknown"
).split()
INSTITUTIONAL_CODES = {
    "institutional_durable": "institutional_only",
    "institutional_cache": "cache_only",
    "local_scratch": "cache_only",
}
ARTIFACT_KINDS = ("json", "sb3_zip", "torch_pt", "unknown")
DEPENDENCY_GROUPS = frozenset("core sb3 torch training rllib carla".split())
LEARNED_ALGORITHMS = frozenset(
    "ppo sac td3 ddpg a2c dreamer imitation predictive_planner cadrl".split()
)
ALIAS_VERSIONS = frozenset("latest best best-success current head".split())
PROBE_CODES = frozenset(
    "artifact_unreadable invalid_artifact missing_data_member missing_custom_object".split()
)
INCOMPATIBILITY_CODES = (
    frozenset(
        "digest_mismatch digest_not_pinned normalizer_missing normalizer_digest_mismatch "
        "normalizer_not_declared observation_contract_mismatch action_contract_mismatch "
        "dependency_unavailable dependency_group_unknown non_finite_parameters "
        "lineage_unresolved".split()
    )
    | PROBE_CODES
)
NON_BLOCKING_CODES = frozenset(("artifact_not_staged",))
MODEL_KEYS = frozenset(
    "model_id artifact_kind artifact_version artifact_path artifact_sha256 locator_class "
    "companion_files algorithm_class policy_class observation_schema observation_shape "
    "action_schema action_shape normalizer_required normalizer_embedded normalizer_path "
    "normalizer_sha256 dependency_group dependency_modules required_custom_objects "
    "environment_class source_config training_commit rights active".split()
)
CONSUMER_KEYS = frozenset(
    "consumer_id owner active required_model_ids observation_schema observation_shape "
    "action_schema action_shape normalizer_required environment_class".split()
)
SUM_KEYS = "model_id algorithm_class policy_class environment_class source_config training_commit rights active dependency_group".split()
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FORBIDDEN_FIELDS = frozenset(
    "account command command_line cmdline env environment host hostname node node_list nodelist "
    "nodes partition password private_path qos scheduler_account secret secrets slurm_account "
    "token url user user_name username".split()
)
_PRIVATE_PATTERNS = (
    (r"://", "URL or private path"),
    (r"(?i)[?&](?:sig|signature|token|expires|x-amz-|x-goog-)", "signed URL"),
    (r"(?i)(?:api[_-]?key|secret|password|passwd|token|credential|bearer)", "credential-like"),
    (r"(?i)(?:[a-z0-9][a-z0-9-]{0,61}\.){2,}[a-z]{2,24}", "hostname-like content"),
)


def _f(code, model_id=None, consumer_id=None, detail=""):
    return {"code": code, "model_id": model_id, "consumer_id": consumer_id, "detail": detail}


def _text(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _sha_ok(value):
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def _module_of(name):
    return name.rsplit(".", 1)[0] if "." in name else name


def _module_missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ModuleNotFoundError, ValueError):
        return True


def _kind_for(path, declared):
    suffixes = {".json": "json", ".zip": "sb3_zip", ".pt": "torch_pt", ".pth": "torch_pt"}
    if declared in ARTIFACT_KINDS and declared != "unknown":
        return str(declared)
    return suffixes.get(Path(path).suffix.lower() if path else "", "unknown")


def _strings(raw):
    return [str(item).strip() for item in raw or [] if isinstance(item, str) and item.strip()]


def _scan_private(value: Any, location: str, findings: list[dict[str, Any]]) -> bool:
    """Append sanitized findings for private-looking values; never echo a value."""
    if isinstance(value, Mapping):
        children = [(f"{location}/{key}", key, item) for key, item in value.items()]
    elif isinstance(value, list):
        children = [(f"{location}[{index}]", None, item) for index, item in enumerate(value)]
    elif isinstance(value, str):
        details = [d for p, d in _PRIVATE_PATTERNS if re.search(p, value) is not None]
        if value.startswith(("/", "~", "\\")) or "@" in value or ".." in value.split("/"):
            details.append("private identity or path")
        details += ["control character"] if any(ord(c) < 32 for c in value) else []
        findings.extend(_f("forbidden_value", detail=f"{location}: {item}") for item in details)
        return bool(details)
    else:
        children = []
    leaked = False
    for child, key, item in children:
        if key is not None and str(key).lower() in _FORBIDDEN_FIELDS:
            findings.append(_f("forbidden_value", detail=f"{child}: private field"))
            leaked = True
        leaked |= _scan_private(item, child, findings)
    return leaked


def _record(raw, keys, location, findings):
    if not isinstance(raw, Mapping) or set(raw) - keys:
        findings.append(_f("invalid_record", detail=f"{location}: unknown keys or non-mapping"))
        return None
    return dict(raw)


def _model(raw, location, findings):
    record = _record(raw, MODEL_KEYS, location, findings)
    if record is None:
        return None
    model_id = _text(record.get("model_id"))
    locator = _text(record.get("locator_class")) or "unknown"
    sha = record.get("artifact_sha256")
    invalid = (
        model_id is None
        or _ID_RE.fullmatch(model_id) is None
        or ".." in model_id
        or locator not in LOCATOR_CLASSES
        or (sha is not None and not (_sha_ok(sha) or sha in {"pending", "placeholder"}))
    )
    if invalid:
        findings.append(_f("invalid_record", detail=f"{location}: invalid identity or value"))
        return None
    record.update(
        model_id=model_id,
        locator_class=locator,
        algorithm_class=_text(record.get("algorithm_class")) or "unknown",
        rights=_text(record.get("rights")) or "unknown",
        active=bool(record.get("active", True)),
        dependency_modules=_strings(record.get("dependency_modules")),
        required_custom_objects=_strings(record.get("required_custom_objects")),
    )
    return record


def _consumer(raw, location, findings):
    record = _record(raw, CONSUMER_KEYS, location, findings)
    model_ids = _strings(record.get("required_model_ids")) if record else []
    if record is not None and (not model_ids or _text(record.get("consumer_id")) is None):
        findings.append(_f("invalid_record", detail=f"{location}: consumer id/models required"))
        record = None
    if record is not None:
        record.update(required_model_ids=model_ids, active=bool(record.get("active", True)))
    return record


def _load_input(path: Path, findings):
    """Validate one sanitized overlay document; reject it wholesale on any hard error."""
    if not path.is_file():
        findings.append(_f("invalid_input", detail=f"{path.name}: input missing"))
        return [], []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        findings.append(_f("invalid_input", detail=f"{path.name}: unreadable JSON"))
        return [], []
    if not isinstance(payload, Mapping) or payload.get("schema") != INPUT_SCHEMA:
        findings.append(_f("invalid_input", detail=f"{path.name}: wrong schema"))
        return [], []
    if _scan_private(payload, path.name, findings):
        return [], []
    raw_models, raw_consumers = payload.get("models", []), payload.get("consumers", [])
    if not isinstance(raw_models, list) or not isinstance(raw_consumers, list):
        findings.append(_f("invalid_input", detail=f"{path.name}: models/consumers lists"))
        return [], []
    models = [
        m for i, r in enumerate(raw_models) if (m := _model(r, f"{path.name}.m{i}", findings))
    ]
    consumers = [
        c for i, r in enumerate(raw_consumers) if (c := _consumer(r, f"{path.name}.c{i}", findings))
    ]
    return models, consumers


def _inspect_artifact(path: Path, kind: str) -> dict[str, Any]:  # noqa: C901
    """Read-only inspection of sanitized contract facts or an SB3 zip member list."""
    facts: dict[str, Any] = {"kind": kind, "checked": False}
    if kind == "json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return facts | {"issues": ["artifact_unreadable"]}
        if not isinstance(payload, Mapping):
            return facts | {"issues": ["invalid_artifact"]}
        facts["policy_class"] = _text(payload.get("policy_class"))
        for prefix in ("observation", "action"):
            raw = payload.get(prefix)
            if isinstance(raw, Mapping):
                facts[f"{prefix}_schema"] = _text(raw.get("schema"))
                if isinstance(raw.get("shape"), list):
                    facts[f"{prefix}_shape"] = raw["shape"]
        normalizer = payload.get("normalizer")
        if isinstance(normalizer, Mapping):
            facts["normalizer_required"] = bool(normalizer.get("required"))
            facts["normalizer_present"] = bool(normalizer.get("embedded"))
        parameters = payload.get("parameters")
        if isinstance(parameters, list):
            finite = [math.isfinite(float(v)) for v in parameters if isinstance(v, (int, float))]
            facts["parameters_finite"] = all(finite) if finite else True
        facts["custom_objects_missing"] = sorted(
            n for n in _strings(payload.get("custom_objects")) if _module_missing(_module_of(n))
        )
        facts["modules_missing"] = sorted(
            n for n in _strings(payload.get("dependency_modules")) if _module_missing(n)
        )
        facts["checked"] = True
    elif kind == "sb3_zip":
        try:
            names = set(zipfile.ZipFile(path).namelist())
        except (OSError, zipfile.BadZipFile):
            return facts | {"issues": ["artifact_unreadable"]}
        if "data" not in names:
            facts["issues"] = ["missing_data_member"]
    return facts


def _availability_codes(record, local, digest):
    """Classify artifact identity, custody, digest, and companion state."""
    codes: set[str] = set()
    locator = record["locator_class"]
    path, sha = record.get("artifact_path"), record.get("artifact_sha256")
    pinned = _sha_ok(sha)
    if record.get("rights") == "blocked":
        codes.add("rights_blocked")
    if locator == "mutable_alias" or (
        (_text(record.get("artifact_version")) or "").lower() in ALIAS_VERSIONS and not pinned
    ):
        codes.add("mutable_alias")
    if sha in {"pending", "placeholder"} or (path and Path(path).name.endswith(".placeholder")):
        codes.add("placeholder_artifact")
    if locator == "missing" or (path is None and locator not in DURABLE_LOCATORS):
        codes.add("artifact_missing")
    elif path and not local:
        codes.add("artifact_not_staged" if locator in DURABLE_LOCATORS else "artifact_missing")
    if locator in INSTITUTIONAL_CODES:
        codes.add(INSTITUTIONAL_CODES[locator])
    if locator == "unknown":
        codes.add("availability_unknown")
    if not pinned and locator != "missing":
        codes.add("digest_not_pinned")
    elif pinned and local and digest != sha:
        codes.add("digest_mismatch")
    return codes


def _compatibility_codes(record, facts, root):  # noqa: C901
    """Classify normalizer, contract, lineage, dependency, and probe compatibility."""
    codes: set[str] = set()
    required = bool(record.get("normalizer_required")) or facts.get("normalizer_required") is True
    present = bool(record.get("normalizer_embedded")) or facts.get("normalizer_present") is True
    path = record.get("normalizer_path")
    if path and not (root / path).is_file():
        codes.add("normalizer_missing")
    elif (
        path
        and _sha_ok(record.get("normalizer_sha256"))
        and (sha256_of_file(root / path) != record["normalizer_sha256"])
    ):
        codes.add("normalizer_digest_mismatch")
    if required and not present:
        codes.add("normalizer_missing")
    learned = record.get("algorithm_class") in LEARNED_ALGORITHMS
    if learned and record.get("normalizer_required") is None:
        codes.add("normalizer_not_declared")
    if learned:
        source = record.get("source_config")
        if not record.get("training_commit") or not source or not (root / source).exists():
            codes.add("lineage_unresolved")
    if record.get("dependency_group") is not None and (
        record.get("dependency_group") not in DEPENDENCY_GROUPS
    ):
        codes.add("dependency_group_unknown")
    if facts.get("parameters_finite") is False:
        codes.add("non_finite_parameters")
    if facts.get("custom_objects_missing"):
        codes.add("missing_custom_object")
    if facts.get("modules_missing"):
        codes.add("dependency_unavailable")
    codes.update(code for code in facts.get("issues") or () if code in PROBE_CODES)
    for prefix in ("observation", "action"):
        for key in ("schema", "shape"):
            declared, observed = record.get(f"{prefix}_{key}"), facts.get(f"{prefix}_{key}")
            if declared and observed and declared != observed:
                codes.add(f"{prefix}_contract_mismatch")
    return codes


def _state(codes, locator, pinned):
    names = {
        "rights_blocked": "rights_blocked",
        "mutable_alias": "mutable_alias",
        "artifact_missing": "missing",
        "placeholder_artifact": "placeholder",
    }
    for code, name in names.items():
        if code in codes:
            return name
    if codes & INCOMPATIBILITY_CODES:
        return "incompatible"
    if locator == "public_release" and pinned:
        return "public"
    if locator in {"cloud_durable", "personal_durable"} and pinned:
        return "private_durable"
    return "institutional_cache_only" if codes & {"cache_only", "institutional_only"} else "unknown"


def _facts_for(record, root):
    """Inspect one local artifact and merge record-declared dependency facts."""
    path = record.get("artifact_path")
    kind = _kind_for(path, record.get("artifact_kind"))
    local = root / path if path else None
    if local is None or not local.is_file() or kind == "unknown":
        return {"kind": kind, "checked": False}
    facts = _inspect_artifact(local, kind)
    missing = {
        n for n in record.get("required_custom_objects") or [] if _module_missing(_module_of(n))
    }
    modules = {n for n in record.get("dependency_modules") or [] if _module_missing(n)}
    facts["custom_objects_missing"] = sorted(
        set(facts.get("custom_objects_missing") or []) | missing
    )
    facts["modules_missing"] = sorted(set(facts.get("modules_missing") or []) | modules)
    facts["issues"] = list(facts.get("issues") or [])
    return facts


def _evaluate_model(record, facts, root):
    """Classify one model record into a terminal state with stable reason codes."""
    path = record.get("artifact_path")
    local = bool(path and (root / path).is_file())
    digest = sha256_of_file(root / path) if local else None
    codes = _availability_codes(record, local, digest) | _compatibility_codes(record, facts, root)
    verified = bool(facts.get("checked") and not facts.get("issues"))
    row = {key: record.get(key) for key in SUM_KEYS}
    row["artifact"] = {
        "kind": _kind_for(path, record.get("artifact_kind")),
        "version": record.get("artifact_version"),
        "path": path,
        "sha256": record.get("artifact_sha256"),
        "locator_class": record["locator_class"],
        "companion_files": record.get("companion_files") or [],
    }
    for prefix in ("observation", "action"):
        row[f"{prefix}_contract"] = {
            "schema": record.get(f"{prefix}_schema"),
            "shape": record.get(f"{prefix}_shape"),
        }
    row["normalizer"] = {
        "required": bool(record.get("normalizer_required"))
        or facts.get("normalizer_required") is True,
        "embedded": bool(record.get("normalizer_embedded"))
        or facts.get("normalizer_present") is True,
        "path": record.get("normalizer_path"),
    }
    row.update(
        state=_state(codes, record["locator_class"], _sha_ok(record.get("artifact_sha256"))),
        load_status="verified" if verified else ("inspected" if local else "not_staged"),
        reason_codes=sorted(codes),
        probe={
            key: facts.get(key)
            for key in "kind parameters_finite custom_objects_missing modules_missing".split()
        },
    )
    return row


def _contract_mismatch(consumer, row):
    for prefix in ("observation", "action"):
        model = row.get(f"{prefix}_contract") or {}
        for key in ("schema", "shape"):
            if consumer.get(f"{prefix}_{key}") and model.get(key):
                if consumer[f"{prefix}_{key}"] != model[key]:
                    return f"{prefix}_contract_mismatch"
    return None


def _evaluate_consumer(consumer, rows, findings):
    """Gate one active consumer on recoverable, compatible, load-verified models."""
    consumer_id = str(consumer["consumer_id"])
    failed: set[str] = set()
    if consumer.get("active", True):
        for model_id in consumer["required_model_ids"]:
            row = rows.get(model_id)
            if row is None:
                findings.append(_f("consumer_model_unresolved", model_id, consumer_id, "absent"))
            elif row["state"] not in RECOVERABLE_STATES:
                findings.append(
                    _f("consumer_model_unrecoverable", model_id, consumer_id, str(row["state"]))
                )
            elif blocking := sorted(
                code for code in row["reason_codes"] if code not in NON_BLOCKING_CODES
            ):
                findings.append(
                    _f("consumer_model_incompatible", model_id, consumer_id, ",".join(blocking))
                )
            elif row["load_status"] != "verified":
                findings.append(
                    _f("consumer_model_load_unverified", model_id, consumer_id, row["load_status"])
                )
            elif mismatched := _contract_mismatch(consumer, row):
                findings.append(_f(mismatched, model_id, consumer_id, "consumer vs model"))
            else:
                continue
            failed.add(model_id)
    active = bool(consumer.get("active", True))
    return {
        "consumer_id": consumer_id,
        "owner": consumer.get("owner"),
        "active": active,
        "required_model_ids": list(consumer["required_model_ids"]),
        "failed_model_ids": sorted(failed),
        "outcome": "fail" if failed else ("pass" if active else "inactive"),
    }


def build_audit(*, inputs=(), root: Path):  # noqa: C901
    """Build the deterministic audit payload from sanitized overlay records."""
    findings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    for input_path in inputs:
        models, loaded = _load_input(Path(input_path), findings)
        records.extend(models)
        consumers.extend(loaded)
    counts = Counter(record["model_id"] for record in records)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(record["model_id"], record)
    for model_id, count in sorted(counts.items()):
        if count > 1:
            findings.append(_f("duplicate_model_id", model_id, detail="duplicate inventory id"))
    rows = []
    for model_id in sorted(unique):
        row = _evaluate_model(unique[model_id], _facts_for(unique[model_id], root), root)
        if counts[model_id] > 1:
            row["reason_codes"] = sorted(set(row["reason_codes"]) | {"duplicate_model_id"})
        rows.append(row)
    for row in rows:
        for code in row["reason_codes"]:
            if code not in {"duplicate_model_id", *NON_BLOCKING_CODES}:
                findings.append(_f(code, row["model_id"], detail=row["state"]))
    row_map = {row["model_id"]: row for row in rows}
    evaluated = sorted(
        (_evaluate_consumer(consumer, row_map, findings) for consumer in consumers),
        key=lambda item: item["consumer_id"],
    )
    ordered = sorted(
        findings, key=lambda item: (item["code"], item["model_id"] or "", item["detail"])
    )
    blocking = [item for item in ordered if item["code"] not in NON_BLOCKING_CODES]
    if (
        not rows
        and not evaluated
        and any(item["code"] in {"invalid_input", "forbidden_value"} for item in ordered)
    ):
        status = "unknown"
    else:
        status = "fail" if blocking else "pass"
    return {
        "schema": REPORT_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "status": status,
        "ok": status == "pass",
        "summary": {
            "model_count": len(rows),
            "consumer_count": len(evaluated),
            "active_consumer_count": sum(1 for item in evaluated if item["active"]),
            "recoverable_count": sum(1 for row in rows if row["state"] in RECOVERABLE_STATES),
            "state_counts": dict(sorted(Counter(row["state"] for row in rows).items())),
            "finding_counts": dict(sorted(Counter(item["code"] for item in ordered).items())),
        },
        "findings": ordered,
        "models": rows,
        "consumers": evaluated,
    }


def render_json(payload):
    """Return byte-stable audit JSON with sorted keys and a trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_markdown(payload):
    """Return the deterministic human-readable Markdown audit projection."""
    summary, findings = payload["summary"], payload["findings"]
    lines = [
        "# Checkpoint Compatibility Audit",
        "",
        f"- Schema: `{payload['schema']}` | Status: `{payload['status']}`",
        f"- Models: {summary['model_count']} (recoverable: {summary['recoverable_count']})"
        f" | Consumers: {summary['consumer_count']} | Findings: {len(findings)}",
        "",
        payload["claim_boundary"],
        "",
        "## Findings",
        "",
        *(
            [
                f"- `{item['code']}` `{item['model_id'] or '-'}`"
                f" `{item['consumer_id'] or '-'}`: {item['detail']}"
                for item in findings
            ]
            or ["No findings."]
        ),
        "",
        "## Models",
        "",
        *(
            f"- `{row['model_id']}`: `{row['state']}`"
            f" locator=`{row['artifact']['locator_class']}`"
            f" digest={'pinned' if _sha_ok(row['artifact'].get('sha256')) else 'unpinned'}"
            f" load={row['load_status']} reasons={','.join(row['reason_codes']) or '-'}"
            for row in payload["models"]
        ),
    ]
    if payload["consumers"]:
        lines += ["", "## Consumers", ""]
        lines += [
            f"- `{item['consumer_id']}`: {item['outcome']}"
            f" failed={','.join(item['failed_model_ids']) or '-'}"
            for item in payload["consumers"]
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and return a shell-friendly exit code."""
    parser = argparse.ArgumentParser(
        prog="audit_checkpoint_compatibility", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--input", action="append", default=[], type=Path, help="Overlay.")
    parser.add_argument("--root", type=Path, default=None, help="Artifact resolution root.")
    parser.add_argument("--check", action="store_true", help="Exit 1 on blocking findings.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args(argv)
    if not args.input:
        sys.stderr.write("FAIL invalid_input: --input is required\n")
        return 2
    root = (args.root or Path.cwd()).resolve()
    payload = build_audit(inputs=args.input, root=root)
    sys.stdout.write(
        render_markdown(payload) if args.format == "markdown" else render_json(payload)
    )
    if payload["status"] == "unknown":
        return 2
    return 0 if payload["ok"] or not args.check else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
