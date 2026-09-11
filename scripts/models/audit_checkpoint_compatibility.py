#!/usr/bin/env python3
"""Read-only checkpoint/model compatibility audit (issues #8896, #8991).

Inventories checkpoints referenced by a sanitized overlay document
(``robot_sf.checkpoint_compatibility_input.v1``) and/or the canonical ``model/registry.yaml``
(plus explicitly supplied referencing configs or release manifests). Each flat row records the
stable model id, artifact version/digest/locator class, companion files, algorithm/policy class,
observation/action contract, normalizer, source/training lineage, consumer ownership, and
environment class, and terminates in exactly one availability state. Records that cannot be
sanitized or probed are explicit exclusions, never silently dropped.

Contract bytes are inspected read-only; the parent process imports no framework, no training and
no benchmark execution is performed. With ``--probe``, local ``.zip``/``.pt`` artifacts get an
opt-in bounded-subprocess loader probe (hard timeout; fail closed on timeout, non-zero exit, or
malformed output) that records sanitized ``policy_class``, observation/action shape,
finite-parameter, and custom-object facts.

Report schema: ``robot_sf.checkpoint_compatibility_audit.v1``. Exit codes: ``0`` pass or
report-only, ``1`` ``--check`` failure, ``2`` unknown or invalid input. The audit writes no
state, starts no download, and emits no private path, host, or credential.

    uv run python scripts/models/audit_checkpoint_compatibility.py \
        --registry model/registry.yaml --check --format json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import re
import subprocess
import sys
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.models.preflight import required_model_ids_for_config
from robot_sf.models.registry import load_registry, sha256_of_file

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
LOADER_PROBE_CODES = frozenset(
    "loader_probe_timeout loader_probe_failed loader_probe_malformed".split()
)
PROBE_CODES = (
    frozenset(
        "artifact_unreadable invalid_artifact missing_data_member missing_custom_object".split()
    )
    | LOADER_PROBE_CODES
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
PROBE_WORKER = Path(__file__).resolve()
DEFAULT_PROBE_TIMEOUT = 30.0
PROBE_FACT_KEYS = (
    "kind loader policy_class observation_shape action_shape parameters_finite "
    "custom_objects_missing modules_missing".split()
)
PROBE_PROBEABLE_KINDS = frozenset(("sb3_zip", "torch_pt"))
PROBE_ERROR_CODES = frozenset(
    "unsupported_kind checkpoint_corrupt missing_custom_object dependency_unavailable "
    "loader_error".split()
)
_PROBE_CODE_MAP = {
    "checkpoint_corrupt": "artifact_unreadable",
    "missing_custom_object": "missing_custom_object",
    "dependency_unavailable": "dependency_unavailable",
}
_REGISTRY_ALGORITHM_TAGS = (
    (frozenset(("sacadrl", "ga3c", "cadrl")), "cadrl"),
    (frozenset(("sac",)), "sac"),
    (frozenset(("ppo",)), "ppo"),
    (frozenset(("predictive",)), "predictive_planner"),
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
        probe={key: facts.get(key) for key in PROBE_FACT_KEYS},
    )
    return row


class _ProbeError(Exception):
    """Fail-closed loader-probe error carrying a stable code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _intake_label(path: Path, root: Path) -> str:
    """Return a private-safe label for an intake document."""
    try:
        label = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        label = path.resolve().name
    return _relpath(label) or "intake-document"


def _relpath(value: Any) -> str | None:
    """Return one private-safe relative POSIX path, or ``None`` when unsafe."""
    raw = _text(value)
    if raw is None:
        return None
    raw = raw.replace("\\", "/")
    if raw.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", raw):
        return None
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        return None
    candidate = "/".join(parts)
    if any(re.search(pattern, candidate) for pattern, _ in _PRIVATE_PATTERNS):
        return None
    return candidate


def _exclusion(source: str, model_id: str | None, reason: str) -> dict[str, Any]:
    return {"source": source, "model_id": model_id, "reason": reason}


def _registry_algorithm(entry: Mapping[str, Any]) -> str:
    """Map canonical registry tags to the audit's algorithm vocabulary."""
    raw_tags = entry.get("tags")
    tags = {str(tag).strip().lower() for tag in raw_tags or [] if str(tag).strip()}
    for candidates, algorithm in _REGISTRY_ALGORITHM_TAGS:
        if tags & candidates:
            return algorithm
    return "unknown"


def _release_sha(release: Mapping[str, Any], path: str | None) -> tuple[str | None, bool]:
    """Return the digest pinning ``path`` and whether invalid release metadata was seen."""
    per_file = release.get("per_file_sha256")
    if path is not None and isinstance(per_file, Mapping):
        candidate = per_file.get(Path(path).name)
        if _sha_ok(candidate):
            return str(candidate).strip().lower(), False
    raw = release.get("sha256")
    if raw is None:
        return None, False
    normalized = str(raw).strip().lower()
    return (normalized, False) if _sha_ok(normalized) else (None, True)


def _registry_entry_record(  # noqa: C901
    source: str, model_id: str, entry: Mapping[str, Any], exclusions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Map one canonical registry entry into the sanitized record schema."""
    local_raw = entry.get("local_path")
    path = _relpath(local_raw) if _text(local_raw) else None
    if _text(local_raw) and path is None:
        exclusions.append(_exclusion(source, model_id, "unsanitized_local_path"))
        return None
    release = entry.get("github_release")
    if release is not None and not isinstance(release, Mapping):
        exclusions.append(_exclusion(source, model_id, "invalid_release_metadata"))
        release = None
    locator, sha, version = "missing", None, None
    if release:
        locator = "public_release"
        version = _text(release.get("version")) or _text(release.get("tag"))
        sha, invalid_sha = _release_sha(release, path)
        if invalid_sha:
            exclusions.append(_exclusion(source, model_id, "invalid_release_sha"))
    elif entry.get("local_only") is True:
        locator = "personal_durable"
    elif entry.get("wandb_artifact_path") or (
        entry.get("wandb_run_path") and entry.get("wandb_file")
    ):
        locator = "cloud_durable"
    elif path is not None:
        locator = "local_scratch"
    source_config = None
    if _text(entry.get("config_path")):
        source_config = _relpath(entry.get("config_path"))
        if source_config is None:
            exclusions.append(_exclusion(source, model_id, "unsanitized_source_config"))
    commit_raw = _text(entry.get("commit"))
    commit = commit_raw if commit_raw and re.fullmatch(r"[0-9a-f]{7,40}", commit_raw) else None
    if commit_raw and commit is None:
        exclusions.append(_exclusion(source, model_id, "invalid_training_commit"))
    return {
        "model_id": model_id,
        "artifact_path": path,
        "artifact_sha256": sha,
        "artifact_version": version,
        "locator_class": locator,
        "algorithm_class": _registry_algorithm(entry),
        "source_config": source_config,
        "training_commit": commit,
        "active": True,
        "rights": "unknown",
    }


def _skipped_registry_entries(registry_path: Path, loaded_ids: set[str]) -> list[str | None]:
    """Return raw registry model ids the canonical loader skipped (explicit accounting only)."""
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return []
    models = data.get("models") if isinstance(data, Mapping) else None
    if not isinstance(models, list):
        return []
    raw_ids = [(_text(e.get("model_id")) if isinstance(e, Mapping) else None) for e in models]
    return [model_id for model_id in raw_ids if model_id is None or model_id not in loaded_ids]


def _registry_records(
    registry_path: Path, *, root: Path, exclusions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Load canonical registry entries and map them into sanitized records."""
    label = _intake_label(registry_path, root)
    try:
        registry = load_registry(registry_path)
    except (AttributeError, FileNotFoundError, OSError, TypeError, ValueError, yaml.YAMLError):
        exclusions.append(_exclusion(label, None, "registry_unreadable"))
        return []
    for model_id in _skipped_registry_entries(registry_path, set(registry)):
        exclusions.append(_exclusion(label, model_id, "registry_entry_skipped"))
    records = []
    for model_id, entry in sorted(registry.items()):
        record = _registry_entry_record(label, model_id, entry, exclusions)
        if record is not None:
            records.append(record)
    return records


def _reference_consumers(
    reference_paths: Sequence[Path],
    *,
    root: Path,
    exclusions: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map referencing configs/release manifests into sanitized consumer records."""
    consumers = []
    for path in reference_paths:
        label = _intake_label(path, root)
        if not path.is_file():
            exclusions.append(_exclusion(label, None, "reference_missing"))
            findings.append(_f("invalid_input", detail=f"{label}: reference missing"))
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            exclusions.append(_exclusion(label, None, "reference_unreadable"))
            findings.append(_f("invalid_input", detail=f"{label}: unreadable YAML"))
            continue
        model_ids = required_model_ids_for_config(payload)
        if not model_ids:
            exclusions.append(_exclusion(label, None, "no_model_references"))
            continue
        record = _consumer(
            {
                "consumer_id": label,
                "owner": "intake",
                "active": True,
                "required_model_ids": model_ids,
            },
            f"intake:{label}",
            findings,
        )
        if record is None:
            exclusions.append(_exclusion(label, None, "consumer_invalid"))
        else:
            consumers.append(record)
    return consumers


def _probe_facts_for(record, facts, root, *, timeout, state):
    """Attach opt-in bounded loader-probe facts to one record's evaluation facts."""
    model_id = record["model_id"]
    path = record.get("artifact_path")
    local = root / path if path else None
    kind = _kind_for(path, record.get("artifact_kind"))
    if local is None or not local.is_file():
        state["skipped"].append({"model_id": model_id, "reason": "not_staged"})
    elif kind not in PROBE_PROBEABLE_KINDS:
        state["skipped"].append({"model_id": model_id, "reason": "unsupported_kind"})
    else:
        observed, codes = _probe_subprocess(local, kind, timeout=timeout)
        if not codes:
            facts.update(observed)
            facts["checked"] = True
            state["verified"].append(model_id)
            return facts
        facts["issues"] = sorted(set(facts.get("issues") or []) | set(codes))
        facts["checked"] = False
        state["failed"].append({"model_id": model_id, "reason_codes": codes})
    return facts


def _child_error(stdout: str) -> str | None:
    try:
        payload = json.loads(stdout)
    except ValueError:
        return None
    if isinstance(payload, Mapping) and payload.get("error") in PROBE_ERROR_CODES:
        return str(payload["error"])
    return None


def _sanitize_probe_facts(raw: Any) -> dict[str, Any] | None:
    """Validate child output strictly; ``None`` means malformed and fails closed."""
    if not isinstance(raw, Mapping):
        return None
    loader, policy_class = _text(raw.get("loader")), raw.get("policy_class")
    if loader not in {"sb3", "torch"}:
        return None
    if policy_class is not None and (
        not isinstance(policy_class, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", policy_class) is None
    ):
        return None
    facts: dict[str, Any] = {"loader": loader, "policy_class": policy_class}
    for key in ("observation_shape", "action_shape"):
        shape = _shape_list(raw.get(key))
        if raw.get(key) is not None and shape is None:
            return None
        facts[key] = shape
    finite = raw.get("parameters_finite")
    if finite is not None and not isinstance(finite, bool):
        return None
    facts["parameters_finite"] = finite
    for key in ("custom_objects_missing", "modules_missing"):
        values = raw.get(key)
        if values is not None and (
            not isinstance(values, list) or len(_strings(values)) != len(values)
        ):
            return None
        facts[key] = sorted(_strings(values))
    return facts


def _probe_subprocess(
    local: Path, kind: str, *, timeout: float
) -> tuple[dict[str, Any], list[str]]:
    """Run the bounded loader-probe child; fail closed on any process anomaly."""
    command = [sys.executable, str(PROBE_WORKER), "--probe-child", str(local), kind]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return {}, ["loader_probe_timeout"]
    except OSError:
        return {}, ["loader_probe_failed"]
    if completed.returncode != 0:
        codes = ["loader_probe_failed"]
        if mapped := _PROBE_CODE_MAP.get(_child_error(completed.stdout) or ""):
            codes.append(mapped)
        return {}, sorted(codes)
    try:
        payload = json.loads(completed.stdout)
    except ValueError:
        return {}, ["loader_probe_malformed"]
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        return {}, ["loader_probe_failed"]
    facts = _sanitize_probe_facts(payload.get("facts"))
    if facts is None:
        return {}, ["loader_probe_malformed"]
    return facts, []


def _probe_child_main(path: Path, kind: str) -> int:
    """Run the probe child contract: exactly one JSON object on stdout, fail closed."""
    try:
        facts = _probe_artifact(path, kind)
    except _ProbeError as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": exc.code}, sort_keys=True) + "\n")
        return 1
    sys.stdout.write(json.dumps({"ok": True, "facts": facts}, sort_keys=True) + "\n")
    return 0


def _class_of(instance: Any) -> str:
    cls = instance if isinstance(instance, type) else type(instance)
    return f"{cls.__module__}.{cls.__qualname__}"


def _shape_list(value: Any) -> list[int] | None:
    if isinstance(value, Mapping):
        value = value.get("shape")
    if not isinstance(value, (list, tuple)):
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    return [int(item) for item in value]


def _collect_tensors(node: Any, tensors: list[Any]) -> None:
    if isinstance(node, Mapping):
        for value in node.values():
            _collect_tensors(value, tensors)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _collect_tensors(value, tensors)
    elif hasattr(node, "isfinite") and hasattr(node, "numel"):
        tensors.append(node)


def _probe_sb3(path: Path) -> dict[str, Any]:
    """Load one Stable-Baselines3 zip via its own loader; return sanitized facts (subprocess)."""
    try:
        import torch
        from stable_baselines3.common.save_util import load_from_zip_file
    except ImportError as exc:
        raise _ProbeError("dependency_unavailable") from exc
    try:
        data, params, _ = load_from_zip_file(str(path), device="cpu")
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise _ProbeError("missing_custom_object") from exc
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise _ProbeError("checkpoint_corrupt") from exc
    if not isinstance(data, Mapping):
        raise _ProbeError("checkpoint_corrupt")
    tensors: list[Any] = []
    _collect_tensors(params, tensors)
    policy = data.get("policy_class")
    return {
        "loader": "sb3",
        "policy_class": _class_of(policy) if isinstance(policy, type) else None,
        "observation_shape": _shape_list(getattr(data.get("observation_space"), "shape", None)),
        "action_shape": _shape_list(getattr(data.get("action_space"), "shape", None)),
        "parameters_finite": (
            all(bool(torch.isfinite(tensor).all()) for tensor in tensors) if tensors else None
        ),
        "custom_objects_missing": [],
        "modules_missing": [],
    }


def _probe_torch(path: Path) -> dict[str, Any]:
    """Safely load one torch checkpoint and return sanitized facts (subprocess only)."""
    try:
        import torch
    except ImportError as exc:
        raise _ProbeError("dependency_unavailable") from exc
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except ImportError as exc:
        raise _ProbeError("dependency_unavailable") from exc
    except (AttributeError, ModuleNotFoundError) as exc:
        raise _ProbeError("missing_custom_object") from exc
    except pickle.UnpicklingError as exc:
        code = "missing_custom_object" if "Unsupported global" in str(exc) else "checkpoint_corrupt"
        raise _ProbeError(code) from exc
    except (EOFError, OSError, RuntimeError, ValueError) as exc:
        raise _ProbeError("checkpoint_corrupt") from exc
    mapping = payload if isinstance(payload, Mapping) else {}
    tensors: list[Any] = []
    _collect_tensors(payload, tensors)
    return {
        "loader": "torch",
        "policy_class": _text(mapping.get("policy_class")) or _text(mapping.get("class_name")),
        "observation_shape": _shape_list(mapping.get("observation_shape")),
        "action_shape": _shape_list(mapping.get("action_shape")),
        "parameters_finite": (
            all(bool(torch.isfinite(tensor).all()) for tensor in tensors) if tensors else None
        ),
        "custom_objects_missing": [],
        "modules_missing": [],
    }


def _probe_artifact(path: Path, kind: str) -> dict[str, Any]:
    if kind == "sb3_zip":
        return _probe_sb3(path)
    if kind == "torch_pt":
        return _probe_torch(path)
    raise _ProbeError("unsupported_kind")


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


def build_audit(  # noqa: C901, PLR0912
    *,
    inputs=(),
    registry: Path | None = None,
    references: Sequence[Path] = (),
    probe: bool = False,
    probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    root: Path,
):
    """Build the deterministic audit payload from sanitized overlay and canonical records."""
    findings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for input_path in inputs:
        models, loaded = _load_input(Path(input_path), findings)
        records.extend(models)
        consumers.extend(loaded)
    if registry is not None:
        for candidate in _registry_records(Path(registry), root=root, exclusions=exclusions):
            record = _model(candidate, f"intake:{candidate.get('model_id')}", findings)
            if record is None:
                exclusions.append(
                    _exclusion("registry", candidate.get("model_id"), "record_invalid")
                )
            else:
                records.append(record)
    reference_paths = [Path(path) for path in references]
    consumers.extend(
        _reference_consumers(reference_paths, root=root, exclusions=exclusions, findings=findings)
    )
    for item in exclusions:
        findings.append(
            _f("intake_excluded", item["model_id"], detail=f"{item['source']}:{item['reason']}")
        )
    probe_state: dict[str, Any] = {
        "enabled": bool(probe),
        "timeout_seconds": float(probe_timeout) if probe else None,
        "verified": [],
        "failed": [],
        "skipped": [],
    }
    counts = Counter(record["model_id"] for record in records)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(record["model_id"], record)
    for model_id, count in sorted(counts.items()):
        if count > 1:
            findings.append(_f("duplicate_model_id", model_id, detail="duplicate inventory id"))
    rows = []
    for model_id in sorted(unique):
        record = unique[model_id]
        facts = _facts_for(record, root)
        if probe:
            facts = _probe_facts_for(record, facts, root, timeout=probe_timeout, state=probe_state)
        row = _evaluate_model(record, facts, root)
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
        "intake": {
            "registry": _intake_label(Path(registry), root) if registry is not None else None,
            "references": sorted({_intake_label(Path(path), root) for path in reference_paths}),
            "exclusions": sorted(
                exclusions,
                key=lambda item: (item["source"], item["model_id"] or "", item["reason"]),
            ),
            "probe": {
                "enabled": probe_state["enabled"],
                "timeout_seconds": probe_state["timeout_seconds"],
                "verified": sorted(probe_state["verified"]),
                "failed": sorted(probe_state["failed"], key=lambda item: item["model_id"]),
                "skipped": sorted(probe_state["skipped"], key=lambda item: item["model_id"]),
            },
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
    intake, probe = payload["intake"], payload["intake"]["probe"]
    lines = [
        "# Checkpoint Compatibility Audit",
        "",
        f"- Schema: `{payload['schema']}` | Status: `{payload['status']}`",
        f"- Models: {summary['model_count']} (recoverable: {summary['recoverable_count']})"
        f" | Consumers: {summary['consumer_count']} | Findings: {len(findings)}",
        f"- Intake: registry=`{intake['registry'] or '-'}`"
        f" references={len(intake['references'])} exclusions={len(intake['exclusions'])}"
        f" | Probe: {'on' if probe['enabled'] else 'off'}"
        f" verified={len(probe['verified'])} failed={len(probe['failed'])}",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_checkpoint_compatibility", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--input", action="append", default=[], type=Path, help="Overlay.")
    parser.add_argument(
        "--registry", type=Path, default=None, help="Canonical model registry YAML to intake."
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        type=Path,
        dest="references",
        metavar="PATH",
        help="Referencing config/release manifest YAML (repeatable).",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run the opt-in bounded loader probe for local .zip/.pt artifacts.",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT,
        help=f"Hard loader-probe subprocess timeout seconds (default: {DEFAULT_PROBE_TIMEOUT}).",
    )
    parser.add_argument("--root", type=Path, default=None, help="Artifact resolution root.")
    parser.add_argument("--check", action="store_true", help="Exit 1 on blocking findings.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument(
        "--probe-child",
        nargs=2,
        default=None,
        metavar=("PATH", "KIND"),
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and return a shell-friendly exit code."""
    args = _build_parser().parse_args(argv)
    if args.probe_child:
        return _probe_child_main(Path(args.probe_child[0]), args.probe_child[1])
    if not args.input and args.registry is None and not args.references:
        sys.stderr.write("FAIL invalid_input: --input, --registry, or --config is required\n")
        return 2
    if not math.isfinite(args.probe_timeout) or args.probe_timeout <= 0:
        sys.stderr.write("FAIL invalid_input: --probe-timeout must be finite and > 0\n")
        return 2
    root = (args.root or Path.cwd()).resolve()
    payload = build_audit(
        inputs=args.input,
        registry=args.registry,
        references=args.references,
        probe=args.probe,
        probe_timeout=args.probe_timeout,
        root=root,
    )
    sys.stdout.write(
        render_markdown(payload) if args.format == "markdown" else render_json(payload)
    )
    if payload["status"] == "unknown":
        return 2
    return 0 if payload["ok"] or not args.check else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
