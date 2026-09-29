#!/usr/bin/env python3
"""Audit paired 0.0.7/0.0.8 episode outcomes and metrics by campaign slot.

The accepted 0.0.7 publication bundle is the only default baseline. A rule
records an analyst's causal claim and evidence; matching a rule is not itself
proof of causality or release admission.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from scripts.analysis.compare_issue_9431_release import (
    EXPECTED_SUCCESSOR_SCENARIO_MANIFEST,
    EXPECTED_SUCCESSOR_SCENARIO_MANIFEST_SHA256,
    _archive_source_commit,
    _bundle_campaign_id,
    _bundle_scenario_identity,
    _row_source_commit,
    _sha256,
    _verify_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

BASELINE_SHA256 = "684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f"
BASELINE_SOURCE = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"
BASELINE_CAMPAIGN = "issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922"
BASELINE_PRESERVATION_DIGEST = "5eb0d68e1483f3d82e75c33c3966a1c597330816f911a0caeaecbde119d4379f"
V4_SLOT_REPLACEMENTS = {
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield": "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4",
    "scenario_adaptive_hybrid_orca_v2_collision_guard": "scenario_adaptive_hybrid_orca_v2_collision_guard_v4",
    "hybrid_rule_v3_fast_progress_static_escape": "hybrid_rule_v4_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous": "hybrid_rule_v4_fast_progress_static_escape_continuous",
}
V4_PREDECESSORS = {new: old for old, new in V4_SLOT_REPLACEMENTS.items()}
TOLERANCE = 1e-12
SLOT_COLUMNS = ("planner", "kinematics", "scenario_id", "seed", "benchmark_track")
FINDING_COLUMNS = (
    "finding_id",
    *SLOT_COLUMNS,
    "replacement_planner",
    "presence",
    "field",
    "old_value",
    "new_value",
    "delta_0_0_8_minus_0_0_7",
    "classification",
    "issue",
    "explanation",
    "evidence",
    "rule_id",
)
SUMMARY_COLUMNS = (
    "planner",
    "kinematics",
    "scenario_id",
    "benchmark_track",
    "field",
    "paired_count",
    "changed_count",
    "mean_0_0_7",
    "mean_0_0_8",
    "mean_paired_delta",
    "only_0_0_7",
    "only_0_0_8",
)
MISSING = object()


def _run_identity(name: str) -> tuple[str, str]:
    if "__" not in name:
        raise ValueError(f"run directory lacks planner__kinematics: {name}")
    planner, kinematics = name.rsplit("__", 1)
    if not planner or not kinematics:
        raise ValueError(f"invalid planner/kinematics run directory: {name}")
    return planner, kinematics


def _slot(row: Mapping[str, Any], run_name: str) -> tuple[str, str, str, int, str]:
    planner, kinematics = _run_identity(run_name)
    scenario = row.get("scenario_id")
    seed = row.get("seed")
    track = row.get("benchmark_track")
    if not isinstance(scenario, str) or not scenario or type(seed) is not int:
        raise ValueError(f"invalid scenario_id/seed in {run_name}: {scenario!r}/{seed!r}")
    if track is not None and (not isinstance(track, str) or not track):
        raise ValueError(f"invalid benchmark_track in {run_name}: {track!r}")
    return planner, kinematics, scenario, seed, track or ""


def _insert_rows(
    rows: dict[tuple[str, str, str, int, str], dict[str, Any]],
    raw_lines: Any,
    run_name: str,
    source: str,
    *,
    retain_provenance: bool = False,
) -> None:
    for line_number, raw in enumerate(raw_lines, 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: episode row must be an object")
        key = _slot(row, run_name)
        if key in rows:
            raise ValueError(f"duplicate slot {key} at {source}:{line_number}")
        if not isinstance(row.get("outcome"), dict) or not isinstance(row.get("metrics"), dict):
            raise ValueError(f"{source}:{line_number}: outcome and metrics must be objects")
        # The published bundle is hundreds of MB uncompressed. Keep only the
        # compared values and a checked source identity for each slot.
        rows[key] = {
            "outcome": row["outcome"],
            "metrics": row["metrics"],
            "_source_commit": _row_source_commit(row),
        }
        if retain_provenance:
            rows[key]["_provenance"] = {
                name: row.get(name)
                for name in (
                    "algo",
                    "planner_key",
                    "scenario_params",
                    "algorithm_metadata",
                    "provenance",
                    "config_hash",
                )
            }


def _bundle_rows(bundle: Path) -> dict[tuple[str, str, str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    with tarfile.open(bundle, "r:gz") as archive:
        members = sorted(
            (
                member
                for member in archive
                if member.isfile()
                and "/payload/runs/" in member.name
                and member.name.endswith("/episodes.jsonl")
            ),
            key=lambda member: member.name,
        )
        if not members:
            raise ValueError("baseline bundle contains no payload/runs/*/episodes.jsonl")
        for member in members:
            relative = member.name.rsplit("/payload/runs/", 1)[1]
            if len(Path(relative).parts) != 2:
                raise ValueError(f"unexpected episode path in baseline: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read baseline member {member.name}")
            _insert_rows(rows, stream, Path(relative).parent.name, member.name)
    return rows


def _preserved_entries(root: Path, expected_digest: str) -> dict[str, dict[str, Any]]:
    path = root / "campaign_preservation_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "campaign-preservation-manifest.v1"
    ):
        raise ValueError("0.0.7 preservation manifest has wrong schema")
    recorded = manifest.pop("manifest_digest", "")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    if recorded != f"sha256:{digest}" or digest != expected_digest:
        raise ValueError("0.0.7 preservation manifest digest mismatch")
    if manifest.get("campaign_id") != BASELINE_CAMPAIGN:
        raise ValueError("0.0.7 preserved campaign ID differs from accepted publication")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("0.0.7 preservation manifest has no file list")
    entries: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("invalid preserved file entry")
        if entry["path"] in entries:
            raise ValueError(f"duplicate preserved file {entry['path']}")
        entries[entry["path"]] = entry
    return entries


def _verified_preserved_path(root: Path, entry: Mapping[str, Any]) -> Path:
    name = entry["path"]
    stored = entry.get("stored_path")
    if (
        not isinstance(name, str)
        or stored != f"{name}.gz"
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError(f"invalid preserved path {name!r}")
    path = root / stored
    if _sha256(path) != entry.get("stored_sha256"):
        raise ValueError(f"stored SHA-256 mismatch for {name}")
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry.get("sha256"):
        raise ValueError(f"uncompressed SHA-256 mismatch for {name}")
    return path


def _preserved_json(root: Path, entries: Mapping[str, dict[str, Any]], name: str) -> dict[str, Any]:
    entry = entries.get(name)
    if entry is None:
        raise ValueError(f"0.0.7 preservation manifest lacks {name}")
    path = _verified_preserved_path(root, entry)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"preserved {name} must be an object")
    return value


def _preserved_rows(
    root: Path, entries: Mapping[str, dict[str, Any]]
) -> dict[tuple[str, str, str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    names = sorted(
        name for name in entries if name.startswith("runs/") and name.endswith("/episodes.jsonl")
    )
    if not names:
        raise ValueError("0.0.7 preservation manifest has no run rows")
    for name in names:
        parts = Path(name).parts
        if len(parts) != 3:
            raise ValueError(f"unexpected preserved episode path: {name}")
        path = _verified_preserved_path(root, entries[name])
        with gzip.open(path, "rb") as stream:
            _insert_rows(rows, stream, parts[1], name)
    return rows


def _baseline_from_root(root: Path, expected_digest: str) -> tuple[dict, dict]:
    entries = _preserved_entries(root, expected_digest)
    campaign = _preserved_json(root, entries, "campaign_manifest.json")
    release = _preserved_json(root, entries, "release/release_manifest.resolved.json")
    git = campaign.get("git")
    source = git.get("commit") if isinstance(git, dict) else None
    if source != BASELINE_SOURCE or release.get("source_sha") != BASELINE_SOURCE:
        raise ValueError("preserved 0.0.7 source differs from accepted publication")
    scenario = release.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError("preserved 0.0.7 release lacks scenario identity")
    identity = {"manifest": scenario.get("matrix_path"), "sha256": scenario.get("matrix_sha256")}
    return _preserved_rows(root, entries), {
        "source_commit": source,
        "scenario": identity,
        "preservation_manifest_digest": expected_digest,
        "campaign_id": campaign.get("campaign_id"),
    }


def _root_rows(root: Path) -> dict[tuple[str, str, str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    paths = sorted(root.glob("runs/*/episodes.jsonl"))
    if not paths:
        raise ValueError(f"0.0.8 root has no runs/*/episodes.jsonl: {root}")
    for path in paths:
        with path.open("rb") as stream:
            _insert_rows(rows, stream, path.parent.name, str(path), retain_provenance=True)
    return rows


def _hex_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} requires a 64-character lowercase SHA-256")
    return value


def _source_bytes(source_root: Path, commit: str, name: str) -> bytes:
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError(f"invalid successor source path: {name!r}")
    result = subprocess.run(
        ["git", "-C", str(source_root), "show", f"{commit}:{name}"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"successor source commit lacks {name}")
    return result.stdout


def _runtime_successor_identity(
    source_root: Path,
    commit: str,
    config_path: str,
    rows: Mapping[tuple[str, str, str, int, str], dict[str, Any]],
) -> tuple[str, str, dict[tuple[str, str, str, int, str], dict[str, Any]]]:
    """Recreate the campaign runner's hashes from a detached source checkout."""
    with tempfile.TemporaryDirectory(prefix="slot-paired-source-") as directory:
        checkout = Path(directory) / "source"
        result = subprocess.run(
            ["git", "-C", str(source_root), "worktree", "add", "--detach", str(checkout), commit],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise ValueError("cannot check out pinned successor source commit")
        try:
            worker = Path(__file__).with_name("_pinned_successor_runtime.py")
            request = {
                "config_path": config_path,
                "rows": [
                    {"slot": slot, "scenario_params": row["_provenance"]["scenario_params"]}
                    for slot, row in rows.items()
                ],
            }
            resolved = subprocess.run(
                [sys.executable, "-I", str(worker)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                cwd=checkout,
                check=False,
            )
            if resolved.returncode:
                raise ValueError(
                    f"pinned successor runtime resolution failed: {resolved.stderr.strip()}"
                )
            payload = json.loads(resolved.stdout)
            runtime_rows = {tuple(item.pop("slot")): item for item in payload["rows"]}
            return payload["config_hash"], payload["scenario_hash"], runtime_rows
        finally:
            subprocess.run(
                ["git", "-C", str(source_root), "worktree", "remove", "--force", str(checkout)],
                capture_output=True,
                check=False,
            )


def _verified_successor_manifest(  # noqa: C901, PLR0912
    path: Path,
    digest: str,
    source_root: Path,
    rows: Mapping[tuple[str, str, str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    _verify_sha256(
        path, _hex_digest(digest, "successor manifest digest"), label="successor manifest SHA-256"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "slot-paired-successor.v1"
    ):
        raise ValueError("successor manifest requires schema_version slot-paired-successor.v1")
    if manifest.get("release") != "0.0.8":
        raise ValueError("successor manifest release must be 0.0.8")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("successor manifest requires a full source commit")
    if not isinstance(manifest.get("campaign_id"), str) or not manifest["campaign_id"]:
        raise ValueError("successor manifest requires campaign_id")
    for key in ("campaign_config", "scenario_matrix"):
        binding = manifest.get(key)
        if not isinstance(binding, dict):
            raise ValueError(f"successor manifest requires {key} binding")
        expected = _hex_digest(binding.get("sha256"), f"{key} sha256")
        actual = hashlib.sha256(_source_bytes(source_root, commit, binding.get("path"))).hexdigest()
        if actual != expected:
            raise ValueError(f"successor {key} SHA-256 mismatch")
        if not isinstance(binding.get("runtime_hash"), str):
            raise ValueError(f"successor {key} requires runtime_hash")
    config = yaml.safe_load(_source_bytes(source_root, commit, manifest["campaign_config"]["path"]))
    if (
        not isinstance(config, dict)
        or config.get("scenario_matrix") != manifest["scenario_matrix"]["path"]
    ):
        raise ValueError("successor config scenario binding mismatch")
    planners = config.get("planners")
    if not isinstance(planners, list):
        raise ValueError("successor config lacks planner bindings")
    configured = {item.get("key"): item for item in planners if isinstance(item, dict)}
    if len(configured) != len(planners) or any(
        not isinstance(key, str) or not key for key in configured
    ):
        raise ValueError("successor config has duplicate or invalid planner keys")
    bindings = manifest.get("versioned_planner_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("successor manifest requires versioned_planner_bindings")
    versioned = set(V4_SLOT_REPLACEMENTS.values())
    if not versioned <= configured.keys():
        raise ValueError("successor config lacks required v4 planner bindings")
    if set(bindings) != versioned:
        raise ValueError("successor versioned planner binding set mismatch")
    for key, binding in bindings.items():
        if not isinstance(binding, dict) or configured[key].get("algo_config") != binding.get(
            "path"
        ):
            raise ValueError(f"successor planner binding mismatch: {key}")
        expected = _hex_digest(binding.get("sha256"), f"{key} sha256")
        actual = hashlib.sha256(_source_bytes(source_root, commit, binding["path"])).hexdigest()
        if actual != expected:
            raise ValueError(f"successor planner binding SHA-256 mismatch: {key}")
    config_hash, scenario_hash, runtime_rows = _runtime_successor_identity(
        source_root, commit, manifest["campaign_config"]["path"], rows
    )
    for key, actual in (("campaign_config", config_hash), ("scenario_matrix", scenario_hash)):
        if manifest[key]["runtime_hash"] != actual:
            raise ValueError(f"successor {key} runtime_hash differs from pinned source")
    manifest["manifest_sha256"] = digest
    manifest["planner_keys"] = sorted(configured)
    manifest["runtime_rows"] = runtime_rows
    return manifest


def _root_identity(root: Path, expected: Mapping[str, Any]) -> dict[str, str]:
    path = root / "campaign_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("0.0.8 campaign manifest must be an object")
    git = manifest.get("git")
    source = git.get("commit") if isinstance(git, dict) else None
    campaign_id = manifest.get("campaign_id")
    if campaign_id != expected["campaign_id"] or source != expected["source_commit"]:
        raise ValueError("0.0.8 campaign ID/source differs from verified successor manifest")
    checks = {
        "config_hash": expected["campaign_config"]["runtime_hash"],
        "scenario_matrix": expected["scenario_matrix"]["path"],
        "scenario_matrix_hash": expected["scenario_matrix"]["runtime_hash"],
    }
    for key, value in checks.items():
        if manifest.get(key) != value:
            raise ValueError(f"0.0.8 {key} differs from verified successor manifest")
    return {
        "campaign_id": campaign_id,
        "source_commit": source,
        "scenario_matrix": str(manifest.get("scenario_matrix", "")),
        "scenario_matrix_hash": str(manifest.get("scenario_matrix_hash", "")),
        "config_hash": str(manifest.get("config_hash", "")),
    }


def _matches_config_path(observed: Any, expected: str | None) -> bool:
    if expected is None:
        return observed is None
    if not isinstance(observed, str) or not observed:
        return False
    observed_parts = Path(observed).parts
    expected_parts = Path(expected).parts
    return observed_parts[-len(expected_parts) :] == expected_parts


def _validate_successor_row(  # noqa: C901 - each provenance assertion fails independently
    slot: tuple[str, str, str, int, str],
    row: Mapping[str, Any],
    runtime_rows: Mapping[str, Any],
    source_commit: str,
) -> None:
    """Check the recorded algorithm and effective config against the pinned arm."""
    planner = runtime_rows[slot]
    recorded = row["_provenance"]
    scenario = recorded["scenario_params"]
    metadata = recorded["algorithm_metadata"]
    provenance = recorded["provenance"]
    if not isinstance(scenario, dict) or not isinstance(metadata, dict):
        raise ValueError(f"0.0.8 row lacks effective planner provenance at {slot}")
    if not any(field in scenario for field in ("name", "id", "scenario_id")) or any(
        scenario[field] != slot[2] for field in ("name", "id", "scenario_id") if field in scenario
    ):
        raise ValueError(f"0.0.8 row scenario provenance differs from run slot at {slot}")
    for field, expected in planner["scenario"].items():
        if field not in {"seed", "seeds"} and (
            field not in scenario or scenario[field] != expected
        ):
            raise ValueError(f"0.0.8 row {field} differs from pinned scenario at {slot}")
    if recorded["algo"] != planner["algo"] or scenario.get("algo") != planner["algo"]:
        raise ValueError(f"0.0.8 row algorithm differs from configured planner at {slot}")
    if metadata.get("algorithm") != planner["algo"]:
        raise ValueError(f"0.0.8 row algorithm metadata differs from configured planner at {slot}")
    if recorded["planner_key"] is not None and recorded["planner_key"] != slot[0]:
        raise ValueError(f"0.0.8 row planner_key differs from run directory at {slot}")
    if (
        metadata.get("config") != planner["config"]
        or metadata.get("config_hash") != planner["config_hash"]
    ):
        raise ValueError(f"0.0.8 row effective planner config differs from pinned source at {slot}")
    if recorded["config_hash"] != planner["scenario_config_hash"]:
        raise ValueError(
            f"0.0.8 row scenario config_hash differs from effective scenario at {slot}"
        )
    if scenario.get("algo_config_hash", planner["config_hash"]) != planner["config_hash"]:
        raise ValueError(
            f"0.0.8 row scenario planner config hash differs from pinned source at {slot}"
        )
    if not isinstance(provenance, dict) or provenance.get("commit_hash") != source_commit:
        raise ValueError(f"0.0.8 row run provenance source differs from campaign at {slot}")
    identity = provenance.get("config_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("algo") != planner["algo"]
        or not _matches_config_path(identity.get("algo_config_path"), planner["path"])
    ):
        raise ValueError(f"0.0.8 row run provenance differs from configured planner at {slot}")


def _fields(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}", child)
        else:
            result[prefix] = value

    for section in ("outcome", "metrics"):
        visit(section, row[section])
    return result


def _number(value: Any) -> float | None:
    if type(value) is bool:
        return float(value)
    if type(value) in (int, float):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "<NaN>"
        return "<Infinity>" if value > 0 else "<-Infinity>"
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    return value


def _different(old: Any, new: Any) -> bool:
    if old is MISSING or new is MISSING:
        return True
    old_number, new_number = _number(old), _number(new)
    if (
        old_number is not None
        and new_number is not None
        and (type(old) is bool) == (type(new) is bool)
    ):
        return abs(new_number - old_number) > TOLERANCE
    return type(old) is not type(new) or _json_safe(old) != _json_safe(new)


def _read_rules(path: Path | None) -> list[dict[str, Any]]:  # noqa: C901, PLR0912
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "slot-paired-classifications.v2":
        raise ValueError(
            "classification file requires schema_version slot-paired-classifications.v2"
        )
    rules = data.get("rules")
    if not isinstance(rules, list):
        raise ValueError("classification file requires a rules list")
    seen_ids: set[str] = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict) or set(rule) != {
            "rule_id",
            "slots",
            "fields",
            "predicate",
            "max_findings",
            "classification",
            "issue",
            "explanation",
            "evidence",
        }:
            raise ValueError(f"rule {index} has missing or unknown keys")
        for key in ("rule_id", "classification", "issue", "explanation", "evidence"):
            if not isinstance(rule.get(key), str) or not rule[key].strip():
                raise ValueError(f"rule {index} requires nonempty {key}")
        if rule["rule_id"] in seen_ids:
            raise ValueError(f"duplicate rule_id {rule['rule_id']}")
        seen_ids.add(rule["rule_id"])
        if rule["classification"].lower() in {"unexplained", "unknown", "todo"}:
            raise ValueError(f"rule {index} cannot classify a finding as unexplained")
        if type(rule["max_findings"]) is not int or rule["max_findings"] < 1:
            raise ValueError(f"rule {index} requires positive max_findings")
        if (
            not isinstance(rule["fields"], list)
            or not rule["fields"]
            or any(not isinstance(field, str) or not field for field in rule["fields"])
        ):
            raise ValueError(f"rule {index} requires fields")
        if not isinstance(rule["slots"], list) or not rule["slots"]:
            raise ValueError(f"rule {index} requires exact slots")
        for slot in rule["slots"]:
            if not isinstance(slot, dict) or set(slot) != {
                *SLOT_COLUMNS[:-2],
                "benchmark_track",
                "seeds",
            }:
                raise ValueError(
                    f"rule {index} requires planner, kinematics, scenario_id, benchmark_track, seeds"
                )
            if any(
                not isinstance(slot[key], str) or not slot[key]
                for key in ("planner", "kinematics", "scenario_id")
            ) or not isinstance(slot["benchmark_track"], str):
                raise ValueError(f"rule {index} has invalid slot")
            seeds = slot["seeds"]
            if seeds != "all" and (
                not isinstance(seeds, list)
                or not seeds
                or any(type(seed) is not int for seed in seeds)
            ):
                raise ValueError(f"rule {index} requires explicit seeds or all")
        predicate = rule["predicate"]
        if not isinstance(predicate, dict):
            raise ValueError(f"rule {index} requires predicate")
        if set(predicate) == {"finding_ids"}:
            if (
                not isinstance(predicate["finding_ids"], list)
                or not predicate["finding_ids"]
                or any(not isinstance(item, str) or not item for item in predicate["finding_ids"])
            ):
                raise ValueError(f"rule {index} requires finding IDs")
        elif set(predicate) == {"sign", "max_abs_delta"}:
            if (
                predicate["sign"] not in ("positive", "negative")
                or type(predicate["max_abs_delta"]) not in (int, float)
                or not math.isfinite(predicate["max_abs_delta"])
                or predicate["max_abs_delta"] <= 0
            ):
                raise ValueError(f"rule {index} requires sign and positive finite bound")
        else:
            raise ValueError(f"rule {index} has invalid predicate")
    return rules


def _rule_matches(rule: Mapping[str, Any], finding: Mapping[str, Any]) -> bool:
    slot_match = any(
        all(
            finding[key] == slot[key]
            for key in ("planner", "kinematics", "scenario_id", "benchmark_track")
        )
        and (slot["seeds"] == "all" or finding["seed"] in slot["seeds"])
        for slot in rule["slots"]
    )
    if not slot_match or finding["field"] not in rule["fields"]:
        return False
    predicate = rule["predicate"]
    if "finding_ids" in predicate:
        return finding["finding_id"] in predicate["finding_ids"]
    delta = finding["delta_0_0_8_minus_0_0_7"]
    return (
        delta is not None
        and (delta > 0 if predicate["sign"] == "positive" else delta < 0)
        and abs(delta) <= predicate["max_abs_delta"]
    )


def _display(value: Any) -> str:
    return (
        "<missing>"
        if value is MISSING
        else json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False, allow_nan=False)
    )


def compare(  # noqa: C901, PLR0912, PLR0913, PLR0915
    baseline_bundle: Path | None,
    successor_root: Path,
    *,
    successor_manifest: Path,
    successor_manifest_sha256: str,
    successor_source_root: Path,
    baseline_root: Path | None = None,
    classification_file: Path | None = None,
    baseline_sha256: str = BASELINE_SHA256,
    baseline_manifest_digest: str = BASELINE_PRESERVATION_DIGEST,
    baseline_source: str = BASELINE_SOURCE,
    expected_scenario_identity: dict[str, str] | None = None,
    broad_rule_bound_threshold: float = 1.0,
) -> dict[str, Any]:
    """Return an audit report; unexplained findings remain visible in the result."""
    if not math.isfinite(broad_rule_bound_threshold) or broad_rule_bound_threshold < 0:
        raise ValueError("broad rule bound threshold must be finite and nonnegative")
    if (baseline_bundle is None) == (baseline_root is None):
        raise ValueError("provide exactly one 0.0.7 bundle or preserved artifact root")
    if baseline_bundle is not None:
        digest = _verify_sha256(baseline_bundle, baseline_sha256, label="0.0.7 bundle SHA-256")
        source = _archive_source_commit(baseline_bundle)
        if source != baseline_source:
            raise ValueError(f"0.0.7 bundle source mismatch: {source} != {baseline_source}")
        if (
            baseline_sha256 == BASELINE_SHA256
            and _bundle_campaign_id(baseline_bundle) != BASELINE_CAMPAIGN
        ):
            raise ValueError("0.0.7 bundle campaign ID differs from accepted publication")
        scenario_identity = _bundle_scenario_identity(baseline_bundle)
        old = _bundle_rows(baseline_bundle)
        baseline_identity = {
            "bundle_sha256": digest,
            "source_commit": source,
            "scenario": scenario_identity,
        }
    else:
        old, baseline_identity = _baseline_from_root(baseline_root, baseline_manifest_digest)
        source = baseline_identity["source_commit"]
        scenario_identity = baseline_identity["scenario"]
    expected = expected_scenario_identity or {
        "manifest": EXPECTED_SUCCESSOR_SCENARIO_MANIFEST,
        "sha256": EXPECTED_SUCCESSOR_SCENARIO_MANIFEST_SHA256,
    }
    if scenario_identity != expected:
        raise ValueError(f"0.0.7 scenario identity mismatch: {scenario_identity} != {expected}")
    new = _root_rows(successor_root)
    verified_successor = _verified_successor_manifest(
        successor_manifest, successor_manifest_sha256, successor_source_root, new
    )
    successor_identity = _root_identity(successor_root, verified_successor)
    if any(key[0] not in verified_successor["planner_keys"] for key in new):
        raise ValueError("0.0.8 row planner is absent from verified successor config")
    for key, row in old.items():
        if row["_source_commit"] != baseline_source:
            raise ValueError(f"0.0.7 row has wrong source at {key}")
    for key, row in new.items():
        if row["_source_commit"] != successor_identity["source_commit"]:
            raise ValueError(f"0.0.8 row source differs from campaign manifest at {key}")
        _validate_successor_row(
            key, row, verified_successor["runtime_rows"], successor_identity["source_commit"]
        )
    rules = _read_rules(classification_file)
    broad_rules = []
    for rule in rules:
        reasons = []
        if any(slot["seeds"] == "all" for slot in rule["slots"]):
            reasons.append("seeds: all")
        bound = rule["predicate"].get("max_abs_delta")
        if bound is not None and bound > broad_rule_bound_threshold:
            reasons.append(f"max_abs_delta: {bound}")
        if reasons:
            broad_rules.append({"rule_id": rule["rule_id"], "reasons": reasons})
    findings: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def summary(slot: tuple[str, str, str, int, str], field: str) -> dict[str, Any]:
        planner, kinematics, scenario, _, track = slot
        key = planner, kinematics, scenario, track, field
        if key not in summaries:
            summaries[key] = {
                "planner": planner,
                "kinematics": kinematics,
                "scenario_id": scenario,
                "benchmark_track": track,
                "field": field,
                "paired_count": 0,
                "changed_count": 0,
                "mean_0_0_7": None,
                "mean_0_0_8": None,
                "mean_paired_delta": None,
                "only_0_0_7": 0,
                "only_0_0_8": 0,
                "_old": [],
                "_new": [],
                "_delta": [],
            }
        return summaries[key]

    def add_finding(
        slot: tuple[str, str, str, int, str],
        presence: str,
        field: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        old_number = _number(old_value) if old_value is not MISSING else None
        new_number = _number(new_value) if new_value is not MISSING else None
        delta = (
            new_number - old_number if old_number is not None and new_number is not None else None
        )
        finding = dict(zip(SLOT_COLUMNS, slot, strict=True))
        finding["replacement_planner"] = (
            V4_SLOT_REPLACEMENTS.get(slot[0], V4_PREDECESSORS.get(slot[0], ""))
            if field == "__row__"
            else ""
        )
        old_display = (
            "<row present>"
            if field == "__row__" and old_value is not MISSING
            else _display(old_value)
        )
        new_display = (
            "<row present>"
            if field == "__row__" and new_value is not MISSING
            else _display(new_value)
        )
        finding.update(
            {
                "presence": presence,
                "field": field,
                "old_value": old_display,
                "new_value": new_display,
                "delta_0_0_8_minus_0_0_7": delta,
            }
        )
        identity = {
            key: finding[key]
            for key in (*SLOT_COLUMNS, "presence", "field", "old_value", "new_value")
        }
        finding["finding_id"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        finding.update(
            {
                "classification": "unexplained",
                "issue": "",
                "explanation": "",
                "evidence": "",
                "rule_id": "",
            }
        )
        findings.append(finding)

    for slot in sorted(set(old) | set(new)):
        old_row, new_row = old.get(slot), new.get(slot)
        if old_row is None or new_row is None:
            presence = "only_0_0_7" if new_row is None else "only_0_0_8"
            add_finding(
                slot,
                presence,
                "__row__",
                old_row if old_row is not None else MISSING,
                new_row if new_row is not None else MISSING,
            )
            for field in _fields(old_row or new_row):
                summary(slot, field)[presence] += 1
            continue
        old_fields, new_fields = _fields(old_row), _fields(new_row)
        for field in sorted(set(old_fields) | set(new_fields)):
            old_value = old_fields.get(field, MISSING)
            new_value = new_fields.get(field, MISSING)
            item = summary(slot, field)
            if old_value is not MISSING and new_value is not MISSING:
                item["paired_count"] += 1
                a, b = _number(old_value), _number(new_value)
                if a is not None and b is not None:
                    item["_old"].append(a)
                    item["_new"].append(b)
                    item["_delta"].append(b - a)
            else:
                item["only_0_0_7" if new_value is MISSING else "only_0_0_8"] += 1
            if _different(old_value, new_value):
                item["changed_count"] += 1
                add_finding(slot, "paired", field, old_value, new_value)

    for item in summaries.values():
        for private, public in (
            ("_old", "mean_0_0_7"),
            ("_new", "mean_0_0_8"),
            ("_delta", "mean_paired_delta"),
        ):
            values = item.pop(private)
            item[public] = sum(values) / len(values) if values else None
    rule_coverage = []
    rule_matches = [
        [finding for finding in findings if _rule_matches(rule, finding)] for rule in rules
    ]
    over_limit_findings = {
        finding["finding_id"]
        for rule, matched in zip(rules, rule_matches, strict=True)
        if len(matched) > rule["max_findings"]
        for finding in matched
    }
    claimed: set[str] = set()
    for rule, matched in zip(rules, rule_matches, strict=True):
        over_limit = len(matched) > rule["max_findings"]
        covered = []
        if not over_limit:
            for finding in matched:
                if finding["finding_id"] in over_limit_findings:
                    continue
                if finding["finding_id"] in claimed:
                    raise ValueError(f"ambiguous classification rules for {finding['finding_id']}")
                claimed.add(finding["finding_id"])
                finding.update(
                    {
                        key: rule[key]
                        for key in ("classification", "issue", "explanation", "evidence", "rule_id")
                    }
                )
                covered.append(finding["finding_id"])
        rule_coverage.append(
            {
                "rule_id": rule["rule_id"],
                "max_findings": rule["max_findings"],
                "over_limit": over_limit,
                "covered_findings": covered,
                "matching_findings": [finding["finding_id"] for finding in matched],
            }
        )
    unexplained = sum(finding["classification"] == "unexplained" for finding in findings)
    return {
        "schema_version": "slot-paired-release-diff.v1",
        "baseline": {"release": "0.0.7", **baseline_identity},
        "successor": {
            "release": "0.0.8",
            "root": str(successor_root),
            "verified_manifest_sha256": successor_manifest_sha256,
            **successor_identity,
        },
        "slot_columns": list(SLOT_COLUMNS),
        "v4_slot_replacements": V4_SLOT_REPLACEMENTS,
        "numeric_tolerance_absolute": TOLERANCE,
        "rows_0_0_7": len(old),
        "rows_0_0_8": len(new),
        "paired_rows": len(set(old) & set(new)),
        "only_0_0_7": len(set(old) - set(new)),
        "only_0_0_8": len(set(new) - set(old)),
        "unexplained_count": unexplained,
        "status": "classified" if unexplained == 0 else "unexplained",
        "findings": findings,
        "rules": rule_coverage,
        "broad_rule_bound_threshold": broad_rule_bound_threshold,
        "broad_rules": broad_rules,
        "planner_scenario_metrics": [summaries[key] for key in sorted(summaries)],
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    """Write machine-readable findings, aggregates, and a short reader report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    for filename, columns, rows in (
        ("findings.csv", FINDING_COLUMNS, report["findings"]),
        ("planner_scenario_metrics.csv", SUMMARY_COLUMNS, report["planner_scenario_metrics"]),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# 0.0.7 to 0.0.8 slot-paired audit",
        "",
        f"- Baseline identity: `{report['baseline'].get('bundle_sha256') or report['baseline']['preservation_manifest_digest']}`",
        f"- Rows: 0.0.7 `{report['rows_0_0_7']}`, 0.0.8 `{report['rows_0_0_8']}`, paired `{report['paired_rows']}`.",
        f"- Release-only rows: 0.0.7 `{report['only_0_0_7']}`, 0.0.8 `{report['only_0_0_8']}`.",
        f"- Findings: `{len(report['findings'])}`; unexplained: `{report['unexplained_count']}`.",
        f"- Verified successor manifest SHA-256: `{report['successor']['verified_manifest_sha256']}`.",
        "- Classification rules are analyst claims; this audit does not prove causality or admit a release.",
        "",
        "## Broad rules",
        "",
        f"Rules with seeds `all` or max_abs_delta above `{report['broad_rule_bound_threshold']}` (review first; not rejected):",
        *(f"- `{rule['rule_id']}`: {', '.join(rule['reasons'])}" for rule in report["broad_rules"]),
        *([] if report["broad_rules"] else ["- (none)"]),
        "",
        "## Rule coverage",
        "",
        *(
            f"- `{rule['rule_id']}`: {len(rule['covered_findings'])} covered, over limit `{rule['over_limit']}`; finding IDs: {', '.join(rule['covered_findings']) or '(none)'}"
            for rule in report["rules"]
        ),
        "",
        "See `findings.csv` for every changed field and release-only row, and",
        "`planner_scenario_metrics.csv` for paired means and differences.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run the audit and fail when any observed difference is unexplained."""
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-bundle", type=Path)
    baseline.add_argument("--baseline-root", type=Path)
    parser.add_argument("--successor-root", required=True, type=Path)
    parser.add_argument("--successor-manifest", required=True, type=Path)
    parser.add_argument("--successor-manifest-sha256", required=True)
    parser.add_argument("--successor-source-root", required=True, type=Path)
    parser.add_argument("--classification-file", type=Path)
    parser.add_argument("--broad-rule-bound-threshold", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = compare(
            args.baseline_bundle,
            args.successor_root,
            successor_manifest=args.successor_manifest,
            successor_manifest_sha256=args.successor_manifest_sha256,
            successor_source_root=args.successor_source_root,
            baseline_root=args.baseline_root,
            classification_file=args.classification_file,
            broad_rule_bound_threshold=args.broad_rule_bound_threshold,
        )
        write_report(report, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        parser.exit(2, f"comparison failed: {exc}\n")
    return 0 if report["unexplained_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
