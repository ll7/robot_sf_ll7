#!/usr/bin/env python3
"""Generate and verify exact expected-row ledgers for staged campaigns.

Expands frozen campaign packets through canonical configuration resolvers
into deterministic, ordered row identities without running workloads.
Supports fail-closed validation of completeness and comparison against observed rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

SCHEMA_VERSION = "campaign_expected_row_ledger.v1"
SCHEMA_PATH = Path("docs/contracts/campaign_expected_row_ledger.v1.schema.json")
SUPPORTED_PACKET_CLASSES = {
    "benchmark",
    "bounded_training",
    "replay",
    "compatibility",
    "platform_smoke",
}
ALIAS_PLACEHOLDER_RE = re.compile(r"^<[^<>]+>$|^\{[^{}]+\}$|^%[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class ExpectedRow:
    """Exact expected row identity in a campaign ledger."""

    row_id: str
    campaign_id: str
    arm: str
    scenario_id: str
    seed: int | str
    replicate: int
    expected_execution_mode: str
    horizon: int | None = None
    config_digest: str | None = None
    model_digest: str | None = None
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert row to a JSON-serializable mapping."""
        res: dict[str, Any] = {
            "row_id": self.row_id,
            "campaign_id": self.campaign_id,
            "arm": self.arm,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "replicate": self.replicate,
            "expected_execution_mode": self.expected_execution_mode,
            "horizon": self.horizon,
            "config_digest": self.config_digest,
            "model_digest": self.model_digest,
        }
        if self.metadata is not None:
            res["metadata"] = self.metadata
        return res


@dataclass(frozen=True)
class GridSpec:
    """Dimensions for cross-product row expansion."""

    campaign_id: str
    arms: list[dict[str, Any]]
    scenarios: list[str]
    seeds: list[int | str]
    replicates: int
    horizon: int | None
    config_digest: str | None
    model_digest: str | None
    excluded_cells: set[tuple[str, str]]


def _sha256_canonical(data: Any) -> str:
    """Return SHA-256 hex digest of canonical JSON serialization."""
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_packet(path: Path) -> dict[str, Any]:
    """Read a campaign packet from JSON or YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"Campaign packet file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) if path.suffix.lower() in (".yaml", ".yml") else json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Campaign packet at {path} must be a mapping")
    return data


def _read_observed_rows(path: Path) -> list[dict[str, Any]]:
    """Read observed rows from JSON, JSONL, or CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Observed rows file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [dict(r) for r in csv.DictReader(text.splitlines())]
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
        return parsed["rows"]
    raise ValueError(f"Unrecognized observed row format in {path}")


def _resolve_packet_class(packet: dict[str, Any]) -> str:
    """Determine normalized packet class."""
    declared = (
        packet.get("packet_class")
        or packet.get("archetype")
        or ("benchmark" if "scenario_matrix" in packet or "planners" in packet else "")
    )
    declared_str = str(declared).strip().lower()
    if declared_str:
        return declared_str
    if isinstance(packet.get("campaign"), dict):
        return "benchmark"
    return ""


def _parse_arm_entry(
    entry: Any,
    default_mode: str,
    config_digest: str | None,
    model_digest: str | None,
    blockers: list[str],
) -> dict[str, Any] | None:
    """Parse one arm/planner entry."""
    if isinstance(entry, str):
        key = entry.strip()
        if ALIAS_PLACEHOLDER_RE.match(key):
            blockers.append(f"unknown_alias: unresolved placeholder in arm: {key}")
        return {"arm": key, "execution_mode": default_mode}
    if isinstance(entry, dict):
        key = str(entry.get("key") or entry.get("algo") or entry.get("planner_id", "")).strip()
        if not key:
            blockers.append("under_specified_dimension: arm entry missing key or planner_id")
            return None
        if ALIAS_PLACEHOLDER_RE.match(key):
            blockers.append(f"unknown_alias: unresolved placeholder in arm: {key}")
        mode = str(entry.get("adapter_mode") or entry.get("execution_mode") or default_mode).strip()
        return {
            "arm": key,
            "execution_mode": mode,
            "config_digest": entry.get("config_digest", config_digest),
            "model_digest": entry.get("model_digest", model_digest),
        }
    return None


def _normalize_arms(
    packet: dict[str, Any],
    default_mode: str,
    config_digest: str | None,
    model_digest: str | None,
    blockers: list[str],
) -> list[dict[str, Any]]:
    """Extract and validate normalized arms."""
    raw: list[Any] = []
    if isinstance(packet.get("arms"), list):
        raw = packet["arms"]
    elif isinstance(packet.get("planners"), list):
        raw = packet["planners"]
    elif isinstance(packet.get("planners"), dict) and "rows" in packet["planners"]:
        raw = packet["planners"]["rows"]

    arms: list[dict[str, Any]] = []
    for entry in raw:
        parsed = _parse_arm_entry(entry, default_mode, config_digest, model_digest, blockers)
        if parsed is not None:
            arms.append(parsed)

    for pair in packet.get("paired_arms", []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            a, b = str(pair[0]).strip(), str(pair[1]).strip()
            arms.extend(
                [
                    {"arm": a, "execution_mode": default_mode, "pair_partner": b},
                    {"arm": b, "execution_mode": default_mode, "pair_partner": a},
                ]
            )
    return arms


def _normalize_scenarios(packet: dict[str, Any], blockers: list[str]) -> list[str]:
    """Extract and validate scenario identifiers."""
    raw: list[Any] = []
    if isinstance(packet.get("scenarios"), list):
        raw = packet["scenarios"]
    elif isinstance(packet.get("scenario_suite"), dict):
        raw = packet["scenario_suite"].get("scenario_ids") or packet["scenario_suite"].get(
            "scenario_families", []
        )
    elif "scenario_matrix" in packet:
        m = packet["scenario_matrix"]
        raw = m if isinstance(m, list) else [Path(m).stem]

    scenarios: list[str] = []
    for sc in raw:
        s = str(sc.get("name") if isinstance(sc, dict) else sc).strip()
        if not s:
            blockers.append("under_specified_dimension: scenario entry is empty")
        elif ALIAS_PLACEHOLDER_RE.match(s):
            blockers.append(f"unknown_alias: unresolved placeholder in scenario: {s}")
        scenarios.append(s)
    return scenarios


def _normalize_seeds(packet: dict[str, Any], blockers: list[str]) -> list[int | str]:
    """Extract and validate seeds."""
    raw: list[Any] = []
    if isinstance(packet.get("seeds"), list):
        raw = packet["seeds"]
    elif isinstance(packet.get("seed_policy"), dict):
        raw = packet["seed_policy"].get("seeds", [])

    seeds: list[int | str] = []
    for sd in raw:
        if isinstance(sd, int):
            seeds.append(sd)
        elif isinstance(sd, str) and sd.strip():
            try:
                seeds.append(int(sd.strip()))
            except ValueError:
                seeds.append(sd.strip())
        else:
            blockers.append("under_specified_dimension: seed entry is invalid")
    return seeds


def _expand_grid_rows(spec: GridSpec, blockers: list[str]) -> list[ExpectedRow]:
    """Ordered cross-product expansion of grid dimensions."""
    sorted_arms = sorted(spec.arms, key=lambda a: a["arm"])
    sorted_scenarios = sorted(spec.scenarios)
    sorted_seeds = sorted(spec.seeds, key=lambda s: (0, s) if isinstance(s, int) else (1, str(s)))

    rows: list[ExpectedRow] = []
    seen: set[str] = set()

    for arm_info in sorted_arms:
        arm_name = arm_info["arm"]
        arm_mode = arm_info["execution_mode"]
        arm_cfg = arm_info.get("config_digest", spec.config_digest)
        arm_mdl = arm_info.get("model_digest", spec.model_digest)
        arm_meta = (
            {"pair_partner": arm_info["pair_partner"]} if "pair_partner" in arm_info else None
        )

        for scenario in sorted_scenarios:
            if (arm_name, scenario) in spec.excluded_cells:
                continue
            for seed in sorted_seeds:
                for rep in range(spec.replicates):
                    row_id = f"{spec.campaign_id}::{arm_name}::{scenario}::{seed}::r{rep}"
                    if row_id in seen:
                        blockers.append(f"duplicate_identity: duplicate row_id generated: {row_id}")
                    seen.add(row_id)
                    rows.append(
                        ExpectedRow(
                            row_id=row_id,
                            campaign_id=spec.campaign_id,
                            arm=arm_name,
                            scenario_id=scenario,
                            seed=seed,
                            replicate=rep,
                            expected_execution_mode=arm_mode,
                            horizon=spec.horizon,
                            config_digest=arm_cfg,
                            model_digest=arm_mdl,
                            metadata=arm_meta,
                        )
                    )
    return rows


def expand_campaign_packet(
    packet: dict[str, Any],
    *,
    packet_path: Path | None = None,
) -> tuple[list[ExpectedRow], list[str], str, str]:
    """Expand campaign packet into ordered deterministic rows."""
    blockers: list[str] = []
    packet_class = _resolve_packet_class(packet)
    if not packet_class or packet_class not in SUPPORTED_PACKET_CLASSES:
        blockers.append(
            f"unsupported_packet_class: '{packet_class}' not in {sorted(SUPPORTED_PACKET_CLASSES)}"
        )
        return [], blockers, "", packet_class

    campaign_id = str(
        packet.get("campaign_id")
        or (
            packet.get("campaign", {}).get("id") if isinstance(packet.get("campaign"), dict) else ""
        )
        or packet.get("name", "")
    ).strip()
    if not campaign_id:
        blockers.append("under_specified_dimension: campaign_id is required and non-empty")

    cfg = packet.get("config_path") or packet.get("config")
    if cfg and (".." in str(cfg).split("/") or ".." in str(cfg).split("\\")):
        blockers.append(f"mutable_input: config path traverses parent directories: {cfg}")

    default_mode = str(packet.get("expected_execution_mode", "native")).strip().lower()
    config_digest = packet.get("config_digest") or packet.get("config_sha256")
    model_digest = packet.get("model_digest")
    horizon = int(packet["horizon"]) if packet.get("horizon") is not None else None

    replicates = packet.get("replicates", packet.get("repeats", 1))
    if not isinstance(replicates, int) or replicates <= 0:
        blockers.append(
            f"under_specified_dimension: replicates must be positive integer, got {replicates}"
        )
        replicates = 1

    arms = _normalize_arms(packet, default_mode, config_digest, model_digest, blockers)
    scenarios = _normalize_scenarios(packet, blockers)
    seeds = _normalize_seeds(packet, blockers)

    if not arms:
        blockers.append("under_specified_dimension: arms / planners must be a non-empty list")
    if not scenarios:
        blockers.append("under_specified_dimension: scenarios must be a non-empty list")
    if not seeds:
        blockers.append("under_specified_dimension: seeds must be a non-empty list")

    if blockers:
        return [], blockers, campaign_id, packet_class

    excluded = {
        (
            str(e.get("arm") or e.get("planner_id", "")).strip(),
            str(e.get("scenario_id") or e.get("scenario", "")).strip(),
        )
        for e in packet.get("excluded_cells", [])
        if isinstance(e, dict)
    }

    spec = GridSpec(
        campaign_id=campaign_id,
        arms=arms,
        scenarios=scenarios,
        seeds=seeds,
        replicates=replicates,
        horizon=horizon,
        config_digest=config_digest,
        model_digest=model_digest,
        excluded_cells=excluded,
    )
    rows = _expand_grid_rows(spec, blockers)

    cnt = (
        packet.get("declared_count")
        if packet.get("declared_count") is not None
        else packet.get("expected_rows", packet.get("declared_rows"))
    )
    if cnt is not None and int(cnt) != len(rows):
        blockers.append(
            f"inconsistent_count: declared_count={cnt} differs from expanded rows={len(rows)}"
        )

    rows.sort(
        key=lambda r: (
            r.arm,
            r.scenario_id,
            (0, r.seed) if isinstance(r.seed, int) else (1, str(r.seed)),
            r.replicate,
            r.row_id,
        )
    )
    return rows, blockers, campaign_id, packet_class


def _classify_observed_status(row: dict[str, Any]) -> str:
    """Classify execution mode and status into ledger category."""
    st = str(row.get("row_status", "")).strip().lower()
    mode = str(row.get("execution_mode", "")).strip().lower()
    if st == "fallback" or mode == "fallback":
        return "fallback"
    if (
        st == "degraded"
        or mode == "degraded"
        or st.startswith(("degraded", "policy_step_timeout_fallback"))
    ):
        return "degraded"
    if st in ("failed", "error", "unavailable") or row.get("success") is False:
        return "failed"
    sha, uri = str(row.get("artifact_sha256", "")).strip(), str(row.get("artifact_uri", "")).strip()
    if ("artifact_sha256" in row and len(sha) != 64) or ("artifact_uri" in row and not uri):
        return "provenance_invalid"
    return "present"


def compare_observed_rows(
    expected_rows: list[ExpectedRow],
    observed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare observed rows against expected ledger under fail-closed semantics."""
    expected_by_id = {r.row_id: r for r in expected_rows}
    seen: set[str] = set()
    summary: dict[str, int] = dict.fromkeys(
        (
            "present",
            "missing",
            "duplicate",
            "unexpected",
            "conflict",
            "fallback",
            "degraded",
            "failed",
            "provenance_invalid",
        ),
        0,
    )
    details: dict[str, list[str]] = {k: [] for k in summary}

    for row in observed_rows:
        row_id = str(row.get("row_id", "")).strip()
        if not row_id:
            c, a = str(row.get("campaign_id", "")), str(row.get("arm") or row.get("planner") or "")
            s, sd, r = (
                str(row.get("scenario_id", "")),
                str(row.get("seed", "")),
                str(row.get("replicate", 0)),
            )
            if c and a and s and sd:
                row_id = f"{c}::{a}::{s}::{sd}::r{r}"

        if not row_id or row_id not in expected_by_id:
            summary["unexpected"] += 1
            details["unexpected"].append(row_id or str(row))
            continue
        if row_id in seen:
            summary["duplicate"] += 1
            details["duplicate"].append(row_id)
            continue

        seen.add(row_id)
        exp = expected_by_id[row_id]
        if (
            ("seed" in row and str(row["seed"]) != str(exp.seed))
            or ("scenario_id" in row and str(row["scenario_id"]) != exp.scenario_id)
            or (
                bool(row.get("arm") or row.get("planner"))
                and str(row.get("arm") or row.get("planner")).strip() != exp.arm
            )
        ):
            summary["conflict"] += 1
            details["conflict"].append(row_id)
            continue

        cat = _classify_observed_status(row)
        summary[cat] += 1
        details[cat].append(row_id)

    for rid in expected_by_id:
        if rid not in seen:
            summary["missing"] += 1
            details["missing"].append(rid)

    return {
        "status": "pass" if all(v == 0 for k, v in summary.items() if k != "present") else "fail",
        "observed_count": len(observed_rows),
        "summary": summary,
        "details": details,
    }


def generate_ledger(
    packet_path: Path,
    *,
    observed_rows_path: Path | None = None,
) -> dict[str, Any]:
    """Generate campaign ledger and optionally verify observed rows."""
    packet = _read_packet(packet_path)
    packet_sha = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    rows, blockers, camp_id, pclass = expand_campaign_packet(packet, packet_path=packet_path)

    if blockers:
        return {
            "ok": False,
            "schema": SCHEMA_VERSION,
            "campaign_id": camp_id,
            "packet_class": pclass,
            "packet_path": str(packet_path),
            "packet_sha256": packet_sha,
            "expected_count": len(rows),
            "identity_digest": "",
            "ledger_sha256": "",
            "rows": [],
            "blockers": blockers,
        }

    row_ids = [r.row_id for r in rows]
    identity_digest = hashlib.sha256(
        json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    row_dicts = [r.as_dict() for r in rows]
    ledger_sha = _sha256_canonical(row_dicts)

    payload: dict[str, Any] = {
        "ok": True,
        "schema": SCHEMA_VERSION,
        "campaign_id": camp_id,
        "packet_class": pclass,
        "packet_path": str(packet_path),
        "packet_sha256": packet_sha,
        "expected_count": len(rows),
        "identity_digest": identity_digest,
        "ledger_sha256": ledger_sha,
        "rows": row_dicts,
    }
    if "issue_id" in packet:
        payload["issue_id"] = packet["issue_id"]

    if observed_rows_path:
        comparison = compare_observed_rows(rows, _read_observed_rows(observed_rows_path))
        payload["observed_comparison"] = comparison
        if comparison["status"] != "pass":
            payload["ok"] = False

    return payload


def format_summary(ledger: dict[str, Any]) -> str:
    """Format human-readable compact summary."""
    lines = [
        f"Campaign: {ledger.get('campaign_id')} ({ledger.get('packet_class')})",
        f"Expected Count: {ledger.get('expected_count', 0)} | Status: {'PASS' if ledger.get('ok') else 'FAIL'}",
        f"Identity Digest: {ledger.get('identity_digest', '')[:16]}... | Ledger SHA: {ledger.get('ledger_sha256', '')[:16]}...",
    ]
    lines.extend(f"  Blocker: {b}" for b in ledger.get("blockers", []))
    obs = ledger.get("observed_comparison")
    if obs:
        lines.append(f"Observed: count={obs['observed_count']} verdict={obs['status'].upper()}")
        lines.extend(f"  {cat:<18}: {cnt}" for cat, cnt in obs["summary"].items())
    return "\n".join(lines)


def validate_schema(data: dict[str, Any], schema_path: Path = SCHEMA_PATH) -> list[str]:
    """Validate ledger payload against JSON Schema draft 2020-12."""
    if not schema_path.is_file():
        return [f"Schema file not found: {schema_path}"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    return [f"{'/'.join(map(str, err.path))}: {err.message}" for err in errors]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for campaign expected-row ledger generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True, help="Campaign packet path")
    parser.add_argument("--observed-rows", type=Path, default=None, help="Observed rows path")
    parser.add_argument("--format", choices=["json", "jsonl", "summary"], default="json")
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    parser.add_argument("--check", action="store_true", help="Fail closed on blockers")
    args = parser.parse_args(argv)

    try:
        ledger = generate_ledger(args.packet, observed_rows_path=args.observed_rows)
    except (ValueError, OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if ledger.get("ok") and SCHEMA_PATH.is_file():
        errs = validate_schema({k: v for k, v in ledger.items() if k != "ok"})
        if errs:
            print(f"ERROR: Schema validation failed: {errs}", file=sys.stderr)
            return 2

    if args.format == "json":
        text = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    elif args.format == "jsonl":
        text = "\n".join(json.dumps(r, sort_keys=True) for r in ledger.get("rows", [])) + "\n"
    else:
        text = format_summary(ledger) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    return 2 if (args.check and not ledger.get("ok")) else 0


if __name__ == "__main__":
    sys.exit(main())
