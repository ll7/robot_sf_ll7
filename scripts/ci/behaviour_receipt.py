"""Validate behaviour-change receipts without admitting scientific conclusions.

The release integration owner supplies the independent scope inventory. Missing
scope, job identity or classifications block admission; there is no success-rate
threshold. Independent review must verify the linked job and durable evidence.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
SCOPE_PATH = ROOT / "configs/benchmarks/releases/behaviour_gate_0_1_0.json"
TRIGGERS = (
    "robot_sf/planner/",
    "robot_sf/baselines/",
    "robot_sf/sim/",
    "robot_sf/robot/",
    "robot_sf/nav/",
    "robot_sf/gym_env/",
    "fast-pysf/pysocialforce/",
    "maps/",
    "configs/scenarios/",
    "configs/benchmarks/",
    # Covers both physical row producers, their compatibility wrappers, episode
    # schemas/types and campaign accounting. None may bypass receipt admission.
    "robot_sf/benchmark/",
)
BEHAVIOUR_FILES = {
    "robot_sf/benchmark/metrics.py",
    "robot_sf/benchmark/metric_definitions.py",
    "robot_sf/benchmark/path_utils.py",
    "robot_sf/benchmark/exact_repeat_campaign.py",
    "robot_sf/benchmark/event_ledger.py",
    "robot_sf/benchmark/termination_reason.py",
}


def current_head() -> str:
    """Resolve the exact checkout evaluated by the contract workflow."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def latest_release(repo: str) -> str:
    """Use the latest published software release, excluding artifact releases."""
    rows = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{repo}/releases", "--paginate", "--slurp"],
            text=True,
        )
    )
    releases = [
        r
        for page in rows
        for r in page
        if not r["draft"]
        and not r["prerelease"]
        and re.fullmatch(r"v?\d+\.\d+\.\d+(?:\.post\d+)?", r["tag_name"])
    ]
    if not releases:
        raise ValueError("published software release inventory is empty")
    return max(releases, key=lambda r: r["published_at"])["tag_name"]


def release_source(release: str) -> str:
    """Resolve the immutable published software tag, never a moving branch."""
    return subprocess.check_output(
        ["git", "rev-parse", f"refs/tags/{release}^{{commit}}"], cwd=ROOT, text=True
    ).strip()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _digest(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _uri(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    return (
        parsed.scheme == "https"
        and bool(host)
        and "." in host
        and not parsed.username
        and not parsed.password
        and host not in {"localhost", "127.0.0.1"}
        and not host.endswith(".local")
    )


def _key(row: dict) -> tuple:
    return row["arm"], row["map"], row["seed"]


def validate_receipt(
    receipt: dict, scope: dict, head: str, release: str, baseline_source: str
) -> None:
    """Check exact coverage, provenance, exceptions and exhaustive classifications."""
    import jsonschema

    schema = json.loads((Path(__file__).with_name("behaviour_receipt.schema.json")).read_text())
    jsonschema.Draft202012Validator(schema).validate(receipt)
    scope_digest = hashlib.sha256(
        json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _require(receipt["scope_sha256"] == scope_digest, "scope inventory identity is stale")
    _require(
        receipt["baseline"]["source_sha"] == baseline_source,
        "baseline source does not match the published tag",
    )
    _require(receipt["head_sha"] == head, "receipt head is stale")
    _require(receipt["scheduler"]["source_sha"] == head, "job source does not match head")
    _require(
        bool(re.fullmatch(r"[1-9]\d*(?:_\d+)?", receipt["scheduler"]["job_id"])),
        "missing scheduler job identity",
    )
    _require(
        receipt["baseline"]["release"] == release,
        "baseline is not the latest published software release",
    )
    _require(
        receipt["baseline"]["comparison"] in {"archived_rows", "development_reconstruction"},
        "invalid baseline comparison",
    )
    _require(
        receipt["baseline"]["differences"],
        "baseline body/config/source differences must be declared",
    )
    _require(
        receipt["vehicle"]["id"] == scope["vehicle_id"], "vehicle does not match the reviewed scope"
    )
    for field in ("arms", "maps"):
        _require(
            isinstance(scope[field], list)
            and bool(scope[field])
            and len(set(scope[field])) == len(scope[field]),
            "required scope inventory is empty or duplicated",
        )
    expected = set(itertools.product(scope["arms"], scope["maps"], range(1001, 1031)))
    rows = receipt["rows"]
    keys = [_key(row) for row in rows]
    _require(
        len(keys) == len(set(keys)) and set(keys) == expected,
        "omitted or duplicate arm/map/development-seed coverage",
    )
    from robot_sf.benchmark.algorithm_metadata import (
        canonical_algorithm_name,
        enrich_algorithm_metadata,
    )

    new = {}
    for row in rows:
        expected_algo = canonical_algorithm_name(
            scope.get("arm_algorithms", {}).get(row["arm"], row["arm"])
        )
        _require(row["algorithm"] == expected_algo, "row algorithm differs from reviewed arm")
        profile = enrich_algorithm_metadata(algo=expected_algo)["planner_kinematics"]
        required_mode = "native" if profile["supports_native_commands"] else "adapter"
        _require(
            row["execution_mode"] == required_mode
            and (profile["supports_native_commands"] or profile["supports_adapter_commands"]),
            "command execution mode does not satisfy the algorithm registry",
        )
        _require(row["controller_executed"], "intended solver/controller did not execute")
        _require(
            not row["fallback"] and not row["degraded"],
            "fallback or degraded execution cannot satisfy the receipt",
        )
        key = _key(row)
        if row["baseline_success"] and not row["success"]:
            new[(*key, "success_to_failure")] = 1
        if row["collisions"] > row["baseline_collisions"]:
            new[(*key, "new_collision")] = row["collisions"] - row["baseline_collisions"]
    classified = [(*_key(row), row["kind"]) for row in receipt["classifications"]]
    _require(
        len(classified) == len(set(classified)) and set(classified) == set(new),
        "every new failure and collision needs exactly one classification",
    )
    _require(
        all(row["count"] == new[(*_key(row), row["kind"])] for row in receipt["classifications"]),
        "classification event counts contradict rows",
    )
    _require(
        all(_uri(row["evidence"]) for row in receipt["classifications"]),
        "classification evidence must be durable",
    )
    _require(
        receipt["exceptions"] == scope["exceptions"],
        "undeclared or omitted vehicle-specific exception",
    )
    declared = {(exception["map"], exception["vehicle"]) for exception in scope["exceptions"]}
    _require(
        all(
            row["class"] != "vehicle_specific_infeasible"
            or (row["map"], scope["vehicle_id"]) in declared
            for row in receipt["classifications"]
        ),
        "infeasible classification requires a reviewed vehicle exception",
    )
    for exception in scope["exceptions"]:
        _require(
            exception["map"] in scope["maps"]
            and exception["vehicle"] == scope["vehicle_id"]
            and _uri(exception["evidence"]),
            "invalid reviewed exception inventory",
        )
    totals = receipt["totals"]
    _require(
        totals
        == {
            "episodes": len(rows),
            "new_failures": sum(k[-1] == "success_to_failure" for k in new),
            "new_collisions": sum(count for k, count in new.items() if k[-1] == "new_collision"),
        },
        "contradictory totals",
    )
    for value, length in (
        (head, 40),
        (receipt["baseline"]["source_sha"], 40),
        (receipt["artifact"]["sha256"], 64),
        (receipt["baseline"]["artifact_sha256"], 64),
        (receipt["baseline"]["config_sha256"], 64),
        (receipt["vehicle"]["body_sha256"], 64),
        (receipt["interaction_audit"]["sha256"], 64),
    ):
        _require(_digest(value, length), "invalid source or artifact identity")
    for uri in (
        receipt["artifact"]["uri"],
        receipt["baseline"]["artifact_uri"],
        receipt["interaction_audit"]["uri"],
        receipt["refute_review"]["uri"],
    ):
        _require(_uri(uri), "missing external evidence identity")
    _require(receipt["interaction_audit"]["source_sha"] == head, "real-row audit head is stale")
    _require(
        receipt["refute_review"]["head_sha"] == head
        and receipt["refute_review"]["verdict"] == "accepted",
        "exact-head refute verdict is missing",
    )


def check_receipt(body: str, changed_files: list[str], repo: str) -> list[str]:
    """Fail closed for in-scope changes; prose and tooling paths are exempt."""
    if not any(path in BEHAVIOUR_FILES or path.startswith(TRIGGERS) for path in changed_files):
        return []
    matches = re.findall(r"<!--\s*behaviour-change-receipt:v1\s*\n(.*?)-->", body, re.DOTALL)
    if len(matches) != 1:
        return ["BLOCKER: behaviour receipt missing or duplicated for a behaviour-changing PR"]
    if not SCOPE_PATH.is_file():
        return [
            "BLOCKER: reviewed scope inventory is not installed; merge the release "
            "integration owner's inventory before enabling this gate"
        ]
    import jsonschema

    try:
        receipt = json.loads(matches[0])
        scope = json.loads(SCOPE_PATH.read_text())
        release = latest_release(repo)
        validate_receipt(receipt, scope, current_head(), release, release_source(release))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        jsonschema.ValidationError,
    ) as exc:
        return [
            f"BLOCKER: behaviour receipt rejected ({type(exc).__name__}); verify scope, exact head, baseline, job and classification inventory"
        ]
    return []
