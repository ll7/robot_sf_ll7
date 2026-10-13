"""Validate behaviour-change receipts without admitting scientific conclusions.

The release integration owner supplies the independent scope inventory. Missing
scope, job identity or classifications block admission; there is no success-rate
threshold. Independent review must verify the linked job and durable evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from scripts.ci.receipt_provenance import (
    HardwareIdentityError,
    canonical_pair_key,
    validate_scope_ids,
)

ROOT = Path(__file__).resolve().parents[2]
SCOPE_FILE = "configs/benchmarks/releases/behaviour_gate_0_1_0.json"
SCOPE_PATH = ROOT / SCOPE_FILE
SOFTWARE_VERSION_RE = re.compile(r"(?<![\w.])v?(\d+\.\d+\.\d+(?:\.post\d+)?)(?![\w.])")
TRIGGERS = (
    "robot_sf/planner/",
    "robot_sf/baselines/",
    "robot_sf/sim/",
    "robot_sf/robot/",
    "robot_sf/nav/",
    "robot_sf/gym_env/",
    "fast-pysf/",
    "configs/algos/",
    "configs/baselines/",
    "configs/planners/",
    "configs/robots/",
    "model/",
    "robot_sf/models/",
    "robot_sf/sensor/",
    "robot_sf/ped_npc/",
    "robot_sf/ped_ego/",
    "robot_sf/common/",
    "robot_sf/training/",
    "robot_sf/prediction/",
    "robot_sf/feature_extractors/",
    "robot_sf/feature_extractor.py",
    "scripts/benchmark",
    "scripts/classic_benchmark",
    "scripts/run_social_navigation_benchmark",
    "scripts/tools/run_camera_ready_benchmark",
    "scripts/tools/run_split_camera_ready_campaign",
    "scripts/tools/run_benchmark",
    "scripts/tools/benchmark",
    "scripts/tools/evaluate_",
    "scripts/evaluate.py",
    "scripts/training/",
    "scripts/validation/run_empty_world_sweep.py",
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
    """Bind the event source even when CI checks out a synthetic merge."""
    source = os.environ.get("BEHAVIOUR_PR_HEAD_SHA")
    if source is not None:
        _require(_digest(source, 40), "invalid PR source SHA")
        return source
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def current_main_source(repo: str) -> str:
    """Read the live GitHub main ref; event/local tracking refs are not freshness proof."""
    environment = dict(os.environ)
    if not environment.get("GH_TOKEN"):
        environment["GH_TOKEN"] = subprocess.check_output(
            ["gh", "auth", "token"], text=True
        ).strip()
    data = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{repo}/git/ref/heads/main"], text=True, env=environment
        )
    )
    _require(
        isinstance(data, dict) and data.get("ref") == "refs/heads/main",
        "equivalence freshness refused: unexpected remote main ref",
    )
    identity = data.get("object")
    _require(
        isinstance(identity, dict)
        and identity.get("type") == "commit"
        and _digest(identity.get("sha"), 40),
        "equivalence freshness refused: missing remote main commit",
    )
    return identity["sha"]


def _release_version_key(release: dict) -> tuple[int, ...] | None:
    """Return the software semver advertised by a GitHub release, if any."""
    tag = str(release.get("tag_name") or "")
    name = str(release.get("name") or "")
    if tag.startswith("artifact/") or tag.startswith("models-"):
        return None
    versions = set(SOFTWARE_VERSION_RE.findall(tag) + SOFTWARE_VERSION_RE.findall(name))
    if len(versions) > 1:
        raise ValueError(f"conflicting software versions for release {tag!r}: {sorted(versions)}")
    if not versions:
        return None
    version = versions.pop()
    parts: list[int] = []
    for chunk in version.replace(".post", ".").split("."):
        parts.append(int(chunk))
    return tuple(parts)


def latest_release(repo: str) -> str:  # noqa: C901 - Keep release ambiguity guards together.
    """Use the latest uniquely identified publication, excluding artifact releases."""
    rows = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{repo}/releases", "--paginate", "--slurp"],
            text=True,
        )
    )
    releases = []
    for page in rows:
        for release in page:
            if release["draft"] or release["prerelease"]:
                continue
            tag = release.get("tag_name")
            if not isinstance(tag, str) or not tag:
                raise ValueError("published release is missing its tag identity")
            if tag.startswith(("artifact/", "models-")):
                continue
            if not release.get("published_at"):
                raise ValueError(f"published release {tag!r} is missing its publication time")
            version_key = _release_version_key(release)
            releases.append((version_key, release["published_at"], tag))
    if not releases:
        raise ValueError("published software release inventory is empty")
    latest_time = max(row[1] for row in releases)
    latest = [row for row in releases if row[1] == latest_time]
    if len(latest) != 1:
        raise ValueError("ambiguous latest software publication")
    if latest[0][0] is None:
        raise ValueError("latest software publication is missing version metadata")
    highest_version = max(row[0] for row in releases if row[0] is not None)
    if sum(row[0] == highest_version for row in releases) != 1:
        raise ValueError("duplicate highest-version software publications")
    return latest[0][2]


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


HARDWARE_FILES = (
    "execution_node.txt",
    "execution_cpu.json",
    "execution_node_class.txt",
    "execution_request.json",
    "execution_scheduler.json",
)


def _hardware_require(condition: bool, reason: str) -> None:
    if not condition:
        raise HardwareIdentityError(f"hardware comparison invalid: {reason}")


def scheduler_job_ids(value: str) -> list[str]:
    """Admit literal allocation IDs, with an explicit comma-separated shard list."""
    _hardware_require(
        isinstance(value, str) and re.fullmatch(r"[1-9]\d*(?:,[1-9]\d*)*", value) is not None,
        "invalid scheduler job IDs",
    )
    jobs = value.split(",")
    _hardware_require(len(set(jobs)) == len(jobs), "duplicate scheduler job IDs")
    return jobs


def scheduler_inventory() -> tuple[dict, dict]:
    """Read inventory beside the trusted policy, never from the candidate checkout."""
    path = Path(__file__).with_name("behaviour_receipt_nodes.json")
    try:
        raw = path.read_bytes()
        inventory = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise HardwareIdentityError(
            "hardware comparison invalid: trusted node inventory unavailable"
        ) from exc
    _hardware_require(
        isinstance(inventory, dict)
        and inventory.get("schema_version") == "behaviour-receipt-node-inventory.v1"
        and isinstance(inventory.get("nodes"), dict),
        "invalid trusted node inventory",
    )
    return inventory["nodes"], {
        "path": "scripts/ci/behaviour_receipt_nodes.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def scheduler_records(jobs: list[str]) -> list[dict[str, str]]:
    """Fetch allocation state directly from the authenticated scheduler read endpoint."""
    scheduler_job_ids(",".join(jobs))
    command = [
        "sacct",
        "-X",
        "--parsable2",
        "--noheader",
        f"--jobs={','.join(jobs)}",
        "--format=JobIDRaw,State,ExitCode,NodeList,Partition",
    ]
    try:
        result = subprocess.run(
            ["ssh", "-oBatchMode=yes", "-oConnectTimeout=10", "imech192", shlex.join(command)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HardwareIdentityError(
            "hardware comparison invalid: authoritative scheduler records unavailable"
        ) from exc
    records = {}
    nodes, _ = scheduler_inventory()
    for line in result.stdout.splitlines():
        values = line.split("|")
        _hardware_require(len(values) == 5, "malformed scheduler allocation record")
        job, state, exit_code, node, partition = values
        _hardware_require(
            job in jobs and job not in records, "unexpected or duplicate scheduler allocation"
        )
        _hardware_require(
            state == "COMPLETED" and exit_code == "0:0",
            "scheduler allocation did not complete successfully",
        )
        _hardware_require(node in nodes, "scheduler node is absent from trusted inventory")
        _hardware_require(
            partition == nodes[node]["node_class"],
            "scheduler node class differs from trusted inventory",
        )
        records[job] = dict(
            zip(("job_id", "state", "exit_code", "node", "partition"), values, strict=True)
        )
    _hardware_require(set(records) == set(jobs), "missing authoritative scheduler allocations")
    _hardware_require(
        len({r["node"] for r in records.values()}) == 1, "mixed scheduler nodes within one side"
    )
    return [records[job] for job in jobs]


def scheduler_snapshot(jobs: list[str]) -> dict:
    """Capture verified records with the exact trusted inventory artifact identity."""
    _, inventory = scheduler_inventory()
    return {
        "schema_version": "behaviour-scheduler-verification.v1",
        "inventory": inventory,
        "records": scheduler_records(jobs),
    }


def validate_scheduler_rows(rows: list[dict], identities: dict[str, dict[str, str]]) -> None:
    """Every joined row must retain both source allocations and their actual nodes."""
    _hardware_require(isinstance(rows, list), "invalid row allocation inventory")
    for row in rows:
        _hardware_require(isinstance(row, dict), "invalid row allocation record")
        for side, prefix in (("head", ""), ("baseline", "baseline_")):
            _hardware_require(
                row.get(prefix + "scheduler_job_id")
                in scheduler_job_ids(identities[side]["job_id"]),
                f"{side} row job differs from scheduler shard",
            )
            _hardware_require(
                row.get(prefix + "scheduler_node") == identities[side]["node"],
                f"{side} row node differs from scheduler allocation",
            )


def validate_source_row_context(row: dict, job: str, node: str) -> None:
    """Reject mixed contexts in source rows before projection into the receipt."""
    inventory, _ = scheduler_inventory()
    _hardware_require(node in inventory, "source row node is absent from trusted inventory")
    _hardware_require(
        row.get("scheduler_job_id") == job, "source row job differs from scheduler shard"
    )
    _hardware_require(
        row.get("scheduler_node") == node, "source row node differs from scheduler allocation"
    )
    contexts = [row]
    for parent in (row, row.get("algorithm_metadata", {}), row.get("result_provenance", {})):
        if isinstance(parent, dict) and "execution_context" in parent:
            context = parent["execution_context"]
            _hardware_require(isinstance(context, dict), "invalid source execution context")
            contexts.append(context)
    expected = {
        "cpu_model": inventory[node]["cpu_model"],
        "hostname": node,
        "node_name": node,
        "scheduler_node": node,
        "scheduler_job_id": job,
    }
    for context in contexts:
        for field, value in expected.items():
            _hardware_require(
                field not in context or context[field] == value,
                f"source {field} differs from scheduler context",
            )


def hardware_identity(files: dict) -> dict[str, str]:
    """Validate hashed identity-file bytes without inferring missing observations."""
    _hardware_require(isinstance(files, dict), "missing hardware identity files")
    _hardware_require(set(files) == set(HARDWARE_FILES), "missing or unexpected hardware files")
    texts = {}
    for name in HARDWARE_FILES:
        entry = files[name]
        _hardware_require(isinstance(entry, dict), f"invalid {name} record")
        content = entry.get("content")
        _hardware_require(isinstance(content, str) and bool(content), f"missing {name} bytes")
        try:
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        except UnicodeError as exc:
            raise HardwareIdentityError(
                f"hardware comparison invalid: invalid {name} encoding"
            ) from exc
        _hardware_require(entry.get("sha256") == digest, f"{name} digest mismatch")
        texts[name] = content
    node = texts["execution_node.txt"].strip()
    node_class = texts["execution_node_class.txt"].strip()
    for label, value in (("observed node", node), ("node class", node_class)):
        _hardware_require(
            re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", value) is not None
            and value.casefold() != "unknown",
            f"invalid {label}",
        )
    try:
        cpu = json.loads(texts["execution_cpu.json"])
        request = json.loads(texts["execution_request.json"])
    except ValueError as exc:
        raise HardwareIdentityError("hardware comparison invalid: malformed identity JSON") from exc
    _hardware_require(
        isinstance(cpu, dict) and isinstance(cpu.get("lscpu"), list), "invalid CPU identity"
    )
    fields = {}
    for label in ("Model name:", "Architecture:"):
        values = [
            row.get("data")
            for row in cpu["lscpu"]
            if isinstance(row, dict) and row.get("field") == label
        ]
        _hardware_require(
            len(values) == 1
            and isinstance(values[0], str)
            and bool(values[0].strip())
            and values[0].strip().casefold() != "unknown"
            and not any(ord(char) < 32 for char in values[0].strip()),
            f"missing or ambiguous CPU {label}",
        )
        fields[label] = values[0].strip()
    _hardware_require(isinstance(request, dict), "invalid hardware request")
    _hardware_require(
        request.get("schema_version") == "behaviour-hardware-request.v1",
        "invalid hardware request schema",
    )
    _hardware_require(request.get("requested_node") == node, "requested and observed nodes differ")
    _hardware_require(_digest(request.get("source_sha"), 40), "invalid hardware source SHA")
    jobs = scheduler_job_ids(request.get("job_id"))
    try:
        snapshot = json.loads(texts["execution_scheduler.json"])
    except ValueError as exc:
        raise HardwareIdentityError(
            "hardware comparison invalid: malformed scheduler verification"
        ) from exc
    _hardware_require(
        snapshot == scheduler_snapshot(jobs),
        "stored scheduler evidence differs from authoritative records",
    )
    # Derive identity from accounting and the trusted inventory. Job-written bytes
    # are compared as claims; their hashes do not grant them authority.
    observed = snapshot["records"][0]["node"]
    inventory, _ = scheduler_inventory()
    trusted = inventory[observed]
    _hardware_require(
        node == observed and request.get("requested_node") == observed,
        "requested or claimed node differs from scheduler allocation",
    )
    _hardware_require(
        fields["Model name:"] == trusted["cpu_model"]
        and fields["Architecture:"] == trusted["architecture"],
        "CPU claim differs from trusted node inventory",
    )
    _hardware_require(
        node_class == trusted["node_class"], "node class claim differs from trusted node inventory"
    )
    return {
        "node": observed,
        "node_class": trusted["node_class"],
        "cpu_model": trusted["cpu_model"],
        "architecture": trusted["architecture"],
        "source_sha": request["source_sha"],
        "job_id": request["job_id"],
    }


def validate_hardware_pair(hardware: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    """Reject unmatched execution contexts independently of episode outcomes."""
    if not isinstance(hardware, dict):
        raise HardwareIdentityError("hardware comparison invalid: missing paired hardware evidence")
    _hardware_require(
        isinstance(hardware, dict) and set(hardware) == {"head", "baseline"},
        "missing paired hardware evidence",
    )
    identities = {side: hardware_identity(hardware[side]) for side in ("head", "baseline")}
    for field in ("cpu_model", "architecture", "node_class", "node"):
        _hardware_require(
            identities["head"][field] == identities["baseline"][field],
            f"head and baseline {field} differ",
        )
    return identities


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
    return canonical_pair_key(row["arm"], row["map"], row["seed"])


def require_receipt_only_ancestor(source: str, head: str, label: str) -> None:
    """Preserve execution bytes when evidence is committed after a run or audit."""
    if source == head:
        return
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source, head],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    _require(ancestry.returncode == 0, f"{label} source is not an ancestor of PR head")
    changed = (
        subprocess.check_output(
            ["git", "diff", "--name-only", "-z", "--no-renames", source, head], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    _require(
        all(path.startswith("receipts/behaviour/") for path in changed if path),
        f"non-receipt changes after {label} source invalidate execution identity",
    )


def validate_receipt(
    receipt: dict, scope: dict, head: str, release: str, baseline_source: str
) -> None:
    """Check exact coverage, provenance, exceptions and exhaustive classifications."""
    import jsonschema

    identities = validate_hardware_pair(receipt.get("hardware"))
    schema = json.loads((Path(__file__).with_name("behaviour_receipt.schema.json")).read_text())
    validate_scheduler_rows(receipt.get("rows", []), identities)
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
    require_receipt_only_ancestor(receipt["scheduler"]["source_sha"], head, "job")
    require_receipt_only_ancestor(
        receipt["interaction_audit"]["source_sha"], head, "real-row audit"
    )
    for side, source in (
        ("head", receipt["scheduler"]["source_sha"]),
        ("baseline", baseline_source),
    ):
        _hardware_require(
            identities[side]["source_sha"] == source,
            f"{side} hardware source differs from execution",
        )
    jobs = [identities[side]["job_id"] for side in ("head", "baseline")]
    expected_job = (jobs[0] if jobs[0] == jobs[1] else "_".join(jobs)).replace(",", "_")
    _hardware_require(
        expected_job == receipt["scheduler"]["job_id"],
        "hardware jobs differ from scheduler identity",
    )
    _require(
        bool(re.fullmatch(r"[1-9]\d*(?:_\d+)*", receipt["scheduler"]["job_id"])),
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
    validate_scope_ids(scope)
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
        enrich_algorithm_metadata,
    )
    from scripts.ci import receipt_arm_execution
    from scripts.ci.make_behaviour_receipt import require_valid_execution

    new = {}
    for row in rows:
        expected_algo = receipt_arm_execution.expected_algorithm(
            row["arm"],
            row["map"],
            scope.get("arm_algorithms", {}).get(row["arm"], row["arm"]),
            row=row,
            source_sha=receipt["scheduler"]["source_sha"],
            repo_root=ROOT,
        )
        profile = enrich_algorithm_metadata(algo=expected_algo)["planner_kinematics"]
        require_valid_execution(row, expected_algo, profile)
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
    _require(
        receipt["refute_review"]["head_sha"] == head
        and receipt["refute_review"]["verdict"] == "accepted",
        "exact-head refute verdict is missing",
    )


# Deliberately separate, base-owned policy switch. An author ruling may disable
# only this dependency rule without weakening production path triggers.
DEPENDENCY_RECEIPTS_ENABLED = True
DEPENDENCY_FILES = {"uv.lock", "pyproject.toml", "fast-pysf/uv.lock", "fast-pysf/pyproject.toml"}
SENSITIVE_DEPENDENCIES = {"torch", "stable-baselines3", "numpy", "gymnasium"}
MAX_ROWS_BYTES = 32 * 1024 * 1024


def dependency_change_requires_receipt(changed_files: list[str], base_ref: str) -> bool:
    """Use the Dependabot parser on exact source changes, excluding base drift."""
    files = set(changed_files) & DEPENDENCY_FILES
    if not DEPENDENCY_RECEIPTS_ENABLED or not files:
        return False
    from scripts.dev.check_dependabot_update_policy import (
        _dependency_group_requirements_from_text,
        _dependency_groups_from_text,
        _project_dependency_rows_from_text,
        changed_lock_package_names,
        git_file_at_ref,
        requirement_package_name,
    )

    head = current_head()
    merge_base = subprocess.check_output(
        ["git", "merge-base", base_ref, head], cwd=ROOT, text=True
    ).strip()
    changed = set()
    for file in sorted(files):
        if not file.endswith("uv.lock"):
            continue
        changed.update(
            changed_lock_package_names(
                git_file_at_ref(ROOT, merge_base, file) or "",
                git_file_at_ref(ROOT, head, file) or "",
            )
        )
    for file in sorted(files):
        if not file.endswith("pyproject.toml"):
            continue

        def sensitive_rows(text: str) -> dict:
            rows = _project_dependency_rows_from_text(text)
            for group in _dependency_groups_from_text(text):
                for requirement in _dependency_group_requirements_from_text(text, group) or []:
                    name = requirement_package_name(requirement)
                    # Unnamed requirements are discarded by the sensitive-name projection below.
                    rows[name] = (*rows.get(name, ()), f"{group}:{requirement}")  # ty: ignore[invalid-assignment]
            return {name: tuple(sorted(rows.get(name, ()))) for name in SENSITIVE_DEPENDENCIES}

        before = sensitive_rows(git_file_at_ref(ROOT, merge_base, file) or "")
        after = sensitive_rows(git_file_at_ref(ROOT, head, file) or "")
        changed.update(name for name in SENSITIVE_DEPENDENCIES if before[name] != after[name])
    return bool(changed & SENSITIVE_DEPENDENCIES)


def read_committed_payload_bytes(
    header: dict,
    head: str,
    *,
    repo_root: Path | None = None,
    git_environment: dict[str, str] | None = None,
    timeout: float | None = None,
) -> bytes:
    """Read hash-verified regular/LFS Git bytes; no PR import or hardware query.

    The caller supplies a trusted checkout and an immutable source commit. This
    stdlib custody primitive also supports queue-time freshness without invoking
    the receipt's controller or scheduler admission logic.
    """
    _require(_digest(head, 40), "invalid committed receipt source SHA")
    artifact = header.get("rows_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("missing committed receipt artifact")
    _require(_digest(artifact.get("sha256")), "invalid committed receipt digest")
    path = artifact.get("path")
    _require(isinstance(path, str), "invalid committed receipt path")
    relative = PurePosixPath(path)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and path.startswith("receipts/behaviour/")
        and path.endswith(".json")
        and relative.as_posix() == path
        and not any(ord(c) < 32 for c in path),
        "receipt path must be a repository-relative receipts/behaviour JSON file",
    )
    git_options: dict[str, Any] = {"cwd": repo_root or ROOT}
    if git_environment is not None:
        git_options["env"] = git_environment
    if timeout is not None:
        git_options["timeout"] = timeout
    entry = subprocess.check_output(
        ["git", "ls-tree", head, "--", path], text=True, **git_options
    ).split()
    _require(
        bool(entry) and entry[0] in {"100644", "100755"}, "receipt must be a committed regular file"
    )
    blob = f"{head}:{path}"
    size = int(subprocess.check_output(["git", "cat-file", "-s", blob], text=True, **git_options))
    _require(size <= MAX_ROWS_BYTES, "receipt payload exceeds the 32 MiB limit")
    raw = subprocess.check_output(["git", "cat-file", "blob", blob], **git_options)
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        pointer = re.fullmatch(
            rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n",
            raw,
        )
        _require(pointer is not None, "invalid receipt LFS pointer")
        # _require raises on None; ty cannot narrow through this validation helper.
        oid, object_size = pointer.groups()  # ty: ignore[unresolved-attribute]
        _require(int(object_size) <= MAX_ROWS_BYTES, "receipt LFS payload exceeds the 32 MiB limit")
        _require(
            oid.decode() == header["rows_artifact"]["sha256"], "receipt LFS pointer digest mismatch"
        )
        # Smudge the exact source pointer, never potentially dirty checkout bytes.
        # Missing downloads must fail, rather than returning the pointer as data.
        raw = subprocess.run(
            ["git", "-c", "lfs.skipdownloaderrors=false", "lfs", "smudge", "--", path],
            input=raw,
            **{key: value for key, value in git_options.items() if key != "env"},
            capture_output=True,
            check=True,
            env={**(git_environment or os.environ), "GIT_LFS_SKIP_SMUDGE": "0"},
        ).stdout
        _require(len(raw) == int(object_size), "receipt LFS payload size mismatch")
    _require(
        hashlib.sha256(raw).hexdigest() == header["rows_artifact"]["sha256"],
        "receipt payload digest mismatch",
    )
    return raw


def decode_committed_payload(
    header: dict, raw: bytes, head: str, *, git_environment: dict[str, str] | None = None
) -> dict:
    """Validate schema, hardware and execution claims in already hash-bound Git bytes."""
    import jsonschema

    schema = json.loads(Path(__file__).with_name("behaviour_receipt.schema.json").read_text())
    header_schema = (
        "equivalence_header"
        if header.get("schema_version") == "default-path-equivalence-header.v1"
        else "header"
    )
    jsonschema.Draft202012Validator(schema["$defs"][header_schema]).validate(header)
    _require(
        hashlib.sha256(raw).hexdigest() == header["rows_artifact"]["sha256"],
        "receipt payload digest mismatch",
    )
    if header_schema == "equivalence_header":
        from scripts.ci import default_path_equivalence as equivalence

        payload = equivalence.strict_json_loads(raw)
    else:
        payload = json.loads(raw)
    _hardware_require(isinstance(payload, dict), "invalid receipt hardware payload")
    if header_schema == "equivalence_header":
        jsonschema.Draft202012Validator(schema).evolve(
            schema=schema["$defs"]["equivalence_receipt"]
        ).validate(payload)
        for files in payload["hardware"].values():
            for name, entry in files.items():
                if name.endswith(".json"):
                    equivalence.strict_json_loads(entry["content"])
    identities = validate_hardware_pair(payload.get("hardware"))
    validate_scheduler_rows(payload.get("rows", []), identities)
    verify_source_artifacts(payload, identities, head, header, git_environment=git_environment)
    if header_schema == "header":
        jsonschema.Draft202012Validator(schema).evolve(
            schema=schema["$defs"]["rows_artifact"]
        ).validate(payload)
    if header_schema == "equivalence_header":
        jsonschema.Draft202012Validator(schema).evolve(
            schema=schema["$defs"]["equivalence_receipt"]
        ).validate(payload)
        return {**payload, "head_sha": header["head_sha"], "refute_review": header["refute_review"]}
    return payload


def load_committed_payload(header: dict, head: str) -> dict:
    """Read the exact committed regular/LFS payload and validate its admission claims."""
    return decode_committed_payload(header, read_committed_payload_bytes(header, head), head)


def load_receipt(header: dict, head: str) -> dict:
    """Load the existing full receipt without changing its admission contract."""
    payload = load_committed_payload(header, head)

    classifications = payload["classifications"]
    _require(
        header["classifications"]
        == {
            "count": len(classifications),
            "sha256": hashlib.sha256(
                json.dumps(classifications, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "classification summary differs from the committed payload",
    )
    receipt = {key: value for key, value in header.items() if key != "rows_artifact"}
    receipt.update(
        schema_version="behaviour-change-receipt.v1",
        rows=payload["rows"],
        classifications=classifications,
        hardware=payload["hardware"],
    )
    return receipt


def admit_equivalence_payload(  # noqa: PLR0913 - explicit trusted source/custody contexts.
    header: dict,
    raw: bytes,
    head: str,
    base_source: str,
    remote_main: str,
    changed_files: list[str],
    *,
    repo_root: Path | None = None,
    is_release: bool = False,
    git_environment: dict[str, str] | None = None,
) -> dict:
    """Apply the same complete base-owned admission to the PR and current queue selection."""
    import jsonschema

    from scripts.ci import default_path_equivalence

    global ROOT
    previous_root = ROOT
    ROOT = repo_root or ROOT
    try:
        _require(header.get("head_sha") == head, "equivalence header is stale")
        try:
            receipt = decode_committed_payload(header, raw, head, git_environment=git_environment)
        except jsonschema.ValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path) or "header"
            raise ValueError(f"equivalence schema refused: {path} [{exc.validator}]") from exc
        scope = json.loads(SCOPE_PATH.read_text())
        _require(
            base_source == remote_main == receipt["baseline"]["source_sha"],
            "equivalence freshness refused: event base and receipt baseline must match current remote main",
        )
        default_path_equivalence.validate_receipt(
            receipt, scope, head, base_source, changed_files, is_release=is_release
        )
        return receipt
    finally:
        ROOT = previous_root


def parse_equivalence_header(body: str) -> dict | None:
    """Dispatch explicit receipt comments independently of well-formed JSON capture."""
    markers = re.findall(r"<!--\s*default-path-equivalence[^\s>]*", body)
    if not markers:
        return None
    matches = re.findall(r"<!--\s*default-path-equivalence:v1\s*\n(.*?)-->", body, re.DOTALL)
    _require(
        len(markers) == 1
        and len(matches) == 1
        and not re.search(r"<!--\s*behaviour-change-receipt", body),
        "malformed, duplicate or mixed equivalence receipt markers",
    )
    _require(len(body) <= 65536, "PR body exceeds GitHub's limit")
    from scripts.ci import default_path_equivalence as equivalence

    header = equivalence.strict_json_loads(matches[0])
    _require(isinstance(header, dict), "malformed equivalence receipt header")
    return header


def _source_artifact_bytes(
    reference: dict, head: str, *, git_environment: dict[str, str] | None = None
) -> bytes:
    """Read exact committed source evidence, including authenticated LFS objects."""
    path, expected = reference["path"], reference["sha256"]
    relative = PurePosixPath(path)
    _hardware_require(
        path.startswith("receipts/behaviour/")
        and path.endswith(".tar.gz")
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == path
        and not any(ord(c) < 32 for c in path)
        and _digest(expected),
        "provenance digest artifact reference is invalid",
    )
    git_options: dict[str, Any] = {"cwd": ROOT, "timeout": 60}
    if git_environment is not None:
        git_options["env"] = git_environment
    entry = subprocess.check_output(
        ["git", "ls-tree", head, "--", path], text=True, **git_options
    ).split()
    _hardware_require(
        bool(entry) and entry[0] == "100644", "provenance digest artifact is not regular"
    )
    blob = f"{head}:{path}"
    limit = 512 * 1024 * 1024
    size = int(subprocess.check_output(["git", "cat-file", "-s", blob], text=True, **git_options))
    _hardware_require(size <= limit, "provenance digest artifact exceeds size limit")
    raw = subprocess.check_output(["git", "cat-file", "blob", blob], **git_options)
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        pointer = re.fullmatch(
            rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n",
            raw,
        )
        if pointer is None:
            raise HardwareIdentityError(
                "hardware comparison invalid: provenance digest LFS pointer is invalid"
            )
        oid, object_size = pointer.groups()
        _hardware_require(
            oid.decode() == expected and int(object_size) <= limit,
            "provenance digest LFS identity differs",
        )
        raw = subprocess.run(
            ["git", "-c", "lfs.skipdownloaderrors=false", "lfs", "smudge", "--", path],
            input=raw,
            **{key: value for key, value in git_options.items() if key != "env"},
            capture_output=True,
            check=True,
            env={**(git_environment or os.environ), "GIT_LFS_SKIP_SMUDGE": "0"},
        ).stdout
        _hardware_require(len(raw) == int(object_size), "provenance digest LFS size differs")
    _hardware_require(
        hashlib.sha256(raw).hexdigest() == expected, "provenance digest artifact mismatch"
    )
    return raw


def _equivalence_source_projection(
    root: Path, raw_rows: dict, payload: dict, identity: dict
) -> dict:
    """Rebuild digest/action proof from sealed raw bytes and original source provenance."""
    from scripts.ci import default_path_equivalence as equivalence
    from scripts.ci import make_behaviour_receipt as producer

    equivalence.validate_source_json(root)
    request = equivalence.strict_json_loads((root / "execution_request.json").read_bytes())
    binding = equivalence.strict_json_loads((root / "execution_binding.json").read_bytes())
    _hardware_require(
        isinstance(request, dict)
        and request.get("source_sha") == identity["source_sha"]
        and request.get("arm_role") == binding.get("arm_role"),
        "provenance digest archived request source/role differs",
    )
    expected_request = equivalence.strict_json_loads(
        payload["hardware"][binding["arm_role"]]["execution_request.json"]["content"]
    )
    expected_request = {**expected_request, "job_id": root.name}
    _hardware_require(
        request.get("schema_version") == "behaviour-hardware-request.v1"
        and request.get("job_id") == root.name
        and root.name in scheduler_job_ids(identity["job_id"])
        and request.get("requested_node") == identity["node"]
        and request == expected_request,
        "provenance digest archived request differs from verified allocation/hardware",
    )
    producer._verify_sweep_metadata(root, identity["source_sha"])
    bound_raw = {path.resolve() for path in root.glob("campaigns/**/episodes.jsonl")}
    for summary in root.glob("episodes_*.jsonl"):
        suite = summary.stem.removeprefix("episodes_")
        campaign = root / "campaigns" / f"empty_world_{suite}"
        for line in summary.read_text().splitlines():
            record = json.loads(line)
            source = record.get("source_file")
            _hardware_require(
                isinstance(source, str)
                and not PurePosixPath(source).is_absolute()
                and ".." not in PurePosixPath(source).parts
                and PurePosixPath(source).as_posix() == source
                and (campaign / source).resolve() in bound_raw,
                "provenance digest summary path outside sealed raw inventory",
            )
    joined_rows = producer._load_sweep_rows(
        root, scheduler_job_id=root.name, scheduler_node=identity["node"], preserve_raw=True
    )
    statuses = producer._sweep_execution_statuses(root)
    audited_rows = {
        key: {**row, "execution_status": statuses[key]} for key, row in joined_rows.items()
    }
    _hardware_require(
        equivalence.same_raw_rows(audited_rows, raw_rows),
        "provenance digest summary/raw join differs",
    )
    run = producer.SweepRun(identity["source_sha"], root, "", "", root.name, shard_dirs=(root,))
    tooling = producer._verify_equivalence_tooling(
        (run, run), payload["tooling_sha"], payload["seeds"], ROOT
    )
    _hardware_require(
        tooling
        == (payload["tooling_path"], payload["tooling_file_sha256"], payload["harness_sha256"]),
        "provenance digest acquisition tooling projection differs",
    )
    for file in root.glob("campaigns/**/episodes.jsonl.provenance.json"):
        provenance = json.loads(file.read_text())
        _hardware_require(
            provenance["run"]["repo_commit"] == identity["source_sha"]
            and all(row["repo_commit"] == identity["source_sha"] for row in provenance["rows"]),
            "provenance digest original execution source differs",
        )
    statuses = producer._sweep_execution_statuses(root)
    paths = producer._sweep_episode_paths(root)
    _hardware_require(
        set(raw_rows) == set(statuses) == set(paths), "provenance digest summary coverage differs"
    )
    from scripts.ci import receipt_arm_execution

    result = {}
    for key, row in joined_rows.items():
        equivalence.validate_runtime_source(row, identity["source_sha"])
        result[key] = {
            "sha256": hashlib.sha256(equivalence.episode_bytes(row)).hexdigest(),
            "execution": producer._execution_evidence(
                receipt_arm_execution.carry_execution_proof(
                    raw_rows[key], {**row, "execution_status": statuses[key]}
                )
            ),
            "path": paths[key],
        }
    return result


def _verify_equivalence_projection(
    payload: dict,
    prefix: str,
    side: str,
    source_rows: dict,
    source_projections: dict,
    jobs: set[str],
) -> dict:
    """Compact rows must reproduce the complete source-bound behavioral and execution proof."""
    verified = {}
    for row in payload["rows"]:
        source = source_rows[_key(row)]
        allocation_matches = (
            source["scheduler_job_id"] == row[prefix + "scheduler_job_id"]
            and source["scheduler_node"] == row[prefix + "scheduler_node"]
        )
        proof = source_projections[_key(row)]
        path = row[f"{side}_path"]
        expected_path = proof["path"]
        job = source["scheduler_job_id"]
        _hardware_require(
            allocation_matches
            and proof["sha256"] == row[f"{side}_sha256"]
            and proof["execution"] == row[f"{side}_execution"]
            and (path == f"{job}/{expected_path}" or (len(jobs) == 1 and path == expected_path)),
            "provenance digest behavioral/execution source projection differs",
        )

        verified[_key(row)] = proof["execution"]
    return verified


def verify_source_artifacts(
    payload: dict,
    identities: dict,
    head: str,
    header: dict,
    *,
    git_environment: dict[str, str] | None = None,
) -> None:
    """Independently verify sealed source bytes and their allocation/row projection."""
    from scripts.ci import make_behaviour_receipt as producer
    from scripts.ci.receipt_provenance import BINDING_FILE, verified_raw_rows

    references = payload.get("source_artifacts")
    if not isinstance(references, dict):
        raise HardwareIdentityError(
            "hardware comparison invalid: provenance digest evidence missing"
        )
    _hardware_require(
        set(references) == {"head", "baseline"},
        "provenance digest evidence missing",
    )
    source_claims = (
        payload if payload.get("schema_version") == "default-path-equivalence.v1" else header
    )
    _hardware_require(
        identities["head"]["source_sha"] == source_claims["scheduler"]["source_sha"]
        and identities["baseline"]["source_sha"] == source_claims["baseline"]["source_sha"]
        and identities["head"]["source_sha"] != identities["baseline"]["source_sha"],
        "producer source invalid: arm claims differ or share a producer commit",
    )
    comparison_sources = {}
    projected_proofs = {}
    derived = []
    for side, prefix in (("head", ""), ("baseline", "baseline_")):
        jobs = set(scheduler_job_ids(identities[side]["job_id"]))
        _hardware_require(
            isinstance(references[side], dict) and set(references[side]) == jobs,
            "provenance digest shard coverage differs",
        )
        source_rows = {}
        source_projections = {}
        for job, reference in references[side].items():
            raw = _source_artifact_bytes(reference, head, git_environment=git_environment)
            with tempfile.TemporaryDirectory(prefix="receipt-sources-") as temporary:
                root = Path(temporary) / job
                root.mkdir()
                seen = set()
                total_size = 0
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
                    for member in archive:
                        path = PurePosixPath(member.name)
                        total_size += member.size
                        _hardware_require(
                            member.isfile()
                            and not path.is_absolute()
                            and ".." not in path.parts
                            and path.as_posix() == member.name
                            and member.name not in seen
                            and total_size <= 2 * 1024 * 1024 * 1024,
                            "provenance digest archive is unsafe or oversized",
                        )
                        seen.add(member.name)
                        target = root / member.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        stream = archive.extractfile(member)
                        _hardware_require(
                            stream is not None, "provenance digest archive member missing"
                        )
                        if stream is None:
                            raise HardwareIdentityError(
                                "hardware comparison invalid: provenance digest archive member missing"
                            )
                        target.write_bytes(stream.read())
                if payload.get("schema_version") == "default-path-equivalence.v1":
                    from scripts.ci import default_path_equivalence as equivalence

                    equivalence.validate_source_json(root)
                binding = reference.get("binding")
                _hardware_require(
                    isinstance(binding, dict)
                    and binding.get("path") == BINDING_FILE
                    and binding.get("sha256")
                    == hashlib.sha256((root / BINDING_FILE).read_bytes()).hexdigest(),
                    "provenance digest allocation binding mismatch",
                )
                rows = verified_raw_rows(
                    root,
                    job,
                    identities[side]["node"],
                    source_sha=identities[side]["source_sha"],
                    arm_role=side,
                )
                if payload.get("schema_version") == "default-path-equivalence.v1":
                    source_projections.update(
                        _equivalence_source_projection(root, rows, payload, identities[side])
                    )

                _hardware_require(
                    not (set(rows) & set(source_rows)), "provenance digest duplicate source rows"
                )
                source_rows.update(rows)
        _hardware_require(
            set(source_rows) == {_key(row) for row in payload["rows"]},
            "provenance digest source row coverage differs",
        )
        if payload.get("schema_version") == "default-path-equivalence.v1":
            projected_proofs[side] = _verify_equivalence_projection(
                payload, prefix, side, source_rows, source_projections, jobs
            )
        comparison_sources[side] = source_rows
    if payload.get("schema_version") != "default-path-equivalence.v1":
        scope = json.loads(SCOPE_PATH.read_text())
        derived, _, totals = producer.derive_valid_comparison(
            comparison_sources["head"],
            comparison_sources["baseline"],
            scope,
            classification_class="known_limitation",
            evidence_base_uri="verified-source",
        )
        _hardware_require(
            {_key(row): row for row in derived} == {_key(row): row for row in payload["rows"]}
            and header["totals"] == totals,
            "raw execution invalid: receipt projection or totals differ from sealed sources",
        )

    _restore_execution_proofs(payload, projected_proofs, derived)


def _restore_execution_proofs(payload: dict, projected_proofs: dict, derived: list) -> None:
    """Restore verifier attributes only after all source projections compare exactly."""
    # Only after both sealed source archives and all exact projections verify may
    # the plain JSON payload receive in-memory verifier attributes.
    from scripts.ci import receipt_arm_execution

    if payload.get("schema_version") == "default-path-equivalence.v1":
        for row in payload["rows"]:
            for side in ("head", "baseline"):
                row[f"{side}_execution"] = receipt_arm_execution.carry_execution_proof(
                    projected_proofs[side][_key(row)], row[f"{side}_execution"]
                )
    else:
        indexed = {_key(row): row for row in derived}
        payload["rows"] = [
            receipt_arm_execution.carry_execution_proof(indexed[_key(row)], row)
            for row in payload["rows"]
        ]


def check_receipt(
    body: str, changed_files: list[str], repo: str, base_ref: str = "origin/main"
) -> list[str]:
    """Fail closed for production changes, loading a compact header and source file."""
    import jsonschema

    try:
        equivalence_header = parse_equivalence_header(body)
        path_trigger = any(
            path != SCOPE_FILE
            and path not in DEPENDENCY_FILES
            and PurePosixPath(path).suffix.lower() != ".md"
            and (path in BEHAVIOUR_FILES or path.startswith(TRIGGERS))
            for path in changed_files
        )
        release_context = os.environ.get("BEHAVIOUR_RECEIPT_CONTEXT") == "release"
        if (
            equivalence_header is None
            and not release_context
            and not path_trigger
            and not dependency_change_requires_receipt(changed_files, base_ref)
        ):
            return []
        matches = re.findall(r"<!--\s*behaviour-change-receipt:v2\s*\n(.*?)-->", body, re.DOTALL)
        if equivalence_header is not None:
            _require(len(body) <= 65536, "PR body exceeds GitHub's limit")
            head = current_head()
            header = equivalence_header
            base_source = subprocess.check_output(
                ["git", "rev-parse", f"{base_ref}^{{commit}}"], cwd=ROOT, text=True
            ).strip()
            admit_equivalence_payload(
                header,
                read_committed_payload_bytes(header, head),
                head,
                base_source,
                current_main_source(repo),
                changed_files,
                is_release=os.environ.get("BEHAVIOUR_RECEIPT_CONTEXT") == "release",
            )
            _require(
                current_main_source(repo) == base_source,
                "equivalence freshness refused: current remote main moved during admission",
            )
            return []
        if len(matches) != 1:
            return ["BLOCKER: behaviour receipt missing or duplicated for a behaviour-changing PR"]
        if not SCOPE_PATH.is_file():
            return [
                "BLOCKER: reviewed scope inventory is not installed; merge the release integration owner's inventory before enabling this gate"
            ]
        _require(len(body) <= 65536, "PR body exceeds GitHub's 65,536 character limit")
        header = json.loads(matches[0])
        head = current_head()
        receipt = load_receipt(header, head)
        scope = json.loads(SCOPE_PATH.read_text())
        release = latest_release(repo)
        validate_receipt(receipt, scope, head, release, release_source(release))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        jsonschema.ValidationError,
    ) as exc:
        if isinstance(exc, HardwareIdentityError):
            return [f"BLOCKER: behaviour receipt invalid: {exc}"]
        if isinstance(exc, ValueError) and "default-path-equivalence:v1" in body:
            return [f"BLOCKER: {exc}"]

        return [
            f"BLOCKER: behaviour receipt rejected ({type(exc).__name__}); verify scope, exact head, baseline, job, payload digest and classification inventory"
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    """Run the base-owned validator against event metadata and source Git objects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-event-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-ref", required=True)
    args = parser.parse_args(argv)
    global ROOT
    ROOT = args.repo_root.resolve()
    event = json.loads(args.github_event_path.read_text())
    if "release" in event or not isinstance(event.get("pull_request"), dict):
        print(
            "BLOCKER: PR-only equivalence cannot admit a release/non-PR event; full release evidence is required"
        )
        return 1
    pr = event["pull_request"]
    _require(
        os.environ.get("BEHAVIOUR_PR_HEAD_SHA") == pr["head"]["sha"],
        "event source SHA is missing or contradictory",
    )
    head = current_head()
    changed_files = (
        subprocess.check_output(
            ["git", "diff", "--name-only", "-z", "--no-renames", f"{args.base_ref}...{head}"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    blockers = check_receipt(
        pr.get("body") or "",
        [p for p in changed_files if p],
        event["repository"]["full_name"],
        args.base_ref,
    )
    for blocker in blockers:
        print(blocker)
    if not blockers:
        print("Behaviour receipt gate: accepted or out of scope")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
