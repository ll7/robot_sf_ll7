"""Validate behaviour-change receipts without admitting scientific conclusions.

The release integration owner supplies the independent scope inventory. Missing
scope, job identity or classifications block admission; there is no success-rate
threshold. Independent review must verify the linked job and durable evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

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


def _release_version_key(release: dict) -> tuple[int, ...] | None:
    """Return the software semver advertised by a GitHub release, if any."""
    tag = str(release.get("tag_name") or "")
    name = str(release.get("name") or "")
    if tag.startswith("artifact/") or tag.startswith("models-"):
        return None
    match = SOFTWARE_VERSION_RE.search(tag) or SOFTWARE_VERSION_RE.search(name)
    if match is None:
        return None
    version = match.group(1)
    parts: list[int] = []
    for chunk in version.replace(".post", ".").split("."):
        parts.append(int(chunk))
    return tuple(parts)


def latest_release(repo: str) -> str:
    """Use the highest published software release, excluding artifact releases."""
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
            version_key = _release_version_key(release)
            if version_key is not None:
                releases.append((version_key, release["published_at"], release["tag_name"]))
    if not releases:
        raise ValueError("published software release inventory is empty")
    return max(releases)[2]


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
    require_receipt_only_ancestor(receipt["scheduler"]["source_sha"], head, "job")
    require_receipt_only_ancestor(
        receipt["interaction_audit"]["source_sha"], head, "real-row audit"
    )
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
                    rows[name] = (*rows.get(name, ()), f"{group}:{requirement}")
            return {name: tuple(sorted(rows.get(name, ()))) for name in SENSITIVE_DEPENDENCIES}

        before = sensitive_rows(git_file_at_ref(ROOT, merge_base, file) or "")
        after = sensitive_rows(git_file_at_ref(ROOT, head, file) or "")
        changed.update(name for name in SENSITIVE_DEPENDENCIES if before[name] != after[name])
    return bool(changed & SENSITIVE_DEPENDENCIES)


def load_receipt(header: dict, head: str) -> dict:
    """Load committed rows/classifications at the source SHA and verify their bytes."""
    import jsonschema

    schema = json.loads(Path(__file__).with_name("behaviour_receipt.schema.json").read_text())
    jsonschema.Draft202012Validator(schema["$defs"]["header"]).validate(header)
    path = header["rows_artifact"]["path"]
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
    entry = subprocess.check_output(
        ["git", "ls-tree", head, "--", path], cwd=ROOT, text=True
    ).split()
    _require(
        bool(entry) and entry[0] in {"100644", "100755"}, "receipt must be a committed regular file"
    )
    blob = f"{head}:{path}"
    size = int(subprocess.check_output(["git", "cat-file", "-s", blob], cwd=ROOT, text=True))
    _require(size <= MAX_ROWS_BYTES, "receipt payload exceeds the 32 MiB limit")
    raw = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=ROOT)
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        pointer = re.fullmatch(
            rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n",
            raw,
        )
        _require(pointer is not None, "invalid receipt LFS pointer")
        oid, object_size = pointer.groups()
        _require(int(object_size) <= MAX_ROWS_BYTES, "receipt LFS payload exceeds the 32 MiB limit")
        _require(
            oid.decode() == header["rows_artifact"]["sha256"], "receipt LFS pointer digest mismatch"
        )
        # Smudge the exact source pointer, never potentially dirty checkout bytes.
        # Missing downloads must fail, rather than returning the pointer as data.
        raw = subprocess.run(
            ["git", "-c", "lfs.skipdownloaderrors=false", "lfs", "smudge", "--", path],
            input=raw,
            cwd=ROOT,
            capture_output=True,
            check=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "0"},
        ).stdout
        _require(len(raw) == int(object_size), "receipt LFS payload size mismatch")
    _require(
        hashlib.sha256(raw).hexdigest() == header["rows_artifact"]["sha256"],
        "receipt payload digest mismatch",
    )
    payload = json.loads(raw)
    jsonschema.Draft202012Validator(schema["$defs"]["rows_artifact"]).validate(payload)
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
    )
    return receipt


def check_receipt(
    body: str, changed_files: list[str], repo: str, base_ref: str = "origin/main"
) -> list[str]:
    """Fail closed for production changes, loading a compact header and source file."""
    import jsonschema

    try:
        path_trigger = any(
            path != SCOPE_FILE
            and path not in DEPENDENCY_FILES
            and PurePosixPath(path).suffix.lower() != ".md"
            and (path in BEHAVIOUR_FILES or path.startswith(TRIGGERS))
            for path in changed_files
        )
        if not path_trigger and not dependency_change_requires_receipt(changed_files, base_ref):
            return []
        matches = re.findall(r"<!--\s*behaviour-change-receipt:v2\s*\n(.*?)-->", body, re.DOTALL)
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
