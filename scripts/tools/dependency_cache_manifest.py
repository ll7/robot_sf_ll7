#!/usr/bin/env python3
"""Preserve offline dependency and build-cache manifests for reconstruction (#8852).

Derives the exact artifact set for one declared environment profile from the
canonical lock, classifies every requirement against an already-populated cache,
and emits a deterministic private manifest plus a sanitized public reconstruction
status. It never downloads and never copies artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from packaging.markers import InvalidMarker, Marker

CACHE_SCHEMA = "dependency_cache_manifest.v1"
STATUS_SCHEMA = "dependency_cache_reconstruction_status.v1"
PROFILE_SCHEMA = "dependency_cache_profile.v1"

ARTIFACT_SUFFIXES = (".whl", ".tar.gz", ".tgz", ".zip")
FALLBACK_AVAILABILITY = ("missing", "wrong_platform", "source_only", "build_required")
AVAILABILITY_CLASSES = (
    "available_verified",
    "checksum_drift",
    "duplicate_artifact",
    *FALLBACK_AVAILABILITY,
)
RIGHTS_CLASSES = ("redistribution-permitted", "redistribution-unknown", "private-companion")
OFFLINE_PERMITTED = frozenset({"available_verified"})
PRIVATE_PATH_RE = re.compile(
    r"(?:^|[\s\"'=])(/(?:home|root|private|opt/secrets|var/run/secrets|scratch|work)/)",
    re.IGNORECASE,
)
CREDENTIAL_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|AWS_SECRET_ACCESS_KEY|bearer\s+[A-Za-z0-9_\-\.]+",
    re.IGNORECASE,
)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, str(exc)
    return (data, None) if isinstance(data, dict) else (None, "content is not a JSON object")


def _read_toml(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        return None, str(exc)
    return (data, None) if isinstance(data, dict) else (None, "content is not a TOML table")


def _marker_env(environment: dict[str, Any], extra: str = "") -> dict[str, str]:
    version = str(environment.get("python_version", "3.11"))
    env = {
        "python_version": version,
        "python_full_version": str(environment.get("python_full_version", f"{version}.0")),
        "sys_platform": str(environment.get("sys_platform", "linux")),
        "platform_system": str(environment.get("platform_system", "Linux")),
        "platform_machine": str(environment.get("platform_machine", "x86_64")),
        "os_name": str(environment.get("os_name", "posix")),
        "implementation_name": str(environment.get("implementation_name", "cpython")),
        "extra": extra,
    }
    return env


def _marker_true(marker: str | None, env: dict[str, str], ignore_extra: bool = False) -> bool:
    if not marker:
        return True
    if ignore_extra:
        marker = re.sub(r"\bextra\s*(?:==|!=)\s*(?:'[^']*'|\"[^\"]*\")", "True", marker)
    try:
        return bool(Marker(marker).evaluate(env))
    except InvalidMarker:
        return False


def _package_index(
    lock: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for package in lock.get("package", []):
        if not isinstance(package, dict) or "name" not in package:
            continue
        by_key[(package["name"], str(package.get("version", "")))] = package
        by_name.setdefault(package["name"], []).append(package)
    return by_key, by_name


def _resolve_edge(
    edge: dict[str, Any],
    by_key: dict[tuple[str, str], dict[str, Any]],
    by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    target = edge.get("name")
    if not target:
        return None
    if edge.get("version"):
        return by_key.get((target, str(edge["version"])))
    candidates = by_name.get(target, [])
    return candidates[0] if candidates else None


def _walk_closure(
    root: dict[str, Any],
    by_key: dict[tuple[str, str], dict[str, Any]],
    by_name: dict[str, list[dict[str, Any]]],
    extras: list[str],
    base_env: dict[str, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Collect the transitive closure for the root package and the selected extras."""
    selected: dict[tuple[str, str], dict[str, Any]] = {
        (root["name"], str(root.get("version", ""))): root
    }
    queue: list[tuple[dict[str, Any], bool]] = [
        (edge, False) for edge in root.get("dependencies", [])
    ]
    for extra in extras:
        queue.extend((edge, True) for edge in root.get("optional-dependencies", {}).get(extra, []))
    while queue:
        edge, from_extra = queue.pop(0)
        if not _marker_true(edge.get("marker"), base_env, ignore_extra=from_extra):
            continue
        package = _resolve_edge(edge, by_key, by_name)
        if package is None:
            continue
        key = (package["name"], str(package.get("version", "")))
        if key in selected:
            continue
        selected[key] = package
        queue.extend((child, False) for child in package.get("dependencies", []))
    return selected


def _wheel_compatible(filename: str, env: dict[str, str]) -> bool:
    parts = filename[:-4].split("-")
    if len(parts) < 5:
        return False
    python_tag, abi_tag, platform_tag = parts[-3], parts[-2], parts[-1]
    version = env["python_version"].replace(".", "")
    py_ok = python_tag in {
        f"py{env['python_version'].split('.')[0]}",
        "py3",
        "py2.py3",
    } or python_tag.startswith(f"cp{version}")
    if not py_ok:
        return False
    if "abi3" in abi_tag or "none" in abi_tag or abi_tag == f"cp{version}":
        pass
    else:
        return False
    if platform_tag == "any":
        return True
    machine = env["platform_machine"]
    tokens = platform_tag.split(".")
    return any(machine in token for token in tokens) and any(
        env["sys_platform"] in token for token in tokens
    )


def _select_artifact(
    package: dict[str, Any], env: dict[str, str]
) -> tuple[str, str, str | None, int | None]:
    wheels = package.get("wheels", [])
    for wheel in wheels:
        filename = str(wheel.get("url", "")).rsplit("/", 1)[-1]
        if _wheel_compatible(filename, env):
            return "wheel", filename, wheel.get("hash"), wheel.get("size")
    sdist = package.get("sdist")
    if sdist:
        filename = str(sdist.get("url", "")).rsplit("/", 1)[-1]
        return "sdist", filename, sdist.get("hash"), sdist.get("size")
    source = package.get("source", {})
    if isinstance(source, dict) and any(
        key in source for key in ("editable", "directory", "git", "path")
    ):
        return "build", f"{package.get('name')}-{package.get('version')}-build", None, None
    return "wheel", "", None, None


def _scan_cache(cache_root: Path, wanted: set[str]) -> tuple[dict[str, list[Path]], set[str]]:
    """Index the cache once: paths for required filenames, plus every artifact name seen."""
    index: dict[str, list[Path]] = {}
    seen: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(cache_root):
        for name in filenames:
            if not name.endswith(ARTIFACT_SUFFIXES):
                continue
            seen.add(name)
            if name in wanted:
                index.setdefault(name, []).append(Path(dirpath) / name)
    return index, seen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def _classify(
    package: dict[str, Any],
    env: dict[str, str],
    cache: dict[str, list[Path]],
    seen: set[str],
    rights: str,
) -> dict[str, Any]:
    kind, filename, lock_hash, size = _select_artifact(package, env)
    expected_hash = str(lock_hash or "").replace("sha256:", "") or None
    matches = cache.get(filename, []) if filename else []
    availability = "missing"
    observed_hash = None
    if kind == "build":
        availability = "build_required"
    elif kind == "sdist":
        availability = "source_only"
        if matches:
            observed_hash = _sha256(matches[0])
            availability = (
                "available_verified" if observed_hash == expected_hash else "checksum_drift"
            )
    elif matches:
        if len(matches) > 1:
            availability = "duplicate_artifact"
        else:
            observed_hash = _sha256(matches[0])
            availability = (
                "available_verified" if observed_hash == expected_hash else "checksum_drift"
            )
    else:
        wheel_names = [str(w.get("url", "")).rsplit("/", 1)[-1] for w in package.get("wheels", [])]
        other_platform = any(name in seen for name in wheel_names)
        availability = "wrong_platform" if other_platform else "missing"
    return {
        "name": package["name"],
        "version": str(package.get("version", "")),
        "artifact_kind": kind,
        "artifact": filename or None,
        "availability": availability,
        "rights": rights,
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "size_bytes": size,
        "cache_copies": len(matches),
    }


def _rights_for(name: str, rights_policy: dict[str, Any]) -> str:
    if name in set(rights_policy.get("private_companions", [])):
        return "private-companion"
    if name in set(rights_policy.get("permitted", [])):
        return "redistribution-permitted"
    return str(rights_policy.get("default", "redistribution-unknown"))


def build_manifest(
    profile: dict[str, Any], lock: dict[str, Any], cache_root: Path
) -> dict[str, Any]:
    """Derive the manifest for one profile against an already-populated cache root."""
    by_key, by_name = _package_index(lock)
    env = _marker_env(profile.get("environment", {}))
    rights_policy = profile.get("rights", {})
    requirements: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for root in profile.get("roots", []):
        package = _resolve_edge(root, by_key, by_name)
        if package is None:
            requirements.append(
                {
                    "name": root.get("name"),
                    "version": "",
                    "artifact_kind": "unresolved",
                    "artifact": None,
                    "availability": "missing",
                    "rights": _rights_for(str(root.get("name")), rights_policy),
                    "expected_sha256": None,
                    "observed_sha256": None,
                    "size_bytes": None,
                    "cache_copies": 0,
                }
            )
            continue
        extras = root.get("extras", [])
        if extras == "*":
            extras = sorted(package.get("optional-dependencies", {}))
        closure = _walk_closure(package, by_key, by_name, list(extras), env)
        members.extend(member for _, member in sorted(closure.items()))
    wanted = {_select_artifact(member, env)[1] for member in members} - {""}
    cache, seen = _scan_cache(cache_root, wanted)
    for member in members:
        requirements.append(
            _classify(member, env, cache, seen, _rights_for(member["name"], rights_policy))
        )
    unused = sorted(name for name in seen if name not in wanted)
    return {
        "schema": CACHE_SCHEMA,
        "profile_id": profile.get("profile_id", "unknown"),
        "lockfile": profile.get("lockfile", "uv.lock"),
        "environment": env,
        "requirements": requirements,
        "summary": _summarize(requirements),
        "cache_artifacts_not_required": unused,
    }


def _summarize(requirements: list[dict[str, Any]]) -> dict[str, Any]:
    availability: dict[str, int] = {}
    rights: dict[str, int] = {}
    for row in requirements:
        availability[row["availability"]] = availability.get(row["availability"], 0) + 1
        rights[row["rights"]] = rights.get(row["rights"], 0) + 1
    return {
        "requirement_count": len(requirements),
        "by_availability": dict(sorted(availability.items())),
        "by_rights": dict(sorted(rights.items())),
    }


def reconstruction_status(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitized public status: no private paths, indexes, or credentials."""
    complete = all(
        row["availability"] == "available_verified" and row["rights"] == "redistribution-permitted"
        for row in manifest["requirements"]
    )
    status = {
        "schema": STATUS_SCHEMA,
        "profile_id": manifest["profile_id"],
        "requirement_count": manifest["summary"]["requirement_count"],
        "by_availability": manifest["summary"]["by_availability"],
        "by_rights": manifest["summary"]["by_rights"],
        "offline_reconstruction_complete": complete,
        "private_preservation_does_not_imply_public_redistribution": True,
        "requirements": [
            {
                "name": row["name"],
                "version": row["version"],
                "availability": row["availability"],
                "rights": row["rights"],
            }
            for row in manifest["requirements"]
        ],
    }
    text = json.dumps(status)
    if PRIVATE_PATH_RE.search(text) or CREDENTIAL_RE.search(text):
        status["offline_reconstruction_complete"] = False
        status["sanitization"] = "blocked"
    return status


def write_sums(manifest: dict[str, Any], path: Path) -> int:
    """Write a deterministic ``SHA256SUMS`` file; returns the number of rows."""
    rows = [
        f"{row['expected_sha256']}  {row['artifact']}"
        for row in manifest["requirements"]
        if row["artifact"] and row["expected_sha256"]
    ]
    path.write_text("\n".join(sorted(rows)) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def offline_install_plan(
    manifest: dict[str, Any], cache_root: Path, timeout: int = 300
) -> dict[str, Any]:
    """Verify an offline install in a temporary environment when policy permits it."""
    rows = manifest["requirements"]
    blockers = [
        f"{row['name']}=={row['version']}:{row['availability']}/{row['rights']}"
        for row in rows
        if row["availability"] not in OFFLINE_PERMITTED
        or row["rights"] != "redistribution-permitted"
    ]
    if blockers:
        return {
            "status": "blocked",
            "reason": "incomplete_or_unpermitted_set",
            "blockers": blockers[:10],
        }
    if not manifest.get("offline_install_permitted", False):
        return {"status": "blocked", "reason": "policy_does_not_permit_offline_install"}
    with tempfile.TemporaryDirectory(prefix="dependency_cache_offline_") as tmp:
        env_dir = Path(tmp) / "venv"
        requirements = Path(tmp) / "requirements.txt"
        requirements.write_text(
            "\n".join(f"{row['name']}=={row['version']}" for row in rows) + "\n", encoding="utf-8"
        )
        python = env_dir / "bin" / "python"
        commands = [
            ["uv", "venv", str(env_dir)],
            [
                "uv",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(cache_root),
                "--python",
                str(python),
                "-r",
                str(requirements),
            ],
        ]
        for command in commands:
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=timeout, check=False
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                return {"status": "failed", "reason": "execution_error", "detail": str(exc)[:200]}
            if completed.returncode != 0:
                return {
                    "status": "failed",
                    "reason": "offline_install_failed",
                    "detail": completed.stderr.strip().splitlines()[-1][:200]
                    if completed.stderr
                    else "",
                }
    return {"status": "verified", "reason": "offline_install_succeeded"}


def render_markdown(status: dict[str, Any]) -> str:
    """Render a deterministic Markdown summary of a reconstruction status."""
    lines = [
        "# Dependency cache reconstruction status",
        "",
        f"Profile: {status['profile_id']}",
        f"Requirements: {status['requirement_count']}",
        f"Offline reconstruction complete: {status['offline_reconstruction_complete']}",
        "",
        "| Availability | Count |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in status["by_availability"].items())
    lines.append("")
    lines.append("| Rights | Count |")
    lines.append("| --- | --- |")
    lines.extend(f"| {key} | {value} |" for key, value in status["by_rights"].items())
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 0 for a complete verified set, 1 for gaps, 2 for input errors."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Run the read-only cache check.")
    parser.add_argument("--profile", required=True, type=Path, help="Profile JSON path.")
    parser.add_argument("--cache-root", required=True, type=Path, help="Populated artifact cache.")
    parser.add_argument("--lock", type=Path, default=None, help="Lock file override.")
    parser.add_argument("--manifest-out", type=Path, default=None, help="Private manifest path.")
    parser.add_argument("--sums-out", type=Path, default=None, help="SHA256SUMS output path.")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    parser.add_argument("--verify-offline-install", action="store_true")
    parser.add_argument("--offline-timeout", type=int, default=300)
    args = parser.parse_args(argv)

    profile, err = _read_json(args.profile)
    if err or profile is None or profile.get("schema") != PROFILE_SCHEMA:
        print(f"invalid profile: {err or 'unsupported schema'}", file=sys.stderr)
        return 2
    lock_path = args.lock or Path(profile.get("lockfile", "uv.lock"))
    lock, lock_err = _read_toml(lock_path)
    if lock_err or lock is None:
        print(f"invalid lock: {lock_err}", file=sys.stderr)
        return 2
    manifest = build_manifest(profile, lock, args.cache_root)
    manifest["offline_install_permitted"] = bool(
        profile.get("offline_install", {}).get("permitted", False)
    )
    if args.manifest_out:
        args.manifest_out.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.sums_out:
        write_sums(manifest, args.sums_out)
    status = reconstruction_status(manifest)
    if args.verify_offline_install:
        status["offline_install"] = offline_install_plan(
            manifest, args.cache_root, args.offline_timeout
        )
    if args.format == "json":
        sys.stdout.write(json.dumps(status, indent=2, sort_keys=True) + "\n")
    elif args.format == "markdown":
        sys.stdout.write(render_markdown(status))
    else:
        sys.stdout.write(f"Profile: {status['profile_id']}\n")
        for key, value in status["by_availability"].items():
            sys.stdout.write(f"- {key}: {value}\n")
    if status.get("offline_install", {}).get("status") == "failed":
        return 1
    return 0 if status["offline_reconstruction_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
