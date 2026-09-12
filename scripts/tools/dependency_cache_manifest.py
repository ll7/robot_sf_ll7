#!/usr/bin/env python3
"""Preserve offline dependency and build-cache manifests for reconstruction (#8852).

Derives the exact artifact set for one declared environment profile from the
canonical lock, classifies every requirement against an already-populated cache,
and emits a deterministic private manifest plus a sanitized public reconstruction
status. It never downloads and never copies artifacts.

Issue #9126 repair: the extra-marker substitution, wheel-tag policy, offline
install guards, public sanitization, and custody inventory are fail-closed.
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
from packaging.utils import InvalidWheelFilename, parse_wheel_filename

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
CUSTODY_CLASSES = (
    "registry-artifact",
    "source-tree",
    "editable-checkout",
    "private-companion",
    "built-local",
)
OFFLINE_PERMITTED = frozenset({"available_verified"})
ABI3_MIN_TAG = re.compile(r"^cp(\d)(\d+)$")
PRIVATE_PATH_RE = re.compile(
    r"(?:^|[\s\"'=])(/(?:home|root|private|opt/secrets|var/run/secrets|scratch|work)/)",
    re.IGNORECASE,
)
CREDENTIAL_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|AWS_SECRET_ACCESS_KEY|bearer\s+[A-Za-z0-9_\-\.]+"
    r"|['\"]?(?:password|passwd|api_key|secret_key)['\"]?\s*[:=]\s*['\"][^'\"]{8,}['\"]",
    re.IGNORECASE,
)
TRUTHY_MARKER = 'python_version >= "0"'
FALSEY_MARKER = 'python_version < "0"'


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


def _redact(text: str) -> str:
    """Strip private paths and credentials from a string bound for public output."""
    text = PRIVATE_PATH_RE.sub("<redacted-path>", text)
    return CREDENTIAL_RE.sub("<redacted-credential>", text)


def _marker_env(environment: dict[str, Any], extra: str = "") -> dict[str, str]:
    version = str(environment.get("python_version", "3.11"))
    return {
        "python_version": version,
        "python_full_version": str(environment.get("python_full_version", f"{version}.0")),
        "sys_platform": str(environment.get("sys_platform", "linux")),
        "platform_system": str(environment.get("platform_system", "Linux")),
        "platform_machine": str(environment.get("platform_machine", "x86_64")),
        "os_name": str(environment.get("os_name", "posix")),
        "implementation_name": str(environment.get("implementation_name", "cpython")),
        "extra": extra,
    }


def _marker_true(marker: str | None, env: dict[str, str], ignore_extra: bool = False) -> bool:
    """Evaluate a lock marker; selected-extra edges treat ``extra`` clauses as satisfied.

    Substitutions must stay valid marker expressions: replacing a clause with a bare
    ``True`` literal makes ``packaging`` reject the whole marker, which previously
    dropped entire dependency sub-graphs (ray/CUDA) from the closure.
    """
    if not marker:
        return True
    if ignore_extra:
        marker = re.sub(r"\bextra\s*(?:==|!=)\s*(?:'[^']*'|\"[^\"]*\")", TRUTHY_MARKER, marker)
    try:
        return bool(Marker(marker).evaluate(env))
    except InvalidMarker:
        return False


def _python_tag_ok(python_tag: str, env: dict[str, str]) -> bool:
    major, minor = (int(part) for part in env["python_version"].split(".")[:2])
    if python_tag in {"py3", f"py{major}", f"py{major}{minor}"}:
        return True
    match = ABI3_MIN_TAG.fullmatch(python_tag)
    if match:
        return (int(match.group(1)), int(match.group(2))) <= (major, minor)
    return False


def _abi_ok(abi_tag: str, python_tag: str, env: dict[str, str]) -> bool:
    if abi_tag == "none":
        return True
    if abi_tag in {"abi3", "abi4"}:
        match = ABI3_MIN_TAG.fullmatch(python_tag)
        return match is not None
    major, minor = env["python_version"].split(".")[:2]
    return abi_tag == f"cp{major}{minor}"


def _platform_ok(platform_tag: str, env: dict[str, str]) -> bool:
    if platform_tag == "any":
        return True
    platform = env["sys_platform"]
    machine = env["platform_machine"]
    tokens = platform_tag.split(".")
    if platform == "linux":
        family = ("manylinux", "musllinux", "linux")
        return any(token.startswith(family) for token in tokens) and any(
            machine in token or "x86_64" in token for token in tokens
        )
    if platform == "darwin":
        return any(token.startswith("macosx") for token in tokens) and any(
            ("arm64" in token or "universal2" in token) if machine == "arm64" else "x86_64" in token
            for token in tokens
        )
    if platform == "win32":
        return any(token.startswith("win") for token in tokens) and any(
            machine in token or "amd64" in token for token in tokens
        )
    return any(machine in token for token in tokens)


def _wheel_rejection(filename: str, env: dict[str, str]) -> str | None:
    """Return ``None`` when the wheel is compatible, else a stable rejection reason."""
    try:
        _, _, _, tags = parse_wheel_filename(filename)
    except InvalidWheelFilename:
        return "unparseable_wheel"
    python_ok = False
    platform_ok = False
    for tag in tags:
        if _python_tag_ok(tag.interpreter, env) and _abi_ok(tag.abi, tag.interpreter, env):
            python_ok = True
            if _platform_ok(tag.platform, env):
                platform_ok = True
    if python_ok and platform_ok:
        return None
    return "incompatible_platform" if python_ok else "incompatible_interpreter"


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


def _queue_edges(
    package: dict[str, Any], extras: list[str], queue: list[tuple[dict[str, Any], bool]]
) -> None:
    queue.extend((edge, False) for edge in package.get("dependencies", []))
    for extra in extras:
        queue.extend(
            (edge, True) for edge in package.get("optional-dependencies", {}).get(extra, [])
        )


def _walk_closure(
    root: dict[str, Any],
    by_key: dict[tuple[str, str], dict[str, Any]],
    by_name: dict[str, list[dict[str, Any]]],
    extras: list[str],
    env: dict[str, str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    """Collect the transitive closure and record every filtered or unresolved edge."""
    selected: dict[tuple[str, str], dict[str, Any]] = {
        (root["name"], str(root.get("version", ""))): root
    }
    filtered: list[dict[str, Any]] = []
    queue: list[tuple[dict[str, Any], bool]] = []
    _queue_edges(root, extras, queue)
    while queue:
        edge, from_extra = queue.pop(0)
        if not _marker_true(edge.get("marker"), env, ignore_extra=from_extra):
            filtered.append(
                {
                    "name": edge.get("name"),
                    "reason": "marker_excluded",
                    "marker": edge.get("marker"),
                }
            )
            continue
        package = _resolve_edge(edge, by_key, by_name)
        if package is None:
            filtered.append(
                {
                    "name": edge.get("name"),
                    "reason": "unresolved_edge",
                    "marker": edge.get("marker"),
                }
            )
            continue
        key = (package["name"], str(package.get("version", "")))
        if key in selected:
            continue
        selected[key] = package
        _queue_edges(package, [str(extra) for extra in edge.get("extra", [])], queue)
    return selected, filtered


def _select_artifact(
    package: dict[str, Any], env: dict[str, str]
) -> tuple[str, str, str | None, int | None, str | None]:
    """Return (kind, filename, lock hash, size, rejection reason)."""
    rejected: list[str] = []
    for wheel in package.get("wheels", []):
        filename = str(wheel.get("url", "")).rsplit("/", 1)[-1]
        reason = _wheel_rejection(filename, env)
        if reason is None:
            return "wheel", filename, wheel.get("hash"), wheel.get("size"), None
        rejected.append(reason)
    sdist = package.get("sdist")
    if sdist:
        filename = str(sdist.get("url", "")).rsplit("/", 1)[-1]
        return "sdist", filename, sdist.get("hash"), sdist.get("size"), None
    source = package.get("source", {})
    if isinstance(source, dict) and any(
        key in source for key in ("editable", "directory", "git", "path")
    ):
        return "build", f"{package.get('name')}-{package.get('version')}-build", None, None, None
    return "wheel", "", None, None, rejected[0] if rejected else "no_artifact"


def _custody_for(package: dict[str, Any], rights: str, kind: str, cached_build: Path | None) -> str:
    if rights == "private-companion":
        return "private-companion"
    if cached_build is not None:
        return "built-local"
    source = package.get("source", {})
    if isinstance(source, dict):
        if "editable" in source:
            return "editable-checkout"
        if any(key in source for key in ("directory", "path", "git")):
            return "source-tree"
    if kind == "sdist":
        return "source-tree"
    return "registry-artifact"


def _scan_cache(
    cache_root: Path, wanted: set[str]
) -> tuple[dict[str, list[Path]], set[str], list[str]]:
    """Index required artifacts once; skip symlinks and any path escaping the root."""
    root = cache_root.resolve()
    index: dict[str, list[Path]] = {}
    seen: set[str] = set()
    rejected: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if not (Path(dirpath) / name).is_symlink()]
        for name in filenames:
            if not name.endswith(ARTIFACT_SUFFIXES):
                continue
            path = Path(dirpath) / name
            if path.is_symlink():
                rejected.append(f"symlink:{name}")
                continue
            try:
                if not path.resolve().is_relative_to(root):
                    rejected.append(f"out_of_root:{name}")
                    continue
            except OSError:
                rejected.append(f"unreadable:{name}")
                continue
            seen.add(name)
            if name in wanted:
                index.setdefault(name, []).append(path)
    return index, seen, sorted(rejected)


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
    kind, filename, lock_hash, size, rejection = _select_artifact(package, env)
    expected_hash = str(lock_hash or "").replace("sha256:", "") or None
    matches = cache.get(filename, []) if filename else []
    observed_hash = None
    if kind == "build":
        availability = "build_required"
    elif matches:
        if len(matches) > 1:
            availability = "duplicate_artifact"
        else:
            observed_hash = _sha256(matches[0])
            availability = (
                "available_verified" if observed_hash == expected_hash else "checksum_drift"
            )
    elif kind == "sdist":
        availability = "source_only"
    else:
        availability = "wrong_platform" if rejection == "incompatible_platform" else "missing"
    custody = _custody_for(
        package, rights, kind, matches[0] if availability == "built-local" else None
    )
    return {
        "name": package["name"],
        "version": str(package.get("version", "")),
        "artifact_kind": kind,
        "artifact": filename or None,
        "availability": availability,
        "rights": rights,
        "custody": custody,
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "size_bytes": size,
        "cache_copies": len(matches),
        "rejection_reason": rejection,
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
    members: dict[tuple[str, str], dict[str, Any]] = {}
    filtered: list[dict[str, Any]] = []
    unresolved_roots: list[str] = []
    for root in profile.get("roots", []):
        package = _resolve_edge(root, by_key, by_name)
        if package is None:
            unresolved_roots.append(str(root.get("name")))
            continue
        extras = root.get("extras", [])
        if extras == "*":
            extras = sorted(package.get("optional-dependencies", {}))
        closure, closure_filtered = _walk_closure(package, by_key, by_name, list(extras), env)
        members.update(closure)
        filtered.extend(closure_filtered)
    wanted = {_select_artifact(member, env)[1] for member in members.values()} - {""}
    cache, seen, rejected = _scan_cache(cache_root, wanted)
    for _, member in sorted(members.items()):
        requirements.append(
            _classify(member, env, cache, seen, _rights_for(member["name"], rights_policy))
        )
    if unresolved_roots:
        for name in sorted(unresolved_roots):
            requirements.append(
                {
                    "name": name,
                    "version": "",
                    "artifact_kind": "unresolved",
                    "artifact": None,
                    "availability": "missing",
                    "rights": _rights_for(name, rights_policy),
                    "custody": "source-tree",
                    "expected_sha256": None,
                    "observed_sha256": None,
                    "size_bytes": None,
                    "cache_copies": 0,
                    "rejection_reason": "unresolved_root",
                }
            )
    unresolved_edges = [row for row in filtered if row["reason"] == "unresolved_edge"]
    return {
        "schema": CACHE_SCHEMA,
        "profile_id": profile.get("profile_id", "unknown"),
        "lockfile": profile.get("lockfile", "uv.lock"),
        "environment": env,
        "requirements": requirements,
        "root_names": sorted(str(root.get("name")) for root in profile.get("roots", [])),
        "summary": _summarize(requirements),
        "closure_audit": {
            "selected_requirements": len(requirements),
            "marker_filtered_edges": len(filtered) - len(unresolved_edges),
            "unresolved_edges": sorted({row["name"] for row in unresolved_edges if row["name"]}),
            "filtered_edges": filtered[:20],
            "complete": not unresolved_edges,
        },
        "cache_rejections": rejected,
        "cache_artifacts_not_required": sorted(name for name in seen if name not in wanted),
    }


def _summarize(requirements: list[dict[str, Any]]) -> dict[str, Any]:
    availability: dict[str, int] = {}
    rights: dict[str, int] = {}
    custody: dict[str, int] = {}
    for row in requirements:
        availability[row["availability"]] = availability.get(row["availability"], 0) + 1
        rights[row["rights"]] = rights.get(row["rights"], 0) + 1
        custody[row["custody"]] = custody.get(row["custody"], 0) + 1
    return {
        "requirement_count": len(requirements),
        "by_availability": dict(sorted(availability.items())),
        "by_rights": dict(sorted(rights.items())),
        "by_custody": dict(sorted(custody.items())),
    }


def reconstruction_status(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitized public status: no private paths, identifiers, or credentials."""
    profile_id = str(manifest["profile_id"])
    profile_public = _redact(profile_id)
    if profile_public != profile_id:
        profile_public = f"profile-sha256:{hashlib.sha256(profile_id.encode()).hexdigest()[:16]}"
    status = {
        "schema": STATUS_SCHEMA,
        "profile_id": profile_public,
        "requirement_count": manifest["summary"]["requirement_count"],
        "by_availability": manifest["summary"]["by_availability"],
        "by_rights": manifest["summary"]["by_rights"],
        "by_custody": manifest["summary"]["by_custody"],
        "closure_complete": manifest.get("closure_audit", {}).get("complete", False),
        "offline_reconstruction_complete": all(
            row["availability"] == "available_verified"
            and row["rights"] == "redistribution-permitted"
            and row["custody"] in {"registry-artifact", "built-local"}
            for row in manifest["requirements"]
        ),
        "private_preservation_does_not_imply_public_redistribution": True,
        "requirements": [
            {
                "name": _redact(str(row["name"])),
                "version": _redact(str(row["version"])),
                "availability": row["availability"],
                "rights": row["rights"],
                "custody": row["custody"],
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


def build_offline_install_plan(
    manifest: dict[str, Any], cache_root: Path, env_dir: Path, requirements_path: Path
) -> dict[str, Any]:
    """Build the guarded offline install plan without running anything.

    Local roots (the workload's own project) are installed by the caller from the
    admitted source, so they are excluded from the dependency install and reported
    explicitly; any other source custody stays blocking.
    """
    roots = {str(name) for name in manifest.get("root_names", [])}
    rows = [row for row in manifest["requirements"] if row["name"] not in roots]
    excluded_roots = sorted(roots)
    preflight = [
        f"{row['name']}=={row['version']}:{row['availability']}/{row['rights']}/{row['custody']}"
        for row in rows
        if row["availability"] not in OFFLINE_PERMITTED
        or row["rights"] != "redistribution-permitted"
        or row["custody"] not in {"registry-artifact", "built-local"}
    ]
    if preflight:
        return {
            "status": "blocked",
            "reason": "incomplete_or_unpermitted_set",
            "blockers": preflight[:10],
            "excluded_roots": excluded_roots,
        }
    if not manifest.get("offline_install_permitted", False):
        return {
            "status": "blocked",
            "reason": "policy_does_not_permit_offline_install",
            "excluded_roots": excluded_roots,
        }
    without_hash = [row["name"] for row in rows if not row["expected_sha256"]]
    if without_hash:
        return {
            "status": "blocked",
            "reason": "missing_hash",
            "blockers": without_hash[:10],
            "excluded_roots": excluded_roots,
        }
    python = env_dir / "bin" / "python"
    return {
        "status": "planned",
        "interpreter": str(python),
        "expected_python_version": manifest["environment"]["python_version"],
        "excluded_roots": excluded_roots,
        "requirements": sorted(
            f"{row['name']}=={row['version']} --hash=sha256:{row['expected_sha256']}"
            for row in rows
        ),
        "requirements_path": str(requirements_path),
        "venv_argv": [
            "uv",
            "venv",
            "--python",
            manifest["environment"]["python_version"],
            str(env_dir),
        ],
        "install_argv": [
            "uv",
            "pip",
            "install",
            "--no-index",
            "--offline",
            "--only-binary",
            ":all:",
            "--require-hashes",
            "--find-links",
            str(cache_root),
            "--python",
            str(python),
            "-r",
            str(requirements_path),
        ],
        "verify_argv": ["uv", "pip", "check", "--python", str(python)],
        "list_argv": ["uv", "pip", "list", "--format", "json", "--python", str(python)],
        "guarded_env": {
            "UV_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "PIP_NO_INPUT": "1",
        },
    }


def offline_install_plan(
    manifest: dict[str, Any], cache_root: Path, timeout: int = 300, runner: Any = None
) -> dict[str, Any]:
    """Verify an offline install in a temporary environment when policy permits it."""
    plan = build_offline_install_plan(manifest, cache_root, Path("VENV"), Path("requirements.txt"))
    if plan["status"] != "planned":
        return plan
    run = runner or (
        lambda argv, env: subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False, env=env
        )
    )
    with tempfile.TemporaryDirectory(prefix="dependency_cache_offline_") as tmp:
        env_dir = Path(tmp) / "venv"
        requirements = Path(tmp) / "requirements.txt"
        plan = build_offline_install_plan(manifest, cache_root, env_dir, requirements)
        requirements.write_text("\n".join(plan["requirements"]) + "\n", encoding="utf-8")
        base_env = {**os.environ, **plan["guarded_env"]}
        try:
            venv = run(plan["venv_argv"], base_env)
            if venv.returncode != 0:
                return {"status": "failed", "reason": "interpreter_unavailable"}
            install = run(plan["install_argv"], base_env)
            if install.returncode != 0:
                tail = (install.stderr or "").strip().splitlines()
                return {
                    "status": "failed",
                    "reason": "offline_install_failed",
                    "detail": _redact(tail[-1][:200]) if tail else "",
                }
            check = run(plan["verify_argv"], base_env)
            if check.returncode != 0:
                return {"status": "failed", "reason": "post_install_check_failed"}
            listed = run(plan["list_argv"], base_env)
            if listed.returncode != 0:
                return {"status": "failed", "reason": "post_install_list_failed"}
            installed = {
                str(row.get("name", "")).lower().replace("_", "-"): str(row.get("version", ""))
                for row in json.loads(listed.stdout or "[]")
            }
            roots = {str(name) for name in manifest.get("root_names", [])}
            mismatched = sorted(
                f"{row['name']}=={row['version']}"
                for row in manifest["requirements"]
                if row["name"] not in roots
                and installed.get(row["name"].lower().replace("_", "-")) != row["version"]
            )
            if mismatched:
                return {
                    "status": "failed",
                    "reason": "post_install_version_mismatch",
                    "blockers": mismatched[:10],
                }
        except (subprocess.TimeoutExpired, OSError) as exc:
            return {
                "status": "failed",
                "reason": "execution_error",
                "detail": _redact(str(exc)[:200]),
            }
    return {"status": "verified", "reason": "offline_install_succeeded"}


def render_markdown(status: dict[str, Any]) -> str:
    """Render a deterministic Markdown summary of a reconstruction status."""
    lines = [
        "# Dependency cache reconstruction status",
        "",
        f"Profile: {status['profile_id']}",
        f"Requirements: {status['requirement_count']}",
        f"Closure complete: {status['closure_complete']}",
        f"Offline reconstruction complete: {status['offline_reconstruction_complete']}",
        "",
        "| Availability | Count |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in status["by_availability"].items())
    lines.extend(["", "| Rights | Count |", "| --- | --- |"])
    lines.extend(f"| {key} | {value} |" for key, value in status["by_rights"].items())
    lines.extend(["", "| Custody | Count |", "| --- | --- |"])
    lines.extend(f"| {key} | {value} |" for key, value in status["by_custody"].items())
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
