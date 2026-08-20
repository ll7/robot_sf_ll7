#!/usr/bin/env python3
"""Check that dependency declarations and independently resolved locks agree.

This is a report-oriented contract for the repository's root project and the
standalone ``fast-pysf`` project.  It deliberately does not decide whether a
dependency version, license, or update lane is acceptable.  Its job is to
identify the lock/profile owners affected by a declaration change and to fail
closed when the required lock or supported-profile proof is unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "scripts/validation/dependency_coherence.v1.json"
DEFAULT_PROFILE_MANIFEST = REPO_ROOT / "scripts/validation/dependency_license_profiles.v1.json"
SCHEMA_VERSION = "robot-sf.dependency-coherence.v1"
REPORT_SCHEMA_VERSION = "robot-sf.dependency-coherence-report.v1"
PACKAGE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
MARKER_ATOM = re.compile(
    r"^(?P<key>[A-Za-z_]+)\s*(?P<operator>==|!=|<=|>=|<|>)\s*['\"](?P<value>[^'\"]+)['\"]$"
)
SPECIFIER = re.compile(r"(===|==|!=|<=|>=|<|>)\s*([0-9][A-Za-z0-9.\-+]*)")
VALID_STATUSES = frozenset(
    {
        "coherent",
        "missing_lock_update",
        "declaration_lock_mismatch",
        "profile_unavailable",
        "conflict",
        "invalid",
    }
)
VALID_COUPLINGS = frozenset({"independent", "workspace"})


class CoherenceError(ValueError):
    """Raised when a dependency-coherence report cannot be proved."""


class ProfileUnavailable(CoherenceError):
    """Raised when a supported Python/profile check cannot be evaluated."""


class LockCheckFailed(CoherenceError):
    """Raised when uv proves that a lock is stale or malformed."""


def normalize_package_name(name: str) -> str:
    """Return the PEP 503 comparison form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _identity_value(value: Any) -> Any:
    """Normalize order-only TOML differences without dropping resolution facts."""
    if isinstance(value, Mapping):
        return {str(key): _identity_value(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        normalized = [_identity_value(child) for child in value]
        return sorted(normalized, key=_canonical_json)
    return value


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CoherenceError(f"{label} must be an object")
    return value


def _as_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CoherenceError(f"{label} must be a list of strings")
    return list(value)


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and validate the machine-readable owner map."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoherenceError(f"unable to load coherence manifest {path}: {exc}") from exc
    manifest = _as_mapping(value, "coherence manifest")
    validate_manifest(manifest)
    return dict(manifest)


def validate_manifest(manifest: Mapping[str, Any]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Validate the declaration-to-lock/profile ownership contract."""
    required = {"schema_version", "profile_manifest", "resolver", "python_profiles", "owners"}
    missing = sorted(required - set(manifest))
    if missing:
        raise CoherenceError("coherence manifest is missing: " + ", ".join(missing))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise CoherenceError(f"unsupported coherence schema: {manifest['schema_version']!r}")
    if not isinstance(manifest["profile_manifest"], str) or not manifest["profile_manifest"]:
        raise CoherenceError("coherence manifest profile_manifest must be a path")

    resolver = _as_mapping(manifest["resolver"], "resolver")
    if resolver.get("name") != "uv":
        raise CoherenceError("coherence resolver must be uv")
    if not isinstance(resolver.get("version"), str) or not resolver["version"]:
        raise CoherenceError("coherence resolver version must be pinned")
    if resolver.get("lock_mode") != "check":
        raise CoherenceError("coherence resolver lock_mode must be check")

    python_profiles = manifest["python_profiles"]
    if not isinstance(python_profiles, list) or not python_profiles:
        raise CoherenceError("coherence manifest needs at least one Python profile")
    profile_ids: set[str] = set()
    for index, raw_profile in enumerate(python_profiles):
        profile = _as_mapping(raw_profile, f"python_profiles[{index}]")
        profile_id = profile.get("id")
        if not isinstance(profile_id, str) or not profile_id or profile_id in profile_ids:
            raise CoherenceError(f"python_profiles[{index}] has a duplicate or invalid id")
        if not isinstance(profile.get("python"), str) or not profile["python"]:
            raise CoherenceError(f"python_profiles[{index}].python must be a version")
        if profile.get("required") is not True:
            raise CoherenceError(f"python_profiles[{index}].required must be true")
        profile_ids.add(profile_id)

    owners = manifest["owners"]
    if not isinstance(owners, list) or not owners:
        raise CoherenceError("coherence manifest needs at least one owner")
    owner_ids: set[str] = set()
    declaration_paths: set[str] = set()
    lock_paths: dict[str, list[str]] = {}
    for index, raw_owner in enumerate(owners):
        owner = _as_mapping(raw_owner, f"owners[{index}]")
        owner_id = owner.get("id")
        declaration = owner.get("declaration")
        lockfile = owner.get("lockfile")
        if not isinstance(owner_id, str) or not owner_id or owner_id in owner_ids:
            raise CoherenceError(f"owners[{index}] has a duplicate or invalid id")
        if not isinstance(declaration, str) or not declaration:
            raise CoherenceError(f"owners[{index}].declaration must be a path")
        if not isinstance(lockfile, str) or not lockfile:
            raise CoherenceError(f"owners[{index}].lockfile must be a path")
        if declaration in declaration_paths:
            raise CoherenceError(f"declaration path has multiple owners: {declaration}")
        if not isinstance(owner.get("root_package"), str) or not owner["root_package"]:
            raise CoherenceError(f"owners[{index}].root_package must be a package name")
        coupling = owner.get("coupling")
        if coupling not in VALID_COUPLINGS:
            raise CoherenceError(
                f"owners[{index}].coupling must be one of {sorted(VALID_COUPLINGS)}"
            )
        owner_profiles = _as_string_list(owner.get("profile_ids"), f"owners[{index}].profile_ids")
        if not owner_profiles:
            raise CoherenceError(f"owners[{index}] needs at least one profile owner")
        owner_ids.add(owner_id)
        declaration_paths.add(declaration)
        lock_paths.setdefault(lockfile, []).append(owner_id)

    for lockfile, lock_owners in lock_paths.items():
        if len(lock_owners) > 1:
            couplings = {
                str(owner.get("coupling"))
                for owner in owners
                if isinstance(owner, Mapping) and owner.get("id") in lock_owners
            }
            if couplings != {"workspace"}:
                raise CoherenceError(
                    f"independent owners cannot share lockfile {lockfile}: {lock_owners}"
                )


def validate_profile_manifest(profile_manifest: Mapping[str, Any]) -> set[str]:
    """Validate the small profile surface needed by the coherence checker."""
    profiles = profile_manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ProfileUnavailable("profile manifest has no profiles")
    ids: set[str] = set()
    for index, raw_profile in enumerate(profiles):
        profile = _as_mapping(raw_profile, f"profiles[{index}]")
        profile_id = profile.get("id")
        if not isinstance(profile_id, str) or not profile_id or profile_id in ids:
            raise ProfileUnavailable(f"profile manifest has an invalid profile at index {index}")
        ids.add(profile_id)
    return ids


def _validate_owner_profiles(
    manifest: Mapping[str, Any], profile_manifest: Mapping[str, Any]
) -> set[str]:
    profile_ids = validate_profile_manifest(profile_manifest)
    missing = [
        f"{owner['id']}: {profile_id}"
        for owner in _owner_records(manifest)
        for profile_id in _as_string_list(owner.get("profile_ids"), f"{owner['id']}.profile_ids")
        if profile_id not in profile_ids
    ]
    if missing:
        raise ProfileUnavailable("profile owner ids are missing: " + ", ".join(missing))
    return profile_ids


def _requirement_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = PACKAGE_TOKEN.match(value.strip())
    return normalize_package_name(match.group(0)) if match else None


def _project_requirements(text: str) -> tuple[str, dict[str, set[str]]]:
    if not text:
        return "", {}
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise CoherenceError(f"invalid project declaration: {exc}") from exc
    project = _as_mapping(document.get("project"), "project declaration project")
    project_name = normalize_package_name(str(project.get("name", "")))
    requirements: dict[str, set[str]] = {}
    sources: list[Any] = [project.get("dependencies", [])]
    optional = project.get("optional-dependencies", {})
    if isinstance(optional, Mapping):
        sources.extend(optional.values())
    groups = document.get("dependency-groups", {})
    if isinstance(groups, Mapping):
        sources.extend(groups.values())
    for source in sources:
        if not isinstance(source, list):
            raise CoherenceError("project dependency groups must be lists")
        for requirement in source:
            name = _requirement_name(requirement)
            if name and name != project_name:
                requirements.setdefault(name, set()).add(str(requirement).strip())
    return project_name, requirements


def _lock_rows(text: str) -> dict[str, list[Mapping[str, Any]]]:
    if not text:
        return {}
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise CoherenceError(f"invalid uv.lock: {exc}") from exc
    rows = document.get("package")
    if not isinstance(rows, list):
        raise CoherenceError("uv.lock must contain a package array")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for index, raw_row in enumerate(rows):
        row = _as_mapping(raw_row, f"uv.lock package[{index}]")
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise CoherenceError(f"uv.lock package[{index}] has no name")
        grouped.setdefault(normalize_package_name(name), []).append(row)
    return grouped


def _lock_header(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise CoherenceError(f"invalid uv.lock: {exc}") from exc
    return {
        key: document.get(key)
        for key in ("requires-python", "resolution-markers")
        if key in document
    }


def _material_row(row: Mapping[str, Any]) -> Any:
    return _identity_value(
        {key: value for key, value in row.items() if key != "resolution-markers"}
    )


def _version_key(value: str) -> tuple[Any, ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"[.\-+]", value)
    )


def _marker_state(expression: str | None, environment: Mapping[str, str]) -> bool | None:  # noqa: C901
    """Evaluate the lock marker subset used by uv's supported profiles."""
    if not expression:
        return True

    def atom_state(atom: str) -> bool | None:
        atom = atom.strip()
        while atom.startswith("(") and atom.endswith(")"):
            atom = atom[1:-1].strip()
        match = MARKER_ATOM.fullmatch(atom)
        if match is None:
            return None
        key = match.group("key")
        observed = environment.get(key)
        if observed is None:
            return None
        expected = match.group("value")
        operator = match.group("operator")
        if key in {"python_version", "python_full_version"}:
            left, right = _version_key(observed), _version_key(expected.removesuffix(".*"))
            if expected.endswith(".*") and operator in {"==", "!="}:
                equal = observed == expected.removesuffix(".*") or observed.startswith(
                    f"{expected.removesuffix('.*')}."
                )
                return equal if operator == "==" else not equal
        else:
            left, right = observed.casefold(), expected.casefold()
        return {
            "==": left == right,
            "!=": left != right,
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
        }.get(operator)

    term_states: list[bool | None] = []
    for term in re.split(r"\s+or\s+", expression, flags=re.IGNORECASE):
        atom_states = [
            atom_state(atom) for atom in re.split(r"\s+and\s+", term, flags=re.IGNORECASE)
        ]
        if any(state is False for state in atom_states):
            term_states.append(False)
        elif all(state is True for state in atom_states):
            term_states.append(True)
        else:
            term_states.append(None)
    if any(state is True for state in term_states):
        return True
    if all(state is False for state in term_states):
        return False
    return None


def _python_environment(profile: Mapping[str, Any]) -> dict[str, str]:
    version = str(profile["python"])
    full_version = version if len(version.split(".")) == 3 else f"{version}.0"
    return {
        "python_version": version,
        "python_full_version": full_version,
        "sys_platform": str(profile.get("sys_platform", "linux")),
        "platform_machine": str(profile.get("platform_machine", "x86_64")),
        "os_name": str(profile.get("os_name", "posix")),
        "platform_python_implementation": str(profile.get("implementation", "CPython")),
        "implementation_name": str(profile.get("implementation_name", "cpython")),
    }


def _active_identity(
    rows: Mapping[str, list[Mapping[str, Any]]], environment: Mapping[str, str]
) -> dict[str, list[Any]]:
    active: dict[str, list[Any]] = {}
    for name, variants in rows.items():
        selected: list[Any] = []
        for row in variants:
            markers = row.get("resolution-markers", [])
            if not isinstance(markers, list):
                raise CoherenceError(f"lock row {name} has invalid resolution-markers")
            states = [
                _marker_state(marker, environment) for marker in markers if isinstance(marker, str)
            ]
            if not states or any(state is True for state in states):
                selected.append(_material_row(row))
            elif all(state is False for state in states):
                continue
            else:
                raise ProfileUnavailable(
                    f"unsupported resolution marker for {name} in profile {environment['python_version']}"
                )
        if selected:
            active[name] = sorted(selected, key=_canonical_json)
    return active


def compare_lock_resolution(
    base_text: str,
    head_text: str,
    python_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare supported-profile lock semantics while exposing normalization-only churn."""
    base_rows = _lock_rows(base_text)
    head_rows = _lock_rows(head_text)
    base_header = _lock_header(base_text)
    head_header = _lock_header(head_text)
    changed_names = sorted(
        name
        for name in set(base_rows) | set(head_rows)
        if _identity_value(base_rows.get(name, [])) != _identity_value(head_rows.get(name, []))
    )
    material_names: set[str] = set()
    material_fields = sorted(
        key
        for key in set(base_header) | set(head_header)
        if _identity_value(base_header.get(key)) != _identity_value(head_header.get(key))
    )
    for profile in python_profiles:
        environment = _python_environment(profile)
        base_active = _active_identity(base_rows, environment)
        head_active = _active_identity(head_rows, environment)
        for name in set(base_active) | set(head_active):
            if base_active.get(name, []) != head_active.get(name, []):
                material_names.add(name)
    return {
        "changed_packages": changed_names,
        "material_packages": sorted(material_names),
        "material_fields": material_fields,
        "material_resolution": bool(material_names or material_fields),
        "normalization_only": bool(changed_names) and not material_names and not material_fields,
    }


def _lock_dependency_names(text: str, root_package: str) -> set[str] | None:  # noqa: C901
    rows = _lock_rows(text)
    row = next(iter(rows.get(normalize_package_name(root_package), [])), None)
    if row is None:
        return None
    names: set[str] = set()
    for dependency in row.get("dependencies", []):
        if isinstance(dependency, Mapping) and isinstance(dependency.get("name"), str):
            names.add(normalize_package_name(dependency["name"]))
    optional = row.get("optional-dependencies", {})
    if isinstance(optional, Mapping):
        for values in optional.values():
            if isinstance(values, list):
                for dependency in values:
                    if isinstance(dependency, Mapping) and isinstance(dependency.get("name"), str):
                        names.add(normalize_package_name(dependency["name"]))
    development = row.get("dev-dependencies", {})
    if isinstance(development, Mapping):
        for values in development.values():
            if isinstance(values, list):
                for dependency in values:
                    if isinstance(dependency, Mapping) and isinstance(dependency.get("name"), str):
                        names.add(normalize_package_name(dependency["name"]))
    return names


def _requirement_constraint(requirement: str) -> dict[str, Any]:
    lower: tuple[tuple[Any, ...], bool] | None = None
    upper: tuple[tuple[Any, ...], bool] | None = None
    exact: set[str] = set()
    excluded: set[str] = set()
    for operator, raw_version in SPECIFIER.findall(requirement.split(";", 1)[0]):
        version = raw_version.strip()
        if operator in {"==", "==="}:
            exact.add(version)
        elif operator == "!=":
            excluded.add(version)
        elif operator in {">", ">="}:
            candidate = (_version_key(version), operator == ">=")
            if (
                lower is None
                or candidate[0] > lower[0]
                or (candidate[0] == lower[0] and not candidate[1])
            ):
                lower = candidate
        elif operator in {"<", "<="}:
            candidate = (_version_key(version), operator == "<=")
            if (
                upper is None
                or candidate[0] < upper[0]
                or (candidate[0] == upper[0] and not candidate[1])
            ):
                upper = candidate
    return {"lower": lower, "upper": upper, "exact": exact, "excluded": excluded}


def _constraint_contains(constraint: Mapping[str, Any], version: str) -> bool:
    key = _version_key(version)
    lower = constraint.get("lower")
    upper = constraint.get("upper")
    if lower is not None and (key < lower[0] or (key == lower[0] and not lower[1])):
        return False
    if upper is not None and (key > upper[0] or (key == upper[0] and not upper[1])):
        return False
    return version not in constraint.get("excluded", set())


def _find_requirement_conflicts(
    requirement_maps: Mapping[str, Mapping[str, set[str]]],
    shared_names: Iterable[str],
) -> list[str]:
    conflicts: list[str] = []
    for name in sorted(set(shared_names)):
        constraints = [
            _requirement_constraint(requirement)
            for requirements in requirement_maps.values()
            for requirement in requirements.get(name, set())
        ]
        exact_versions = {version for constraint in constraints for version in constraint["exact"]}
        if len(exact_versions) > 1:
            conflicts.append(
                f"incompatible exact requirements for {name}: {sorted(exact_versions)}"
            )
            continue
        if exact_versions:
            version = next(iter(exact_versions))
            if any(not _constraint_contains(constraint, version) for constraint in constraints):
                conflicts.append(f"incompatible requirements for {name}: {sorted(exact_versions)}")
                continue
        lowers = [constraint["lower"] for constraint in constraints if constraint["lower"]]
        uppers = [constraint["upper"] for constraint in constraints if constraint["upper"]]
        if lowers and uppers:
            lower = max(lower[0] for lower in lowers)
            upper = min(upper[0] for upper in uppers)
            lower_closed = all(item[1] for item in lowers if item[0] == lower)
            upper_closed = all(item[1] for item in uppers if item[0] == upper)
            if lower > upper or (lower == upper and (not lower_closed or not upper_closed)):
                conflicts.append(f"incompatible bounds for {name}")
    return conflicts


def _owner_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_as_mapping(value, "owner") for value in manifest["owners"]]


def _profile_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        _as_mapping(value, "python profile")
        for value in manifest["python_profiles"]
        if isinstance(value, Mapping) and value.get("required") is True
    ]


def _run_command(
    args: Sequence[str], cwd: Path, runner: Callable[[Sequence[str], Path], Any] | None
) -> Any:
    if runner is not None:
        return runner(args, cwd)
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_profile_checks(
    repo_root: Path,
    manifest: Mapping[str, Any],
    owners: Sequence[Mapping[str, Any]],
    profile_manifest: Mapping[str, Any],
    runner: Callable[[Sequence[str], Path], Any] | None,
) -> list[dict[str, Any]]:
    """Run pinned uv lock checks for every required owner/Python profile."""
    _validate_owner_profiles(manifest, profile_manifest)

    resolver = _as_mapping(manifest["resolver"], "resolver")
    uv_command = shutil.which("uv") or "uv"
    version_result = _run_command([uv_command, "--version"], repo_root, runner)
    version_output = str(getattr(version_result, "stdout", "") or "").strip()
    expected_version = str(resolver["version"])
    if version_result.returncode != 0 or not version_output.startswith(f"uv {expected_version}"):
        raise ProfileUnavailable(
            f"pinned uv {expected_version} unavailable (observed {version_output or 'missing'})"
        )

    checked: list[dict[str, Any]] = []
    for python_profile in _profile_records(manifest):
        version = str(python_profile["python"])
        find_result = _run_command(
            [uv_command, "python", "find", version, "--no-python-downloads"], repo_root, runner
        )
        if find_result.returncode != 0:
            raise ProfileUnavailable(f"Python profile {version} is unavailable")
        for owner in owners:
            owner_directory = (repo_root / str(owner["lockfile"])).parent
            result = _run_command(
                [
                    uv_command,
                    "lock",
                    "--check",
                    "--python",
                    version,
                    "--directory",
                    str(owner_directory),
                ],
                repo_root,
                runner,
            )
            if result.returncode != 0:
                detail = str(getattr(result, "stderr", "") or getattr(result, "stdout", "")).strip()
                raise LockCheckFailed(
                    f"uv lock check failed for {owner['id']} Python {version}: {detail}"
                )
            checked.append({"owner": owner["id"], "python": version, "status": "checked"})
    return checked


def evaluate_coherence(  # noqa: C901, PLR0912, PLR0915
    *,
    manifest: Mapping[str, Any],
    profile_manifest: Mapping[str, Any],
    base_files: Mapping[str, str],
    head_files: Mapping[str, str],
    changed_files: Sequence[str],
    run_profile_checks: bool = True,
    runner: Callable[[Sequence[str], Path], Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Evaluate one declaration/lock change through the public report contract."""
    try:
        validate_manifest(manifest)
        _validate_owner_profiles(manifest, profile_manifest)
        owners = _owner_records(manifest)
        profile_records = _profile_records(manifest)
        owner_by_declaration = {str(owner["declaration"]): owner for owner in owners}
        owner_by_lock = {str(owner["lockfile"]): owner for owner in owners}
        changed = sorted({str(path) for path in changed_files})
        changed_declarations = [path for path in changed if path in owner_by_declaration]
        changed_locks = [path for path in changed if path in owner_by_lock]
        dependency_paths = sorted(set(changed_declarations) | set(changed_locks))
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "coherent",
            "scope": "not_applicable",
            "changed_files": changed,
            "changed_declarations": changed_declarations,
            "changed_lockfiles": changed_locks,
            "required_lockfiles": [],
            "changed_packages": [],
            "material_packages": [],
            "material_fields": [],
            "material_resolution": False,
            "profiles": [],
            "reasons": [],
        }
        if not dependency_paths:
            report["message"] = "no mapped dependency declaration or lockfile changed"
            return report

        declaration_owners = [owner_by_declaration[path] for path in changed_declarations]
        lock_owners = [owner_by_lock[path] for path in changed_locks]
        required_owners = {str(owner["id"]): owner for owner in declaration_owners}
        required_lockfiles = {str(owner["lockfile"]) for owner in declaration_owners}
        report["required_lockfiles"] = sorted(required_lockfiles)
        report["owners"] = sorted(
            {str(owner["id"]) for owner in [*declaration_owners, *lock_owners]}
        )

        requirement_maps: dict[str, dict[str, set[str]]] = {}
        direct_names: set[str] = set()
        changed_direct_names: set[str] = set()
        for owner in owners:
            owner_id = str(owner["id"])
            declaration = str(owner["declaration"])
            base_project_name, base_requirements = _project_requirements(
                base_files.get(declaration, "")
            )
            head_project_name, head_requirements = _project_requirements(
                head_files.get(declaration, "")
            )
            if base_project_name and head_project_name and base_project_name != head_project_name:
                raise CoherenceError(f"project name changed for owner {owner_id}")
            requirement_maps[owner_id] = head_requirements
            direct_names.update(head_requirements)
            changed_direct_names.update(
                name
                for name in set(base_requirements) | set(head_requirements)
                if base_requirements.get(name, set()) != head_requirements.get(name, set())
            )

        if len(declaration_owners) == 1:
            report["scope"] = (
                "root-only" if declaration_owners[0]["id"] == "root" else "fast-pysf-only"
            )
        elif declaration_owners:
            coupling_values = {str(owner["coupling"]) for owner in declaration_owners}
            if "workspace" in coupling_values:
                report["scope"] = "workspace/member-coupling"
            else:
                shared_names = set.intersection(
                    *(set(requirement_maps[str(owner["id"])]) for owner in declaration_owners)
                )
                report["scope"] = (
                    "shared-declaration-independently-resolved" if shared_names else "multi-owner"
                )
                conflicts = _find_requirement_conflicts(requirement_maps, shared_names)
                if conflicts:
                    report["status"] = "conflict"
                    report["reasons"].extend(conflicts)
        else:
            report["scope"] = "transitive-only"

        unexpected_locks = sorted(
            str(owner["lockfile"])
            for owner in lock_owners
            if str(owner["id"]) not in required_owners and declaration_owners
        )
        if unexpected_locks:
            report["status"] = "conflict"
            report["reasons"].append(
                "lockfiles changed without their declaration owner: " + ", ".join(unexpected_locks)
            )

        missing_locks = sorted(
            lockfile for lockfile in required_lockfiles if lockfile not in changed_locks
        )
        if missing_locks and report["status"] == "coherent":
            report["status"] = "missing_lock_update"
            report["reasons"].append(
                "declaration changed without lock update: " + ", ".join(missing_locks)
            )

        resolution_reports: dict[str, dict[str, Any]] = {}
        for lockfile in changed_locks:
            resolution = compare_lock_resolution(
                base_files.get(lockfile, ""), head_files.get(lockfile, ""), profile_records
            )
            resolution_reports[lockfile] = resolution
            report["changed_packages"] = sorted(
                set(report["changed_packages"]) | set(resolution["changed_packages"])
            )
            report["material_packages"] = sorted(
                set(report["material_packages"]) | set(resolution["material_packages"])
            )
            report["material_fields"] = sorted(
                set(report["material_fields"]) | set(resolution["material_fields"])
            )
            report["material_resolution"] = (
                report["material_resolution"] or resolution["material_resolution"]
            )
            report.setdefault("lock_resolution", {})[lockfile] = resolution
        report["changed_packages"] = sorted(set(report["changed_packages"]) | changed_direct_names)
        if not declaration_owners and report["changed_packages"]:
            report["classification"] = (
                "lock_normalization"
                if all(item["normalization_only"] for item in resolution_reports.values())
                else "transitive_only"
                if set(report["changed_packages"]).isdisjoint(direct_names)
                else "lock_only_direct"
            )
        elif (
            resolution_reports
            and any(item["changed_packages"] for item in resolution_reports.values())
            and all(item["normalization_only"] for item in resolution_reports.values())
        ):
            report["classification"] = "lock_normalization"
        else:
            report["classification"] = (
                "material_resolution" if report["material_resolution"] else "declaration_only"
            )

        for owner in declaration_owners:
            declaration = str(owner["declaration"])
            lockfile = str(owner["lockfile"])
            if lockfile not in head_files or not head_files.get(lockfile):
                if report["status"] == "coherent":
                    report["status"] = "declaration_lock_mismatch"
                    report["reasons"].append(f"required lockfile is absent: {lockfile}")
                continue
            lock_names = _lock_dependency_names(head_files[lockfile], str(owner["root_package"]))
            if lock_names is None:
                if report["status"] == "coherent":
                    report["status"] = "declaration_lock_mismatch"
                    report["reasons"].append(
                        f"lockfile {lockfile} has no root package {owner['root_package']!r}"
                    )
                continue
            _, requirements = _project_requirements(head_files.get(declaration, ""))
            expected_names = set(requirements)
            if lock_names != expected_names:
                if report["status"] == "coherent":
                    report["status"] = "declaration_lock_mismatch"
                report["reasons"].append(
                    f"{lockfile} root dependencies disagree with {declaration}: "
                    f"missing={sorted(expected_names - lock_names)} "
                    f"extra={sorted(lock_names - expected_names)}"
                )

        if report["status"] == "coherent" and run_profile_checks:
            try:
                profile_owners = list(required_owners.values()) or lock_owners
                report["profiles"] = _run_profile_checks(
                    repo_root, manifest, profile_owners, profile_manifest, runner
                )
            except ProfileUnavailable as exc:
                report["status"] = "profile_unavailable"
                report["reasons"].append(str(exc))
            except LockCheckFailed as exc:
                report["status"] = "declaration_lock_mismatch"
                report["reasons"].append(str(exc))
        elif not run_profile_checks:
            report["profiles"] = [{"status": "not_run"}]
        return report
    except ProfileUnavailable as exc:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "profile_unavailable",
            "scope": "invalid",
            "reasons": [str(exc)],
        }
    except CoherenceError as exc:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "invalid",
            "scope": "invalid",
            "reasons": [str(exc)],
        }


def _git_text(repo_root: Path, args: Sequence[str]) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, check=False, capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else None


def _changed_files(repo_root: Path, base_ref: str, changed_file_path: Path | None) -> list[str]:
    if changed_file_path is not None:
        return [
            line.strip()
            for line in changed_file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    output = _git_text(repo_root, ["diff", "--name-only", f"{base_ref}...HEAD"])
    if output is None:
        output = _git_text(repo_root, ["diff", "--name-only", base_ref, "HEAD"])
    if output is None:
        raise CoherenceError(f"unable to compare HEAD with {base_ref!r}")
    return [line.strip() for line in output.splitlines() if line.strip()]


def _file_at_ref(repo_root: Path, ref: str, relative_path: str) -> str:
    return _git_text(repo_root, ["show", f"{ref}:{relative_path}"]) or ""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoherenceError(f"unable to load JSON {path}: {exc}") from exc
    return _as_mapping(value, str(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--profile-manifest", type=Path, default=DEFAULT_PROFILE_MANIFEST)
    parser.add_argument("--changed-files-file", type=Path)
    parser.add_argument("--skip-profile-check", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exact-base dependency coherence check."""
    args = _build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        profile_manifest = _load_json(args.profile_manifest)
        if _git_text(REPO_ROOT, ["rev-parse", "--verify", f"{args.base_ref}^{{commit}}"]) is None:
            raise CoherenceError(f"base ref is unavailable: {args.base_ref}")
        changed = _changed_files(REPO_ROOT, args.base_ref, args.changed_files_file)
        owner_paths = [
            str(path)
            for owner in _owner_records(manifest)
            for path in (owner["declaration"], owner["lockfile"])
        ]
        base_files = {path: _file_at_ref(REPO_ROOT, args.base_ref, path) for path in owner_paths}
        head_files = {
            path: (REPO_ROOT / path).read_text(encoding="utf-8")
            if (REPO_ROOT / path).is_file()
            else ""
            for path in owner_paths
        }
        report = evaluate_coherence(
            manifest=manifest,
            profile_manifest=profile_manifest,
            base_files=base_files,
            head_files=head_files,
            changed_files=changed,
            run_profile_checks=not args.skip_profile_check,
            repo_root=REPO_ROOT,
        )
    except (CoherenceError, OSError) as exc:
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "invalid",
            "scope": "invalid",
            "reasons": [str(exc)],
        }
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"DEPENDENCY COHERENCE: {str(report['status']).upper()}")
        print(f"Scope: {report.get('scope', 'unknown')}")
        for reason in report.get("reasons", []):
            print(f"Reason: {reason}")
    return 0 if report.get("status") == "coherent" else 1


if __name__ == "__main__":
    raise SystemExit(main())
