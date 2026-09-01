#!/usr/bin/env python3
"""Assemble and verify a build-once, non-publishing Robot SF candidate bundle.

The helper never builds or publishes packages. ``assemble`` admits exactly one
wheel and one source distribution that were built elsewhere, normalises a uv
CycloneDX export, and binds every payload byte to deterministic provenance.
``verify`` is deliberately offline and only revalidates the admitted bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from scripts.tools.check_distribution_licenses import (
    DistributionLicenseError,
    check_distribution,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

SCHEMA_VERSION = "robot_sf.software_candidate.v1"
SCHEMA_ID = "https://robot-sf.dev/schema/software-candidate-manifest.v1.json"
SCHEMA_SHA256 = "d7bd1f2d7c4146b85fb23ee2d6462bb363f94c79c749b50336832742caf6bdad"
PROVENANCE_VERSION = "robot_sf.software_candidate.provenance.v1"
SCHEMA_PATH = Path(__file__).with_name("software_candidate_manifest.v1.schema.json")
DEFAULT_MATERIALIZATION_POLICY = (
    Path(__file__).resolve().parents[1] / "validation" / "software_candidate_policy.v1.json"
)
DEFAULT_MATERIALIZATION_POLICY_RELATIVE = "scripts/validation/software_candidate_policy.v1.json"
MANIFEST_NAME = "candidate-manifest.json"
PROVENANCE_NAME = "candidate-provenance.json"
MATERIALIZATION_SCHEMA_VERSION = "robot_sf.software_candidate_materialization.v1"
MATERIALIZATION_POLICY_SCHEMA_VERSION = "robot_sf.software_candidate_policy.v1"
MATERIALIZATION_INVENTORY_SCHEMA_VERSION = "robot_sf.asset_rights_inventory.v1"
MATERIALIZATION_METADATA_NAME = "SOFTWARE_CANDIDATE.json"
DEFAULT_MATERIALIZATION_INVENTORY = "scripts/validation/software_candidate_asset_rights.v1.json"
RIGHTS_POLICY_ID = "robot_sf.software_release_rights_policy.v1"
RIGHTS_POLICY_PATH = "scripts/validation/software_release_rights_policy.v1.json"
RIGHTS_POLICY_SCHEMA_PATH = "scripts/validation/software_release_rights_policy.v1.schema.json"
SANITIZED_CANDIDATE_SCHEMA = "robot_sf.software_sanitized_candidate.v1"
SANITIZED_MANIFEST_NAME = "sanitized-candidate.json"
RIGHTS_ADMISSION_SCHEMA = "robot_sf.software_rights_admission.v1"
RIGHTS_ADMISSION_NAME = "rights-admission.json"
RIGHTS_ADMISSION_SCHEMA_PATH = "scripts/validation/software_rights_admission.v1.schema.json"
RIGHTS_GATE_ID = "strict-distribution-rights"
RIGHTS_GATE_COMMAND = (
    "python scripts/tools/check_distribution_licenses.py $DIST_DIR "
    "--strict-asset-rights --repo-root $BUILD_SOURCE --source-tree-ref $SOURCE_SHA"
)
SUPPORTED_DEPENDENCY_SCHEMA_VERSION = "robot-sf.dependency-license-inventory.v1"
SUPPORTED_DEPENDENCY_POLICY_SCHEMA_VERSION = "robot-sf.dependency-license-policy.v1"
SUPPORTED_DEPENDENCY_PROFILE_SCHEMA_VERSION = "robot-sf.dependency-license-profiles.v1"
SUPPORTED_DEPENDENCY_GATE_ID = "strict-supported-dependency-surface"
SUPPORTED_DEPENDENCY_REPORT_NAME = "dependency-license-inventory.json"
SUPPORTED_DEPENDENCY_POLICY_PATH = "scripts/validation/dependency_license_policy.v1.json"
SUPPORTED_DEPENDENCY_PROFILE_PATH = "scripts/validation/dependency_license_profiles.v1.json"
# The profile manifest's reviewed ``all`` closure is the v0.0.6 public surface.
# Keep the explicit extra roster in the receipt so a core-only report cannot be
# mistaken for admission of the wheel's supported optional dependencies.
SUPPORTED_DEPENDENCY_PROFILE_IDS = ("all",)
SUPPORTED_DEPENDENCY_EXTRA_IDS = (
    "viz",
    "maps",
    "benchmark",
    "training",
    "gpu",
    "recurrent",
    "progress",
    "analytics",
    "browser",
    "sacadrl",
    "socnav",
    "criticality",
)
SUPPORTED_DEPENDENCY_GATE_COMMAND = (
    "python scripts/tools/check_dependency_license_inventory.py "
    "--repo-root $BUILD_SOURCE --output $DEPENDENCY_REPORT "
    "--candidate-bundle $CANDIDATE_BUNDLE --fail-on-unresolved"
)
UV_OUT_DIR_MARKER_NAME = ".gitignore"
UV_OUT_DIR_MARKER_BYTES = b"*"
SDIST_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
LFS_POINTER_PATTERN = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\n"
    rb"size (0|[1-9][0-9]*)\n\Z"
)
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
RUN_ID_PATTERN = re.compile(r"[1-9][0-9]*\Z")
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+!_-]*\Z")
VALIDATION_COMMANDS = (
    (
        "version-alignment",
        "python scripts/dev/check_version_alignment.py",
    ),
    (
        "metadata",
        "twine check --strict $DIST_DIR/*.whl $DIST_DIR/*.tar.gz",
    ),
    (
        "archive-license",
        "cd $BUILD_SOURCE && python scripts/tools/check_distribution_licenses.py "
        "$DIST_DIR --strict-asset-rights --repo-root $BUILD_SOURCE "
        "--inventory $BUILD_SOURCE/scripts/validation/software_candidate_asset_rights.v1.json "
        "--source-tree-ref HEAD",
    ),
    (
        "wheel-install",
        "cd $BUILD_SOURCE && bash scripts/validation/wheel_install_smoke.sh "
        "$DIST_DIR/robot_sf-*.whl",
    ),
)
VALIDATOR_IDS = tuple(identifier for identifier, _command in VALIDATION_COMMANDS)
MEMBER_KINDS = ("wheel", "sdist", "sbom", "provenance")
MATERIALIZATION_ALLOWED_ASSET_STATUSES = frozenset({"cleared", "project-authored"})
MATERIALIZATION_ASSET_SUFFIXES = frozenset(
    {
        ".bag",
        ".geojson",
        ".gif",
        ".jpeg",
        ".jpg",
        ".mp4",
        ".mov",
        ".osm",
        ".pbf",
        ".pkl",
        ".png",
        ".svg",
        ".wav",
    }
)
MATERIALIZATION_ASSET_PATH_HINTS = frozenset(
    {"assets", "data", "datasets", "maps", "media", "recordings"}
)
MATERIALIZATION_NON_ASSET_SUFFIXES = frozenset(
    {".md", ".py", ".pyi", ".rst", ".sh", ".toml", ".txt"}
)
# The development checkout keeps the standalone RLlib integration available,
# but it is not part of the rights-clean software surface.  The candidate
# materializer removes only that optional-dependency stanza from the copied
# ``pyproject.toml``.  The public ``all`` aggregator and its twelve reviewed
# members remain unchanged in the candidate package metadata.
CANDIDATE_PYPROJECT_PATH = "pyproject.toml"
SUPPORTED_CANDIDATE_EXTRA_IDS = (
    "viz",
    "maps",
    "benchmark",
    "training",
    "gpu",
    "recurrent",
    "progress",
    "analytics",
    "browser",
    "sacadrl",
    "socnav",
    "criticality",
)
SUPPORTED_CANDIDATE_DISTRIBUTION_EXTRA_IDS = (
    *SUPPORTED_CANDIDATE_EXTRA_IDS,
    "all",
)
MATERIALIZATION_PAYLOAD_FIELDS = (
    "candidate_commit_sha",
    "candidate_tree_sha",
    "policy_path",
    "policy_sha256",
    "source_inventory_path",
    "source_inventory_sha256",
    "candidate_inventory_path",
    "candidate_metadata_path",
)
MATERIALIZATION_REPORT_FIELDS = {
    "schema_version",
    "package",
    "source_sha",
    "policy_path",
    "policy_sha256",
    "source_inventory_path",
    "source_inventory_sha256",
    "candidate_inventory_path",
    "candidate_metadata_path",
    "candidate_commit_sha",
    "candidate_tree_sha",
    "members",
    "excluded_paths",
    "excluded_non_regular_paths",
}
SYSTEM_GIT = Path("/usr/bin/git")
SYSTEM_TEMP = Path("/tmp")


class CandidateError(ValueError):
    """Raised when candidate admission or offline verification fails closed."""


def _trusted_git_environment() -> dict[str, str]:
    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "",
    }


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise CandidateError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path, *, label: str) -> Any:
    try:
        raw = path.read_bytes()
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except CandidateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc


def _json_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    except ValueError as exc:
        raise CandidateError("candidate output contains a non-finite JSON value") from exc
    return (text + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CandidateError(f"cannot hash candidate member {path}: {exc}") from exc
    return digest.hexdigest()


def _normalise_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _safe_member_name(name: str, *, archive: Path) -> str:
    if not name or "\\" in name or "\x00" in name or name.startswith("/"):
        raise CandidateError(f"{archive.name}: unsafe archive member path: {name!r}")
    raw_parts = name.rstrip("/").split("/")
    if not raw_parts or any(
        not part
        or part in {".", ".."}
        or ":" in part
        or part.endswith((".", " "))
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in part)
        for part in raw_parts
    ):
        raise CandidateError(f"{archive.name}: unsafe archive member path: {name!r}")
    return PurePosixPath(*raw_parts).as_posix()


def _validate_unique_archive_members(archive: Path) -> None:
    try:
        if archive.suffix in {".whl", ".zip"}:
            with zipfile.ZipFile(archive) as source:
                infos = source.infolist()
                links = sorted(
                    info.filename for info in infos if stat.S_ISLNK(info.external_attr >> 16)
                )
                if links:
                    raise CandidateError(
                        f"{archive.name}: symbolic-link archive members are forbidden: "
                        f"{', '.join(links)}"
                    )
                names = [_safe_member_name(info.filename, archive=archive) for info in infos]
        else:
            with tarfile.open(archive, mode="r:*") as source:
                members = source.getmembers()
                non_regular = sorted(
                    member.name for member in members if not member.isfile() and not member.isdir()
                )
                if non_regular:
                    raise CandidateError(
                        f"{archive.name}: non-regular archive members are forbidden: "
                        f"{', '.join(non_regular)}"
                    )
                names = [_safe_member_name(member.name, archive=archive) for member in members]
    except CandidateError:
        raise
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise CandidateError(f"cannot inspect archive {archive}: {exc}") from exc

    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise CandidateError(
            f"{archive.name}: duplicate archive member names: {', '.join(duplicates)}"
        )


def _metadata_fields(raw: bytes, *, archive: Path) -> tuple[str, str]:
    message = BytesParser(policy=email_policy).parsebytes(raw)
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise CandidateError(f"{archive.name}: package metadata must contain Name and Version")
    if _normalise_distribution_name(str(name)) != "robot-sf":
        raise CandidateError(f"{archive.name}: package metadata is not for Robot SF: {name!r}")
    version_text = str(version)
    if not VERSION_PATTERN.fullmatch(version_text):
        raise CandidateError(
            f"{archive.name}: unsafe or ambiguous package version: {version_text!r}"
        )
    return str(name), version_text


def _wheel_metadata(archive: Path) -> tuple[str, str]:
    _validate_unique_archive_members(archive)
    try:
        with zipfile.ZipFile(archive) as source:
            matches = [
                info
                for info in source.infolist()
                if PurePosixPath(info.filename).name == "METADATA"
                and PurePosixPath(info.filename).parent.name.endswith(".dist-info")
            ]
            if len(matches) != 1:
                raise CandidateError(
                    f"{archive.name}: expected exactly one .dist-info/METADATA member; "
                    f"found {len(matches)}"
                )
            metadata = _metadata_fields(source.read(matches[0]), archive=archive)
    except CandidateError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise CandidateError(f"cannot read wheel metadata from {archive}: {exc}") from exc

    filename_parts = archive.name.removesuffix(".whl").split("-")
    if len(filename_parts) < 5 or filename_parts[0] != "robot_sf":
        raise CandidateError(f"unexpected Robot SF wheel filename: {archive.name}")
    if filename_parts[1] != metadata[1].replace("-", "_"):
        raise CandidateError(
            f"{archive.name}: filename version {filename_parts[1]!r} does not match "
            f"METADATA version {metadata[1]!r}"
        )
    return metadata


def _sdist_metadata(archive: Path) -> tuple[str, str]:
    _validate_unique_archive_members(archive)
    try:
        with tarfile.open(archive, mode="r:*") as source:
            matches = [
                member
                for member in source.getmembers()
                if member.isfile()
                and len(PurePosixPath(member.name).parts) == 2
                and PurePosixPath(member.name).name == "PKG-INFO"
            ]
            if len(matches) != 1:
                raise CandidateError(
                    f"{archive.name}: expected exactly one root PKG-INFO member; "
                    f"found {len(matches)}"
                )
            extracted = source.extractfile(matches[0])
            if extracted is None:
                raise CandidateError(f"{archive.name}: cannot read root PKG-INFO")
            metadata = _metadata_fields(extracted.read(), archive=archive)
    except CandidateError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise CandidateError(f"cannot read sdist metadata from {archive}: {exc}") from exc

    suffix = next((item for item in SDIST_SUFFIXES if archive.name.endswith(item)), None)
    if suffix is None:
        raise CandidateError(f"unsupported Robot SF sdist filename: {archive.name}")
    expected = f"robot_sf-{metadata[1]}{suffix}"
    if archive.name != expected:
        raise CandidateError(
            f"{archive.name}: filename does not match PKG-INFO version; expected {expected}"
        )
    return metadata


def _distribution_inputs(dist_dir: Path) -> tuple[Path, Path, str]:
    if not dist_dir.is_dir() or dist_dir.is_symlink():
        raise CandidateError(f"distribution input is not a real directory: {dist_dir}")
    entries = sorted(dist_dir.iterdir(), key=lambda path: path.name)
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise CandidateError(f"unclassified distribution member: {path.name}")
    marker = next((path for path in entries if path.name == UV_OUT_DIR_MARKER_NAME), None)
    if marker is not None and marker.read_bytes() != UV_OUT_DIR_MARKER_BYTES:
        raise CandidateError(
            f"pinned uv out-dir marker has unexpected content: {UV_OUT_DIR_MARKER_NAME}"
        )
    artifacts = [path for path in entries if path.name != UV_OUT_DIR_MARKER_NAME]
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [
        path
        for path in artifacts
        if any(path.name.endswith(suffix) for suffix in SDIST_SUFFIXES) and path.suffix != ".whl"
    ]
    classified = {*wheels, *sdists}
    unclassified = [path.name for path in artifacts if path not in classified]
    if unclassified:
        raise CandidateError(
            f"unclassified distribution members: {', '.join(sorted(unclassified))}"
        )
    if len(wheels) != 1 or len(sdists) != 1 or len(artifacts) != 2:
        raise CandidateError(
            "candidate requires exactly one Robot SF wheel and one Robot SF sdist "
            f"(found {len(wheels)} wheel(s), {len(sdists)} sdist(s), "
            f"{len(artifacts)} candidate artifact(s))"
        )

    wheel_name, wheel_version = _wheel_metadata(wheels[0])
    sdist_name, sdist_version = _sdist_metadata(sdists[0])
    if _normalise_distribution_name(wheel_name) != _normalise_distribution_name(sdist_name):
        raise CandidateError("wheel and sdist package names do not match")
    if wheel_version != sdist_version:
        raise CandidateError(
            f"wheel and sdist versions do not match: {wheel_version!r} != {sdist_version!r}"
        )
    return wheels[0], sdists[0], wheel_version


def _read_control_file(path: Path, *, label: str) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CandidateError(f"cannot read {label}: {path}: {exc}") from exc
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise CandidateError(f"{label} is not ASCII: {path}") from exc
    if not text or "\n" in text or "\r" in text or "\x00" in text:
        raise CandidateError(f"{label} is malformed: {path}")
    return text


def _resolve_git_dir(repo_root: Path) -> Path:
    dot_git = repo_root / ".git"
    if dot_git.is_symlink():
        raise CandidateError(f"source .git entry cannot be a symlink: {dot_git}")
    if dot_git.is_dir():
        return dot_git.resolve()
    if not dot_git.is_file():
        raise CandidateError(f"source repository has no real .git directory: {repo_root}")
    pointer = _read_control_file(dot_git, label="source .git pointer")
    if not pointer.startswith("gitdir: "):
        raise CandidateError(f"source .git pointer is malformed: {dot_git}")
    candidate = Path(pointer.removeprefix("gitdir: "))
    if not candidate.is_absolute():
        candidate = dot_git.parent / candidate
    if candidate.is_symlink() or not candidate.is_dir():
        raise CandidateError(f"source git directory is not a real directory: {candidate}")
    return candidate.resolve()


def _resolve_common_dir(git_dir: Path) -> Path:
    commondir_file = git_dir / "commondir"
    if not commondir_file.exists():
        return git_dir
    if commondir_file.is_symlink() or not commondir_file.is_file():
        raise CandidateError(f"source commondir pointer is not a real file: {commondir_file}")
    candidate = Path(_read_control_file(commondir_file, label="source commondir"))
    if not candidate.is_absolute():
        candidate = git_dir / candidate
    if candidate.is_symlink() or not candidate.is_dir():
        raise CandidateError(f"source common git directory is not real: {candidate}")
    return candidate.resolve()


def _git_storage(repo_root: Path) -> tuple[Path, Path, Path]:
    git_dir = _resolve_git_dir(repo_root)
    common_dir = _resolve_common_dir(git_dir)
    objects_dir = common_dir / "objects"
    if not objects_dir.is_dir() or objects_dir.is_symlink():
        raise CandidateError(f"source object store is not a real directory: {objects_dir}")
    return git_dir, common_dir, objects_dir.resolve()


def _safe_ref_name(ref: str) -> bool:
    return (
        ref.startswith("refs/")
        and not ref.endswith(("/", "."))
        and "//" not in ref
        and ".." not in ref
        and "\\" not in ref
        and all(character.isalnum() or character in "-._/" for character in ref)
    )


def _packed_ref(common_dir: Path, ref: str) -> str | None:
    path = common_dir / "packed-refs"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise CandidateError(f"source packed-refs is not a real file: {path}")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CandidateError(f"cannot read source packed-refs: {path}: {exc}") from exc
    for line in lines:
        if not line or line.startswith(("#", "^")):
            continue
        try:
            digest, name = line.split(" ", maxsplit=1)
        except ValueError as exc:
            raise CandidateError(f"source packed-refs is malformed: {path}") from exc
        if name == ref:
            return digest
    return None


def _source_head(git_dir: Path, common_dir: Path) -> str:
    value = _read_control_file(git_dir / "HEAD", label="source HEAD")
    visited: set[str] = set()
    for _depth in range(8):
        if SHA_PATTERN.fullmatch(value):
            return value
        if not value.startswith("ref: "):
            raise CandidateError("source HEAD does not contain an exact commit or symbolic ref")
        ref = value.removeprefix("ref: ")
        if not _safe_ref_name(ref) or ref in visited:
            raise CandidateError(f"source HEAD has an unsafe or cyclic ref: {ref!r}")
        visited.add(ref)
        value = ""
        for root in (git_dir, common_dir):
            candidate = root / PurePosixPath(ref)
            if candidate.exists():
                if candidate.is_symlink() or not candidate.is_file():
                    raise CandidateError(f"source ref is not a real file: {candidate}")
                value = _read_control_file(candidate, label=f"source ref {ref}")
                break
        if not value:
            value = _packed_ref(common_dir, ref) or ""
        if not value:
            raise CandidateError(f"source HEAD ref is missing: {ref}")
    raise CandidateError("source HEAD symbolic-ref depth exceeds the fail-closed limit")


def _safe_workspace_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or path.as_posix() != value
        or path.parts[0] == ".git"
        or any(
            part in {"", ".", ".."}
            or "\\" in part
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in part)
            for part in path.parts
        )
    ):
        raise CandidateError(f"source workspace path is unsafe or ambiguous: {value!r}")
    return path


@contextmanager
def _source_carrier(
    objects_dir: Path,
    expected_sha: str,
) -> Iterator[tuple[Path, dict[str, str]]]:
    if not SYSTEM_GIT.is_file() or SYSTEM_GIT.is_symlink() or not os.access(SYSTEM_GIT, os.X_OK):
        raise CandidateError(f"trusted system Git is unavailable: {SYSTEM_GIT}")
    objects_text = str(objects_dir)
    if os.pathsep in objects_text or "\n" in objects_text or "\x00" in objects_text:
        raise CandidateError(f"source object-store path is unsafe: {objects_dir}")
    if not SYSTEM_TEMP.is_dir() or SYSTEM_TEMP.is_symlink():
        raise CandidateError(f"trusted external temporary directory is unavailable: {SYSTEM_TEMP}")
    with tempfile.TemporaryDirectory(
        prefix="robot-sf-source-carrier-",
        dir=SYSTEM_TEMP,
    ) as carrier_text:
        carrier = Path(carrier_text)
        (carrier / "objects" / "info").mkdir(parents=True)
        (carrier / "objects" / "pack").mkdir()
        (carrier / "refs" / "heads").mkdir(parents=True)
        (carrier / "refs" / "tags").mkdir()
        (carrier / "HEAD").write_text(f"{expected_sha}\n", encoding="ascii")
        (carrier / "config").write_text(
            "[core]\n"
            "\trepositoryformatversion = 0\n"
            "\tbare = true\n"
            "\tfilemode = true\n"
            "\tsymlinks = true\n",
            encoding="ascii",
        )
        env = _trusted_git_environment()
        env["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = objects_text
        yield carrier, env


def _run_carrier(
    carrier: Path,
    env: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            [
                str(SYSTEM_GIT),
                "--no-replace-objects",
                f"--git-dir={carrier}",
                *arguments,
            ],
            check=False,
            capture_output=True,
            cwd=carrier,
            env=env,
        )
    except OSError as exc:
        raise CandidateError(f"cannot execute trusted source carrier: {exc}") from exc


def _carrier_failure(label: str, result: subprocess.CompletedProcess[bytes]) -> CandidateError:
    detail = result.stderr.decode("utf-8", errors="replace").strip() or f"exit {result.returncode}"
    return CandidateError(f"trusted source carrier cannot {label}: {detail}")


def _carrier_tree(objects_dir: Path, expected_sha: str) -> dict[str, tuple[str, str]]:
    with _source_carrier(objects_dir, expected_sha) as (carrier, env):
        object_type = _run_carrier(carrier, env, "cat-file", "-t", expected_sha)
        if object_type.returncode != 0:
            raise _carrier_failure("resolve expected commit", object_type)
        if object_type.stdout != b"commit\n":
            raise CandidateError(f"source identity is not an exact commit: {expected_sha}")
        result = _run_carrier(
            carrier,
            env,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            expected_sha,
        )
    if result.returncode != 0:
        raise _carrier_failure("read commit tree", result)

    entries: dict[str, tuple[str, str]] = {}
    for raw_entry in result.stdout.split(b"\x00"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", maxsplit=1)
            mode, object_type, object_id = header.decode("ascii").split(" ")
            path_text = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise CandidateError("trusted source carrier returned a malformed tree entry") from exc
        _safe_workspace_path(path_text)
        if object_type != "blob" or mode not in {"100644", "100755", "120000"}:
            raise CandidateError(
                f"source tree contains unsupported type or mode at {path_text}: "
                f"{mode} {object_type}"
            )
        if not SHA_PATTERN.fullmatch(object_id) or path_text in entries:
            raise CandidateError(f"source tree entry is ambiguous: {path_text}")
        entries[path_text] = (mode, object_id)
    return entries


def _small_carrier_blob(objects_dir: Path, object_id: str) -> bytes | None:
    with _source_carrier(objects_dir, object_id) as (carrier, env):
        size_result = _run_carrier(carrier, env, "cat-file", "-s", object_id)
        if size_result.returncode != 0:
            raise _carrier_failure("inspect expected blob", size_result)
        try:
            size = int(size_result.stdout.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as exc:
            raise CandidateError("trusted source carrier returned an invalid blob size") from exc
        if size > 256:
            return None
        blob_result = _run_carrier(carrier, env, "cat-file", "blob", object_id)
    if blob_result.returncode != 0:
        raise _carrier_failure("read expected blob", blob_result)
    if len(blob_result.stdout) != size:
        raise CandidateError("trusted source carrier returned a size-drifted expected blob")
    return blob_result.stdout


def _carrier_path_uses_lfs(objects_dir: Path, expected_sha: str, name: str) -> bool:
    with _source_carrier(objects_dir, expected_sha) as (carrier, env):
        read_tree = _run_carrier(carrier, env, "read-tree", expected_sha)
        if read_tree.returncode != 0:
            raise _carrier_failure("prepare expected attribute index", read_tree)
        attribute = _run_carrier(carrier, env, "check-attr", "--cached", "-z", "filter", "--", name)
    if attribute.returncode != 0:
        raise _carrier_failure("read expected path attributes", attribute)
    return attribute.stdout == f"{name}\0filter\0lfs\0".encode()


def _lfs_contract(
    objects_dir: Path,
    expected_sha: str,
    name: str,
    object_id: str,
) -> tuple[str, int] | None:
    raw_pointer = _small_carrier_blob(objects_dir, object_id)
    match = LFS_POINTER_PATTERN.fullmatch(raw_pointer) if raw_pointer is not None else None
    if match is None or not _carrier_path_uses_lfs(objects_dir, expected_sha, name):
        return None
    return match.group(1).decode("ascii"), int(match.group(2))


def _blob_hashes(path: Path, initial: os.stat_result) -> tuple[str, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CandidateError(
            f"cannot open source file without following links: {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != initial.st_dev
            or opened.st_ino != initial.st_ino
            or opened.st_size != initial.st_size
            or opened.st_mode != initial.st_mode
            or opened.st_mtime_ns != initial.st_mtime_ns
            or opened.st_ctime_ns != initial.st_ctime_ns
        ):
            raise CandidateError(f"source file changed type or identity while hashing: {path}")
        git_digest = hashlib.sha1(usedforsecurity=False)
        git_digest.update(f"blob {opened.st_size}\0".encode("ascii"))
        content_digest = hashlib.sha256()
        try:
            while chunk := os.read(descriptor, 1024 * 1024):
                git_digest.update(chunk)
                content_digest.update(chunk)
        except OSError as exc:
            raise CandidateError(f"cannot hash source file {path}: {exc}") from exc
        final = os.fstat(descriptor)
        if (
            final.st_size != opened.st_size
            or final.st_mode != opened.st_mode
            or final.st_mtime_ns != opened.st_mtime_ns
            or final.st_ctime_ns != opened.st_ctime_ns
        ):
            raise CandidateError(f"source file changed while hashing: {path}")
        return git_digest.hexdigest(), content_digest.hexdigest()
    finally:
        os.close(descriptor)


def _expected_directories(entries: dict[str, tuple[str, str]]) -> set[str]:
    directories: set[str] = set()
    for name in entries:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _enumerate_workspace(
    repo_root: Path,
    expected_dirs: set[str],
) -> tuple[list[tuple[str, Path, os.stat_result]], set[str], list[str]]:
    pending: list[tuple[Path, PurePosixPath | None]] = [(repo_root, None)]
    entries: list[tuple[str, Path, os.stat_result]] = []
    observed_dirs: set[str] = set()
    issues: list[str] = []
    while pending:
        directory, prefix = pending.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name))
        except OSError as exc:
            raise CandidateError(f"cannot enumerate source workspace: {directory}: {exc}") from exc
        for child in children:
            if prefix is None and child.name == ".git":
                continue
            try:
                child.name.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise CandidateError("source workspace contains a non-UTF-8 path") from exc
            relative = PurePosixPath(child.name) if prefix is None else prefix / child.name
            name = _safe_workspace_path(relative.as_posix()).as_posix()
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise CandidateError(f"cannot inspect source workspace path {name}: {exc}") from exc
            if stat.S_ISDIR(metadata.st_mode):
                observed_dirs.add(name)
                if name in expected_dirs:
                    pending.append((Path(child.path), relative))
                else:
                    issues.append(f"untracked directory: {name}/")
                continue
            entries.append((name, Path(child.path), metadata))
    return entries, observed_dirs, issues


def _symlink_blob(
    repo_root: Path,
    path: Path,
    name: str,
    initial: os.stat_result,
) -> tuple[str, str | None]:
    try:
        target_bytes = os.readlink(os.fsencode(path))
        final = os.lstat(path)
    except OSError as exc:
        raise CandidateError(f"cannot read source symlink {name}: {exc}") from exc
    if (
        not stat.S_ISLNK(final.st_mode)
        or final.st_dev != initial.st_dev
        or final.st_ino != initial.st_ino
        or final.st_mtime_ns != initial.st_mtime_ns
        or final.st_ctime_ns != initial.st_ctime_ns
    ):
        raise CandidateError(f"source symlink changed while hashing: {name}")
    try:
        target_text = target_bytes.decode("utf-8")
        resolved_target = (path.parent / target_text).resolve(strict=False)
    except (UnicodeDecodeError, OSError, RuntimeError) as exc:
        raise CandidateError(f"source symlink target is ambiguous: {name}: {exc}") from exc
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(target_bytes)}\0".encode("ascii"))
    digest.update(target_bytes)
    if not resolved_target.is_relative_to(repo_root):
        return digest.hexdigest(), f"unsafe symlink target: {name} -> {target_text}"
    return digest.hexdigest(), None


def _workspace_entry_issue(
    repo_root: Path,
    objects_dir: Path,
    expected_sha: str,
    name: str,
    path: Path,
    metadata: os.stat_result,
    contract: tuple[str, str] | None,
) -> str | None:
    if contract is None:
        return f"untracked or ignored entry: {name}"
    expected_mode, expected_blob = contract
    if stat.S_ISREG(metadata.st_mode):
        actual_mode = "100755" if metadata.st_mode & 0o111 else "100644"
        if expected_mode not in {"100644", "100755"} or actual_mode != expected_mode:
            return f"mode or type drift: {name} expected {expected_mode}, found {actual_mode}"
        actual_blob, actual_sha256 = _blob_hashes(path, metadata)
    elif stat.S_ISLNK(metadata.st_mode):
        if expected_mode != "120000":
            return f"mode or type drift: {name} expected {expected_mode}, found symlink"
        actual_blob, issue = _symlink_blob(repo_root, path, name, metadata)
        if issue:
            return issue
        actual_sha256 = None
    else:
        return f"unsupported workspace entry type: {name}"
    if actual_blob != expected_blob:
        if actual_sha256 is not None:
            lfs_contract = _lfs_contract(objects_dir, expected_sha, name, expected_blob)
            if lfs_contract == (actual_sha256, metadata.st_size):
                return None
        return f"content or symlink-target drift: {name}"
    return None


def _workspace_issues(
    repo_root: Path,
    objects_dir: Path,
    expected_sha: str,
    expected: dict[str, tuple[str, str]],
) -> list[str]:
    expected_dirs = _expected_directories(expected)
    entries, observed_dirs, issues = _enumerate_workspace(repo_root, expected_dirs)
    observed_files = {name for name, _path, _metadata in entries}
    issues.extend(
        issue
        for name, path, metadata in entries
        if (
            issue := _workspace_entry_issue(
                repo_root,
                objects_dir,
                expected_sha,
                name,
                path,
                metadata,
                expected.get(name),
            )
        )
    )
    for name in sorted(expected.keys() - observed_files):
        issues.append(f"missing tracked entry: {name}")
    for name in sorted(expected_dirs - observed_dirs):
        issues.append(f"missing tracked directory: {name}/")
    return sorted(set(issues))


def _validate_source(repo_root: Path, expected_sha: str) -> None:
    if not SHA_PATTERN.fullmatch(expected_sha):
        raise CandidateError("source SHA must be one exact lowercase 40-hex commit identity")
    if not repo_root.is_dir() or repo_root.is_symlink():
        raise CandidateError(f"source repository is not a real directory: {repo_root}")
    git_dir, common_dir, objects_dir = _git_storage(repo_root)
    head = _source_head(git_dir, common_dir)
    if head != expected_sha:
        raise CandidateError(f"source SHA drift: expected {expected_sha}, found {head}")
    issues = _workspace_issues(
        repo_root,
        objects_dir,
        expected_sha,
        _carrier_tree(objects_dir, expected_sha),
    )
    if issues:
        raise CandidateError("source repository is dirty or ambiguous:\n" + "\n".join(issues))


def _require_external(path: Path, *, repo_root: Path, label: str) -> None:
    resolved = path.resolve(strict=False)
    root = repo_root.resolve()
    if resolved == root or resolved.is_relative_to(root):
        raise CandidateError(f"{label} must be outside the source repository: {path}")


def _run_staging_git(
    *arguments: str,
    cwd: Path,
) -> subprocess.CompletedProcess[bytes]:
    if not SYSTEM_GIT.is_file() or SYSTEM_GIT.is_symlink() or not os.access(SYSTEM_GIT, os.X_OK):
        raise CandidateError(f"trusted system Git is unavailable: {SYSTEM_GIT}")
    try:
        return subprocess.run(
            [str(SYSTEM_GIT), *arguments],
            check=False,
            capture_output=True,
            cwd=cwd,
            env=_trusted_git_environment(),
        )
    except OSError as exc:
        raise CandidateError(f"cannot execute trusted staging Git: {exc}") from exc


def _require_staging_git_success(
    result: subprocess.CompletedProcess[bytes],
    *,
    operation: str,
) -> None:
    if result.returncode == 0:
        return
    detail = result.stderr.decode("utf-8", errors="replace").strip()
    raise CandidateError(
        f"trusted staging Git cannot {operation}: {detail or f'exit {result.returncode}'}"
    )


def _stage_build_source(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    _validate_source(repo_root, args.source_sha)

    build_root = Path(os.path.abspath(args.build_root))
    resolved_build_root = args.build_root.resolve(strict=False)
    if build_root != resolved_build_root:
        raise CandidateError(
            f"disposable build-root path cannot traverse a symlink: {args.build_root}"
        )
    _require_external(build_root, repo_root=repo_root, label="disposable build root")
    if build_root.is_symlink() or build_root.exists():
        raise CandidateError(f"disposable build root must not already exist: {build_root}")

    try:
        build_root.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CandidateError(
            f"cannot create disposable build-root parent: {build_root.parent}: {exc}"
        ) from exc
    if build_root.parent.is_symlink() or build_root.parent.resolve() != build_root.parent:
        raise CandidateError(
            f"disposable build-root parent is symlinked or ambiguous: {build_root.parent}"
        )

    try:
        with tempfile.TemporaryDirectory(
            prefix=".robot-sf-build-source-",
            dir=build_root.parent,
        ) as staging_parent_text:
            staging_parent = Path(staging_parent_text)
            staging_root = staging_parent / "source"
            empty_template = staging_parent / "empty-template"
            empty_template.mkdir()
            clone = _run_staging_git(
                "-c",
                "core.hooksPath=/dev/null",
                "clone",
                "--local",
                "--shared",
                "--no-checkout",
                "--template",
                str(empty_template),
                "--",
                str(repo_root),
                str(staging_root),
                cwd=build_root.parent,
            )
            _require_staging_git_success(clone, operation="clone the frozen object database")
            checkout = _run_staging_git(
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(staging_root),
                "checkout",
                "--detach",
                "--force",
                args.source_sha,
                "--",
                cwd=build_root.parent,
            )
            _require_staging_git_success(checkout, operation="materialize the exact source commit")
            hooks = _run_staging_git(
                "-C",
                str(staging_root),
                "config",
                "--local",
                "--replace-all",
                "core.hooksPath",
                "/dev/null",
                cwd=build_root.parent,
            )
            _require_staging_git_success(hooks, operation="disable disposable-repository hooks")
            _validate_source(staging_root, args.source_sha)
            _validate_source(repo_root, args.source_sha)
            os.rename(staging_root, build_root)
    except CandidateError:
        raise
    except OSError as exc:
        raise CandidateError(
            f"cannot materialize disposable build root {build_root}: {exc}"
        ) from exc

    _validate_source(build_root, args.source_sha)
    _validate_source(repo_root, args.source_sha)
    print(f"PASS: staged disposable build source at {args.source_sha}: {build_root}")


def _materialization_path(value: Any, *, label: str) -> str:
    """Validate one candidate-relative path without permitting traversal."""
    if not isinstance(value, str) or not value:
        raise CandidateError(f"{label} must be a non-empty repository-relative path")
    if (
        value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).as_posix() != value
    ):
        raise CandidateError(f"{label} is unsafe or ambiguous: {value!r}")
    parts = PurePosixPath(value).parts
    if any(part in {"", ".", ".."} for part in parts) or parts[0] == ".git":
        raise CandidateError(f"{label} is unsafe or ambiguous: {value!r}")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise CandidateError(f"{label} contains a control character: {value!r}")
    return value


def _materialization_pattern(value: Any, *, label: str) -> str:
    """Validate one simple POSIX glob used by the candidate policy."""
    pattern = _materialization_path(value, label=label)
    if "[" in pattern or "]" in pattern:
        raise CandidateError(f"{label} uses unsupported character-class syntax: {pattern!r}")
    return pattern


def _materialization_glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile the repository's small, cross-platform candidate glob dialect."""
    chunks: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            chunks.append("(?:.*/)?")
            index += 3
            continue
        if pattern.startswith("**", index):
            chunks.append(".*")
            index += 2
            continue
        character = pattern[index]
        if character == "*":
            chunks.append("[^/]*")
        elif character == "?":
            chunks.append("[^/]")
        else:
            chunks.append(re.escape(character))
        index += 1
    return re.compile("^" + "".join(chunks) + "$")


def _materialization_matches(path: str, pattern: str) -> bool:
    """Return whether one candidate path matches one policy pattern."""
    return bool(_materialization_glob_regex(pattern).match(path))


def _materialization_string_list(value: Any, *, label: str, patterns: bool) -> tuple[str, ...]:
    """Validate and normalise a non-empty policy string list."""
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise CandidateError(f"{label} must be a non-empty list of strings")
    result = tuple(
        (
            _materialization_pattern(item, label=f"{label}[{index}]")
            if patterns
            else _materialization_path(item, label=f"{label}[{index}]")
        )
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise CandidateError(f"{label} contains duplicate entries")
    return result


def _validate_materialization_asset_rule(raw_rule: Any, *, index: int) -> dict[str, Any]:
    """Validate one release-safe asset classification in the candidate policy."""
    if not isinstance(raw_rule, dict):
        raise CandidateError(f"asset_rules[{index}] must be an object")
    required_keys = {
        "id",
        "scope",
        "patterns",
        "status",
        "source",
        "source_revision_or_access_date",
        "license_or_rights",
        "attribution",
        "checksum_policy",
        "modification_status",
        "evidence",
    }
    if set(raw_rule) != required_keys:
        raise CandidateError(f"asset_rules[{index}] has missing or unclassified fields")
    rule_id = raw_rule.get("id")
    if not isinstance(rule_id, str) or not rule_id:
        raise CandidateError(f"asset_rules[{index}] has an invalid id")
    scope = raw_rule.get("scope")
    if not isinstance(scope, str) or not scope:
        raise CandidateError(f"asset_rules[{index}].scope must be non-empty text")
    patterns = _materialization_string_list(
        raw_rule.get("patterns"), label=f"asset_rules[{index}].patterns", patterns=True
    )
    status = raw_rule.get("status")
    if status not in MATERIALIZATION_ALLOWED_ASSET_STATUSES:
        raise CandidateError(
            f"asset_rules[{index}].status must be one of the release-safe statuses: "
            f"{sorted(MATERIALIZATION_ALLOWED_ASSET_STATUSES)}"
        )
    text_fields = (
        "source",
        "source_revision_or_access_date",
        "license_or_rights",
        "attribution",
        "checksum_policy",
        "modification_status",
    )
    if any(
        not isinstance(raw_rule.get(field), str) or not raw_rule[field].strip()
        for field in text_fields
    ):
        raise CandidateError(f"asset_rules[{index}] has empty rights metadata")
    evidence = _materialization_string_list(
        raw_rule.get("evidence"), label=f"asset_rules[{index}].evidence", patterns=False
    )
    return {
        "id": rule_id,
        "scope": scope,
        "patterns": list(patterns),
        "status": status,
        **{field: raw_rule[field] for field in text_fields},
        "evidence": list(evidence),
    }


def _validate_materialization_policy(payload: Any) -> dict[str, Any]:
    """Validate the closed, deterministic source-selection policy."""
    if not isinstance(payload, dict):
        raise CandidateError("materialization policy must be a JSON object")
    expected_keys = {
        "schema_version",
        "package",
        "source_inventory_path",
        "candidate_inventory_path",
        "metadata_path",
        "include",
        "exclude",
        "required",
        "asset_rules",
    }
    if set(payload) != expected_keys:
        raise CandidateError("materialization policy has missing or unclassified fields")
    if payload.get("schema_version") != MATERIALIZATION_POLICY_SCHEMA_VERSION:
        raise CandidateError("materialization policy schema_version is invalid")
    package = payload.get("package")
    if not isinstance(package, dict) or set(package) != {"name", "version"}:
        raise CandidateError("materialization policy package identity is invalid")
    if package.get("name") != "robot_sf" or not isinstance(package.get("version"), str):
        raise CandidateError("materialization policy must identify robot_sf")
    if not VERSION_PATTERN.fullmatch(package["version"]):
        raise CandidateError("materialization policy package version is invalid")

    source_inventory_path = _materialization_path(
        payload.get("source_inventory_path"), label="source_inventory_path"
    )
    candidate_inventory_path = _materialization_path(
        payload.get("candidate_inventory_path"), label="candidate_inventory_path"
    )
    metadata_path = _materialization_path(payload.get("metadata_path"), label="metadata_path")
    if len({source_inventory_path, candidate_inventory_path, metadata_path}) != 3:
        raise CandidateError("materialization policy generated paths must be distinct")

    include = _materialization_string_list(payload.get("include"), label="include", patterns=True)
    exclude = _materialization_string_list(payload.get("exclude"), label="exclude", patterns=True)
    required = _materialization_string_list(
        payload.get("required"), label="required", patterns=True
    )

    asset_rules_raw = payload.get("asset_rules")
    if not isinstance(asset_rules_raw, list):
        raise CandidateError("materialization policy asset_rules must be a list")
    asset_rules = [
        _validate_materialization_asset_rule(raw_rule, index=index)
        for index, raw_rule in enumerate(asset_rules_raw)
    ]
    rule_ids = [rule["id"] for rule in asset_rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise CandidateError("materialization policy asset rule IDs must be unique")

    return {
        "schema_version": MATERIALIZATION_POLICY_SCHEMA_VERSION,
        "package": {"name": "robot_sf", "version": package["version"]},
        "source_inventory_path": source_inventory_path,
        "candidate_inventory_path": candidate_inventory_path,
        "metadata_path": metadata_path,
        "include": list(include),
        "exclude": list(exclude),
        "required": list(required),
        "asset_rules": asset_rules,
    }


def _run_candidate_git(
    *arguments: str,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run the absolute system Git with configuration and replacement objects disabled."""
    if not SYSTEM_GIT.is_file() or SYSTEM_GIT.is_symlink() or not os.access(SYSTEM_GIT, os.X_OK):
        raise CandidateError(f"trusted system Git is unavailable: {SYSTEM_GIT}")
    try:
        return subprocess.run(
            [str(SYSTEM_GIT), "--no-replace-objects", *arguments],
            check=False,
            capture_output=True,
            cwd=cwd,
            env=env or _trusted_git_environment(),
        )
    except OSError as exc:
        raise CandidateError(f"cannot execute trusted candidate Git: {exc}") from exc


def _candidate_git_output(result: subprocess.CompletedProcess[bytes], *, operation: str) -> bytes:
    """Return successful Git output or a useful fail-closed error."""
    if result.returncode == 0:
        return result.stdout
    detail = result.stderr.decode("utf-8", errors="replace").strip()
    raise CandidateError(f"trusted candidate Git cannot {operation}: {detail or result.returncode}")


def _candidate_source_tree(
    repo_root: Path,
    source_sha: str,
) -> dict[str, tuple[str, str, str]]:
    """Return exact source-tree entries after proving a clean reviewed checkout."""
    if not SHA_PATTERN.fullmatch(source_sha):
        raise CandidateError("source SHA must be one exact lowercase 40-hex commit identity")
    if not repo_root.is_dir() or repo_root.is_symlink():
        raise CandidateError(f"source repository is not a real directory: {repo_root}")
    head = (
        _candidate_git_output(
            _run_candidate_git("rev-parse", "--verify", "HEAD^{commit}", cwd=repo_root),
            operation="resolve source HEAD",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if head != source_sha:
        raise CandidateError(f"source SHA drift: expected {source_sha}, found {head}")
    status = _candidate_git_output(
        _run_candidate_git("status", "--porcelain=v1", "--untracked-files=no", cwd=repo_root),
        operation="check source cleanliness",
    )
    if status:
        raise CandidateError(
            "source checkout has tracked changes; materialization requires the reviewed clean tree"
        )
    raw_tree = _candidate_git_output(
        _run_candidate_git("ls-tree", "-r", "-z", "--full-tree", source_sha, cwd=repo_root),
        operation="read source tree",
    )
    entries: dict[str, tuple[str, str, str]] = {}
    for raw_entry in raw_tree.split(b"\x00"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", maxsplit=1)
            mode_bytes, object_type_bytes, object_id_bytes = header.split(b" ", maxsplit=2)
            path = raw_path.decode("utf-8")
            mode = mode_bytes.decode("ascii")
            object_type = object_type_bytes.decode("ascii")
            object_id = object_id_bytes.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise CandidateError("source tree contains a malformed Git entry") from exc
        _materialization_path(path, label="source tree path")
        if path in entries:
            raise CandidateError(f"source tree contains duplicate path: {path}")
        if not SHA_PATTERN.fullmatch(object_id):
            raise CandidateError(f"source tree object identity is invalid: {path}")
        entries[path] = (mode, object_type, object_id)
    return entries


def _candidate_blob(repo_root: Path, object_id: str) -> bytes:
    """Read one reviewed Git blob by object identity."""
    result = _run_candidate_git("cat-file", "blob", object_id, cwd=repo_root)
    return _candidate_git_output(result, operation=f"read source blob {object_id}")


def _sanitized_candidate_source_bytes(  # noqa: C901, PLR0912 - closed metadata transform
    path: str, content: bytes
) -> bytes:
    """Remove the non-release RLlib extra from the candidate packaging metadata.

    The source checkout intentionally retains ``rllib`` for normal development.
    A rights-clean candidate is a separate, deterministic Git materialization,
    so its copied ``pyproject.toml`` may remove that one optional dependency
    before the candidate commit is built.  This keeps the development checkout
    untouched while making the wheel and sdist metadata match the admitted
    twelve-extra plus ``all`` surface.
    """
    if path != CANDIDATE_PYPROJECT_PATH:
        return content
    try:
        source_text = content.decode("utf-8")
        source_document = tomllib.loads(source_text)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CandidateError("candidate pyproject.toml is not valid UTF-8 TOML") from exc

    project = source_document.get("project")
    optional = project.get("optional-dependencies") if isinstance(project, dict) else None
    if not isinstance(optional, dict):
        raise CandidateError("candidate pyproject.toml has no optional-dependencies table")
    expected_source_extras = {
        *SUPPORTED_CANDIDATE_DISTRIBUTION_EXTRA_IDS,
        "rllib",
    }
    if set(optional) != expected_source_extras:
        raise CandidateError(
            "candidate pyproject.toml optional extras do not match the reviewed development "
            f"surface: expected {sorted(expected_source_extras)}, found {sorted(optional)}"
        )
    if "rllib" not in optional:
        raise CandidateError("candidate pyproject.toml has no standalone rllib extra to exclude")

    lines = source_text.splitlines(keepends=True)
    optional_section = None
    for index, line in enumerate(lines):
        if line.strip() == "[project.optional-dependencies]":
            optional_section = index
            break
    if optional_section is None:
        raise CandidateError("candidate pyproject.toml optional-dependencies section is missing")

    rllib_start = None
    section_end = len(lines)
    for index in range(optional_section + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section_end = index
            break
        if re.fullmatch(r"rllib\s*=\s*\[\s*", stripped):
            rllib_start = index
            break
    if rllib_start is None:
        raise CandidateError(
            "candidate pyproject.toml rllib extra is not a deterministic multiline array"
        )
    rllib_end = None
    for index in range(rllib_start + 1, section_end):
        if lines[index].strip() == "]":
            rllib_end = index
            break
    if rllib_end is None:
        raise CandidateError("candidate pyproject.toml rllib extra has no closing array")

    sanitized_text = "".join(lines[:rllib_start] + lines[rllib_end + 1 :])
    try:
        sanitized_document = tomllib.loads(sanitized_text)
    except tomllib.TOMLDecodeError as exc:
        raise CandidateError("sanitized candidate pyproject.toml is not valid TOML") from exc
    sanitized_project = sanitized_document.get("project")
    sanitized_optional = (
        sanitized_project.get("optional-dependencies")
        if isinstance(sanitized_project, dict)
        else None
    )
    if not isinstance(sanitized_optional, dict) or set(sanitized_optional) != set(
        SUPPORTED_CANDIDATE_DISTRIBUTION_EXTRA_IDS
    ):
        raise CandidateError(
            "sanitized candidate pyproject.toml does not have the exact supported extra roster"
        )
    return sanitized_text.encode("utf-8")


def _materialization_is_asset_like(path: str) -> bool:
    """Use the inventory's conservative asset heuristic for selected source paths."""
    suffix = PurePosixPath(path).suffix.lower()
    return suffix in MATERIALIZATION_ASSET_SUFFIXES or (
        suffix not in MATERIALIZATION_NON_ASSET_SUFFIXES
        and bool(MATERIALIZATION_ASSET_PATH_HINTS.intersection(PurePosixPath(path).parts))
    )


def _candidate_inventory_payload(
    policy: dict[str, Any],
    selected_paths: tuple[str, ...],
    *,
    source_sha: str,
    policy_sha256: str,
    source_inventory_sha256: str,
) -> dict[str, Any]:
    """Build the candidate-local inventory that strict archive/tree checks can consume."""
    selected = set(selected_paths)
    used_rules = [
        rule
        for rule in policy["asset_rules"]
        if any(
            path in selected
            and any(_materialization_matches(path, pattern) for pattern in rule["patterns"])
            for path in selected_paths
        )
    ]
    scopes: list[dict[str, Any]] = []
    for scope in dict.fromkeys(rule["scope"] for rule in used_rules):
        scope_patterns = list(
            dict.fromkeys(
                pattern
                for rule in used_rules
                if rule["scope"] == scope
                for pattern in rule["patterns"]
            )
        )
        scopes.append(
            {
                "id": scope,
                "globs": scope_patterns,
                "release_relevant": True,
            }
        )
    rows: list[dict[str, Any]] = []
    for rule in used_rules:
        rows.append(
            {
                "id": rule["id"],
                "scope": rule["scope"],
                "globs": rule["patterns"],
                "status": rule["status"],
                "source": rule["source"],
                "source_revision_or_access_date": rule["source_revision_or_access_date"],
                "license_or_rights": rule["license_or_rights"],
                "attribution": rule["attribution"],
                "checksum_policy": rule["checksum_policy"],
                "modification_status": rule["modification_status"],
                "evidence": rule["evidence"],
            }
        )
    return {
        "schema_version": MATERIALIZATION_INVENTORY_SCHEMA_VERSION,
        "claim_boundary": (
            "Candidate-local release inventory generated only from the reviewed materialization "
            "policy; it does not clear excluded source assets or model weights."
        ),
        "source_sha": source_sha,
        "source_inventory_sha256": source_inventory_sha256,
        "materialization_policy_sha256": policy_sha256,
        "tracked_scopes": scopes,
        "rows": rows,
    }


def _write_candidate_file(
    root: Path, relative_path: str, content: bytes, *, mode: str = "100644"
) -> None:
    """Write one generated candidate file with a regular-file mode."""
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
        os.chmod(destination, 0o755 if mode == "100755" else 0o644)
    except FileExistsError as exc:
        raise CandidateError(f"candidate generated path already exists: {relative_path}") from exc
    except OSError as exc:
        raise CandidateError(f"cannot write candidate member {relative_path}: {exc}") from exc


def _materialization_inputs(
    args: argparse.Namespace,
    source_root: Path,
    source_entries: dict[str, tuple[str, str, str]],
) -> tuple[dict[str, Any], str, str, bytes, bytes, str, str]:
    """Load policy and inventory bytes from the exact reviewed source tree."""
    policy_input = args.policy
    if policy_input is None:
        policy_input = source_root / DEFAULT_MATERIALIZATION_POLICY_RELATIVE
    elif not policy_input.is_absolute():
        policy_input = source_root / policy_input
    if policy_input.is_symlink():
        raise CandidateError(
            f"materialization policy must be a regular file in source: {policy_input}"
        )
    policy_path = policy_input.resolve(strict=False)
    if not policy_path.is_relative_to(source_root) or not policy_path.is_file():
        raise CandidateError(
            f"materialization policy must be a regular file in source: {policy_path}"
        )
    policy_rel = policy_path.relative_to(source_root).as_posix()
    policy_raw = policy_path.read_bytes()
    policy = _validate_materialization_policy(
        _load_json(policy_path, label="materialization policy")
    )
    policy_entry = source_entries.get(policy_rel)
    if (
        policy_entry is None
        or policy_entry[0] not in {"100644", "100755"}
        or policy_entry[1] != "blob"
    ):
        raise CandidateError(f"materialization policy is not a regular tracked file: {policy_rel}")
    if _candidate_blob(source_root, policy_entry[2]) != policy_raw:
        raise CandidateError(
            f"materialization policy changed outside the reviewed source tree: {policy_rel}"
        )

    source_inventory_rel = policy["source_inventory_path"]
    inventory_path = source_root / source_inventory_rel
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise CandidateError(
            f"source rights inventory is not a regular file: {source_inventory_rel}"
        )
    inventory_entry = source_entries.get(source_inventory_rel)
    if (
        inventory_entry is None
        or inventory_entry[0] not in {"100644", "100755"}
        or inventory_entry[1] != "blob"
    ):
        raise CandidateError(
            f"source rights inventory is not a regular tracked file: {source_inventory_rel}"
        )
    inventory_raw = _candidate_blob(source_root, inventory_entry[2])
    if inventory_raw != inventory_path.read_bytes():
        raise CandidateError(
            f"source rights inventory changed outside the reviewed source tree: {source_inventory_rel}"
        )
    return (
        policy,
        policy_rel,
        source_inventory_rel,
        policy_raw,
        inventory_raw,
        hashlib.sha256(policy_raw).hexdigest(),
        hashlib.sha256(inventory_raw).hexdigest(),
    )


def _select_materialization_entries(
    source_entries: dict[str, tuple[str, str, str]],
    policy: dict[str, Any],
    *,
    policy_rel: str,
) -> tuple[dict[str, tuple[str, str, str]], list[str], list[str]]:
    """Select regular source members and report policy/non-regular exclusions."""
    metadata_rel = policy["metadata_path"]
    candidate_inventory_rel = policy["candidate_inventory_path"]
    if metadata_rel in source_entries or candidate_inventory_rel in source_entries:
        raise CandidateError("generated candidate path already exists in the reviewed source tree")
    selected: dict[str, tuple[str, str, str]] = {}
    excluded_paths: list[str] = []
    excluded_non_regular: list[str] = []
    for path, entry in sorted(source_entries.items()):
        if not any(_materialization_matches(path, pattern) for pattern in policy["include"]):
            continue
        if entry[0] not in {"100644", "100755"} or entry[1] != "blob":
            excluded_non_regular.append(path)
        elif any(_materialization_matches(path, pattern) for pattern in policy["exclude"]):
            excluded_paths.append(path)
        else:
            selected[path] = entry
    selected_paths = tuple(sorted(selected))
    for required in policy["required"]:
        if not any(_materialization_matches(path, required) for path in selected_paths):
            raise CandidateError(f"materialization policy requirement is not selected: {required}")
    if not selected_paths:
        raise CandidateError("materialization policy selected no regular source members")
    _validate_materialization_assets(selected_paths, selected, policy, policy_rel=policy_rel)
    return selected, sorted(excluded_paths), sorted(excluded_non_regular)


def _validate_materialization_assets(
    selected_paths: tuple[str, ...],
    selected: dict[str, tuple[str, str, str]],
    policy: dict[str, Any],
    *,
    policy_rel: str,
) -> None:
    """Require every selected asset-like path to have one release-safe rule."""
    del policy_rel  # The parameter keeps this check's error context tied to the policy caller.
    for path in selected_paths:
        if "model" in PurePosixPath(path).parts:
            raise CandidateError(f"model member selected by materialization policy: {path}")
        if not _materialization_is_asset_like(path):
            continue
        matches = [
            rule
            for rule in policy["asset_rules"]
            if any(_materialization_matches(path, pattern) for pattern in rule["patterns"])
        ]
        if len(matches) != 1:
            raise CandidateError(
                f"asset member is not covered by exactly one release-safe asset rule: {path}"
            )
        rule = matches[0]
        if rule["status"] not in MATERIALIZATION_ALLOWED_ASSET_STATUSES:
            raise CandidateError(f"asset member has a non-release status: {path}")
        if any(evidence not in selected for evidence in rule["evidence"]):
            raise CandidateError(
                f"asset rule evidence is not present in the candidate source: {rule['id']}"
            )


def _materialization_members(
    source_root: Path,
    selected: dict[str, tuple[str, str, str]],
) -> tuple[tuple[str, ...], dict[str, bytes], list[dict[str, Any]]]:
    """Read selected source blobs and build their deterministic member records."""
    selected_paths = tuple(sorted(selected))
    member_bytes: dict[str, bytes] = {}
    members: list[dict[str, Any]] = []
    for path in selected_paths:
        mode, _object_type, object_id = selected[path]
        content = _sanitized_candidate_source_bytes(
            path,
            _candidate_blob(source_root, object_id),
        )
        member_bytes[path] = content
        members.append(
            {
                "mode": mode,
                "object_sha": object_id,
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    return selected_paths, member_bytes, members


def _materialization_metadata(  # noqa: PLR0913 - explicit provenance fields stay visible
    policy: dict[str, Any],
    *,
    source_sha: str,
    policy_rel: str,
    source_inventory_rel: str,
    policy_sha256: str,
    source_inventory_sha256: str,
    members: list[dict[str, Any]],
    excluded_paths: list[str],
    excluded_non_regular: list[str],
) -> tuple[bytes, bytes]:
    """Build generated candidate inventory and provenance envelope bytes."""
    candidate_inventory = _candidate_inventory_payload(
        policy,
        tuple(member["path"] for member in members),
        source_sha=source_sha,
        policy_sha256=policy_sha256,
        source_inventory_sha256=source_inventory_sha256,
    )
    candidate_inventory_bytes = _json_bytes(candidate_inventory)
    metadata = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "package": policy["package"],
        "source": {
            "commit_sha": source_sha,
            "inventory_path": source_inventory_rel,
            "inventory_sha256": source_inventory_sha256,
        },
        "policy": {"path": policy_rel, "sha256": policy_sha256},
        "candidate_inventory": {
            "path": policy["candidate_inventory_path"],
            "sha256": hashlib.sha256(candidate_inventory_bytes).hexdigest(),
            "size": len(candidate_inventory_bytes),
        },
        "members": members,
        "excluded_paths": excluded_paths,
        "excluded_non_regular_paths": excluded_non_regular,
        "envelope_path": policy["metadata_path"],
    }
    return candidate_inventory_bytes, _json_bytes(metadata)


def _candidate_commit(staging_root: Path, empty_template: Path) -> tuple[str, str]:
    """Create the fixed-identity candidate commit and return commit/tree SHAs."""
    init = _run_candidate_git(
        "-c",
        "core.hooksPath=/dev/null",
        "init",
        "--quiet",
        "--initial-branch=main",
        f"--template={empty_template}",
        cwd=staging_root,
    )
    _candidate_git_output(init, operation="initialize candidate repository")
    for key, value in (
        ("user.name", "Robot SF candidate"),
        ("user.email", "candidate@robot-sf.invalid"),
        ("core.hooksPath", "/dev/null"),
    ):
        _candidate_git_output(
            _run_candidate_git("config", "--local", key, value, cwd=staging_root),
            operation=f"configure candidate {key}",
        )
    candidate_env = _trusted_git_environment()
    candidate_env.update(
        {
            "GIT_AUTHOR_NAME": "Robot SF candidate",
            "GIT_AUTHOR_EMAIL": "candidate@robot-sf.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_NAME": "Robot SF candidate",
            "GIT_COMMITTER_EMAIL": "candidate@robot-sf.invalid",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        }
    )
    _candidate_git_output(
        _run_candidate_git("add", "--all", "--", ".", cwd=staging_root, env=candidate_env),
        operation="stage candidate members",
    )
    _candidate_git_output(
        _run_candidate_git(
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "Materialize Robot SF software candidate",
            cwd=staging_root,
            env=candidate_env,
        ),
        operation="commit candidate members",
    )
    commit_sha = (
        _candidate_git_output(
            _run_candidate_git("rev-parse", "--verify", "HEAD^{commit}", cwd=staging_root),
            operation="resolve candidate commit",
        )
        .decode("ascii")
        .strip()
    )
    tree_sha = (
        _candidate_git_output(
            _run_candidate_git("rev-parse", "--verify", "HEAD^{tree}", cwd=staging_root),
            operation="resolve candidate tree",
        )
        .decode("ascii")
        .strip()
    )
    return commit_sha, tree_sha


def _build_candidate_root(  # noqa: PLR0913 - explicit candidate write inputs stay visible
    candidate_root: Path,
    *,
    parent: Path,
    selected_paths: tuple[str, ...],
    selected: dict[str, tuple[str, str, str]],
    member_bytes: dict[str, bytes],
    candidate_inventory_rel: str,
    candidate_inventory_bytes: bytes,
    metadata_rel: str,
    metadata_bytes: bytes,
) -> tuple[str, str]:
    """Write and commit one candidate root atomically."""
    try:
        with tempfile.TemporaryDirectory(
            prefix=".robot-sf-materialization-", dir=parent
        ) as temp_text:
            staging_root = Path(temp_text) / "candidate"
            staging_root.mkdir()
            for path in selected_paths:
                _write_candidate_file(
                    staging_root,
                    path,
                    member_bytes[path],
                    mode=selected[path][0],
                )
            _write_candidate_file(staging_root, candidate_inventory_rel, candidate_inventory_bytes)
            _write_candidate_file(staging_root, metadata_rel, metadata_bytes)
            empty_template = Path(temp_text) / "empty-template"
            empty_template.mkdir()
            candidate_commit_sha, candidate_tree_sha = _candidate_commit(
                staging_root, empty_template
            )
            candidate_entries = _candidate_source_tree(staging_root, candidate_commit_sha)
            expected_paths = set(selected_paths) | {candidate_inventory_rel, metadata_rel}
            if set(candidate_entries) != expected_paths or any(
                mode not in {"100644", "100755"} or object_type != "blob"
                for mode, object_type, _object_id in candidate_entries.values()
            ):
                raise CandidateError("candidate commit contains unexpected or non-regular members")
            os.rename(staging_root, candidate_root)
            return candidate_commit_sha, candidate_tree_sha
    except CandidateError:
        raise
    except OSError as exc:
        raise CandidateError(f"cannot finalize candidate root {candidate_root}: {exc}") from exc


def _write_materialization_report(
    report_path: Path,
    report_payload: dict[str, Any],
    *,
    source_root: Path,
    candidate_root: Path,
) -> None:
    """Write the external candidate report without overwriting any existing path."""
    _require_external(report_path, repo_root=source_root, label="materialization report")
    if report_path.resolve(strict=False).is_relative_to(candidate_root):
        raise CandidateError(
            f"materialization report must be outside candidate root: {report_path}"
        )
    if report_path.exists() or report_path.is_symlink():
        raise CandidateError(f"materialization report must not already exist: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(_json_bytes(report_payload))


def _materialize_source(args: argparse.Namespace) -> None:
    """Materialize a deterministic, rights-scoped Git candidate and commit it once."""
    source_root = args.repo_root.resolve()
    source_entries = _candidate_source_tree(source_root, args.source_sha)
    (
        policy,
        policy_rel,
        source_inventory_rel,
        policy_raw,
        inventory_raw,
        policy_sha256,
        source_inventory_sha256,
    ) = _materialization_inputs(args, source_root, source_entries)
    del policy_raw, inventory_raw
    selected, excluded_paths, excluded_non_regular = _select_materialization_entries(
        source_entries, policy, policy_rel=policy_rel
    )
    selected_paths, member_bytes, members = _materialization_members(source_root, selected)
    candidate_inventory_bytes, metadata_bytes = _materialization_metadata(
        policy,
        source_sha=args.source_sha,
        policy_rel=policy_rel,
        source_inventory_rel=source_inventory_rel,
        policy_sha256=policy_sha256,
        source_inventory_sha256=source_inventory_sha256,
        members=members,
        excluded_paths=excluded_paths,
        excluded_non_regular=excluded_non_regular,
    )
    candidate_root = Path(os.path.abspath(args.candidate_root))
    _require_external(candidate_root, repo_root=source_root, label="candidate root")
    if candidate_root.exists() or candidate_root.is_symlink():
        raise CandidateError(f"candidate root must not already exist: {candidate_root}")
    parent = candidate_root.parent
    if parent.is_symlink() or parent.resolve() != parent:
        raise CandidateError(f"candidate root parent is symlinked or ambiguous: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or parent.resolve() != parent:
        raise CandidateError(f"candidate root parent is symlinked or ambiguous: {parent}")

    candidate_commit_sha, candidate_tree_sha = _build_candidate_root(
        candidate_root,
        parent=parent,
        selected_paths=selected_paths,
        selected=selected,
        member_bytes=member_bytes,
        candidate_inventory_rel=policy["candidate_inventory_path"],
        candidate_inventory_bytes=candidate_inventory_bytes,
        metadata_rel=policy["metadata_path"],
        metadata_bytes=metadata_bytes,
    )
    report_payload = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "package": policy["package"],
        "source_sha": args.source_sha,
        "policy_path": policy_rel,
        "policy_sha256": policy_sha256,
        "source_inventory_path": source_inventory_rel,
        "source_inventory_sha256": source_inventory_sha256,
        "candidate_inventory_path": policy["candidate_inventory_path"],
        "candidate_metadata_path": policy["metadata_path"],
        "candidate_commit_sha": candidate_commit_sha,
        "candidate_tree_sha": candidate_tree_sha,
        "members": members,
        "excluded_paths": excluded_paths,
        "excluded_non_regular_paths": excluded_non_regular,
    }
    if args.report is not None:
        _write_materialization_report(
            Path(os.path.abspath(args.report)),
            report_payload,
            source_root=source_root,
            candidate_root=candidate_root,
        )
    print(
        f"PASS: materialized Robot SF {policy['package']['version']} candidate "
        f"from {args.source_sha} at {candidate_commit_sha}"
    )


def _normalised_sbom(raw_sbom: Path, version: str) -> bytes:
    payload = _load_json(raw_sbom, label="raw SBOM")
    if not isinstance(payload, dict):
        raise CandidateError("raw SBOM must be a JSON object")
    if payload.get("bomFormat") != "CycloneDX" or payload.get("specVersion") != "1.5":
        raise CandidateError("raw SBOM must be CycloneDX 1.5")
    if payload.get("version") != 1:
        raise CandidateError("raw SBOM must have document version 1")
    if not isinstance(payload.get("components"), list) or not isinstance(
        payload.get("dependencies"), list
    ):
        raise CandidateError("raw SBOM must contain components and dependencies arrays")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CandidateError("raw SBOM must contain metadata")
    component = metadata.get("component")
    if (
        not isinstance(component, dict)
        or _normalise_distribution_name(str(component.get("name", ""))) != "robot-sf"
    ):
        raise CandidateError("raw SBOM root component must be Robot SF")

    payload.pop("serialNumber", None)
    metadata.pop("timestamp", None)
    component["bom-ref"] = f"pkg:pypi/robot-sf@{version}"
    component["name"] = "robot-sf"
    component["purl"] = f"pkg:pypi/robot-sf@{version}"
    component["type"] = "library"
    component["version"] = version
    return _json_bytes(payload)


def _member(path: Path, kind: str) -> dict[str, Any]:
    return {
        "filename": path.name,
        "kind": kind,
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _validation_payload() -> dict[str, Any]:
    return {
        "checks": [
            {"command": command, "id": identifier, "status": "passed"}
            for identifier, command in VALIDATION_COMMANDS
        ],
        "status": "passed",
    }


def _load_schema(schema_path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw_schema = schema_path.read_bytes()
        schema = json.loads(
            raw_schema.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except CandidateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(
            f"candidate manifest schema is not valid UTF-8 JSON: {schema_path}: {exc}"
        ) from exc
    if not isinstance(schema, dict):
        raise CandidateError("candidate manifest schema must be a JSON object")
    return raw_schema, schema


def _schema_properties(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise CandidateError("candidate manifest schema must use JSON Schema draft 2020-12")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise CandidateError("candidate manifest schema must be a closed object schema")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise CandidateError("candidate manifest schema is missing properties")
    version_schema = properties.get("schema_version")
    if not isinstance(version_schema, dict) or version_schema.get("const") != SCHEMA_VERSION:
        raise CandidateError("candidate manifest schema has the wrong schema_version contract")
    required = schema.get("required")
    if not isinstance(required, list) or set(required) != {
        "schema_version",
        "repository",
        "source_sha",
        "workflow",
        "package",
        "validation",
        "members",
    }:
        raise CandidateError("candidate manifest schema has the wrong required fields")
    return properties


def _validate_members_schema(properties: dict[str, Any]) -> None:
    members_schema = properties.get("members")
    if not isinstance(members_schema, dict):
        raise CandidateError("candidate manifest schema has no members contract")
    cardinality = tuple(members_schema.get(key) for key in ("type", "minItems", "maxItems"))
    if cardinality != ("array", len(MEMBER_KINDS), len(MEMBER_KINDS)):
        raise CandidateError("candidate manifest schema has the wrong members cardinality")
    member_item = members_schema.get("items")
    if not isinstance(member_item, dict):
        raise CandidateError("candidate manifest schema has no member object contract")
    item_shape = (member_item.get("type"), member_item.get("additionalProperties"))
    if item_shape != ("object", False):
        raise CandidateError("candidate manifest schema has the wrong member object contract")
    if set(member_item.get("required", ())) != {"filename", "kind", "sha256", "size"}:
        raise CandidateError("candidate manifest schema has the wrong member required fields")
    member_properties = member_item.get("properties")
    if not isinstance(member_properties, dict):
        raise CandidateError("candidate manifest schema has no member properties")
    kind_schema = member_properties.get("kind")
    if not isinstance(kind_schema, dict) or kind_schema.get("enum") != list(MEMBER_KINDS):
        raise CandidateError("candidate manifest schema has the wrong member kinds")


def _validate_materialization_schema(properties: dict[str, Any]) -> None:
    """Require the optional materialization property to retain its closed contract."""
    materialization = properties.get("materialization")
    if not isinstance(materialization, dict):
        raise CandidateError("candidate manifest schema has no materialization contract")
    if (
        materialization.get("type") != "object"
        or materialization.get("additionalProperties") is not False
    ):
        raise CandidateError(
            "candidate manifest schema has the wrong materialization object contract"
        )
    if set(materialization.get("required", ())) != set(MATERIALIZATION_PAYLOAD_FIELDS):
        raise CandidateError(
            "candidate manifest schema has the wrong materialization required fields"
        )
    materialization_properties = materialization.get("properties")
    if not isinstance(materialization_properties, dict) or set(materialization_properties) != set(
        MATERIALIZATION_PAYLOAD_FIELDS
    ):
        raise CandidateError("candidate manifest schema has the wrong materialization properties")


def _validate_schema_file(schema_path: Path) -> None:
    raw_schema, schema = _load_schema(schema_path)
    properties = _schema_properties(schema)
    schema_sha256 = hashlib.sha256(raw_schema).hexdigest()
    if schema_sha256 != SCHEMA_SHA256:
        raise CandidateError(
            "candidate manifest schema has unreviewed contract drift: "
            f"expected sha256 {SCHEMA_SHA256}, found {schema_sha256}"
        )
    if schema.get("$id") != SCHEMA_ID:
        raise CandidateError("candidate manifest schema has the wrong stable schema ID")
    _validate_members_schema(properties)
    _validate_materialization_schema(properties)


def _validate_workflow_identity(workflow: Any) -> None:
    if not isinstance(workflow, dict) or set(workflow) != {"run_id", "run_attempt"}:
        raise CandidateError("candidate manifest workflow identity is invalid")
    run_id = workflow["run_id"]
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise CandidateError("candidate manifest workflow run_id is invalid")
    attempt = workflow["run_attempt"]
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise CandidateError("candidate manifest workflow run_attempt is invalid")


def _validate_package_identity(package: Any) -> str:
    if not isinstance(package, dict) or set(package) != {"name", "version"}:
        raise CandidateError("candidate manifest package identity is invalid")
    version = package["version"]
    if package["name"] != "robot_sf" or not isinstance(version, str):
        raise CandidateError("candidate manifest package must identify robot_sf")
    if not VERSION_PATTERN.fullmatch(version):
        raise CandidateError("candidate manifest package version is invalid")
    return version


def _validate_materialization_payload(payload: Any) -> dict[str, Any]:
    """Validate the materialized-source identity carried by a bundle envelope."""
    if not isinstance(payload, dict) or set(payload) != set(MATERIALIZATION_PAYLOAD_FIELDS):
        raise CandidateError("candidate materialization identity is missing or unclassified")
    for field in ("candidate_commit_sha", "candidate_tree_sha"):
        value = payload[field]
        if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
            raise CandidateError(f"candidate materialization {field} is invalid")
    for field in ("policy_sha256", "source_inventory_sha256"):
        value = payload[field]
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise CandidateError(f"candidate materialization {field} is invalid")
    paths = []
    for field in (
        "policy_path",
        "source_inventory_path",
        "candidate_inventory_path",
        "candidate_metadata_path",
    ):
        paths.append(_materialization_path(payload[field], label=f"materialization.{field}"))
    if len(set(paths)) != len(paths):
        raise CandidateError("candidate materialization paths must be distinct")
    return payload


def _validate_materialization_report(payload: Any) -> dict[str, Any]:
    """Validate the external report shape before binding it to the source tree."""
    if not isinstance(payload, dict) or set(payload) != MATERIALIZATION_REPORT_FIELDS:
        raise CandidateError("materialization report has missing or unclassified fields")
    if payload.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        raise CandidateError("materialization report schema_version is invalid")
    _validate_package_identity(payload.get("package"))
    source_sha = payload.get("source_sha")
    if not isinstance(source_sha, str) or not SHA_PATTERN.fullmatch(source_sha):
        raise CandidateError("materialization report source_sha is invalid")
    identity = {field: payload[field] for field in MATERIALIZATION_PAYLOAD_FIELDS}
    _validate_materialization_payload(identity)
    if not isinstance(payload.get("members"), list):
        raise CandidateError("materialization report members must be a list")
    for field in ("excluded_paths", "excluded_non_regular_paths"):
        values = payload.get(field)
        if not isinstance(values, list) or any(
            _materialization_path(value, label=f"materialization report {field}") != value
            for value in values
        ):
            raise CandidateError(f"materialization report {field} is invalid")
    return payload


def _load_rights_policy(  # noqa: C901, PLR0912 - closed policy contract
    policy_path: Path, *, repo_root: Path
) -> tuple[bytes, dict[str, Any]]:
    """Load the tracked rights-admission policy with a closed, stable identity."""
    try:
        policy_rel = policy_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise CandidateError(
            f"rights admission policy is outside the source repository: {policy_path}"
        ) from exc
    if policy_rel != RIGHTS_POLICY_PATH:
        raise CandidateError(f"rights admission policy must be {RIGHTS_POLICY_PATH}")
    if policy_path.is_symlink() or not policy_path.is_file():
        raise CandidateError(f"rights admission policy is not a regular file: {policy_path}")
    raw = policy_path.read_bytes()
    payload = _load_json(policy_path, label="software release rights policy")
    expected_keys = {
        "$schema",
        "schema_version",
        "policy_id",
        "candidate_schema_version",
        "claim_boundary",
        "supported_dependency_surface",
        "source_selection",
        "exclusion_reasons",
        "external_only",
        "materialization",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise CandidateError("software release rights policy has missing or unclassified fields")
    if (
        payload["$schema"]
        != "https://robot-sf.dev/schema/software_release_rights_policy.v1.schema.json"
        or payload["schema_version"] != RIGHTS_POLICY_ID
        or payload["policy_id"] != RIGHTS_POLICY_ID
        or payload["candidate_schema_version"] != SANITIZED_CANDIDATE_SCHEMA
    ):
        raise CandidateError("software release rights policy identity is unsupported")
    if not isinstance(payload["claim_boundary"], str) or not payload["claim_boundary"].strip():
        raise CandidateError("software release rights policy claim boundary is empty")
    supported_surface = payload["supported_dependency_surface"]
    if not isinstance(supported_surface, dict) or set(supported_surface) != {
        "profile_manifest_path",
        "profile_ids",
        "extra_ids",
        "excluded_extra_ids",
    }:
        raise CandidateError(
            "software release rights policy supported dependency surface is malformed"
        )
    if (
        supported_surface["profile_manifest_path"] != SUPPORTED_DEPENDENCY_PROFILE_PATH
        or supported_surface["profile_ids"] != list(SUPPORTED_DEPENDENCY_PROFILE_IDS)
        or supported_surface["extra_ids"] != list(SUPPORTED_DEPENDENCY_EXTRA_IDS)
        or supported_surface["excluded_extra_ids"] != ["rllib"]
    ):
        raise CandidateError(
            "software release rights policy supported dependency roster is invalid"
        )
    selection = payload["source_selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "allow_globs",
        "exclude_globs",
        "required_paths",
    }:
        raise CandidateError("software release rights policy source selection is malformed")
    for field in ("allow_globs", "exclude_globs", "required_paths"):
        values = selection[field]
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise CandidateError(f"software release rights policy {field} is malformed")
        for index, value in enumerate(values):
            _materialization_pattern(value, label=f"rights policy {field}[{index}]")
        if len(values) != len(set(values)):
            raise CandidateError(f"software release rights policy {field} is not unique")
    if not selection["allow_globs"] or not selection["required_paths"]:
        raise CandidateError("software release rights policy source selection is empty")
    reasons = payload["exclusion_reasons"]
    if not isinstance(reasons, list):
        raise CandidateError("software release rights policy exclusion reasons are malformed")
    for index, entry in enumerate(reasons):
        if not isinstance(entry, dict) or set(entry) != {"glob", "reason"}:
            raise CandidateError(
                f"software release rights policy exclusion reason {index} is malformed"
            )
        _materialization_pattern(entry["glob"], label=f"rights policy exclusion reason {index}")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise CandidateError(
                f"software release rights policy exclusion reason {index} is empty"
            )
    external_only = payload["external_only"]
    if not isinstance(external_only, list) or not external_only:
        raise CandidateError("software release rights policy external-only boundary is empty")
    for index, entry in enumerate(external_only):
        if not isinstance(entry, dict) or set(entry) != {"glob", "name", "reason"}:
            raise CandidateError(
                f"software release rights policy external-only entry {index} is malformed"
            )
        _materialization_pattern(entry["glob"], label=f"rights policy external-only {index}")
        if any(
            not isinstance(entry[key], str) or not entry[key].strip() for key in ("name", "reason")
        ):
            raise CandidateError(
                f"software release rights policy external-only entry {index} is invalid"
            )
    materialization = payload["materialization"]
    if not isinstance(materialization, dict) or set(materialization) != {
        "commit_parent",
        "commit_message",
        "commit_timestamp",
        "tree_digest",
    }:
        raise CandidateError("software release rights policy materialization contract is malformed")
    if (
        materialization["commit_parent"] != "root_commit"
        or materialization["commit_message"] != ("Materialize Robot SF software candidate")
        or materialization["commit_timestamp"] != "2000-01-01T00:00:00Z"
    ):
        raise CandidateError("software release rights policy materialization identity is invalid")
    if (
        not isinstance(materialization["tree_digest"], str)
        or not materialization["tree_digest"].strip()
    ):
        raise CandidateError("software release rights policy tree digest description is empty")
    return raw, payload


def _rights_exclusion_reason(path: str, policy: dict[str, Any]) -> str:
    """Explain an excluded path, including paths outside every named exclusion glob."""
    matches = [
        entry["reason"]
        for entry in policy["exclusion_reasons"]
        if _materialization_matches(path, entry["glob"])
    ]
    if matches:
        return matches[0]
    return "path is outside the reviewed rights-clean software allowlist"


def _candidate_tree_sha1(repo_root: Path, commit_sha: str) -> str:
    """Resolve one exact commit's Git tree identity through the trusted carrier."""
    result = _run_candidate_git("rev-parse", "--verify", f"{commit_sha}^{{tree}}", cwd=repo_root)
    return _candidate_git_output(result, operation="resolve source tree").decode("ascii").strip()


def _validate_candidate_commit_identity(candidate_root: Path, report: dict[str, Any]) -> None:
    """Require the materializer's root commit identity, not just its tree bytes."""
    commit = report["candidate_commit_sha"]
    metadata = _candidate_git_output(
        _run_candidate_git(
            "show",
            "-s",
            "--format=%H%n%P%n%T%n%an%n%ae%n%cn%n%ce%n%aI%n%cI%n%B",
            commit,
            cwd=candidate_root,
        ),
        operation="read candidate commit identity",
    ).decode("utf-8")
    lines = metadata.splitlines()
    expected = [
        commit,
        "",
        report["candidate_tree_sha"],
        "Robot SF candidate",
        "candidate@robot-sf.invalid",
        "Robot SF candidate",
        "candidate@robot-sf.invalid",
        "2000-01-01T00:00:00+00:00",
        "2000-01-01T00:00:00+00:00",
        "Materialize Robot SF software candidate",
        "",
    ]
    if lines != expected:
        raise CandidateError("materialized candidate commit metadata or parent identity drifted")


def _sanitized_manifest_from_report(  # noqa: C901 - closed manifest adapter
    report: dict[str, Any],
    *,
    source_tree_sha1: str,
    policy_raw: bytes,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Translate the materializer's checked report into the shared sanitized schema."""
    members: list[dict[str, Any]] = []
    paths: set[str] = set()
    for raw_member in report["members"]:
        if not isinstance(raw_member, dict) or set(raw_member) != {
            "mode",
            "object_sha",
            "path",
            "sha256",
            "size",
        }:
            raise CandidateError("materialization report member is malformed")
        path = raw_member["path"]
        if not isinstance(path, str) or _safe_workspace_path(path).as_posix() != path:
            raise CandidateError(f"materialization report member path is unsafe: {path!r}")
        mode = raw_member["mode"]
        if mode not in {"100644", "100755"}:
            raise CandidateError(
                f"materialization report member mode is not a regular blob: {path}"
            )
        object_sha = raw_member["object_sha"]
        if not isinstance(object_sha, str) or not SHA_PATTERN.fullmatch(object_sha):
            raise CandidateError(f"materialization report member Git blob is invalid: {path}")
        digest = raw_member["sha256"]
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise CandidateError(f"materialization report member SHA-256 is invalid: {path}")
        size = raw_member["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CandidateError(f"materialization report member size is invalid: {path}")
        if path in paths:
            raise CandidateError(f"materialization report contains duplicate member: {path}")
        if not any(
            _materialization_matches(path, glob)
            for glob in policy["source_selection"]["allow_globs"]
        ):
            raise CandidateError(f"materialization report member is outside rights policy: {path}")
        if any(
            _materialization_matches(path, glob)
            for glob in policy["source_selection"]["exclude_globs"]
        ):
            raise CandidateError(
                f"materialization report member is excluded by rights policy: {path}"
            )
        paths.add(path)
        members.append(
            {
                "git_blob_sha1": object_sha,
                "mode": mode,
                "path": path,
                "sha256": digest,
                "size": size,
            }
        )
    members.sort(key=lambda member: member["path"])
    excluded: list[dict[str, str]] = []
    excluded_paths: set[str] = set()
    for value in (*report["excluded_paths"], *report["excluded_non_regular_paths"]):
        if not isinstance(value, str) or _safe_workspace_path(value).as_posix() != value:
            raise CandidateError(f"materialization report excluded path is unsafe: {value!r}")
        if value in paths or value in excluded_paths:
            raise CandidateError(
                f"materialization report repeats selected or excluded path: {value}"
            )
        excluded_paths.add(value)
        is_non_regular = value in report["excluded_non_regular_paths"]
        excluded.append(
            {
                "kind": "non-regular" if is_non_regular else "policy-excluded",
                "path": value,
                "reason": (
                    "non-regular source member is excluded from software surfaces"
                    if is_non_regular
                    else _rights_exclusion_reason(value, policy)
                ),
            }
        )
    excluded.sort(key=lambda member: member["path"])
    payload = {
        "candidate_commit_sha": report["candidate_commit_sha"],
        "candidate_tree_sha1": report["candidate_tree_sha"],
        "excluded_members": excluded,
        "members": members,
        "policy_id": RIGHTS_POLICY_ID,
        "policy_path": RIGHTS_POLICY_PATH,
        "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
        "schema_version": SANITIZED_CANDIDATE_SCHEMA,
        "source_sha": report["source_sha"],
        "source_tree_sha1": source_tree_sha1,
        "tree_sha256": "",
    }
    tree_binding = {
        "members": members,
        "schema_version": SANITIZED_CANDIDATE_SCHEMA,
        "source_sha": report["source_sha"],
    }
    payload["tree_sha256"] = hashlib.sha256(_json_bytes(tree_binding)).hexdigest()
    _validate_sanitized_manifest(payload, policy_path=None, policy_raw=policy_raw)
    return payload


def _validate_sanitized_manifest(  # noqa: C901, PLR0912, PLR0915 - closed manifest gate
    payload: Any,
    *,
    policy_path: Path | None,
    policy_raw: bytes,
) -> dict[str, Any]:
    """Validate the generated sanitized source/member manifest and tree digest."""
    expected_keys = {
        "candidate_commit_sha",
        "candidate_tree_sha1",
        "excluded_members",
        "members",
        "policy_id",
        "policy_path",
        "policy_sha256",
        "schema_version",
        "source_sha",
        "source_tree_sha1",
        "tree_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise CandidateError("sanitized candidate manifest has missing or unclassified fields")
    if (
        payload["schema_version"] != SANITIZED_CANDIDATE_SCHEMA
        or payload["policy_id"] != RIGHTS_POLICY_ID
    ):
        raise CandidateError(
            "sanitized candidate manifest schema or policy identity is unsupported"
        )
    if payload["policy_path"] != RIGHTS_POLICY_PATH:
        raise CandidateError("sanitized candidate manifest policy path is unsupported")
    if policy_path is not None:
        try:
            actual_path = (
                policy_path.resolve().relative_to(policy_path.resolve().parents[2]).as_posix()
            )
        except ValueError:
            actual_path = payload["policy_path"]
        if actual_path != payload["policy_path"]:
            raise CandidateError(
                "sanitized candidate manifest policy path is not repository-relative"
            )
    for field in ("source_sha", "source_tree_sha1", "candidate_commit_sha", "candidate_tree_sha1"):
        if not isinstance(payload[field], str) or not SHA_PATTERN.fullmatch(payload[field]):
            raise CandidateError(f"sanitized candidate manifest {field} is invalid")
    if payload["candidate_commit_sha"] == payload["source_sha"]:
        raise CandidateError("sanitized candidate commit must not self-reference the source commit")
    if payload["policy_sha256"] != hashlib.sha256(policy_raw).hexdigest():
        raise CandidateError(
            "sanitized candidate manifest policy digest does not match policy bytes"
        )
    if not isinstance(payload["tree_sha256"], str) or not SHA256_PATTERN.fullmatch(
        payload["tree_sha256"]
    ):
        raise CandidateError("sanitized candidate manifest tree digest is invalid")
    members = payload["members"]
    if not isinstance(members, list) or not members:
        raise CandidateError("sanitized candidate manifest must contain admitted members")
    paths: list[str] = []
    for member in members:
        if not isinstance(member, dict) or set(member) != {
            "git_blob_sha1",
            "mode",
            "path",
            "sha256",
            "size",
        }:
            raise CandidateError("sanitized candidate member record is malformed")
        path = member["path"]
        if not isinstance(path, str) or _safe_workspace_path(path).as_posix() != path:
            raise CandidateError(f"sanitized candidate member path is unsafe: {path!r}")
        if member["mode"] not in {"100644", "100755"}:
            raise CandidateError(f"sanitized candidate member mode is invalid: {path}")
        if not isinstance(member["git_blob_sha1"], str) or not SHA_PATTERN.fullmatch(
            member["git_blob_sha1"]
        ):
            raise CandidateError(f"sanitized candidate member Git blob is invalid: {path}")
        if not isinstance(member["sha256"], str) or not SHA256_PATTERN.fullmatch(member["sha256"]):
            raise CandidateError(f"sanitized candidate member SHA-256 is invalid: {path}")
        if (
            isinstance(member["size"], bool)
            or not isinstance(member["size"], int)
            or member["size"] < 0
        ):
            raise CandidateError(f"sanitized candidate member size is invalid: {path}")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise CandidateError("sanitized candidate members must be sorted and unique")
    excluded = payload["excluded_members"]
    if not isinstance(excluded, list):
        raise CandidateError("sanitized candidate excluded-member manifest is malformed")
    excluded_paths: list[str] = []
    for member in excluded:
        if not isinstance(member, dict) or set(member) != {"path", "kind", "reason"}:
            raise CandidateError("sanitized candidate excluded-member record is malformed")
        path = member["path"]
        if not isinstance(path, str) or _safe_workspace_path(path).as_posix() != path:
            raise CandidateError(f"sanitized candidate excluded path is unsafe: {path!r}")
        if member["kind"] not in {"policy-excluded", "non-regular"}:
            raise CandidateError(f"sanitized candidate excluded-member kind is invalid: {path}")
        if not isinstance(member["reason"], str) or not member["reason"].strip():
            raise CandidateError(f"sanitized candidate excluded-member reason is empty: {path}")
        excluded_paths.append(path)
    if excluded_paths != sorted(excluded_paths) or len(excluded_paths) != len(set(excluded_paths)):
        raise CandidateError("sanitized candidate excluded members must be sorted and unique")
    if set(paths).intersection(excluded_paths):
        raise CandidateError("sanitized candidate member cannot also be excluded")
    tree_binding = {
        "members": members,
        "schema_version": SANITIZED_CANDIDATE_SCHEMA,
        "source_sha": payload["source_sha"],
    }
    if payload["tree_sha256"] != hashlib.sha256(_json_bytes(tree_binding)).hexdigest():
        raise CandidateError("sanitized candidate tree digest does not match its member manifest")
    return payload


def _validated_member(member: Any) -> dict[str, Any]:
    if not isinstance(member, dict) or set(member) != {"filename", "kind", "sha256", "size"}:
        raise CandidateError("candidate manifest member record is invalid")
    filename = member["filename"]
    if (
        not isinstance(filename, str)
        or not filename
        or filename != Path(filename).name
        or filename in {".", "..", MANIFEST_NAME}
    ):
        raise CandidateError("candidate manifest member filename is unsafe or reserved")
    digest = member["sha256"]
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise CandidateError(f"candidate member {filename} has an invalid SHA-256")
    size = member["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise CandidateError(f"candidate member {filename} has an invalid size")
    return member


def _validate_members(members: Any, *, version: str) -> None:
    if not isinstance(members, list) or len(members) != len(MEMBER_KINDS):
        raise CandidateError("candidate manifest must bind exactly four payload members")
    validated = [_validated_member(member) for member in members]
    if [member["kind"] for member in validated] != list(MEMBER_KINDS):
        raise CandidateError("candidate manifest member kinds or ordering are invalid")
    filenames = [member["filename"] for member in validated]
    if len(filenames) != len(set(filenames)):
        raise CandidateError("candidate manifest contains duplicate filenames")
    if validated[-1]["filename"] != PROVENANCE_NAME:
        raise CandidateError("candidate manifest provenance filename is invalid")
    if validated[2]["filename"] != f"robot_sf-{version}.cyclonedx.json":
        raise CandidateError("candidate manifest SBOM filename is invalid")


def _validate_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CandidateError("candidate manifest must be a JSON object")
    expected_keys = {
        "schema_version",
        "repository",
        "source_sha",
        "workflow",
        "package",
        "validation",
        "members",
    }
    actual_keys = set(payload)
    if actual_keys not in (expected_keys, expected_keys | {"materialization"}):
        raise CandidateError("candidate manifest has missing or unclassified top-level fields")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CandidateError("candidate manifest schema_version is invalid")
    repository = payload.get("repository")
    if not isinstance(repository, str) or not REPOSITORY_PATTERN.fullmatch(repository):
        raise CandidateError("candidate manifest repository is invalid")
    source_sha = payload.get("source_sha")
    if not isinstance(source_sha, str) or not SHA_PATTERN.fullmatch(source_sha):
        raise CandidateError("candidate manifest source_sha is invalid")
    _validate_workflow_identity(payload.get("workflow"))
    version = _validate_package_identity(payload.get("package"))
    if payload.get("validation") != _validation_payload():
        raise CandidateError("candidate manifest validation roster is missing or invalid")
    _validate_members(payload.get("members"), version=version)
    if "materialization" in payload:
        _validate_materialization_payload(payload["materialization"])
    return payload


def _provenance_payload(
    *,
    repository: str,
    source_sha: str,
    workflow: dict[str, Any],
    package: dict[str, str],
    wheel: dict[str, Any],
    sdist: dict[str, Any],
    sbom: dict[str, Any],
    materialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "build": {
            "command": "cd $BUILD_SOURCE && uv build --out-dir $DIST_DIR",
            "count": 1,
            "source_role": "disposable-exact-commit",
        },
        "package": package,
        "repository": repository,
        "sbom": sbom,
        "schema_version": PROVENANCE_VERSION,
        "source_sha": source_sha,
        "subjects": [wheel, sdist],
        "validation": _validation_payload(),
        "workflow": workflow,
    }
    if materialization is not None:
        payload["materialization"] = materialization
    return payload


def _fresh_bundle_dir(path: Path) -> None:
    if path.is_symlink():
        raise CandidateError(f"bundle directory cannot be a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise CandidateError(f"bundle path is not a directory: {path}")
        if any(path.iterdir()):
            raise CandidateError(f"bundle directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def _load_assembly_materialization_report(
    args: argparse.Namespace,
    repo_root: Path,
) -> tuple[Path, dict[str, Any]] | None:
    """Load and validate the external materialization report envelope."""
    candidate_source_root = getattr(args, "candidate_source_root", None)
    materialization_report = getattr(args, "materialization_report", None)
    if (candidate_source_root is None) != (materialization_report is None):
        raise CandidateError(
            "candidate source root and materialization report must be supplied together"
        )
    if candidate_source_root is None:
        return None
    report_path = Path(os.path.abspath(materialization_report))
    _require_external(report_path, repo_root=repo_root, label="materialization report")
    if report_path.is_symlink() or not report_path.is_file():
        raise CandidateError(f"materialization report must be a regular file: {report_path}")
    report = _validate_materialization_report(
        _load_json(report_path, label="materialization report")
    )
    if report["source_sha"] != args.source_sha:
        raise CandidateError(
            "materialization report source drift: "
            f"expected {args.source_sha}, found {report['source_sha']}"
        )
    candidate_root = Path(os.path.abspath(candidate_source_root))
    _require_external(candidate_root, repo_root=repo_root, label="materialized candidate source")
    _validate_source(candidate_root, report["candidate_commit_sha"])
    candidate_tree_sha = (
        _candidate_git_output(
            _run_candidate_git("rev-parse", "--verify", "HEAD^{tree}", cwd=candidate_root),
            operation="resolve materialized candidate tree",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if candidate_tree_sha != report["candidate_tree_sha"]:
        raise CandidateError(
            "materialization candidate tree drift: "
            f"expected {report['candidate_tree_sha']}, found {candidate_tree_sha}"
        )
    return candidate_root, report


def _recompute_materialization(
    args: argparse.Namespace,
    repo_root: Path,
    report: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, tuple[str, str, str]],
    tuple[str, ...],
    dict[str, bytes],
    bytes,
    bytes,
]:
    """Recompute source selection, generated metadata, and candidate inventory bytes."""
    source_entries = _candidate_source_tree(repo_root, args.source_sha)
    policy_input = repo_root / report["policy_path"]
    policy_args = argparse.Namespace(policy=policy_input)
    (
        policy,
        policy_rel,
        source_inventory_rel,
        _policy_raw,
        _inventory_raw,
        policy_sha256,
        source_inventory_sha256,
    ) = _materialization_inputs(policy_args, repo_root, source_entries)
    if report["package"] != policy["package"]:
        raise CandidateError("materialization report package does not match its policy")
    if (
        report["policy_path"],
        report["source_inventory_path"],
        report["policy_sha256"],
        report["source_inventory_sha256"],
    ) != (policy_rel, source_inventory_rel, policy_sha256, source_inventory_sha256):
        raise CandidateError("materialization report policy or inventory identity drifted")
    if (
        report["candidate_inventory_path"],
        report["candidate_metadata_path"],
    ) != (policy["candidate_inventory_path"], policy["metadata_path"]):
        raise CandidateError("materialization report generated-path identity drifted")

    selected, excluded_paths, excluded_non_regular = _select_materialization_entries(
        source_entries, policy, policy_rel=policy_rel
    )
    selected_paths, member_bytes, members = _materialization_members(repo_root, selected)
    if report["members"] != members:
        raise CandidateError("materialization report selected-member inventory drifted")
    if report["excluded_paths"] != excluded_paths:
        raise CandidateError("materialization report excluded-path inventory drifted")
    if report["excluded_non_regular_paths"] != excluded_non_regular:
        raise CandidateError("materialization report non-regular exclusion inventory drifted")
    candidate_inventory_bytes, metadata_bytes = _materialization_metadata(
        policy,
        source_sha=args.source_sha,
        policy_rel=policy_rel,
        source_inventory_rel=source_inventory_rel,
        policy_sha256=policy_sha256,
        source_inventory_sha256=source_inventory_sha256,
        members=members,
        excluded_paths=excluded_paths,
        excluded_non_regular=excluded_non_regular,
    )
    return policy, selected, selected_paths, member_bytes, candidate_inventory_bytes, metadata_bytes


def _validate_materialized_candidate_tree(
    candidate_root: Path,
    report: dict[str, Any],
    policy: dict[str, Any],
    selected: dict[str, tuple[str, str, str]],
    selected_paths: tuple[str, ...],
    member_bytes: dict[str, bytes],
    candidate_inventory_bytes: bytes,
    metadata_bytes: bytes,
) -> None:
    """Verify every candidate tree member against the recomputed materialization."""
    _validate_candidate_commit_identity(candidate_root, report)
    candidate_entries = _candidate_source_tree(candidate_root, report["candidate_commit_sha"])
    expected_paths = set(selected_paths) | {
        policy["candidate_inventory_path"],
        policy["metadata_path"],
    }
    if set(candidate_entries) != expected_paths:
        raise CandidateError("materialized candidate tree contains unexpected paths")
    for path in selected_paths:
        expected_mode = selected[path][0]
        actual_mode, actual_type, actual_object = candidate_entries[path]
        if actual_mode != expected_mode or actual_type != "blob":
            raise CandidateError(f"materialized candidate member mode or type drifted: {path}")
        if _candidate_blob(candidate_root, actual_object) != member_bytes[path]:
            raise CandidateError(f"materialized candidate member bytes drifted: {path}")
    for path, expected_bytes in (
        (policy["candidate_inventory_path"], candidate_inventory_bytes),
        (policy["metadata_path"], metadata_bytes),
    ):
        mode, object_type, object_id = candidate_entries[path]
        if (mode, object_type) != ("100644", "blob"):
            raise CandidateError(f"materialized generated member mode or type drifted: {path}")
        if _candidate_blob(candidate_root, object_id) != expected_bytes:
            raise CandidateError(f"materialized generated member bytes drifted: {path}")


def _materialization_identity(
    args: argparse.Namespace,
    repo_root: Path,
    *,
    expected_version: str,
) -> dict[str, Any] | None:
    """Recompute and bind the reviewed source materialization used for the build."""
    loaded = _load_assembly_materialization_report(args, repo_root)
    if loaded is None:
        return None
    candidate_root, report = loaded
    if report["package"]["version"] != expected_version:
        raise CandidateError(
            "materialization package version does not match the built distribution: "
            f"expected {expected_version}, found {report['package']['version']}"
        )
    (
        policy,
        selected,
        selected_paths,
        member_bytes,
        candidate_inventory_bytes,
        metadata_bytes,
    ) = _recompute_materialization(args, repo_root, report)
    _validate_materialized_candidate_tree(
        candidate_root,
        report,
        policy,
        selected,
        selected_paths,
        member_bytes,
        candidate_inventory_bytes,
        metadata_bytes,
    )
    return _validate_materialization_payload(
        {field: report[field] for field in MATERIALIZATION_PAYLOAD_FIELDS}
    )


def _assemble(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    _validate_schema_file(args.schema)
    _validate_source(repo_root, args.source_sha)
    for path, label in (
        (args.dist_dir, "distribution directory"),
        (args.raw_sbom, "raw SBOM"),
        (args.bundle_dir, "bundle directory"),
    ):
        _require_external(path, repo_root=repo_root, label=label)
    if list(args.validated) != list(VALIDATOR_IDS):
        raise CandidateError(
            "validated checks must be supplied exactly once in canonical order: "
            + ", ".join(VALIDATOR_IDS)
        )
    if not REPOSITORY_PATTERN.fullmatch(args.repository):
        raise CandidateError("repository must be an exact owner/name identity")
    if not RUN_ID_PATTERN.fullmatch(args.workflow_run_id):
        raise CandidateError("workflow run ID must be a positive decimal identity")
    if args.workflow_run_attempt < 1:
        raise CandidateError("workflow run attempt must be positive")

    wheel_input, sdist_input, version = _distribution_inputs(args.dist_dir)
    materialization = _materialization_identity(
        args,
        repo_root,
        expected_version=version,
    )
    sbom_bytes = _normalised_sbom(args.raw_sbom, version)
    _validate_source(repo_root, args.source_sha)
    _fresh_bundle_dir(args.bundle_dir)

    wheel = args.bundle_dir / wheel_input.name
    sdist = args.bundle_dir / sdist_input.name
    sbom = args.bundle_dir / f"robot_sf-{version}.cyclonedx.json"
    provenance = args.bundle_dir / PROVENANCE_NAME
    shutil.copyfile(wheel_input, wheel)
    shutil.copyfile(sdist_input, sdist)
    sbom.write_bytes(sbom_bytes)

    wheel_member = _member(wheel, "wheel")
    sdist_member = _member(sdist, "sdist")
    sbom_member = _member(sbom, "sbom")
    workflow = {
        "run_attempt": args.workflow_run_attempt,
        "run_id": args.workflow_run_id,
    }
    package = {"name": "robot_sf", "version": version}
    provenance.write_bytes(
        _json_bytes(
            _provenance_payload(
                repository=args.repository,
                source_sha=args.source_sha,
                workflow=workflow,
                package=package,
                wheel=wheel_member,
                sdist=sdist_member,
                sbom=sbom_member,
                materialization=materialization,
            )
        )
    )
    manifest = {
        "members": [wheel_member, sdist_member, sbom_member, _member(provenance, "provenance")],
        "package": package,
        "repository": args.repository,
        "schema_version": SCHEMA_VERSION,
        "source_sha": args.source_sha,
        "validation": _validation_payload(),
        "workflow": workflow,
    }
    if materialization is not None:
        manifest["materialization"] = materialization
    _validate_manifest(manifest)
    (args.bundle_dir / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
    _validate_source(repo_root, args.source_sha)
    print(
        f"PASS: admitted immutable Robot SF candidate {version} at {args.source_sha} "
        f"({len(manifest['members'])} bound payload members)"
    )


def _bundle_entries(bundle_dir: Path) -> list[Path]:
    if not bundle_dir.is_dir() or bundle_dir.is_symlink():
        raise CandidateError(f"candidate bundle is not a real directory: {bundle_dir}")
    entries = sorted(bundle_dir.iterdir(), key=lambda path: path.name)
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise CandidateError(f"unclassified candidate bundle member: {path.name}")
    return entries


def _verify_bundle_membership(
    bundle_dir: Path,
    entries: list[Path],
    manifest: dict[str, Any],
) -> None:
    expected_names = {MANIFEST_NAME, *(member["filename"] for member in manifest["members"])}
    actual_names = {path.name for path in entries}
    if actual_names != expected_names or len(entries) != len(expected_names):
        missing = sorted(expected_names - actual_names)
        unclassified = sorted(actual_names - expected_names)
        raise CandidateError(
            "candidate bundle membership drift "
            f"(missing={missing or 'none'}, unclassified={unclassified or 'none'})"
        )
    for member in manifest["members"]:
        path = bundle_dir / member["filename"]
        size = path.stat().st_size
        digest = _sha256(path)
        if size != member["size"] or digest != member["sha256"]:
            raise CandidateError(
                f"candidate member drift: {path.name} expected "
                f"size={member['size']} sha256={member['sha256']}, "
                f"found size={size} sha256={digest}"
            )


def _verify_archives_and_sbom(
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> None:
    wheel_member, sdist_member, sbom_member, _provenance_member = manifest["members"]
    _wheel_name, wheel_version = _wheel_metadata(bundle_dir / wheel_member["filename"])
    _sdist_name, sdist_version = _sdist_metadata(bundle_dir / sdist_member["filename"])
    if wheel_version != manifest["package"]["version"] or sdist_version != wheel_version:
        raise CandidateError("candidate archive metadata drifted from the manifest package version")
    sbom = _load_json(bundle_dir / sbom_member["filename"], label="normalised SBOM")
    if not isinstance(sbom, dict) or "serialNumber" in sbom:
        raise CandidateError("candidate SBOM is not deterministically normalised")
    metadata = sbom.get("metadata")
    if not isinstance(metadata, dict) or "timestamp" in metadata:
        raise CandidateError("candidate SBOM contains a nondeterministic timestamp")
    component = metadata.get("component")
    if not isinstance(component, dict) or component.get("version") != wheel_version:
        raise CandidateError("candidate SBOM package version drifted from the archives")


def _verify_provenance(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    wheel_member, sdist_member, sbom_member, provenance_member = manifest["members"]
    provenance = _load_json(
        bundle_dir / provenance_member["filename"], label="candidate provenance"
    )
    expected_provenance = _provenance_payload(
        repository=manifest["repository"],
        source_sha=manifest["source_sha"],
        workflow=manifest["workflow"],
        package=manifest["package"],
        wheel=wheel_member,
        sdist=sdist_member,
        sbom=sbom_member,
        materialization=manifest.get("materialization"),
    )
    if provenance != expected_provenance:
        raise CandidateError("candidate provenance does not exactly bind the manifest subjects")


def _candidate_receipt_identity(
    bundle_dir: Path,
    *,
    source_sha: str,
    candidate_run_id: str,
    candidate_run_attempt: int,
    candidate_artifact_id: str,
    candidate_artifact_name: str,
    candidate_artifact_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a candidate bundle and construct the exact rights-receipt identity."""
    _validate_schema_file(SCHEMA_PATH)
    entries = _bundle_entries(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    manifest = _validate_manifest(_load_json(manifest_path, label="candidate manifest"))
    if manifest["source_sha"] != source_sha:
        raise CandidateError("candidate manifest source SHA differs from rights admission input")
    if manifest["workflow"]["run_id"] != candidate_run_id:
        raise CandidateError("candidate manifest workflow run differs from rights admission input")
    if manifest["workflow"]["run_attempt"] != candidate_run_attempt:
        raise CandidateError(
            "candidate manifest workflow attempt differs from rights admission input"
        )
    _verify_bundle_membership(bundle_dir, entries, manifest)
    _verify_archives_and_sbom(bundle_dir, manifest)
    _verify_provenance(bundle_dir, manifest)
    digest = candidate_artifact_digest.removeprefix("sha256:")
    if not SHA256_PATTERN.fullmatch(digest):
        raise CandidateError("candidate artifact digest must be sha256:<64 lowercase characters>")
    if not RUN_ID_PATTERN.fullmatch(candidate_artifact_id):
        raise CandidateError("candidate artifact ID must be a positive decimal identity")
    candidate_name_pattern = re.compile(
        rf"robot-sf-software-candidate-{re.escape(source_sha)}-{re.escape(candidate_run_id)}-"
        rf"([1-9][0-9]*)\Z"
    )
    candidate_name_match = candidate_name_pattern.fullmatch(candidate_artifact_name)
    if candidate_name_match is None or candidate_name_match.group(1) != str(candidate_run_attempt):
        raise CandidateError("candidate artifact name is not bound to source and workflow identity")
    members = manifest["members"]
    members_by_kind = {member["kind"]: member for member in members}
    return manifest, {
        "artifact_digest": f"sha256:{digest}",
        "artifact_id": candidate_artifact_id,
        "artifact_name": candidate_artifact_name,
        "manifest_sha256": _sha256(manifest_path),
        "members": members,
        "package": manifest["package"],
        "provenance_sha256": members_by_kind["provenance"]["sha256"],
        "sbom_sha256": members_by_kind["sbom"]["sha256"],
        "source_sha": source_sha,
        "workflow_run_id": candidate_run_id,
    }


def _strict_archive_gate(
    bundle_dir: Path,
    *,
    candidate_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Re-run the canonical archive/tree gate over the exact candidate payload.

    The candidate manifest already binds every archive byte.  This second check
    deliberately runs against the materialized Git root rather than the source
    checkout, so a rebound archive carrying a forbidden model or blocked member
    cannot inherit the producer's earlier ``passed`` claim.
    """
    materialization = manifest.get("materialization")
    if not isinstance(materialization, dict):
        raise CandidateError("rights admission candidate has no materialization identity")
    inventory_path = candidate_root / materialization["candidate_inventory_path"]
    try:
        result = check_distribution(
            bundle_dir,
            strict_asset_rights=True,
            repo_root=candidate_root,
            inventory_path=inventory_path,
            source_tree_ref=materialization["candidate_commit_sha"],
        )
    except (DistributionLicenseError, OSError, ValueError) as exc:
        raise CandidateError(f"strict candidate archive/tree gate failed: {exc}") from exc
    expected = {
        member["filename"]: member["sha256"]
        for member in manifest["members"]
        if member["kind"] in {"wheel", "sdist"}
    }
    actual = {archive.name: _sha256(archive) for archive in (*result.wheels, *result.sdists)}
    if actual != expected:
        raise CandidateError(
            "strict candidate archive/tree gate did not cover the exact manifest archives"
        )
    # Keep the result canonical and hashable for focused callers/tests.  The
    # receipt's existing closed schema binds this report through the exact
    # archive member hashes and candidate manifest digest.
    report = {
        "archives": [{"filename": name, "sha256": actual[name]} for name in sorted(actual)],
        "candidate_commit_sha": materialization["candidate_commit_sha"],
        "candidate_tree_sha": materialization["candidate_tree_sha"],
        "schema_version": "robot_sf.software_strict_archive_gate.v1",
        "source_sha": manifest["source_sha"],
        "status": "passed",
    }
    report["report_sha256"] = hashlib.sha256(_json_bytes(report)).hexdigest()
    return report


def _report_input_digest(report: dict[str, Any], path: str, *, label: str) -> str:
    """Return one canonical repository-input digest from a dependency report."""
    inputs = report.get("repository_inputs")
    if not isinstance(inputs, list):
        raise CandidateError("supported dependency report has no repository input digest list")
    matches = [item for item in inputs if isinstance(item, dict) and item.get("path") == path]
    if len(matches) != 1:
        raise CandidateError(
            f"supported dependency report must bind exactly one {label} input: {path}"
        )
    digest = matches[0].get("sha256")
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise CandidateError(f"supported dependency report {label} digest is invalid")
    return digest


def _validate_supported_dependency_report(  # noqa: C901, PLR0912, PLR0915 - closed dependency gate
    report_path: Path,
    *,
    identity: dict[str, Any],
    source_sha: str,
    tree_sha256: str,
    workflow_run_attempt: int,
    materialization: dict[str, Any] | None,
    candidate_bundle: Path | None = None,
) -> dict[str, Any]:
    """Require a passed, candidate-bound supported-dependency inventory report."""
    if report_path.is_symlink() or not report_path.is_file():
        raise CandidateError(f"supported dependency report is not a regular file: {report_path}")
    if report_path.name != SUPPORTED_DEPENDENCY_REPORT_NAME:
        raise CandidateError(
            f"supported dependency report must be named {SUPPORTED_DEPENDENCY_REPORT_NAME}"
        )
    report = _load_json(report_path, label="supported dependency report")
    if not isinstance(report, dict):
        raise CandidateError("supported dependency report must be a JSON object")
    if report.get("schema_version") != SUPPORTED_DEPENDENCY_SCHEMA_VERSION:
        raise CandidateError("supported dependency report schema version is unsupported")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise CandidateError("supported dependency report has no summary")
    packages = report.get("packages")
    if not isinstance(packages, list) or not packages:
        raise CandidateError("supported dependency report has no package rows")
    selected_rows = []
    for row in packages:
        if not isinstance(row, dict):
            raise CandidateError("supported dependency report package row is malformed")
        selected_profiles = row.get("selected_profiles")
        if selected_profiles is not None and (
            not isinstance(selected_profiles, list)
            or any(
                not isinstance(profile_id, str) or not profile_id
                for profile_id in selected_profiles
            )
        ):
            raise CandidateError("supported dependency report package selection is malformed")
        if selected_profiles:
            selected_rows.append(row)
    pending_rows = [
        row for row in selected_rows if row.get("policy_disposition") == "review_required"
    ]
    failures = report.get("failures")
    structural_issues = report.get("structural_issues")
    if not isinstance(failures, list) or not isinstance(structural_issues, list):
        raise CandidateError("supported dependency report findings are malformed")
    expected_summary = {
        "selected_package_count": len(selected_rows),
        "policy_pending_package_count": len(pending_rows),
        "unresolved_count": len(failures),
        "structural_issue_count": len(structural_issues),
    }
    for field, expected in expected_summary.items():
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise CandidateError(
                f"supported dependency report summary.{field} is inconsistent with rows"
            )
    if summary.get("status") != "complete" or summary.get("candidate_bound") is not True:
        raise CandidateError("supported dependency report is not a complete candidate binding")
    unresolved_count = summary.get("unresolved_count")
    if (
        isinstance(unresolved_count, bool)
        or not isinstance(unresolved_count, int)
        or unresolved_count != 0
    ):
        raise CandidateError("supported dependency report contains unresolved rows")
    surface = report.get("surface")
    profile_ids = surface.get("profile_ids") if isinstance(surface, dict) else None
    if (
        not isinstance(profile_ids, list)
        or not profile_ids
        or not all(isinstance(profile_id, str) and profile_id for profile_id in profile_ids)
    ):
        raise CandidateError("supported dependency report has no non-empty profile surface")
    if profile_ids != list(SUPPORTED_DEPENDENCY_PROFILE_IDS):
        raise CandidateError(
            "supported dependency report profile surface does not match the closed v0.0.6 "
            f"roster: expected {list(SUPPORTED_DEPENDENCY_PROFILE_IDS)}, found {profile_ids}"
        )
    profile_manifest = report.get("profile_manifest")
    if (
        not isinstance(profile_manifest, dict)
        or profile_manifest.get("path") != SUPPORTED_DEPENDENCY_PROFILE_PATH
        or profile_manifest.get("schema_version") != SUPPORTED_DEPENDENCY_PROFILE_SCHEMA_VERSION
    ):
        raise CandidateError("supported dependency report profile manifest is unsupported")
    policy = report.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("path") != SUPPORTED_DEPENDENCY_POLICY_PATH
        or policy.get("schema_version") != SUPPORTED_DEPENDENCY_POLICY_SCHEMA_VERSION
    ):
        raise CandidateError("supported dependency report policy is unsupported")
    profiles = report.get("profiles")
    all_profiles = (
        [
            profile
            for profile in profiles
            if isinstance(profile, dict) and profile.get("id") == "all"
        ]
        if isinstance(profiles, list)
        else []
    )
    if len(all_profiles) != 1:
        raise CandidateError("supported dependency report has no unique canonical all profile")
    all_package_ids = all_profiles[0].get("package_ids")
    if (
        not isinstance(all_package_ids, list)
        or not all_package_ids
        or any(not isinstance(package_id, str) or not package_id for package_id in all_package_ids)
        or len(all_package_ids) != len(set(all_package_ids))
    ):
        raise CandidateError("supported dependency report all profile package closure is invalid")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in packages:
        package_id = row.get("package_id")
        if not isinstance(package_id, str) or not package_id or package_id in rows_by_id:
            raise CandidateError("supported dependency report package identity coverage is invalid")
        rows_by_id[package_id] = row
        expected_profiles = ["all"] if package_id in all_package_ids else []
        if row.get("selected_profiles") != expected_profiles:
            raise CandidateError(
                "supported dependency report selected-profile membership differs from canonical all closure"
            )
    selected_package_ids = {
        package_id for package_id, row in rows_by_id.items() if row.get("selected_profiles")
    }
    if selected_package_ids != set(all_package_ids):
        raise CandidateError("supported dependency report selected package closure is incomplete")
    if summary["selected_package_count"] != len(all_package_ids):
        raise CandidateError("supported dependency report selected package count is not canonical")
    all_profile = all_profiles[0]
    if (
        not isinstance(all_profile, dict)
        or all_profile.get("extras") != list(SUPPORTED_DEPENDENCY_EXTRA_IDS)
        or all_profile.get("excluded_extras") != ["rllib"]
    ):
        raise CandidateError(
            "supported dependency report all profile does not match the closed v0.0.6 extra roster"
        )
    if report.get("failures") != [] or report.get("structural_issues") != []:
        raise CandidateError("supported dependency report contains failures")
    binding = report.get("candidate_binding")
    if not isinstance(binding, dict) or binding.get("status") != "bound":
        raise CandidateError("supported dependency report candidate binding is not bound")
    if binding.get("source_sha") != source_sha:
        raise CandidateError("supported dependency report source SHA differs from candidate")
    if binding.get("manifest_sha256") != identity["manifest_sha256"]:
        raise CandidateError(
            "supported dependency report is bound to a different candidate manifest"
        )
    if binding.get("package") != identity["package"]:
        raise CandidateError("supported dependency report package identity differs from candidate")
    workflow = binding.get("workflow")
    if workflow != {
        "run_id": identity["workflow_run_id"],
        "run_attempt": workflow_run_attempt,
    }:
        raise CandidateError("supported dependency report workflow identity differs from candidate")
    if binding.get("materialization") != materialization:
        raise CandidateError(
            "supported dependency report materialization identity differs from candidate"
        )
    report_members = binding.get("members")
    if not isinstance(report_members, list):
        raise CandidateError("supported dependency report has no candidate member binding")
    report_members_by_kind = {
        member.get("kind"): member for member in report_members if isinstance(member, dict)
    }
    identity_members_by_kind = {member["kind"]: member for member in identity["members"]}
    if report_members_by_kind != identity_members_by_kind:
        raise CandidateError("supported dependency report candidate members differ from bundle")
    sbom = binding.get("sbom")
    if not isinstance(sbom, dict) or sbom.get("sha256") != identity["sbom_sha256"]:
        raise CandidateError("supported dependency report SBOM binding differs from candidate")
    if candidate_bundle is not None:
        sbom_filename = sbom.get("filename")
        if not isinstance(sbom_filename, str) or Path(sbom_filename).name != sbom_filename:
            raise CandidateError("supported dependency candidate SBOM filename is invalid")
        sbom_path = candidate_bundle / sbom_filename
        if sbom_path.is_symlink() or not sbom_path.is_file():
            raise CandidateError("supported dependency candidate SBOM is not a regular file")
        if _sha256(sbom_path) != identity["sbom_sha256"]:
            raise CandidateError("supported dependency candidate SBOM bytes differ from candidate")
        sbom_payload = _load_json(sbom_path, label="supported dependency candidate SBOM")
        components = sbom_payload.get("components")
        if not isinstance(components, list):
            raise CandidateError("supported dependency candidate SBOM components are invalid")
        component_ids = set()
        for component in components:
            if not isinstance(component, dict):
                raise CandidateError("supported dependency candidate SBOM component is invalid")
            name, version = component.get("name"), component.get("version")
            if not isinstance(name, str) or not isinstance(version, str):
                raise CandidateError("supported dependency candidate SBOM identity is invalid")
            component_ids.add((name.lower().replace("_", "-").replace(".", "-"), version))
        expected_components = {
            (row.get("normalized_name"), row.get("version"))
            for package_id, row in rows_by_id.items()
            if package_id in selected_package_ids and row.get("normalized_name") != "robot-sf"
        }
        if any(
            not isinstance(name, str) or not isinstance(version, str)
            for name, version in expected_components
        ):
            raise CandidateError("supported dependency report selected package identity is invalid")
        inactive_components = {
            tuple(value.split("@", 1))
            for value in sbom.get("target_inactive_components", [])
            if isinstance(value, str) and "@" in value
        }
        if component_ids != expected_components | inactive_components:
            raise CandidateError(
                "supported dependency candidate SBOM closure differs from all profile"
            )
        if sbom.get("component_count") != len(component_ids):
            raise CandidateError("supported dependency candidate SBOM component count is invalid")
    policy_sha256 = _report_input_digest(
        report,
        SUPPORTED_DEPENDENCY_POLICY_PATH,
        label="policy",
    )
    profile_manifest_sha256 = _report_input_digest(
        report,
        SUPPORTED_DEPENDENCY_PROFILE_PATH,
        label="profile manifest",
    )
    return {
        "candidate_manifest_sha256": identity["manifest_sha256"],
        "candidate_tree_sha256": tree_sha256,
        "command": SUPPORTED_DEPENDENCY_GATE_COMMAND,
        "id": SUPPORTED_DEPENDENCY_GATE_ID,
        "policy_path": SUPPORTED_DEPENDENCY_POLICY_PATH,
        "policy_sha256": policy_sha256,
        "profile_manifest_path": SUPPORTED_DEPENDENCY_PROFILE_PATH,
        "profile_manifest_sha256": profile_manifest_sha256,
        "report_filename": SUPPORTED_DEPENDENCY_REPORT_NAME,
        "report_sha256": _sha256(report_path),
        "schema_version": SUPPORTED_DEPENDENCY_SCHEMA_VERSION,
        "source_sha": source_sha,
        "status": "passed",
        "unresolved_count": 0,
    }


def _fresh_external_directory(path: Path, *, repo_root: Path, label: str) -> None:
    """Require an empty, non-symlink directory outside the source checkout."""
    _require_external(path, repo_root=repo_root, label=label)
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise CandidateError(f"{label} is not a real directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise CandidateError(f"{label} must be empty: {path}")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CandidateError(f"cannot create {label}: {path}: {exc}") from exc


def _validate_rights_receipt(receipt: Any) -> dict[str, Any]:  # noqa: C901 - closed receipt gate
    """Validate the exact rights receipt contract shared with the trusted publisher."""
    if not isinstance(receipt, dict) or set(receipt) != {
        "candidate",
        "sanitized",
        "strict_gate",
        "status",
        "schema_version",
        "supported_dependency_gate",
    }:
        raise CandidateError("rights admission receipt has missing or unclassified fields")
    if receipt["schema_version"] != RIGHTS_ADMISSION_SCHEMA or receipt["status"] != "accepted":
        raise CandidateError("rights admission receipt status or schema is invalid")
    candidate = receipt["candidate"]
    expected_candidate_keys = {
        "artifact_digest",
        "artifact_id",
        "artifact_name",
        "manifest_sha256",
        "members",
        "package",
        "provenance_sha256",
        "sbom_sha256",
        "source_sha",
        "workflow_run_id",
    }
    if not isinstance(candidate, dict) or set(candidate) != expected_candidate_keys:
        raise CandidateError("rights admission candidate binding is incomplete")
    if (
        not isinstance(candidate["artifact_digest"], str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", candidate["artifact_digest"])
        or not isinstance(candidate["artifact_id"], str)
        or not RUN_ID_PATTERN.fullmatch(candidate["artifact_id"])
        or not isinstance(candidate["artifact_name"], str)
        or not re.fullmatch(
            r"robot-sf-software-candidate-[0-9a-f]{40}-[1-9][0-9]*-[1-9][0-9]*",
            candidate["artifact_name"],
        )
        or not isinstance(candidate["manifest_sha256"], str)
        or not SHA256_PATTERN.fullmatch(candidate["manifest_sha256"])
        or not isinstance(candidate["provenance_sha256"], str)
        or not SHA256_PATTERN.fullmatch(candidate["provenance_sha256"])
        or not isinstance(candidate["sbom_sha256"], str)
        or not SHA256_PATTERN.fullmatch(candidate["sbom_sha256"])
        or not isinstance(candidate["source_sha"], str)
        or not SHA_PATTERN.fullmatch(candidate["source_sha"])
        or not isinstance(candidate["workflow_run_id"], str)
        or not RUN_ID_PATTERN.fullmatch(candidate["workflow_run_id"])
    ):
        raise CandidateError("rights admission candidate binding is invalid")
    package = candidate["package"]
    if (
        not isinstance(package, dict)
        or set(package) != {"name", "version"}
        or package.get("name") != "robot_sf"
        or not isinstance(package.get("version"), str)
        or not VERSION_PATTERN.fullmatch(package["version"])
    ):
        raise CandidateError("rights admission candidate package identity is invalid")
    _validate_members(candidate["members"], version=package["version"])
    sanitized = receipt["sanitized"]
    if not isinstance(sanitized, dict) or set(sanitized) != {
        "policy_id",
        "policy_path",
        "policy_sha256",
        "schema_version",
        "source_sha",
        "tree_sha256",
    }:
        raise CandidateError("rights admission sanitized binding is incomplete")
    if (
        sanitized["policy_id"] != RIGHTS_POLICY_ID
        or sanitized["policy_path"] != RIGHTS_POLICY_PATH
        or sanitized["schema_version"] != SANITIZED_CANDIDATE_SCHEMA
        or not isinstance(sanitized["source_sha"], str)
        or not SHA_PATTERN.fullmatch(sanitized["source_sha"])
        or not isinstance(sanitized["policy_sha256"], str)
        or not SHA256_PATTERN.fullmatch(sanitized["policy_sha256"])
        or not isinstance(sanitized["tree_sha256"], str)
        or not SHA256_PATTERN.fullmatch(sanitized["tree_sha256"])
    ):
        raise CandidateError("rights admission sanitized binding is invalid")
    strict_gate = receipt["strict_gate"]
    if not isinstance(strict_gate, dict) or set(strict_gate) != {
        "command",
        "findings",
        "id",
        "policy_sha256",
        "source_sha",
        "status",
    }:
        raise CandidateError("rights admission strict-gate binding is incomplete")
    if (
        strict_gate["command"] != RIGHTS_GATE_COMMAND
        or strict_gate["findings"] != 0
        or strict_gate["id"] != RIGHTS_GATE_ID
        or strict_gate["policy_sha256"] != sanitized["policy_sha256"]
        or strict_gate["source_sha"] != sanitized["source_sha"]
        or strict_gate["status"] != "passed"
    ):
        raise CandidateError("rights admission strict-gate binding is invalid")
    dependency_gate = receipt["supported_dependency_gate"]
    expected_dependency_keys = {
        "candidate_manifest_sha256",
        "candidate_tree_sha256",
        "command",
        "id",
        "policy_path",
        "policy_sha256",
        "profile_manifest_path",
        "profile_manifest_sha256",
        "report_filename",
        "report_sha256",
        "schema_version",
        "source_sha",
        "status",
        "unresolved_count",
    }
    if not isinstance(dependency_gate, dict) or set(dependency_gate) != expected_dependency_keys:
        raise CandidateError(
            "rights admission supported-dependency gate is missing or unclassified"
        )
    if (
        dependency_gate["schema_version"] != SUPPORTED_DEPENDENCY_SCHEMA_VERSION
        or dependency_gate["id"] != SUPPORTED_DEPENDENCY_GATE_ID
        or dependency_gate["command"] != SUPPORTED_DEPENDENCY_GATE_COMMAND
        or dependency_gate["policy_path"] != SUPPORTED_DEPENDENCY_POLICY_PATH
        or dependency_gate["profile_manifest_path"] != SUPPORTED_DEPENDENCY_PROFILE_PATH
        or dependency_gate["report_filename"] != SUPPORTED_DEPENDENCY_REPORT_NAME
        or dependency_gate["status"] != "passed"
        or dependency_gate["source_sha"] != sanitized["source_sha"]
        or dependency_gate["candidate_manifest_sha256"] != candidate["manifest_sha256"]
        or dependency_gate["candidate_tree_sha256"] != sanitized["tree_sha256"]
        or isinstance(dependency_gate["unresolved_count"], bool)
        or dependency_gate["unresolved_count"] != 0
        or any(
            not isinstance(dependency_gate[field], str)
            or not SHA256_PATTERN.fullmatch(dependency_gate[field])
            for field in ("policy_sha256", "profile_manifest_sha256", "report_sha256")
        )
    ):
        raise CandidateError("rights admission supported-dependency gate is invalid")
    return receipt


def _admit_rights(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915 - closed receipt workflow
    """Create the separately downloadable exact rights-admission receipt."""
    repo_root = args.repo_root.resolve()
    if not SHA_PATTERN.fullmatch(args.source_sha):
        raise CandidateError("rights admission source SHA is invalid")
    if not RUN_ID_PATTERN.fullmatch(args.workflow_run_id):
        raise CandidateError("rights admission workflow run ID is invalid")
    if args.workflow_run_attempt < 1:
        raise CandidateError("rights admission workflow attempt must be positive")
    _validate_source(repo_root, args.source_sha)
    policy_path = (args.policy if args.policy.is_absolute() else repo_root / args.policy).resolve()
    policy_raw, policy = _load_rights_policy(policy_path, repo_root=repo_root)

    materialization_report_path = Path(os.path.abspath(args.materialization_report))
    candidate_root = Path(os.path.abspath(args.candidate_root))
    materialization_args = argparse.Namespace(
        candidate_source_root=candidate_root,
        materialization_report=materialization_report_path,
        policy=repo_root
        / _load_json(materialization_report_path, label="materialization report")["policy_path"],
        source_sha=args.source_sha,
    )
    loaded = _load_assembly_materialization_report(materialization_args, repo_root)
    if loaded is None:  # pragma: no cover - both arguments are required by argparse.
        raise CandidateError("rights admission requires a materialization report")
    _materialization_identity(
        materialization_args,
        repo_root,
        expected_version=args.candidate_version,
    )
    _candidate_root, report = loaded
    source_tree_sha1 = _candidate_tree_sha1(repo_root, args.source_sha)
    sanitized = _sanitized_manifest_from_report(
        report,
        source_tree_sha1=source_tree_sha1,
        policy_raw=policy_raw,
        policy=policy,
    )
    manifest, identity = _candidate_receipt_identity(
        args.candidate_bundle,
        source_sha=args.source_sha,
        candidate_run_id=args.workflow_run_id,
        candidate_run_attempt=args.workflow_run_attempt,
        candidate_artifact_id=args.candidate_artifact_id,
        candidate_artifact_name=args.candidate_artifact_name,
        candidate_artifact_digest=args.candidate_artifact_digest,
    )
    _strict_archive_gate(
        args.candidate_bundle,
        candidate_root=_candidate_root,
        manifest=manifest,
    )
    if (
        manifest["package"]["name"] != "robot_sf"
        or manifest["package"]["version"] != args.candidate_version
    ):
        raise CandidateError("rights admission candidate package differs from requested version")
    dependency_report_path = args.dependency_report.resolve()
    supported_dependency_gate = _validate_supported_dependency_report(
        dependency_report_path,
        identity=identity,
        source_sha=args.source_sha,
        tree_sha256=sanitized["tree_sha256"],
        workflow_run_attempt=manifest["workflow"]["run_attempt"],
        materialization=manifest.get("materialization"),
        candidate_bundle=args.candidate_bundle,
    )
    sanitized_path = Path(os.path.abspath(args.sanitized_manifest))
    _require_external(sanitized_path, repo_root=repo_root, label="sanitized candidate manifest")
    if sanitized_path.resolve(strict=False).is_relative_to(candidate_root):
        raise CandidateError("sanitized candidate manifest must be outside candidate source")
    try:
        if sanitized_path.exists() or sanitized_path.is_symlink():
            raise CandidateError(
                f"sanitized candidate manifest must not already exist: {sanitized_path}"
            )
        sanitized_path.parent.mkdir(parents=True, exist_ok=True)
        sanitized_path.write_bytes(_json_bytes(sanitized))
    except CandidateError:
        raise
    except OSError as exc:
        raise CandidateError(
            f"cannot write sanitized candidate manifest: {sanitized_path}: {exc}"
        ) from exc

    output_dir = Path(os.path.abspath(args.output_dir))
    _fresh_external_directory(
        output_dir,
        repo_root=repo_root,
        label="rights admission artifact directory",
    )
    dependency_report_copy = output_dir / SUPPORTED_DEPENDENCY_REPORT_NAME
    try:
        shutil.copyfile(dependency_report_path, dependency_report_copy)
    except OSError as exc:
        raise CandidateError(
            "cannot copy supported dependency report into rights admission artifact: "
            f"{dependency_report_copy}: {exc}"
        ) from exc
    if _sha256(dependency_report_copy) != supported_dependency_gate["report_sha256"]:
        raise CandidateError("copied supported dependency report bytes differ from validated input")
    receipt = {
        "candidate": identity,
        "sanitized": {
            "policy_id": RIGHTS_POLICY_ID,
            "policy_path": RIGHTS_POLICY_PATH,
            "policy_sha256": sanitized["policy_sha256"],
            "schema_version": SANITIZED_CANDIDATE_SCHEMA,
            "source_sha": args.source_sha,
            "tree_sha256": sanitized["tree_sha256"],
        },
        "strict_gate": {
            "command": RIGHTS_GATE_COMMAND,
            "findings": 0,
            "id": RIGHTS_GATE_ID,
            "policy_sha256": sanitized["policy_sha256"],
            "source_sha": args.source_sha,
            "status": "passed",
        },
        "status": "accepted",
        "schema_version": RIGHTS_ADMISSION_SCHEMA,
        "supported_dependency_gate": supported_dependency_gate,
    }
    _validate_rights_receipt(receipt)
    receipt_path = output_dir / RIGHTS_ADMISSION_NAME
    try:
        with receipt_path.open("xb") as stream:
            stream.write(_json_bytes(receipt))
    except FileExistsError as exc:
        raise CandidateError(
            f"refusing to overwrite rights admission receipt: {receipt_path}"
        ) from exc
    except OSError as exc:
        raise CandidateError(
            f"cannot write rights admission receipt: {receipt_path}: {exc}"
        ) from exc
    print(
        f"PASS: accepted rights-clean candidate {args.source_sha}; "
        f"receipt={receipt_path} tree_sha256={sanitized['tree_sha256']}"
    )


def _verify(args: argparse.Namespace) -> None:
    _validate_schema_file(args.schema)
    if not SHA_PATTERN.fullmatch(args.expected_source_sha):
        raise CandidateError("expected source SHA must be one exact lowercase 40-hex identity")
    if not RUN_ID_PATTERN.fullmatch(args.expected_workflow_run_id):
        raise CandidateError("expected workflow run ID must be a positive decimal identity")
    entries = _bundle_entries(args.bundle_dir)
    manifest = _validate_manifest(
        _load_json(args.bundle_dir / MANIFEST_NAME, label="candidate manifest")
    )
    if manifest["source_sha"] != args.expected_source_sha:
        raise CandidateError(
            f"candidate source drift: expected {args.expected_source_sha}, "
            f"found {manifest['source_sha']}"
        )
    if manifest["workflow"]["run_id"] != args.expected_workflow_run_id:
        raise CandidateError(
            f"candidate workflow-run drift: expected {args.expected_workflow_run_id}, "
            f"found {manifest['workflow']['run_id']}"
        )
    _verify_bundle_membership(args.bundle_dir, entries, manifest)
    _verify_archives_and_sbom(args.bundle_dir, manifest)
    _verify_provenance(args.bundle_dir, manifest)
    print(
        f"PASS: offline candidate verification reused exact bytes from run "
        f"{manifest['workflow']['run_id']} at {manifest['source_sha']}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("check-source", help="fail unless source identity is exact")
    source.add_argument("--repo-root", type=Path, required=True)
    source.add_argument("--source-sha", required=True)

    stage = subparsers.add_parser(
        "stage-build-source",
        help="materialize an exact commit in a disposable external build root",
    )
    stage.add_argument("--repo-root", type=Path, required=True)
    stage.add_argument("--build-root", type=Path, required=True)
    stage.add_argument("--source-sha", required=True)

    materialize = subparsers.add_parser(
        "materialize-source",
        help="materialize and commit a deterministic rights-scoped source candidate",
    )
    materialize.add_argument("--repo-root", type=Path, required=True)
    materialize.add_argument("--candidate-root", type=Path, required=True)
    materialize.add_argument("--source-sha", required=True)
    materialize.add_argument("--policy", type=Path, default=None)
    materialize.add_argument("--report", type=Path, default=None)

    admit = subparsers.add_parser(
        "rights-admission",
        help="create a separate rights receipt after strict rights and dependency gates pass",
    )
    admit.add_argument("--repo-root", type=Path, required=True)
    admit.add_argument("--candidate-root", type=Path, required=True)
    admit.add_argument("--materialization-report", type=Path, required=True)
    admit.add_argument("--sanitized-manifest", type=Path, required=True)
    admit.add_argument("--candidate-bundle", type=Path, required=True)
    admit.add_argument("--dependency-report", type=Path, required=True)
    admit.add_argument("--output-dir", type=Path, required=True)
    admit.add_argument("--source-sha", required=True)
    admit.add_argument("--candidate-version", required=True)
    admit.add_argument("--workflow-run-id", required=True)
    admit.add_argument("--workflow-run-attempt", type=int, required=True)
    admit.add_argument("--candidate-artifact-id", required=True)
    admit.add_argument("--candidate-artifact-name", required=True)
    admit.add_argument("--candidate-artifact-digest", required=True)
    admit.add_argument("--policy", type=Path, default=Path(RIGHTS_POLICY_PATH))

    assemble = subparsers.add_parser("assemble", help="admit already-built candidate bytes")
    assemble.add_argument("--repo-root", type=Path, required=True)
    assemble.add_argument("--dist-dir", type=Path, required=True)
    assemble.add_argument("--raw-sbom", type=Path, required=True)
    assemble.add_argument("--bundle-dir", type=Path, required=True)
    assemble.add_argument("--source-sha", required=True)
    assemble.add_argument("--repository", required=True)
    assemble.add_argument("--workflow-run-id", required=True)
    assemble.add_argument("--workflow-run-attempt", type=int, required=True)
    assemble.add_argument("--validated", action="append", default=[])
    assemble.add_argument("--candidate-source-root", type=Path, default=None)
    assemble.add_argument("--materialization-report", type=Path, default=None)
    assemble.add_argument("--schema", type=Path, default=SCHEMA_PATH)

    verify = subparsers.add_parser("verify", help="offline verification without rebuilding")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    verify.add_argument("--expected-source-sha", required=True)
    verify.add_argument("--expected-workflow-run-id", required=True)
    verify.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed candidate command."""
    args = _parse_args(argv)
    try:
        if args.command == "check-source":
            _validate_source(args.repo_root.resolve(), args.source_sha)
            print(f"PASS: source identity is clean and exact at {args.source_sha}")
        elif args.command == "stage-build-source":
            _stage_build_source(args)
        elif args.command == "materialize-source":
            _materialize_source(args)
        elif args.command == "rights-admission":
            _admit_rights(args)
        elif args.command == "assemble":
            _assemble(args)
        elif args.command == "verify":
            _verify(args)
        else:  # pragma: no cover - argparse prevents this branch.
            raise CandidateError(f"unsupported candidate command: {args.command}")
    except CandidateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
