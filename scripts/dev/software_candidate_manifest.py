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
import zipfile
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

SCHEMA_VERSION = "robot_sf.software_candidate.v1"
SCHEMA_ID = "https://robot-sf.dev/schema/software-candidate-manifest.v1.json"
SCHEMA_SHA256 = "ffa6635a7a37e21a36881ff8a89be59ee706c41107b94771ace8ed663d2f6469"
PROVENANCE_VERSION = "robot_sf.software_candidate.provenance.v1"
SCHEMA_PATH = Path(__file__).with_name("software_candidate_manifest.v1.schema.json")
MANIFEST_NAME = "candidate-manifest.json"
PROVENANCE_NAME = "candidate-provenance.json"
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
        "python scripts/tools/check_distribution_licenses.py $DIST_DIR",
    ),
    (
        "wheel-install",
        "bash scripts/validation/wheel_install_smoke.sh $DIST_DIR/robot_sf-*.whl",
    ),
)
VALIDATOR_IDS = tuple(identifier for identifier, _command in VALIDATION_COMMANDS)
MEMBER_KINDS = ("wheel", "sdist", "sbom", "provenance")
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


def _load_json(path: Path, *, label: str) -> Any:
    try:
        raw = path.read_bytes()
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object)
    except CandidateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


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
        schema = json.loads(raw_schema.decode("utf-8"), object_pairs_hook=_json_object)
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
    if set(payload) != expected_keys:
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
) -> dict[str, Any]:
    return {
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
    )
    if provenance != expected_provenance:
        raise CandidateError("candidate provenance does not exactly bind the manifest subjects")


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
