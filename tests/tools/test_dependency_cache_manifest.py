"""Tests for dependency_cache_manifest (#8852, repaired by #9126).

Scenario coverage: exact wheel set, platform mismatch, missing transitive package,
local editable package, source-only build, private companion, duplicate artifact,
checksum drift, offline install failure, plus the five P1 fail-closed contracts:
complete extra closure, universal/ABI3 tag acceptance, guarded offline install,
sanitized public output, and symlink-safe custody inventory.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from scripts.tools.dependency_cache_manifest import (
    build_manifest,
    build_offline_install_plan,
    main,
    offline_install_plan,
    reconstruction_status,
    write_sums,
)

if TYPE_CHECKING:
    from pathlib import Path

WHEEL = "wheelpkg-1.0.0-py3-none-any.whl"
SDIST = "sdistpkg-2.0.0.tar.gz"
MAC_WHEEL = "platpkg-3.0.0-cp311-cp311-macosx_11_0_arm64.whl"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha(data)


def _lock(*, wheel_hash: str, sdist_hash: str, mac_wheel_hash: str, editables: bool = True) -> str:
    editable = 'source = { editable = "." }\n' if editables else ""
    return f"""
version = 1
revision = 3
requires-python = ">=3.11"

[[package]]
name = "app"
version = "0.1.0"
{editable}dependencies = [
    {{ name = "wheelpkg" }},
    {{ name = "sdistpkg" }},
    {{ name = "platpkg" }},
    {{ name = "missingpkg" }},
]

[[package]]
name = "wheelpkg"
version = "1.0.0"
source = {{ registry = "https://pypi.org/simple" }}
wheels = [
    {{ url = "https://files.example/{WHEEL}", hash = "sha256:{wheel_hash}", size = 10 }},
]

[[package]]
name = "sdistpkg"
version = "2.0.0"
source = {{ registry = "https://pypi.org/simple" }}
sdist = {{ url = "https://files.example/{SDIST}", hash = "sha256:{sdist_hash}", size = 20 }}

[[package]]
name = "platpkg"
version = "3.0.0"
source = {{ registry = "https://pypi.org/simple" }}
wheels = [
    {{ url = "https://files.example/{MAC_WHEEL}", hash = "sha256:{mac_wheel_hash}", size = 30 }},
]

[[package]]
name = "missingpkg"
version = "4.0.0"
source = {{ registry = "https://pypi.org/simple" }}
wheels = [
    {{ url = "https://files.example/missingpkg-4.0.0-py3-none-any.whl", hash = "sha256:{wheel_hash}", size = 40 }},
]
"""


def _profile(**rights: object) -> dict:
    return {
        "schema": "dependency_cache_profile.v1",
        "profile_id": "fixture-linux-x86_64-py311",
        "lockfile": "uv.lock",
        "roots": [{"name": "app", "extras": []}],
        "environment": {
            "python_version": "3.11",
            "sys_platform": "linux",
            "platform_machine": "x86_64",
        },
        "rights": {"default": "redistribution-unknown", **rights},
    }


def _setup(tmp_path: Path, **rights: object) -> tuple[dict, Path]:
    cache = tmp_path / "cache"
    wheel_hash = _write(cache / WHEEL, b"wheel-bytes")
    sdist_hash = _write(cache / SDIST, b"sdist-bytes")
    mac_hash = _write(cache / "other" / MAC_WHEEL, b"mac-bytes")
    editables = bool(rights.pop("editables", True))
    lock = _lock(
        wheel_hash=wheel_hash, sdist_hash=sdist_hash, mac_wheel_hash=mac_hash, editables=editables
    )
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(lock, encoding="utf-8")
    import tomllib

    with lock_path.open("rb") as handle:
        return tomllib.load(handle), cache


def test_exact_wheel_set_source_only_and_editable(tmp_path: Path) -> None:
    lock, cache = _setup(tmp_path)
    manifest = build_manifest(_profile(), lock, cache)
    rows = {row["name"]: row for row in manifest["requirements"]}
    assert rows["wheelpkg"]["availability"] == "available_verified"
    assert rows["wheelpkg"]["custody"] == "registry-artifact"
    assert rows["sdistpkg"]["availability"] == "available_verified"
    assert rows["sdistpkg"]["artifact_kind"] == "sdist"
    assert rows["sdistpkg"]["custody"] == "source-tree"
    assert rows["app"]["availability"] == "build_required"
    assert rows["app"]["custody"] == "editable-checkout"


def test_platform_mismatch_and_missing_transitive(tmp_path: Path) -> None:
    lock, cache = _setup(tmp_path)
    rows = {row["name"]: row for row in build_manifest(_profile(), lock, cache)["requirements"]}
    assert rows["platpkg"]["availability"] == "wrong_platform"
    assert rows["platpkg"]["rejection_reason"] == "incompatible_platform"
    assert rows["missingpkg"]["availability"] == "missing"


def test_source_only_when_sdist_not_cached(tmp_path: Path) -> None:
    lock, cache = _setup(tmp_path)
    (cache / SDIST).unlink()
    rows = {row["name"]: row for row in build_manifest(_profile(), lock, cache)["requirements"]}
    assert rows["sdistpkg"]["availability"] == "source_only"


def test_duplicate_artifact_and_checksum_drift(tmp_path: Path) -> None:
    lock, cache = _setup(tmp_path)
    _write(cache / "second" / WHEEL, b"wheel-bytes")
    _write(cache / SDIST, b"different-bytes")
    rows = {row["name"]: row for row in build_manifest(_profile(), lock, cache)["requirements"]}
    assert rows["wheelpkg"]["availability"] == "duplicate_artifact"
    assert rows["wheelpkg"]["cache_copies"] == 2
    assert rows["sdistpkg"]["availability"] == "checksum_drift"


def test_private_companion_stays_private(tmp_path: Path) -> None:
    lock, cache = _setup(tmp_path)
    profile = _profile(private_companions=["wheelpkg"], permitted=["sdistpkg"])
    status = reconstruction_status(build_manifest(profile, lock, cache))
    rows = {row["name"]: row for row in status["requirements"]}
    assert rows["wheelpkg"]["rights"] == "private-companion"
    assert rows["wheelpkg"]["custody"] == "private-companion"
    assert rows["sdistpkg"]["rights"] == "redistribution-permitted"
    assert status["private_preservation_does_not_imply_public_redistribution"] is True
    assert status["offline_reconstruction_complete"] is False


def test_write_sums_is_deterministic(tmp_path: Path) -> None:
    lock, cache = _setup(tmp_path)
    manifest = build_manifest(_profile(), lock, cache)
    first, second = tmp_path / "SHA256SUMS.a", tmp_path / "SHA256SUMS.b"
    assert write_sums(manifest, first) == 3
    write_sums(manifest, second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    assert WHEEL in first.read_text(encoding="utf-8")


def test_extra_marker_edges_keep_the_complete_closure(tmp_path: Path) -> None:
    """P1: extra-marker substitution must not drop a dependency sub-graph."""
    lock_text = """
version = 1
revision = 3

[[package]]
name = "app"
version = "0.1.0"
source = { editable = "." }
optional-dependencies = { rllib = [
    { name = "ray", extra = ["rllib"], marker = "extra == 'extra-8-app-rllib'" },
] }

[[package]]
name = "ray"
version = "2.9.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "ormsgpack" }]
optional-dependencies = { rllib = [{ name = "lz4" }] }
wheels = [{ url = "https://files.example/ray-2.9.0-py3-none-any.whl", hash = "sha256:aa", size = 1 }]

[[package]]
name = "ormsgpack"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://files.example/ormsgpack-1.0.0-py311-none-any.whl", hash = "sha256:bb", size = 1 }]

[[package]]
name = "lz4"
version = "4.0.0"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://files.example/lz4-4.0.0-cp39-abi3-manylinux_2_28_x86_64.whl", hash = "sha256:cc", size = 1 }]
"""
    import tomllib

    lock = tomllib.loads(lock_text)
    profile = _profile()
    profile["roots"] = [{"name": "app", "extras": ["rllib"]}]
    manifest = build_manifest(profile, lock, tmp_path / "cache")
    names = {row["name"] for row in manifest["requirements"]}
    assert {"ray", "ormsgpack", "lz4"} <= names
    assert manifest["closure_audit"]["complete"] is True
    assert manifest["closure_audit"]["selected_requirements"] == len(names)


def test_universal_and_abi3_wheels_are_accepted(tmp_path: Path) -> None:
    """P1: usable py311-none-any and cp39-abi3 wheels must be selectable."""
    lock_text = """
version = 1
[[package]]
name = "app"
version = "0.1.0"
dependencies = [{ name = "universal" }, { name = "abi3" }, { name = "future" }]
[[package]]
name = "universal"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://files.example/universal-1.0.0-py311-none-any.whl", hash = "sha256:dd", size = 1 }]
[[package]]
name = "abi3"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://files.example/abi3-1.0.0-cp39-abi3-manylinux_2_28_x86_64.whl", hash = "sha256:ee", size = 1 }]
[[package]]
name = "future"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://files.example/future-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl", hash = "sha256:ff", size = 1 }]
"""
    import tomllib

    lock = tomllib.loads(lock_text)
    manifest = build_manifest(_profile(), lock, tmp_path / "empty-cache")
    rows = {row["name"]: row for row in manifest["requirements"]}
    assert rows["universal"]["artifact"] == "universal-1.0.0-py311-none-any.whl"
    assert rows["abi3"]["artifact"] == "abi3-1.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
    assert rows["future"]["availability"] == "missing"
    assert rows["future"]["artifact"] is None
    assert rows["future"]["rejection_reason"] == "incompatible_interpreter"


def _all_wheel_manifest(tmp_path: Path, **rights: object) -> tuple[dict, Path]:
    wheel_hash = _sha(b"wheel-bytes")
    lock_text = f"""
version = 1
[[package]]
name = "app"
version = "0.1.0"
dependencies = [{{ name = "wheelpkg" }}]
[[package]]
name = "wheelpkg"
version = "1.0.0"
source = {{ registry = "https://pypi.org/simple" }}
wheels = [{{ url = "https://files.example/{WHEEL}", hash = "sha256:{wheel_hash}", size = 10 }}]
"""
    import tomllib

    manifest = build_manifest(
        _profile(permitted=["wheelpkg", "app"], **rights),
        tomllib.loads(lock_text),
        tmp_path / "cache",
    )
    manifest["offline_install_permitted"] = True
    return manifest, tmp_path / "cache"


def _installable_manifest(tmp_path: Path, **rights: object) -> tuple[dict, Path]:
    """A manifest whose non-root rows are all verified registry artifacts with hashes."""
    manifest, cache = _all_wheel_manifest(tmp_path, **rights)
    roots = {str(name) for name in manifest["root_names"]}
    manifest["requirements"] = [row for row in manifest["requirements"] if row["name"] not in roots]
    for row in manifest["requirements"]:
        row["availability"] = "available_verified"
        row["custody"] = "registry-artifact"
        row["expected_sha256"] = "a" * 64
    return manifest, cache


def test_offline_plan_is_guarded_and_hashed(tmp_path: Path) -> None:
    """P1: offline mode proves no-download, interpreter, hash, and allowlist guards."""
    manifest, cache = _installable_manifest(tmp_path)
    plan = build_offline_install_plan(manifest, cache, tmp_path / "venv", tmp_path / "req.txt")
    assert plan["status"] == "planned"
    argv = plan["install_argv"]
    for flag in ("--no-index", "--offline", "--only-binary", "--require-hashes", "--find-links"):
        assert flag in argv
    assert plan["guarded_env"]["UV_OFFLINE"] == "1"
    assert plan["guarded_env"]["PIP_NO_INDEX"] == "1"
    assert plan["guarded_env"]["UV_PYTHON_DOWNLOADS"] == "never"
    assert all("--hash=sha256:" in line for line in plan["requirements"])
    assert plan["expected_python_version"] == "3.11"
    assert plan["excluded_roots"] == ["app"]


def test_offline_plan_refuses_missing_hash_and_source_custody(tmp_path: Path) -> None:
    manifest, cache = _installable_manifest(tmp_path)
    for row in manifest["requirements"]:
        row["expected_sha256"] = None
    plan = build_offline_install_plan(manifest, cache, tmp_path / "venv", tmp_path / "req.txt")
    assert plan["status"] == "blocked"
    assert plan["reason"] == "missing_hash"

    lock, cache2 = _setup(tmp_path)
    locked = build_manifest(
        _profile(permitted=["wheelpkg", "sdistpkg", "platpkg", "missingpkg"]), lock, cache2
    )
    locked["offline_install_permitted"] = True
    for row in locked["requirements"]:
        row["availability"] = "available_verified"
        row["expected_sha256"] = "a" * 64
    plan = build_offline_install_plan(locked, cache2, tmp_path / "venv2", tmp_path / "req2.txt")
    assert plan["status"] == "blocked"
    assert plan["reason"] == "incomplete_or_unpermitted_set"
    assert any("source-tree" in blocker for blocker in plan["blockers"])


def test_offline_install_failure_is_reported(tmp_path: Path) -> None:
    manifest, cache = _installable_manifest(tmp_path)

    class Result:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode, self.stdout, self.stderr = returncode, stdout, stderr

    calls = {"n": 0}

    def runner(argv: list[str], env: dict[str, str]) -> Result:
        calls["n"] += 1
        if calls["n"] == 1:  # uv venv
            return Result(0)
        return Result(1, "", "ERROR: no matching distribution found")

    result = offline_install_plan(manifest, cache, runner=runner)
    assert result["status"] == "failed"
    assert result["reason"] == "offline_install_failed"


def test_offline_install_verifies_post_install_versions(tmp_path: Path) -> None:
    manifest, cache = _installable_manifest(tmp_path)

    class Ok:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str = "") -> None:
            self.stdout = stdout

    def runner(argv: list[str], env: dict[str, str]) -> Ok:
        if "list" in argv:
            return Ok(json.dumps([{"name": "wheelpkg", "version": "9.9.9"}]))
        return Ok()

    result = offline_install_plan(manifest, cache, runner=runner)
    assert result["status"] == "failed"
    assert result["reason"] == "post_install_version_mismatch"


def test_public_status_redacts_paths_and_profile_identifiers(tmp_path: Path) -> None:
    """P1: public output has no absolute private paths or profile identifiers."""
    lock, cache = _setup(tmp_path)
    profile = _profile()
    profile["profile_id"] = "/home/private-owner/secret-profile"
    status = reconstruction_status(build_manifest(profile, lock, cache))
    text = json.dumps(status)
    assert "/home/" not in text
    assert "private-owner" not in text
    assert status["profile_id"].startswith("profile-sha256:")
    assert "/home/" not in json.dumps(
        reconstruction_status(build_manifest(_profile(), lock, cache))
    )


def test_symlinked_artifacts_are_rejected_before_hashing(tmp_path: Path) -> None:
    """P1: custody inventory stays inside the declared root and never follows symlinks."""
    lock, cache = _setup(tmp_path)
    outside = tmp_path / "outside" / WHEEL
    _write(outside, b"outside-bytes")
    link = cache / "linked" / WHEEL
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)

    manifest = build_manifest(_profile(), lock, cache)
    assert manifest["cache_rejections"], "symlinked artifact must be reported as rejected"
    assert any("symlink" in entry for entry in manifest["cache_rejections"])
    row = {r["name"]: r for r in manifest["requirements"]}["wheelpkg"]
    # The real file in the cache root still verifies; the symlink is never hashed.
    assert row["availability"] == "available_verified"


def test_cli_reports_gaps_and_json(tmp_path: Path, capsys) -> None:
    _, cache = _setup(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile()), encoding="utf-8")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        _lock(
            wheel_hash=_sha(b"wheel-bytes"),
            sdist_hash=_sha(b"sdist-bytes"),
            mac_wheel_hash=_sha(b"mac-bytes"),
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "--check",
            "--profile",
            str(profile_path),
            "--cache-root",
            str(cache),
            "--lock",
            str(lock_path),
            "--format",
            "json",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["by_availability"]["available_verified"] == 2
    assert payload["closure_complete"] is True
    sums = tmp_path / "SHA256SUMS"
    man = tmp_path / "manifest.json"
    main(
        [
            "--check",
            "--profile",
            str(profile_path),
            "--cache-root",
            str(cache),
            "--lock",
            str(lock_path),
            "--manifest-out",
            str(man),
            "--sums-out",
            str(sums),
            "--format",
            "markdown",
        ]
    )
    assert sums.is_file() and man.is_file()
    manifest = json.loads(man.read_text(encoding="utf-8"))
    assert manifest["closure_audit"]["complete"] is True
    assert manifest["summary"]["by_custody"]
    assert WHEEL in sums.read_text(encoding="utf-8")
