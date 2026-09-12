"""Tests for dependency_cache_manifest (#8852).

Fixture scenarios: exact wheel set, platform mismatch, missing transitive package,
local editable package, source-only build, private companion, duplicate artifact,
checksum drift, offline install failure.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from scripts.tools.dependency_cache_manifest import (
    build_manifest,
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
    assert rows["sdistpkg"]["availability"] == "available_verified"
    assert rows["sdistpkg"]["artifact_kind"] == "sdist"
    assert rows["app"]["availability"] == "build_required"
    assert rows["app"]["artifact_kind"] == "build"


def test_platform_mismatch_and_missing_transitive(tmp_path: Path) -> None:
    lock, cache = _setup(tmp_path)
    rows = {row["name"]: row for row in build_manifest(_profile(), lock, cache)["requirements"]}
    assert rows["platpkg"]["availability"] == "wrong_platform"
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
    lock, cache = _setup(tmp_path, private_companions=["wheelpkg"], permitted=["sdistpkg"])
    status = reconstruction_status(
        build_manifest(
            _profile(
                **{
                    "private_companions": ["wheelpkg"],
                    "permitted": ["sdistpkg"],
                }
            ),
            lock,
            cache,
        )
    )
    rows = {row["name"]: row for row in status["requirements"]}
    assert rows["wheelpkg"]["rights"] == "private-companion"
    assert rows["sdistpkg"]["rights"] == "redistribution-permitted"
    assert status["private_preservation_does_not_imply_public_redistribution"] is True
    assert status["offline_reconstruction_complete"] is False


def test_public_status_has_no_private_paths(tmp_path: Path) -> None:
    lock, cache = _setup(tmp_path)
    status = reconstruction_status(build_manifest(_profile(), lock, cache))
    text = json.dumps(status)
    assert str(tmp_path) not in text
    assert "/home/" not in text
    assert status["sanitization"] if "sanitization" in status else True


def test_write_sums_is_deterministic(tmp_path: Path) -> None:
    lock, cache = _setup(tmp_path)
    manifest = build_manifest(_profile(), lock, cache)
    first, second = tmp_path / "SHA256SUMS.a", tmp_path / "SHA256SUMS.b"
    assert write_sums(manifest, first) == 3  # cached artifacts plus the missing expected entry
    write_sums(manifest, second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    assert WHEEL in first.read_text(encoding="utf-8")


def test_offline_install_blocked_without_policy(tmp_path: Path) -> None:
    lock, cache = _setup(tmp_path)
    manifest = build_manifest(_profile(), lock, cache)
    assert offline_install_plan(manifest, cache)["reason"] == "incomplete_or_unpermitted_set"
    manifest["offline_install_permitted"] = True
    assert offline_install_plan(manifest, cache)["reason"] == "incomplete_or_unpermitted_set"


def test_offline_install_failure_is_reported(tmp_path: Path, monkeypatch) -> None:
    lock, cache = _setup(
        tmp_path, permitted=["wheelpkg", "sdistpkg", "platpkg", "missingpkg", "app"]
    )
    manifest = build_manifest(
        _profile(permitted=["wheelpkg", "sdistpkg", "platpkg", "missingpkg", "app"]), lock, cache
    )
    manifest["offline_install_permitted"] = True
    for row in manifest["requirements"]:
        row["availability"] = "available_verified"

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "ERROR: no matching distribution found"

    monkeypatch.setattr(
        "scripts.tools.dependency_cache_manifest.subprocess.run", lambda *a, **k: Failed()
    )
    result = offline_install_plan(manifest, cache)
    assert result["status"] == "failed"
    assert result["reason"] == "offline_install_failed"


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
