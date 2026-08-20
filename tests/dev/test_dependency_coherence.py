"""Behavioral tests for the cross-lock dependency coherence contract."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import jsonschema

from scripts.dev.check_dependency_coherence import (
    compare_lock_resolution,
    evaluate_coherence,
    load_manifest,
    validate_profile_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "scripts/validation/dependency_coherence.v1.json"
SCHEMA_PATH = REPO_ROOT / "scripts/validation/dependency_coherence.v1.schema.json"


def test_live_coherence_manifest_matches_its_schema() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(manifest)

    loaded = load_manifest(MANIFEST_PATH)
    profile_manifest = json.loads(
        (REPO_ROOT / "scripts/validation/dependency_license_profiles.v1.json").read_text(
            encoding="utf-8"
        )
    )
    profile_ids = validate_profile_manifest(profile_manifest)
    assert {
        profile_id for owner in loaded["owners"] for profile_id in owner["profile_ids"]
    } <= profile_ids


def _manifest() -> dict:
    return {
        "schema_version": "robot-sf.dependency-coherence.v1",
        "profile_manifest": "profiles.json",
        "resolver": {"name": "uv", "version": "0.11.21", "lock_mode": "check"},
        "python_profiles": [
            {"id": "linux-py311", "python": "3.11", "required": True},
            {"id": "linux-py313", "python": "3.13", "required": True},
        ],
        "owners": [
            {
                "id": "root",
                "declaration": "pyproject.toml",
                "lockfile": "uv.lock",
                "root_package": "robot-sf",
                "profile_ids": ["root"],
                "coupling": "independent",
            },
            {
                "id": "fast-pysf",
                "declaration": "fast-pysf/pyproject.toml",
                "lockfile": "fast-pysf/uv.lock",
                "root_package": "pysocialforce",
                "profile_ids": ["fast-pysf"],
                "coupling": "independent",
            },
        ],
    }


def _project(name: str, dependency: str) -> str:
    return f'''[project]\nname = "{name}"\nversion = "1.0"\nrequires-python = ">=3.11"\ndependencies = ["{dependency}"]\n'''


def _lock(root: str, dependency: str, version: str = "1.0.0") -> str:
    return f'''version = 1\nrevision = 3\nrequires-python = ">=3.11"\n\n[[package]]\nname = "{root}"\nsource = {{ editable = "." }}\ndependencies = [{{ name = "{dependency}" }}]\n\n[[package]]\nname = "{dependency}"\nversion = "{version}"\nsource = {{ registry = "https://pypi.org/simple" }}\n'''


def test_root_only_declaration_and_lock_update_is_coherent() -> None:
    base_files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": _lock("robot-sf", "alpha"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }
    head_files = {
        **base_files,
        "pyproject.toml": _project("robot-sf", "beta>=1"),
        "uv.lock": _lock("robot-sf", "beta"),
    }

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=["pyproject.toml", "uv.lock"],
        run_profile_checks=False,
    )

    assert report["status"] == "coherent"
    assert report["scope"] == "root-only"
    assert report["required_lockfiles"] == ["uv.lock"]


def test_declaration_without_its_lock_is_missing_lock_update() -> None:
    base_files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": _lock("robot-sf", "alpha"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }
    head_files = {**base_files, "pyproject.toml": _project("robot-sf", "beta>=1")}

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=["pyproject.toml"],
        run_profile_checks=False,
    )

    assert report["status"] == "missing_lock_update"
    assert report["required_lockfiles"] == ["uv.lock"]


def test_changed_lock_with_stale_root_dependency_edges_is_a_mismatch() -> None:
    base_files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": _lock("robot-sf", "alpha"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }
    head_files = {
        **base_files,
        "pyproject.toml": _project("robot-sf", "beta>=1"),
    }

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=["pyproject.toml", "uv.lock"],
        run_profile_checks=False,
    )

    assert report["status"] == "declaration_lock_mismatch"
    assert any("root dependencies disagree" in reason for reason in report["reasons"])


def test_supported_python_range_change_is_material_resolution_evidence() -> None:
    base_files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": _lock("robot-sf", "alpha"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }
    head_project = _project("robot-sf", "alpha>=1").replace(
        'requires-python = ">=3.11"', 'requires-python = ">=3.11,<3.14"'
    )
    head_lock = _lock("robot-sf", "alpha").replace(
        'requires-python = ">=3.11"', 'requires-python = ">=3.11,<3.14"'
    )
    head_files = {**base_files, "pyproject.toml": head_project, "uv.lock": head_lock}

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=["pyproject.toml", "uv.lock"],
        run_profile_checks=False,
    )

    assert report["status"] == "coherent"
    assert report["classification"] == "material_resolution"
    assert report["material_fields"] == ["requires-python"]


def test_profile_checks_use_the_pinned_resolver_and_affected_owner_only(tmp_path: Path) -> None:
    files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": _lock("robot-sf", "alpha"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }
    calls: list[tuple[str, ...]] = []

    def runner(args: tuple[str, ...] | list[str], _cwd: Path) -> SimpleNamespace:
        calls.append(tuple(args))
        stdout = "uv 0.11.21\n" if args[1:] == ["--version"] else "ok\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=files,
        head_files=files,
        changed_files=["pyproject.toml", "uv.lock"],
        runner=runner,
        repo_root=tmp_path,
    )

    assert report["status"] == "coherent"
    assert report["profiles"] == [
        {"owner": "root", "python": "3.11", "status": "checked"},
        {"owner": "root", "python": "3.13", "status": "checked"},
    ]
    lock_calls = [call for call in calls if "lock" in call]
    assert len(lock_calls) == 2
    assert all("fast-pysf" not in call for call in lock_calls)
    assert all("--check" in call and "--python" in call for call in lock_calls)


def test_fast_pysf_only_update_has_a_standalone_owner() -> None:
    base_files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": _lock("robot-sf", "alpha"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }
    head_files = {
        **base_files,
        "fast-pysf/pyproject.toml": _project("pysocialforce", "beta>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "beta"),
    }

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=["fast-pysf/pyproject.toml", "fast-pysf/uv.lock"],
        run_profile_checks=False,
    )

    assert report["status"] == "coherent"
    assert report["scope"] == "fast-pysf-only"
    assert report["required_lockfiles"] == ["fast-pysf/uv.lock"]


def test_shared_declaration_reports_independent_lock_owners() -> None:
    base_files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": _lock("robot-sf", "alpha"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }
    head_files = {
        "pyproject.toml": _project("robot-sf", "beta>=1"),
        "uv.lock": _lock("robot-sf", "beta"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "beta>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "beta"),
    }

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=[
            "pyproject.toml",
            "uv.lock",
            "fast-pysf/pyproject.toml",
            "fast-pysf/uv.lock",
        ],
        run_profile_checks=False,
    )

    assert report["status"] == "coherent"
    assert report["scope"] == "shared-declaration-independently-resolved"
    assert report["required_lockfiles"] == ["fast-pysf/uv.lock", "uv.lock"]


def test_workspace_member_coupling_is_explicit() -> None:
    manifest = deepcopy(_manifest())
    for owner in manifest["owners"]:
        owner["coupling"] = "workspace"
    base_files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": _lock("robot-sf", "alpha"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }
    head_files = {
        "pyproject.toml": _project("robot-sf", "beta>=1"),
        "uv.lock": _lock("robot-sf", "beta"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "beta>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "beta"),
    }

    report = evaluate_coherence(
        manifest=manifest,
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=[
            "pyproject.toml",
            "uv.lock",
            "fast-pysf/pyproject.toml",
            "fast-pysf/uv.lock",
        ],
        run_profile_checks=False,
    )

    assert report["status"] == "coherent"
    assert report["scope"] == "workspace/member-coupling"


def test_incompatible_shared_exact_ranges_are_a_conflict() -> None:
    base_files = {
        "pyproject.toml": _project("robot-sf", "shared==0"),
        "uv.lock": _lock("robot-sf", "shared", "0"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "shared==0"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "shared", "0"),
    }
    head_files = {
        "pyproject.toml": _project("robot-sf", "shared==2"),
        "uv.lock": _lock("robot-sf", "shared", "2"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "shared==1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "shared", "1"),
    }

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=[
            "pyproject.toml",
            "uv.lock",
            "fast-pysf/pyproject.toml",
            "fast-pysf/uv.lock",
        ],
        run_profile_checks=False,
    )

    assert report["status"] == "conflict"
    assert any("shared" in reason for reason in report["reasons"])


def test_incompatible_shared_bounds_are_a_conflict() -> None:
    base_files = {
        "pyproject.toml": _project("robot-sf", "shared>=1"),
        "uv.lock": _lock("robot-sf", "shared", "1"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "shared>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "shared", "1"),
    }
    head_files = {
        "pyproject.toml": _project("robot-sf", "shared<2"),
        "uv.lock": _lock("robot-sf", "shared", "1"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "shared>=2"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "shared", "2"),
    }

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=[
            "pyproject.toml",
            "uv.lock",
            "fast-pysf/pyproject.toml",
            "fast-pysf/uv.lock",
        ],
        run_profile_checks=False,
    )

    assert report["status"] == "conflict"


def test_transitive_only_lock_change_is_distinguished_from_declaration_change() -> None:
    base_files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": _lock("robot-sf", "alpha"),
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }
    transitive_row = '\n[[package]]\nname = "helper"\nversion = "1.0.0"\n'
    head_files = {**base_files, "uv.lock": base_files["uv.lock"] + transitive_row}

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=["uv.lock"],
        run_profile_checks=False,
    )

    assert report["status"] == "coherent"
    assert report["scope"] == "transitive-only"
    assert report["classification"] == "transitive_only"


def test_marker_only_lock_churn_is_reported_without_material_resolution_change() -> None:
    base_lock = _lock("robot-sf", "alpha")
    head_lock = base_lock.replace(
        'name = "alpha"\nversion = "1.0.0"',
        'name = "alpha"\nversion = "1.0.0"\nresolution-markers = ["python_full_version >= \'3.11\'"]',
    )
    base_files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": base_lock,
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }
    head_files = {**base_files, "uv.lock": head_lock}

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=base_files,
        head_files=head_files,
        changed_files=["uv.lock"],
        run_profile_checks=False,
    )

    assert report["status"] == "coherent"
    assert report["classification"] == "lock_normalization"
    assert report["material_packages"] == []
    profile = report["lock_resolution"]["uv.lock"]["profiles"][0]
    assert profile["state"] == "lock_normalization"
    assert profile["before_resolution_digest"] == profile["after_resolution_digest"]


def test_unsupported_profile_marker_is_not_equivalence_evidence() -> None:
    base_lock = _lock("robot-sf", "alpha")
    head_lock = base_lock.replace(
        'name = "alpha"\nversion = "1.0.0"',
        'name = "alpha"\nversion = "1.0.0"\nresolution-markers = ["unsupported_marker == \'x\'"]',
    )
    files = {
        "pyproject.toml": _project("robot-sf", "alpha>=1"),
        "uv.lock": base_lock,
        "fast-pysf/pyproject.toml": _project("pysocialforce", "alpha>=1"),
        "fast-pysf/uv.lock": _lock("pysocialforce", "alpha"),
    }

    report = evaluate_coherence(
        manifest=_manifest(),
        profile_manifest={"profiles": [{"id": "root"}, {"id": "fast-pysf"}]},
        base_files=files,
        head_files={**files, "uv.lock": head_lock},
        changed_files=["uv.lock"],
        run_profile_checks=False,
    )

    assert report["status"] == "profile_unavailable"


def test_profile_evidence_retains_digests_predicates_and_artifact_identity() -> None:
    base_lock = _lock("robot-sf", "alpha")
    head_lock = base_lock.replace(
        'version = "1.0.0"\nsource = { registry = "https://pypi.org/simple" }',
        'version = "2.0.0"\nsource = { registry = "https://pypi.org/simple" }\n'
        'sdist = { url = "https://example.invalid/alpha-2.0.0.tar.gz", '
        'hash = "sha256:head", size = 10 }',
    )

    report = compare_lock_resolution(
        base_lock,
        head_lock,
        [{"id": "linux-py311", "python": "3.11", "required": True}],
    )

    profile = report["profiles"][0]
    assert profile["environment"]["python_version"] == "3.11"
    assert profile["before_resolution_digest"] != profile["after_resolution_digest"]
    assert profile["before_closure_digest"] == profile["after_closure_digest"]
    assert profile["state"] == "material"
    assert profile["selected_identities"]["alpha"]["after"][0]["version"] == "2.0.0"
    assert profile["selected_identities"]["alpha"]["after"][0]["sdist"]["hash"] == "sha256:head"
