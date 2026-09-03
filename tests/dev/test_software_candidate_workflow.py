"""Policy tests for the credential-free build-once software-candidate workflow."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts" / "dev" / "software_candidate_manifest.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "software-candidate.yml"
WHEEL_INSTALL_SMOKE = REPO_ROOT / "scripts" / "validation" / "wheel_install_smoke.sh"
ACTION_PINS = {
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/setup-python": "5fda3b95a4ea91299a34e894583c3862153e4b97",
    "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
}


def _workflow() -> tuple[str, dict[str, Any]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert isinstance(payload, dict)
    return text, payload


def _workflow_at(repo_root: Path) -> tuple[str, dict[str, Any]]:
    """Load the candidate workflow from a specific checkout."""
    workflow_path = repo_root / ".github" / "workflows" / "software-candidate.yml"
    text = workflow_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert isinstance(payload, dict)
    return text, payload


def _trigger(payload: dict[str, Any]) -> dict[str, Any]:
    # PyYAML 1.1 treats the unquoted workflow key ``on`` as boolean true.
    trigger = payload.get("on", payload.get(True))
    assert isinstance(trigger, dict)
    return trigger


def _steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = payload["jobs"]
    assert list(jobs) == ["build-candidate"]
    return jobs["build-candidate"]["steps"]


def test_workflow_is_directly_dispatchable_single_job_and_least_privilege() -> None:
    text, workflow = _workflow()

    assert set(_trigger(workflow)) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    assert len(workflow["jobs"]) == 1
    job = workflow["jobs"]["build-candidate"]
    assert "permissions" not in job
    assert "environment" not in job
    assert all("runner." not in str(value) for value in job.get("env", {}).values())
    identity_run = next(step["run"] for step in _steps(workflow) if step.get("id") == "identity")
    assert "RUNNER_TEMP" in identity_run
    assert "GITHUB_ENV" in identity_run
    for name in (
        "CANDIDATE_ROOT",
        "CANDIDATE_SOURCE",
        "MATERIALIZATION_REPORT",
        "BUILD_SOURCE",
        "DIST_DIR",
        "RAW_SBOM",
        "BUNDLE_DIR",
        "DEPENDENCY_REPORT",
        "RIGHTS_MANIFEST",
        "RIGHTS_DIR",
        "DIAGNOSTIC_ROOT",
        "STRICT_RIGHTS_LOG",
        "DEPENDENCY_LOG",
        "RIGHTS_ADMISSION_LOG",
        "STRICT_RIGHTS_EXIT",
        "DEPENDENCY_EXIT",
        "RIGHTS_ADMISSION_EXIT",
    ):
        assert name in identity_run
    assert "secrets" not in text.lower()
    assert "id-token" not in text.lower()


def test_workflow_keeps_python_bytecode_outside_the_frozen_source() -> None:
    _text, workflow = _workflow()

    assert workflow["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_workflow_bootstraps_one_isolated_pinned_helper_environment() -> None:
    """Every candidate helper uses one clean, reviewed dependency surface."""
    text, workflow = _workflow()
    assert workflow["env"]["PYTHONPATH"] == ""
    steps = _steps(workflow)
    bootstrap_index, bootstrap = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("name") == "Bootstrap isolated software-candidate helper environment"
    )
    bootstrap_run = bootstrap["run"]
    assert bootstrap["env"]["HELPER_ENV"] == "${{ runner.temp }}/robot-sf-software-candidate-helper"
    assert 'uv venv --python "${python_bin}" "${HELPER_ENV}"' in bootstrap_run
    assert 'uv pip install --python "${HELPER_ENV}/bin/python"' in bootstrap_run
    assert '"packaging==26.0"' in bootstrap_run
    assert '"pyyaml==6.0.3"' in bootstrap_run
    assert bootstrap_run.count('"packaging==26.0"') == 1
    assert bootstrap_run.count('"pyyaml==6.0.3"') == 1
    assert "SOFTWARE_CANDIDATE_PYTHON=%s/bin/python" in bootstrap_run

    helper_runs = [
        (index, step["run"])
        for index, step in enumerate(steps)
        if "software_candidate_manifest.py" in step.get("run", "")
    ]
    assert len(helper_runs) == 9
    assert all(
        '"${SOFTWARE_CANDIDATE_PYTHON}" scripts/dev/software_candidate_manifest.py' in run
        for _index, run in helper_runs
    )
    assert bootstrap_index < min(index for index, _run in helper_runs)
    assert "python scripts/dev/software_candidate_manifest.py" not in text

    version_run = next(
        step["run"] for step in steps if step.get("name") == "Validate version alignment"
    )
    assert '"${SOFTWARE_CANDIDATE_PYTHON}" scripts/dev/check_version_alignment.py' in version_run
    license_run = next(
        step["run"]
        for step in steps
        if step.get("name") == "Validate candidate archive members and strict rights"
    )
    assert (
        '"${SOFTWARE_CANDIDATE_PYTHON}" scripts/tools/check_distribution_licenses.py' in license_run
    )
    assert "uv run --no-project --with" not in license_run


def test_clean_runner_bootstraps_and_checks_the_cloned_helper_offline(tmp_path: Path) -> None:
    """The real bootstrap reaches semantic validation from an offline source clone."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("uv is required to exercise the clean-runner bootstrap contract")
    clean_repo = tmp_path / "clean-source"
    clone_env = os.environ.copy()
    # This fixture validates the clean helper checkout, not availability of unrelated LFS blobs.
    clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
    clone_result = subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(clean_repo)],
        check=False,
        capture_output=True,
        text=True,
        env=clone_env,
    )
    assert clone_result.returncode == 0, (
        "git clone failed with exit code "
        f"{clone_result.returncode}: {clone_result.args!r}\n"
        f"stdout:\n{clone_result.stdout}\n"
        f"stderr:\n{clone_result.stderr}"
    )
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clean_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    workflow_text, workflow = _workflow_at(clean_repo)
    bootstrap = next(
        step
        for step in _steps(workflow)
        if step.get("name") == "Bootstrap isolated software-candidate helper environment"
    )
    bootstrap_run = bootstrap["run"]
    assert "uv run" not in bootstrap_run
    clone_helper = clean_repo / "scripts" / "dev" / "software_candidate_manifest.py"
    assert clone_helper.is_file()
    assert clone_helper != HELPER
    assert "software_candidate_manifest.py" in workflow_text

    python312 = shutil.which("python3.12")
    if python312 is None:
        pytest.fail("python3.12 is required to match the GitHub Actions bootstrap runtime")
    cache_dir = os.environ.get("UV_CACHE_DIR", "")
    if cache_dir:
        # The child runs from tmp_path; anchor relative configured paths before handing them over.
        cache_dir = str(Path(cache_dir).resolve())
    if not cache_dir:
        cache_dir = subprocess.run(
            [uv, "cache", "dir"],
            check=True,
            capture_output=True,
            text=True,
            env={
                key: value
                for key, value in os.environ.items()
                if key != "XDG_CACHE_HOME" and not key.startswith("UV_") and key != "UV"
            },
        ).stdout.strip()
    assert cache_dir
    tool_bin = tmp_path / "tool-bin"
    tool_bin.mkdir()
    (tool_bin / "python").symlink_to(python312)
    (tool_bin / "uv").symlink_to(uv)

    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONUSERBASE",
            "UV_PROJECT_ENVIRONMENT",
            "VIRTUAL_ENV",
        }
        and not key.startswith("UV_")
        and key != "UV"
    }
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
            "UV_NO_CONFIG": "1",
            "UV_OFFLINE": "1",
            "UV_CACHE_DIR": cache_dir,
            "HELPER_ENV": str(tmp_path / "helper-env"),
            "GITHUB_ENV": str(tmp_path / "github-env"),
            "PATH": f"{tool_bin}{os.pathsep}{os.environ['PATH']}",
        }
    )
    bootstrap_result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", bootstrap_run],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert bootstrap_result.returncode == 0, (
        f"bootstrap failed with exit code {bootstrap_result.returncode}:\n"
        f"stdout:\n{bootstrap_result.stdout}\n"
        f"stderr:\n{bootstrap_result.stderr}"
    )
    github_env = dict(
        line.split("=", maxsplit=1)
        for line in Path(env["GITHUB_ENV"]).read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    helper_python = Path(github_env["SOFTWARE_CANDIDATE_PYTHON"])
    assert helper_python == Path(env["HELPER_ENV"]) / "bin" / "python"
    assert helper_python.is_file()

    helper_command = [
        str(helper_python),
        str(clone_helper),
        "check-source",
        "--repo-root",
        str(clean_repo),
        "--source-sha",
        source_sha,
    ]
    assert helper_command[1] == str(clone_helper)
    assert helper_command[1] != str(HELPER)
    result = subprocess.run(
        helper_command,
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"PASS: source identity is clean and exact at {source_sha}" in result.stdout
    assert "No module named 'yaml'" not in result.stderr


def test_workflow_builds_only_from_staged_external_exact_commit() -> None:
    _text, workflow = _workflow()
    steps = _steps(workflow)
    identity_run = next(step["run"] for step in steps if step.get("id") == "identity")
    stage_index, stage_run = next(
        (index, step["run"])
        for index, step in enumerate(steps)
        if "software_candidate_manifest.py stage-build-source" in step.get("run", "")
    )
    build_index, build_run = next(
        (index, step["run"])
        for index, step in enumerate(steps)
        if "uv build" in step.get("run", "")
    )

    assert "BUILD_SOURCE" in identity_run
    assert '--repo-root "${CANDIDATE_SOURCE}"' in stage_run
    assert '--build-root "${BUILD_SOURCE}"' in stage_run
    assert '--source-sha "${CANDIDATE_SOURCE_SHA}"' in stage_run
    assert stage_index < build_index
    assert 'cd "${BUILD_SOURCE}"' in build_run
    assert 'SETUPTOOLS_SCM_PRETEND_VERSION="${CANDIDATE_VERSION}"' in build_run
    assert 'uv build --out-dir "${DIST_DIR}"' in build_run


def test_workflow_confines_wheel_smoke_side_effects_to_disposable_source() -> None:
    _text, workflow = _workflow()
    steps = _steps(workflow)
    wrapper_text = WHEEL_INSTALL_SMOKE.read_text(encoding="utf-8")
    smoke_index, smoke_run = next(
        (index, step["run"])
        for index, step in enumerate(steps)
        if "wheel_install_smoke.sh" in step.get("run", "")
    )
    assemble_index = next(
        index
        for index, step in enumerate(steps)
        if "software_candidate_manifest.py assemble" in step.get("run", "")
    )

    assert 'VALIDATION_DIR="${REPO_ROOT}/output/validation"' in wrapper_text
    assert 'mkdir -p "${VALIDATION_DIR}"' in wrapper_text
    assert "programmatic-core-map" in wrapper_text
    materialize = next(step["run"] for step in steps if step.get("id") == "materialize")
    assert "software_candidate_manifest.py materialize-source" in materialize
    assert '--candidate-root "${CANDIDATE_SOURCE}"' in materialize
    assert '--report "${MATERIALIZATION_REPORT}"' in materialize
    assert 'bash "${BUILD_SOURCE}/scripts/validation/wheel_install_smoke.sh"' in smoke_run
    assert "bash scripts/validation/wheel_install_smoke.sh" not in smoke_run
    assert smoke_index < assemble_index
    assert '--repo-root "${GITHUB_WORKSPACE}"' in steps[assemble_index]["run"]
    assert '--candidate-source-root "${CANDIDATE_SOURCE}"' in steps[assemble_index]["run"]
    assert '--materialization-report "${MATERIALIZATION_REPORT}"' in steps[assemble_index]["run"]


def test_workflow_builds_once_then_only_validates_and_admits_same_dist_bytes() -> None:
    _text, workflow = _workflow()
    steps = _steps(workflow)
    run_steps = [(index, step.get("run", "")) for index, step in enumerate(steps)]
    build_steps = [(index, run) for index, run in run_steps if re.search(r"\buv build\b", run)]

    assert len(build_steps) == 1
    build_index, build_run = build_steps[0]
    assert 'uv build --out-dir "${DIST_DIR}"' in build_run
    assert all("uv build" not in run for index, run in run_steps if index > build_index)
    assert all("python -m build" not in run for _index, run in run_steps)

    required_after_build = (
        "check_version_alignment.py",
        "twine check --strict",
        "check_distribution_licenses.py",
        "wheel_install_smoke.sh",
        "uv export",
        "software_candidate_manifest.py assemble",
        "software_candidate_manifest.py verify",
        "check_dependency_license_inventory.py",
    )
    positions = []
    for command in required_after_build:
        matches = [index for index, run in run_steps if command in run]
        assert len(matches) == 1, command
        positions.append(matches[0])
    assert build_index < positions[0]
    assert positions == sorted(positions)
    assert all(
        "${DIST_DIR}" in run
        for _index, run in run_steps
        if any(
            owner in run
            for owner in ("twine check", "check_distribution_licenses.py", "wheel_install_smoke.sh")
        )
    )
    strict_license_run = next(
        run for _index, run in run_steps if "check_distribution_licenses.py" in run
    )
    assert "--strict-asset-rights" in strict_license_run
    assert '--repo-root "${BUILD_SOURCE}"' in strict_license_run
    assert "software_candidate_asset_rights.v1.json" in strict_license_run

    assemble_run = next(run for _index, run in run_steps if " assemble" in run)
    for validator in (
        "version-alignment",
        "metadata",
        "archive-license",
        "wheel-install",
    ):
        assert f"--validated {validator}" in assemble_run
    assert assemble_run.count("--validated") == 4

    dependency_index, dependency_run = next(
        (index, run) for index, run in run_steps if "check_dependency_license_inventory.py" in run
    )
    assert dependency_run.startswith("set +e\n")
    assert '"${SOFTWARE_CANDIDATE_PYTHON}" \\' in dependency_run
    assert '"${BUILD_SOURCE}/scripts/tools/' in dependency_run
    assert 'python "${BUILD_SOURCE}/scripts/tools/check_dependency_license_inventory.py"' not in (
        dependency_run
    )
    assert '--repo-root "${BUILD_SOURCE}"' in dependency_run
    assert '--candidate-bundle "${BUNDLE_DIR}"' in dependency_run
    assert '--output "${DEPENDENCY_REPORT}"' in dependency_run
    assert "--profile all" in dependency_run
    assert "--fail-on-unresolved" in dependency_run
    assert dependency_index == positions[-1]
    assert "--profile core" not in dependency_run

    upload_index = next(index for index, step in enumerate(steps) if step.get("id") == "upload")
    rights_index, rights_run = next(
        (index, run)
        for index, run in run_steps
        if "software_candidate_manifest.py rights-admission" in run
    )
    assert dependency_index < upload_index < rights_index
    assert '--dependency-report "${DEPENDENCY_REPORT}"' in rights_run
    assert '--candidate-artifact-id "${{ steps.upload.outputs.artifact-id }}"' in rights_run

    sbom_run = next(run for _index, run in run_steps if "uv export" in run)
    assert "--extra all" in sbom_run


def test_workflow_checks_hermetic_source_identity_around_the_only_build() -> None:
    _text, workflow = _workflow()
    steps = _steps(workflow)
    source_checks = [
        (index, step)
        for index, step in enumerate(steps)
        if "software_candidate_manifest.py check-source" in step.get("run", "")
    ]
    assert [step["name"] for _index, step in source_checks] == [
        "Require hermetic exact source identity before build",
        "Prove disposable build source is the exact commit",
        "Reject any source workspace mutation by the build",
    ]
    build_index = next(
        index for index, step in enumerate(steps) if "uv build" in step.get("run", "")
    )
    assert source_checks[0][0] < source_checks[1][0] < build_index < source_checks[2][0]
    for _index, step in (source_checks[0], source_checks[2]):
        assert '--repo-root "${GITHUB_WORKSPACE}"' in step["run"]
        assert '--source-sha "${GITHUB_SHA}"' in step["run"]
    assert '--repo-root "${BUILD_SOURCE}"' in source_checks[1][1]["run"]
    assert '--source-sha "${CANDIDATE_SOURCE_SHA}"' in source_checks[1][1]["run"]

    assemble = next(step["run"] for step in steps if " assemble" in step.get("run", ""))
    assert '--repo-root "${GITHUB_WORKSPACE}"' in assemble


def test_workflow_uploads_checked_bundle_once_and_exposes_artifact_identity() -> None:
    _text, workflow = _workflow()
    steps = _steps(workflow)
    upload_steps = [
        step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert len(upload_steps) == 3
    upload = next(step for step in upload_steps if step.get("id") == "upload")
    assert upload["id"] == "upload"
    assert upload["with"]["path"].endswith("/bundle/")
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["compression-level"] == 0
    assert upload["with"]["overwrite"] is False
    rights_upload = next(step for step in upload_steps if step.get("id") == "upload-rights")
    assert rights_upload["with"]["path"].endswith("/rights-admission/")
    assert rights_upload["with"]["if-no-files-found"] == "error"
    assert rights_upload["with"]["compression-level"] == 0
    assert rights_upload["with"]["overwrite"] is False
    rejected_upload = next(step for step in upload_steps if step.get("id") == "upload-rejected")
    assert rejected_upload["with"]["name"] == "${{ env.REJECTED_DIAGNOSTIC_ARTIFACT_NAME }}"
    assert rejected_upload["with"]["path"].endswith("/rejected-diagnostic/")
    assert rejected_upload["with"]["if-no-files-found"] == "error"
    assert rejected_upload["with"]["compression-level"] == 0
    assert rejected_upload["with"]["overwrite"] is False
    assert "always()" in rejected_upload["if"]

    assert "workflow_call" not in _trigger(workflow)
    assert "outputs" not in workflow["jobs"]["build-candidate"]


def test_workflow_captures_strict_gate_failures_for_rejected_diagnostics() -> None:
    _text, workflow = _workflow()
    steps = _steps(workflow)
    by_id = {step["id"]: step for step in steps if "id" in step}

    strict_rights = by_id["strict-rights"]["run"]
    assert strict_rights.startswith("set +e\n")
    assert '"${STRICT_RIGHTS_LOG}"' in strict_rights
    assert "STRICT_RIGHTS_EXIT" in strict_rights
    assert 'exit "${status}"' in strict_rights

    strict_dependency = by_id["strict-dependency"]["run"]
    assert strict_dependency.startswith("set +e\n")
    assert '"${DEPENDENCY_LOG}"' in strict_dependency
    assert "DEPENDENCY_EXIT" in strict_dependency
    assert 'exit "${status}"' in strict_dependency

    rights_admission = by_id["rights-admission"]["run"]
    assert rights_admission.startswith("set +e\n")
    assert '"${RIGHTS_ADMISSION_LOG}"' in rights_admission
    assert "RIGHTS_ADMISSION_EXIT" in rights_admission
    assert 'exit "${status}"' in rights_admission

    diagnostic = next(
        step for step in steps if step.get("name") == "Assemble rejected diagnostic evidence"
    )
    assert "always()" in diagnostic["if"]
    assert "rejected-diagnostic" in diagnostic["run"]
    assert '"${STRICT_RIGHTS_EXIT}"' in diagnostic["run"]
    assert '"${DEPENDENCY_EXIT}"' in diagnostic["run"]
    assert '"${RIGHTS_ADMISSION_EXIT}"' in diagnostic["run"]

    # The accepted path still uploads the ordinary candidate and rights receipt;
    # the diagnostic upload is conditional on a recorded strict-gate failure.
    assert "if" not in by_id["upload"]
    assert "if" not in by_id["upload-rights"]
    rejected_upload = next(
        step for step in steps if step.get("name") == "Upload rejected diagnostic evidence"
    )
    assert "always()" in rejected_upload["if"]


def test_workflow_pins_actions_and_contains_no_publication_or_promotion_surface() -> None:
    text, workflow = _workflow()
    steps = _steps(workflow)
    uses = [str(step["uses"]) for step in steps if "uses" in step]

    assert len(uses) == len(ACTION_PINS) + 2
    for use in uses:
        owner, digest = use.split("@", maxsplit=1)
        assert re.fullmatch(r"[0-9a-f]{40}", digest)
        assert digest == ACTION_PINS[owner]
    checkout = next(
        step for step in steps if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["fetch-depth"] == 0

    lowered = text.lower()
    for forbidden in (
        "twine upload",
        "testpypi",
        "pypi.org",
        "gh release",
        "github release",
        "zenodo",
        "id-token",
        "download-artifact",
        "continue-on-error",
        "|| true",
    ):
        assert forbidden not in lowered
