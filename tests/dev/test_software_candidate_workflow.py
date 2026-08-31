"""Policy tests for the credential-free build-once software-candidate workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _trigger(payload: dict[str, Any]) -> dict[str, Any]:
    # PyYAML 1.1 treats the unquoted workflow key ``on`` as boolean true.
    trigger = payload.get("on", payload.get(True))
    assert isinstance(trigger, dict)
    return trigger


def _steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = payload["jobs"]
    assert list(jobs) == ["build-candidate"]
    return jobs["build-candidate"]["steps"]


def test_workflow_is_reusable_single_job_and_least_privilege() -> None:
    text, workflow = _workflow()

    assert set(_trigger(workflow)) == {"workflow_call"}
    assert "workflow_dispatch" not in text
    assert workflow["permissions"] == {"contents": "read"}
    assert len(workflow["jobs"]) == 1
    job = workflow["jobs"]["build-candidate"]
    assert "permissions" not in job
    assert "environment" not in job
    assert all("runner." not in str(value) for value in job.get("env", {}).values())
    identity_run = next(step["run"] for step in _steps(workflow) if step.get("id") == "identity")
    assert "RUNNER_TEMP" in identity_run
    assert "GITHUB_ENV" in identity_run
    for name in ("CANDIDATE_ROOT", "DIST_DIR", "RAW_SBOM", "BUNDLE_DIR"):
        assert name in identity_run
    assert "secrets" not in text.lower()
    assert "id-token" not in text.lower()


def test_workflow_keeps_python_bytecode_outside_the_frozen_source() -> None:
    _text, workflow = _workflow()

    assert workflow["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


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
    assert '--repo-root "${GITHUB_WORKSPACE}"' in stage_run
    assert '--build-root "${BUILD_SOURCE}"' in stage_run
    assert '--source-sha "${GITHUB_SHA}"' in stage_run
    assert stage_index < build_index
    assert 'cd "${BUILD_SOURCE}"' in build_run
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
    assert 'bash "${BUILD_SOURCE}/scripts/validation/wheel_install_smoke.sh"' in smoke_run
    assert "bash scripts/validation/wheel_install_smoke.sh" not in smoke_run
    assert smoke_index < assemble_index
    assert '--repo-root "${GITHUB_WORKSPACE}"' in steps[assemble_index]["run"]


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

    assemble_run = next(run for _index, run in run_steps if " assemble" in run)
    for validator in (
        "version-alignment",
        "metadata",
        "archive-license",
        "wheel-install",
    ):
        assert f"--validated {validator}" in assemble_run
    assert assemble_run.count("--validated") == 4


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
    assert '--source-sha "${GITHUB_SHA}"' in source_checks[1][1]["run"]

    assemble = next(step["run"] for step in steps if " assemble" in step.get("run", ""))
    assert '--repo-root "${GITHUB_WORKSPACE}"' in assemble


def test_workflow_uploads_checked_bundle_once_and_exposes_artifact_identity() -> None:
    _text, workflow = _workflow()
    steps = _steps(workflow)
    upload_steps = [
        step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert len(upload_steps) == 1
    upload = upload_steps[0]
    assert upload["id"] == "upload"
    assert upload["with"]["path"].endswith("/bundle/")
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["compression-level"] == 0
    assert upload["with"]["overwrite"] is False

    outputs = workflow["jobs"]["build-candidate"]["outputs"]
    assert "steps.upload.outputs.artifact-id" in outputs["artifact-id"]
    assert "steps.upload.outputs.artifact-digest" in outputs["artifact-digest"]
    call_outputs = _trigger(workflow)["workflow_call"]["outputs"]
    assert set(call_outputs) == {
        "artifact-id",
        "artifact-digest",
        "artifact-name",
        "source-sha",
    }


def test_workflow_pins_actions_and_contains_no_publication_or_promotion_surface() -> None:
    text, workflow = _workflow()
    steps = _steps(workflow)
    uses = [str(step["uses"]) for step in steps if "uses" in step]

    assert len(uses) == len(ACTION_PINS)
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
        "workflow_dispatch",
        "download-artifact",
        "continue-on-error",
        "|| true",
    ):
        assert forbidden not in lowered
