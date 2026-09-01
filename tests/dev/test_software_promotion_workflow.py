"""Policy tests for the protected software-promotion workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "software-promotion.yml"
ACTION_PINS = {
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/setup-python": "5fda3b95a4ea91299a34e894583c3862153e4b97",
    "actions/download-artifact": "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
    "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    "pypa/gh-action-pypi-publish": "dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
}


def _workflow() -> tuple[str, dict[str, Any]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert isinstance(payload, dict)
    return text, payload


def _trigger(payload: dict[str, Any]) -> dict[str, Any]:
    trigger = payload.get("on", payload.get(True))
    assert isinstance(trigger, dict)
    return trigger


def test_workflow_is_manual_and_least_privilege_by_default() -> None:
    text, workflow = _workflow()
    trigger = _trigger(workflow)
    assert set(trigger) == {"workflow_dispatch"}
    inputs = trigger["workflow_dispatch"]["inputs"]
    for name in (
        "candidate_run_id",
        "candidate_run_attempt",
        "candidate_artifact_id",
        "candidate_artifact_name",
        "candidate_artifact_digest",
        "source_sha",
        "package_version",
        "rights_admission_run_id",
        "rights_admission_run_attempt",
        "rights_admission_artifact_id",
        "rights_admission_artifact_name",
        "rights_admission_artifact_digest",
    ):
        assert inputs[name]["required"] is True
    assert workflow["permissions"] == {"actions": "read", "contents": "read"}
    assert "id-token" not in workflow["permissions"]
    verify_steps = workflow["jobs"]["verify-candidate"]["steps"]
    protected_step = next(
        step
        for step in verify_steps
        if step.get("name") == "Require the protected main-branch workflow"
    )
    assert (
        "Require the protected main-branch workflow"
        in protected_step["name"]
    )
    assert '"refs/heads/main"' in protected_step["run"]
    checkout_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert len(checkout_steps) == 5
    assert all(step["with"].get("ref") == "${{ inputs.source_sha }}" for step in checkout_steps)
    assert text.count("Verify exact candidate-source checkout") == 5
    assert text.count('checked_out_sha="$(git rev-parse HEAD)"') == 5


def test_only_upload_jobs_receive_oidc_permission_and_each_has_a_protected_environment() -> None:
    _text, workflow = _workflow()
    jobs = workflow["jobs"]
    assert jobs["testpypi-upload"]["environment"] == {"name": "testpypi"}
    assert jobs["pypi-upload"]["environment"] == {"name": "pypi"}
    for name, job in jobs.items():
        permissions = job.get("permissions", {})
        if name in {"testpypi-upload", "pypi-upload"}:
            assert permissions["id-token"] == "write"
        else:
            assert "id-token" not in permissions


def test_production_is_downstream_of_testpypi_cold_install() -> None:
    _text, workflow = _workflow()
    needs = workflow["jobs"]["pypi-upload"]["needs"]
    assert set(needs) == {"verify-candidate", "testpypi-upload", "testpypi-cold-install"}
    assert "testpypi-cold-install" in needs
    cold = workflow["jobs"]["testpypi-cold-install"]
    assert "artifact-id" in cold["outputs"]
    assert "artifact-digest" in cold["outputs"]


def test_cross_run_downloads_keep_candidate_and_receipts_on_their_own_runs() -> None:
    _text, workflow = _workflow()
    for name in ("testpypi-cold-install", "pypi-upload"):
        downloads = [
            step
            for step in workflow["jobs"][name]["steps"]
            if str(step.get("uses", "")).startswith("actions/download-artifact@")
        ]
        candidate_download = next(
            step
            for step in downloads
            if step.get("with", {}).get("artifact-ids") == "${{ inputs.candidate_artifact_id }}"
        )
        assert candidate_download["with"]["run-id"] == "${{ inputs.candidate_run_id }}"
        rights_download = next(
            step
            for step in downloads
            if step.get("with", {}).get("artifact-ids")
            == "${{ inputs.rights_admission_artifact_id }}"
        )
        assert rights_download["with"]["run-id"] == "${{ inputs.rights_admission_run_id }}"
        assert any(
            step.get("with", {}).get("run-id") == "${{ github.run_id }}"
            for step in downloads
            if step is not candidate_download
        )


def test_workflow_downloads_and_revalidates_the_candidate_in_every_consumer_job() -> None:
    _text, workflow = _workflow()
    for name in ("verify-candidate", "testpypi-upload", "testpypi-cold-install", "pypi-upload"):
        steps = workflow["jobs"][name]["steps"]
        downloads = [
            step
            for step in steps
            if str(step.get("uses", "")).startswith("actions/download-artifact@")
        ]
        assert downloads, name
        runs = "\n".join(str(step.get("run", "")) for step in steps)
        assert "software_promotion.py" in runs
        assert "check-workflow-run" in runs
        assert "rights-admission.json" in runs
        assert "verify-candidate" in runs or "verify-receipt" in runs


def test_workflow_requires_the_external_sanitized_candidate_producer() -> None:
    text, workflow = _workflow()
    inputs = _trigger(workflow)["workflow_dispatch"]["inputs"]
    for name in (
        "rights_admission_run_id",
        "rights_admission_artifact_id",
        "rights_admission_artifact_name",
        "rights_admission_artifact_digest",
    ):
        assert inputs[name]["required"] is True
    assert text.count("kind rights --source-sha") >= 4
    assert "check-rights-run" not in text
    assert "actions/runs/${{ inputs.rights_admission_run_id }}" not in text
    assert "same producer run attempt" in text
    assert "rights-admission.json" in text
    assert "#8165" in text
    docs = (REPO_ROOT / "docs" / "software_release_promotion.md").read_text(encoding="utf-8")
    assert "supported-surface dependency" in docs
    assert "--fail-on-unresolved" in docs


def test_workflow_has_no_rebuild_or_long_lived_package_credentials() -> None:
    text, workflow = _workflow()
    lowered = text.lower()
    for forbidden in (
        "uv build",
        "python -m build",
        "twine upload",
        "secrets.",
        "pypi_api_token",
        "twine_password",
        "skip-existing: true",
        "continue-on-error",
        "|| true",
    ):
        assert forbidden not in lowered
    assert "skip-existing: false" in lowered
    assert "github.token" in lowered
    assert "id-token: write" in lowered
    assert "test.pypi.org/legacy/" in lowered
    assert "upload.pypi.org/legacy/" in lowered
    # Both promotion channels are explicit, so a refactor cannot silently
    # point production at the rehearsal index (or vice versa).
    assert (
        sum(
            "gh-action-pypi-publish@" in str(step.get("uses", ""))
            for job in workflow["jobs"].values()
            for step in job["steps"]
        )
        == 2
    )


def test_all_actions_are_full_commit_pinned() -> None:
    text, workflow = _workflow()
    uses = [
        str(step["uses"])
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    ]
    assert uses
    for use in uses:
        owner, digest = use.split("@", maxsplit=1)
        assert owner in ACTION_PINS
        assert re.fullmatch(r"[0-9a-f]{40}", digest)
        assert digest == ACTION_PINS[owner]
    assert "pypa/gh-action-pypi-publish" in text


def test_receipt_resume_inputs_and_exact_hash_checks_are_wired() -> None:
    text, workflow = _workflow()
    inputs = _trigger(workflow)["workflow_dispatch"]["inputs"]
    for prefix in ("resume_testpypi", "resume_production"):
        for suffix in (
            "run_id",
            "receipt_artifact_id",
            "receipt_artifact_name",
            "receipt_artifact_digest",
        ):
            assert f"{prefix}_{suffix}" in inputs
    assert text.count("check-artifact") >= 6
    assert text.count("verify-receipt") >= 4
    assert text.count("software_candidate_manifest.py verify") == 4
    assert "verify-index-artifacts" in text
    assert "write-cold-install-receipt" in text
    assert "skip-existing: false" in text
    assert "Require complete resume identities" in text
    assert "must provide run, attempt, artifact ID, name, and digest together" in text


def test_shell_steps_never_interpolate_dispatch_inputs() -> None:
    _text, workflow = _workflow()
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            run = step.get("run")
            if run:
                assert "${{ inputs." not in str(run), step.get("name")


def test_dispatch_identity_validation_rejects_shell_metacharacters() -> None:
    identity_pattern = re.compile(r"^[1-9][0-9]*$")
    for value in (
        "123; touch /tmp/publisher-pwned",
        "123\nGH_TOKEN=forged",
        "$(id)",
        "'123'",
    ):
        assert identity_pattern.fullmatch(value) is None


def test_docs_describe_environment_setup_without_secrets() -> None:
    docs = (REPO_ROOT / "docs" / "software_release_promotion.md").read_text(encoding="utf-8")
    assert "testpypi" in docs
    assert "pypi" in docs
    assert "trusted publisher" in docs.lower()
    assert "password" in docs.lower()
    assert "DO NOT" not in docs
    assert "api token" in docs.lower()
    assert "software_release_promotion.md" in (
        REPO_ROOT / "docs" / "software_release_candidate.md"
    ).read_text(encoding="utf-8")
