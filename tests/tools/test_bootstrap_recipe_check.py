"""Tests for bootstrap_recipe_check (#8853): uv, module, container, driver, CARLA order,
overlay, mutable tag, unavailable recipe."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.tools.bootstrap_recipe_check import check_recipes, main, render_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_RECIPES = REPO_ROOT / "configs" / "bootstrap_recipes"
DIGEST = "sha256:" + "a" * 64
LOCKFILE_SHA256 = "d01cbbf7fb7215d140b3c78f66202e0c48e449601f809036f5e02e9a9bfb79c7"


def _base_recipe() -> dict:
    return {
        "schema": "bootstrap_recipe.v1",
        "recipe_id": "cpu-batch-fixture-v1",
        "execution_class": "cpu_batch",
        "title": "Fixture CPU environment",
        "verification_status": "verified",
        "source_identity": {
            "repository": "ll7/robot_sf_ll7",
            "lockfile": "uv.lock",
            "lockfile_sha256": LOCKFILE_SHA256,
        },
        "prerequisites": [{"id": "uv", "kind": "binary", "identity": "uv>=0.11"}],
        "identity_bindings": [{"kind": "python-package", "name": "numpy", "version": "2.4.6"}],
        "steps": [
            {"id": "sync", "phase": "setup", "argv": ["uv", "sync", "--all-extras"]},
            {"id": "version", "phase": "probe", "argv": ["python3", "-V"], "safe_check": True},
            {"id": "drop", "phase": "cleanup", "argv": ["rm", "-rf", "$RECIPE_ROOT/x"]},
        ],
        "outputs": ["$PROJECT_ROOT/output/cpu_batch/"],
        "private_substitutions": [],
        "verification": {"mode": "structural", "checked_at": "2026-09-12"},
    }


def _write_recipe(tmp_path: Path, recipe: dict) -> Path:
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    (recipes_dir / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    return recipes_dir


def _check_case(tmp_path: Path, recipe: dict) -> dict:
    report = check_recipes(recipes_path=_write_recipe(tmp_path, recipe))
    assert len(report["recipes"]) == 1
    return report["recipes"][0]


def test_uv_environment_module_stack_and_container_digest(tmp_path: Path) -> None:
    assert _check_case(tmp_path, _base_recipe())["reasons"] == []

    pinned = _base_recipe()
    pinned["identity_bindings"].append({"kind": "module", "name": "cuda", "identity": "cuda/13.0"})
    assert _check_case(tmp_path, pinned)["reasons"] == []

    unpinned = _base_recipe()
    unpinned["identity_bindings"].append({"kind": "module", "name": "cuda", "identity": "default"})
    entry = _check_case(tmp_path, unpinned)
    assert entry["status"] == "blocked"
    assert "unpinned_module_alias" in entry["reasons"]

    digest_recipe = _base_recipe()
    digest_recipe["execution_class"] = "carla_platform"
    digest_recipe["identity_bindings"] = [
        {"kind": "container", "name": "carlasim/carla", "tag": "0.9.16", "digest": DIGEST}
    ]
    assert _check_case(tmp_path, digest_recipe)["status"] == "verified"

    mutable = _base_recipe()
    mutable["execution_class"] = "carla_platform"
    mutable["identity_bindings"] = [{"kind": "container", "name": "carlasim/carla", "tag": "x"}]
    entry = _check_case(tmp_path, mutable)
    assert entry["status"] == "blocked"
    assert "mutable_container_alias" in entry["reasons"]


def test_malformed_identity_bindings(tmp_path: Path) -> None:
    non_list = _base_recipe()
    non_list["identity_bindings"] = "not-a-list"
    assert "malformed_identity_bindings" in _check_case(tmp_path / "t1", non_list)["reasons"]

    non_dict = _base_recipe()
    non_dict["identity_bindings"] = ["not-a-dict"]
    assert "malformed_identity_binding" in _check_case(tmp_path / "t2", non_dict)["reasons"]

    missing_kind = _base_recipe()
    missing_kind["identity_bindings"] = [{"name": "numpy"}]
    assert "malformed_identity_binding" in _check_case(tmp_path / "t3", missing_kind)["reasons"]

    missing_name = _base_recipe()
    missing_name["identity_bindings"] = [{"kind": "python-package"}]
    assert "malformed_identity_binding" in _check_case(tmp_path / "t4", missing_name)["reasons"]


def test_gpu_driver_prerequisite(tmp_path: Path) -> None:
    gpu = _base_recipe()
    gpu["execution_class"] = "gpu_training"
    gpu["prerequisites"].append(
        {"id": "nvidia-driver", "kind": "driver", "identity": "nvidia>=560"}
    )
    gpu["steps"] = [
        gpu["steps"][0],
        {"id": "driver", "phase": "probe", "argv": ["nvidia-smi"], "safe_check": True},
        *gpu["steps"][1:],
    ]
    assert _check_case(tmp_path, gpu)["status"] == "verified"


def test_carla_process_order_enforced(tmp_path: Path) -> None:
    recipe = _base_recipe()
    recipe["steps"] = [
        {"id": "probe-first", "phase": "probe", "argv": ["python3", "-V"]},
        {"id": "setup-late", "phase": "setup", "argv": ["uv", "sync"]},
    ]
    entry = _check_case(tmp_path, recipe)
    assert entry["status"] == "invalid"
    assert "wrong_step_order" in entry["reasons"]


def test_private_overlay_and_undeclared_placeholder(tmp_path: Path) -> None:
    declared = _base_recipe()
    declared["private_substitutions"] = [
        {"placeholder": "PRIVATE_ARTIFACT_ROOT", "capability_class": "durable-storage"}
    ]
    declared["outputs"] = ["$PRIVATE_ARTIFACT_ROOT/analysis/"]
    recipes_dir = _write_recipe(tmp_path, declared)
    assert check_recipes(recipes_path=recipes_dir)["recipes"][0]["status"] == "verified"

    overlay = tmp_path / "overlay.json"
    payload = {"schema": "bootstrap_recipe_private_overlay.v1", "placeholders": {}}
    payload["placeholders"] = {"PRIVATE_ARTIFACT_ROOT": {"capability_class": "durable-storage"}}
    overlay.write_text(json.dumps(payload), encoding="utf-8")
    assert check_recipes(recipes_path=recipes_dir, overlay_path=overlay)["private_overlay"]["ok"]

    payload["placeholders"] = {"X": {}}
    overlay.write_text(json.dumps(payload), encoding="utf-8")
    ok = check_recipes(recipes_path=recipes_dir, overlay_path=overlay)["private_overlay"]["ok"]
    assert not ok

    undeclared = _base_recipe()
    undeclared["outputs"] = ["$PRIVATE_ARTIFACT_ROOT/analysis/"]
    entry = _check_case(tmp_path, undeclared)
    assert entry["status"] == "blocked"
    assert "unresolved_placeholder" in entry["reasons"]


def test_unavailable_recipe_status(tmp_path: Path) -> None:
    unavailable = _base_recipe()
    unavailable["verification_status"] = "unavailable"
    unavailable["unavailable_reason"] = "host access ended; no current owner"
    unavailable["steps"] = []
    assert _check_case(tmp_path, unavailable)["status"] == "unavailable"

    missing_reason = _base_recipe()
    missing_reason["verification_status"] = "unavailable"
    entry = _check_case(tmp_path, missing_reason)
    assert entry["status"] == "invalid"
    assert "missing_unavailable_reason" in entry["reasons"]


@pytest.mark.parametrize(
    ("case_name", "expected_status", "expected_reason"),
    [
        ("credential_leak", "blocked", "credential_leak"),
        ("private_path_leak", "blocked", "private_path_leak"),
        ("stale_path", "blocked", "stale_source_path"),
        ("destructive_cleanup", "blocked", "destructive_cleanup"),
        ("shell_string_step", "blocked", "shell_string_step"),
        ("hidden_pythonpath", "invalid", "hidden_environment_dependence"),
        ("missing_probe_step", "invalid", "missing_step_contract"),
    ],
)
def test_blocking_cases(
    tmp_path: Path, case_name: str, expected_status: str, expected_reason: str
) -> None:
    recipe = _base_recipe()
    if case_name == "credential_leak":
        recipe["steps"][0]["env"] = {"AWS_SECRET_ACCESS_KEY": "abc1234567890"}
    elif case_name == "private_path_leak":
        recipe["steps"][0]["workdir"] = "/root/.ssh/keys"
    elif case_name == "stale_path":
        recipe["steps"][0]["argv"] = ["python3", "/home/x/project/bootstrap.py"]
    elif case_name == "destructive_cleanup":
        recipe["steps"].append({"id": "wipe", "phase": "cleanup", "argv": ["rm", "-rf", "/"]})
    elif case_name == "shell_string_step":
        recipe["steps"][0]["argv"] = ["bash", "-c", "uv sync --all-extras"]
    elif case_name == "hidden_pythonpath":
        recipe["steps"][0]["env"] = {"PYTHONPATH": "$PROJECT_ROOT"}
    elif case_name == "missing_probe_step":
        recipe["steps"] = [s for s in recipe["steps"] if s["phase"] != "probe"]
    entry = _check_case(tmp_path, recipe)
    assert entry["status"] == expected_status
    assert expected_reason in entry["reasons"]


def test_safe_check_execution_is_isolated_and_nonzero_fails(tmp_path: Path) -> None:
    recipe = _base_recipe()
    recipe["steps"] = [
        {"id": "no-write", "phase": "setup", "argv": ["python3", "-c", "open('x', 'w')"]},
        {"id": "ok-probe", "phase": "probe", "argv": ["python3", "-V"], "safe_check": True},
        {"id": "drop", "phase": "cleanup", "argv": ["rm", "-rf", "$RECIPE_ROOT/x"]},
    ]
    report = check_recipes(recipes_path=_write_recipe(tmp_path, recipe), execute_safe_checks=True)
    assert report["verdict"] == "pass"
    assert report["host_mutation"] is False
    assert [c["step_id"] for c in report["recipes"][0]["safe_checks"]] == ["ok-probe"]
    assert not (tmp_path / "recipes" / "x").exists()

    failing = copy.deepcopy(recipe)
    failing["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "fails",
            "phase": "probe",
            "argv": ["false"],
            "safe_check": True,
        },
    ]
    report = check_recipes(recipes_path=_write_recipe(tmp_path, failing), execute_safe_checks=True)
    assert report["verdict"] == "fail"
    assert "safe_check_failed" in report["recipes"][0]["reasons"]
    assert report["recipes"][0]["safe_checks"][0]["host_mutation"] is False


def test_safe_check_blocks_arbitrary_python_and_shell(tmp_path: Path) -> None:
    # Python -c arbitrary script execution
    recipe = _base_recipe()
    outside_marker = tmp_path / "outside_marker.txt"
    recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "escape-probe",
            "phase": "probe",
            "argv": [
                "python3",
                "-c",
                f"from pathlib import Path; Path('{outside_marker}').write_text('probe')",
            ],
            "safe_check": True,
        },
    ]
    entry = _check_case(tmp_path, recipe)
    assert entry["status"] == "blocked"
    assert "unsafe_safe_check" in entry["reasons"]
    assert not outside_marker.exists()

    # Shell execution in safe_check
    shell_recipe = _base_recipe()
    shell_recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "shell-probe",
            "phase": "probe",
            "argv": ["bash", "-c", "echo hello"],
            "safe_check": True,
        },
    ]
    entry_shell = _check_case(tmp_path, shell_recipe)
    assert entry_shell["status"] == "blocked"
    assert "unsafe_safe_check" in entry_shell["reasons"]

    # Script file execution in safe_check
    script_recipe = _base_recipe()
    script_recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "script-probe",
            "phase": "probe",
            "argv": ["python3", "external_probe.py"],
            "safe_check": True,
        },
    ]
    entry_script = _check_case(tmp_path, script_recipe)
    assert entry_script["status"] == "blocked"
    assert "unsafe_safe_check" in entry_script["reasons"]


def test_safe_check_workdir_escape_is_blocked_and_cannot_escape(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent_outside"
    parent_dir.mkdir(parents=True, exist_ok=True)
    probe_root = tmp_path / "probe_root"
    probe_root.mkdir(parents=True, exist_ok=True)

    # Static rejection of parent-relative workdir
    recipe_parent = _base_recipe()
    recipe_parent["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "escape-workdir",
            "phase": "probe",
            "argv": ["python3", "-V"],
            "workdir": "$RECIPE_ROOT/../outside",
            "safe_check": True,
        },
    ]
    entry = _check_case(tmp_path, recipe_parent)
    assert entry["status"] == "blocked"
    assert "unsafe_safe_check" in entry["reasons"]

    # Direct runner execution containment validation: parent traversal cannot mkdir outside
    escaped_marker_dir = probe_root / ".." / "outside"
    from scripts.tools.bootstrap_recipe_check import _run_safe_checks

    escape_step_recipe = {
        "steps": [
            {
                "id": "escape-runner",
                "phase": "probe",
                "argv": ["python3", "-V"],
                "workdir": "$RECIPE_ROOT/../outside",
                "safe_check": True,
            }
        ]
    }
    results = _run_safe_checks(
        escape_step_recipe, temp_root=probe_root, project_root=tmp_path, timeout=5
    )
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert results[0]["isolated"] is False
    assert not escaped_marker_dir.exists()

    # Direct runner execution containment validation: symlink escaping temp_root cannot be used
    symlink_dir = probe_root / "symlink_outside"
    symlink_dir.symlink_to(parent_dir)
    symlink_step_recipe = {
        "steps": [
            {
                "id": "symlink-runner",
                "phase": "probe",
                "argv": ["python3", "-V"],
                "workdir": "$RECIPE_ROOT/symlink_outside",
                "safe_check": True,
            }
        ]
    }
    symlink_results = _run_safe_checks(
        symlink_step_recipe, temp_root=probe_root, project_root=tmp_path, timeout=5
    )
    assert len(symlink_results) == 1
    assert symlink_results[0]["status"] == "error"
    assert symlink_results[0]["isolated"] is False


def test_safe_check_environment_canary_not_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.tools.bootstrap_recipe_check import _run_safe_checks

    monkeypatch.setenv("CANARY_SECRET_TOKEN", "super-secret-canary-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-access-key-12345")

    probe_root = tmp_path / "probe_env_root"
    probe_root.mkdir(parents=True, exist_ok=True)

    # Run bounded probe that succeeds
    recipe = {
        "steps": [
            {
                "id": "check-env",
                "phase": "probe",
                "argv": ["python3", "-V"],
                "workdir": "$RECIPE_ROOT",
                "safe_check": True,
            }
        ]
    }
    results = _run_safe_checks(recipe, temp_root=probe_root, project_root=tmp_path, timeout=5)
    assert len(results) == 1
    assert results[0]["status"] == "passed"
    assert results[0]["isolated"] is True
    assert results[0]["host_mutation"] is False


def test_skipped_probe_does_not_claim_host_mutation_zero(tmp_path: Path) -> None:
    from scripts.tools.bootstrap_recipe_check import _run_safe_checks

    probe_root = tmp_path / "probe_root"
    probe_root.mkdir(parents=True, exist_ok=True)

    recipe = {
        "steps": [
            {
                "id": "missing-tool",
                "phase": "probe",
                "argv": ["nvidia-smi"],
                "workdir": "$RECIPE_ROOT",
                "safe_check": True,
            }
        ]
    }
    # Simulate missing program
    import shutil

    original_which = shutil.which
    try:
        shutil.which = lambda prog: None if prog == "nvidia-smi" else original_which(prog)
        results = _run_safe_checks(recipe, temp_root=probe_root, project_root=tmp_path, timeout=5)
        assert len(results) == 1
        assert results[0]["status"] == "skipped_missing_program"
        assert results[0]["isolated"] is True
        assert results[0]["host_mutation"] is None
    finally:
        shutil.which = original_which


def test_shipped_recipes_cover_all_execution_classes() -> None:
    report = check_recipes(
        recipes_path=SHIPPED_RECIPES,
        overlay_path=SHIPPED_RECIPES / "private_overlay.example.json",
        project_root=REPO_ROOT,
        require_verified=True,
    )
    assert report["verdict"] == "pass"
    assert report["missing_verified_classes"] == []
    assert report["private_overlay"]["ok"] is True
    classes = {r["execution_class"] for r in report["recipes"]}
    assert classes == {"cpu_batch", "gpu_training", "carla_platform", "local_analysis"}


def test_cli_is_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recipes_dir = _write_recipe(tmp_path, _base_recipe())
    argv = ["--check", "--recipes", str(recipes_dir), "--format", "json"]
    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(argv) == 0
    assert capsys.readouterr().out == first
    assert render_markdown(json.loads(first)) == render_markdown(copy.deepcopy(json.loads(first)))
    assert main(["--check", "--recipes", str(tmp_path / "missing"), "--format", "json"]) == 2


def test_missing_and_unreadable_lockfile_checks(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)

    # 1. Missing declared lockfile under project_root
    recipe_missing = _base_recipe()
    recipe_missing["source_identity"]["lockfile"] = "missing.lock"
    recipe_missing["source_identity"]["lockfile_sha256"] = LOCKFILE_SHA256
    report_missing = check_recipes(
        recipes_path=_write_recipe(tmp_path / "r1", recipe_missing),
        project_root=project_root,
    )
    assert report_missing["recipes"][0]["status"] == "invalid"
    assert "missing_lockfile" in report_missing["recipes"][0]["reasons"]

    # 2. Unreadable lockfile (e.g. directory instead of regular file)
    unreadable_lock = project_root / "directory.lock"
    unreadable_lock.mkdir(parents=True, exist_ok=True)
    recipe_dir = _base_recipe()
    recipe_dir["source_identity"]["lockfile"] = "directory.lock"
    recipe_dir["source_identity"]["lockfile_sha256"] = LOCKFILE_SHA256
    report_dir = check_recipes(
        recipes_path=_write_recipe(tmp_path / "r2", recipe_dir),
        project_root=project_root,
    )
    assert report_dir["recipes"][0]["status"] == "invalid"
    assert "unreadable_lockfile" in report_dir["recipes"][0]["reasons"]

    # 3. Missing lockfile checksum
    recipe_no_checksum = _base_recipe()
    del recipe_no_checksum["source_identity"]["lockfile_sha256"]
    report_no_sum = check_recipes(
        recipes_path=_write_recipe(tmp_path / "r3", recipe_no_checksum),
        project_root=project_root,
    )
    assert report_no_sum["recipes"][0]["status"] == "invalid"
    assert "missing_lockfile_checksum" in report_no_sum["recipes"][0]["reasons"]

    # 4. Malformed lockfile checksum
    recipe_bad_checksum = _base_recipe()
    recipe_bad_checksum["source_identity"]["lockfile_sha256"] = "short-or-invalid"
    report_bad_sum = check_recipes(
        recipes_path=_write_recipe(tmp_path / "r4", recipe_bad_checksum),
        project_root=project_root,
    )
    assert report_bad_sum["recipes"][0]["status"] == "invalid"
    assert "malformed_lockfile_checksum" in report_bad_sum["recipes"][0]["reasons"]

    # 5. Malformed source_identity object
    recipe_bad_source = _base_recipe()
    recipe_bad_source["source_identity"] = "not-a-dict"
    report_bad_source = check_recipes(
        recipes_path=_write_recipe(tmp_path / "r5", recipe_bad_source),
        project_root=project_root,
    )
    assert report_bad_source["recipes"][0]["status"] == "invalid"
    assert "malformed_source_identity" in report_bad_source["recipes"][0]["reasons"]


def test_lockfile_drift_and_present_lock_passes(tmp_path: Path) -> None:
    import hashlib

    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    lock_path = project_root / "uv.lock"
    lock_content = b"fake lockfile content for testing\n"
    lock_path.write_bytes(lock_content)
    correct_hash = hashlib.sha256(lock_content).hexdigest()

    # Matching lockfile passes
    matching_recipe = _base_recipe()
    matching_recipe["source_identity"]["lockfile"] = "uv.lock"
    matching_recipe["source_identity"]["lockfile_sha256"] = correct_hash
    report_ok = check_recipes(
        recipes_path=_write_recipe(tmp_path / "ok", matching_recipe),
        project_root=project_root,
    )
    assert report_ok["verdict"] == "pass"
    assert report_ok["recipes"][0]["status"] == "verified"
    assert "lockfile_drift" not in report_ok["recipes"][0]["reasons"]

    # Changed/drifted lockfile fails
    drift_recipe = _base_recipe()
    drift_recipe["source_identity"]["lockfile"] = "uv.lock"
    drift_recipe["source_identity"]["lockfile_sha256"] = "0" * 64
    report_drift = check_recipes(
        recipes_path=_write_recipe(tmp_path / "drift", drift_recipe),
        project_root=project_root,
    )
    assert report_drift["verdict"] == "fail"
    assert report_drift["recipes"][0]["status"] == "invalid"
    assert "lockfile_drift" in report_drift["recipes"][0]["reasons"]


def test_missing_executable_and_require_verified_fails(tmp_path: Path) -> None:
    import shutil

    gpu_recipe = _base_recipe()
    gpu_recipe["execution_class"] = "gpu_training"
    gpu_recipe["recipe_id"] = "gpu-training-fixture-v1"
    gpu_recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["uv", "sync"]},
        {"id": "driver", "phase": "probe", "argv": ["nvidia-smi"], "safe_check": True},
    ]

    # Structural mode passes without certifying runtime availability
    report_structural = check_recipes(
        recipes_path=_write_recipe(tmp_path / "struct", gpu_recipe),
        execute_safe_checks=False,
    )
    assert report_structural["verdict"] == "pass"
    assert report_structural["mode"] == "structural"
    assert report_structural["recipes"][0]["status"] == "verified"

    # Executed mode with missing program marks status as unavailable and records reason
    orig_which = shutil.which
    try:
        shutil.which = lambda prog: None if prog == "nvidia-smi" else orig_which(prog)
        report_exec = check_recipes(
            recipes_path=_write_recipe(tmp_path / "exec", gpu_recipe),
            execute_safe_checks=True,
            require_verified=False,
        )
        assert report_exec["verdict"] == "pass"
        assert report_exec["mode"] == "structural+safe_checks"
        assert report_exec["recipes"][0]["status"] == "unavailable"
        assert "probe_program_unavailable" in report_exec["recipes"][0]["reasons"]
        assert report_exec["execution_classes"]["gpu_training"] == "unavailable"

        # --require-verified cannot be satisfied by skipped required checks
        report_req = check_recipes(
            recipes_path=_write_recipe(tmp_path / "req", gpu_recipe),
            execute_safe_checks=True,
            require_verified=True,
        )
        assert report_req["verdict"] == "fail"
        assert "gpu_training" in report_req["missing_verified_classes"]
    finally:
        shutil.which = orig_which


def test_unresolved_required_substitution_and_zero_required_probes(tmp_path: Path) -> None:
    # 1. Unresolved required private substitution in safe check probe
    sub_recipe = _base_recipe()
    sub_recipe["private_substitutions"] = [
        {"placeholder": "PRIVATE_CONTAINER_TAG", "capability_class": "container-image"}
    ]
    sub_recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["uv", "sync"]},
        {
            "id": "probe-sub",
            "phase": "probe",
            "argv": [
                "docker",
                "image",
                "inspect",
                "$PRIVATE_CONTAINER_TAG",
                "--format",
                "{{.Id}}",
            ],
            "safe_check": True,
        },
    ]
    report_sub = check_recipes(
        recipes_path=_write_recipe(tmp_path / "sub", sub_recipe),
        execute_safe_checks=True,
        require_verified=True,
    )
    assert report_sub["verdict"] == "fail"
    assert report_sub["recipes"][0]["status"] == "unavailable"
    assert "unresolved_required_substitution" in report_sub["recipes"][0]["reasons"]

    # 2. Zero executed required probes for verified recipe
    zero_recipe = _base_recipe()
    zero_recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["uv", "sync"]},
        {"id": "unsafeprobe", "phase": "probe", "argv": ["python3", "-c", "import os"]},
    ]
    report_zero = check_recipes(
        recipes_path=_write_recipe(tmp_path / "zero", zero_recipe),
        execute_safe_checks=True,
        require_verified=True,
    )
    assert report_zero["verdict"] == "fail"
    assert report_zero["recipes"][0]["status"] == "unavailable"
    assert "zero_executed_required_probes" in report_zero["recipes"][0]["reasons"]


def test_unavailable_fixture_preserves_unavailable_status(tmp_path: Path) -> None:
    unavailable_gpu = _base_recipe()
    unavailable_gpu["execution_class"] = "gpu_training"
    unavailable_gpu["verification_status"] = "unavailable"
    unavailable_gpu["unavailable_reason"] = "no GPU hardware available in CI"
    unavailable_gpu["steps"] = [
        {"id": "driver", "phase": "probe", "argv": ["python3", "-V"], "safe_check": True}
    ]

    report = check_recipes(
        recipes_path=_write_recipe(tmp_path / "unavail", unavailable_gpu),
        execute_safe_checks=True,
        require_verified=False,
    )
    assert report["verdict"] == "pass"
    # Status must be preserved as unavailable even when safe check passes
    assert report["recipes"][0]["status"] == "unavailable"
    assert report["recipes"][0]["safe_checks"][0]["status"] == "passed"
    assert report["execution_classes"]["gpu_training"] == "unavailable"

    # With require_verified=True, missing verified class gpu_training causes fail
    report_req = check_recipes(
        recipes_path=_write_recipe(tmp_path / "unavail_req", unavailable_gpu),
        execute_safe_checks=True,
        require_verified=True,
    )
    assert report_req["verdict"] == "fail"
    assert "gpu_training" in report_req["missing_verified_classes"]


def test_safe_check_symlinked_parent_missing_descendant_cannot_create_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.tools.bootstrap_recipe_check import _run_safe_checks

    outside = tmp_path / "outside_dir"
    outside.mkdir(parents=True, exist_ok=True)
    probe_root = tmp_path / "probe_root"
    probe_root.mkdir(parents=True, exist_ok=True)

    symlink_parent = probe_root / "symlink_parent"
    symlink_parent.symlink_to(outside)

    recipe = _base_recipe()
    recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "escape-descendant",
            "phase": "probe",
            "argv": ["python3", "-V"],
            "workdir": "$RECIPE_ROOT/symlink_parent/missing_descendant",
            "safe_check": True,
        },
    ]

    # Direct runner execution: missing descendant must not be created outside recipe root
    results = _run_safe_checks(recipe, temp_root=probe_root, project_root=tmp_path, timeout=5)
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert results[0]["isolated"] is False
    assert results[0]["host_mutation"] is None
    assert "workdir escapes isolated root" in results[0]["error"]
    assert not (outside / "missing_descendant").exists()

    import tempfile

    recipes_dir = _write_recipe(tmp_path / "symlink_recipe_dir", recipe)
    orig_tempdir = tempfile.TemporaryDirectory

    class _PatchedTempDir:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.temp_dir = orig_tempdir(*args, **kwargs)

        def __enter__(self) -> str:
            path_str = self.temp_dir.__enter__()
            p = Path(path_str)
            (p / "symlink_parent").symlink_to(outside)
            return path_str

        def __exit__(self, *args: object) -> None:
            self.temp_dir.__exit__(*args)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _PatchedTempDir)
    report = check_recipes(recipes_path=recipes_dir, execute_safe_checks=True)
    assert not (outside / "missing_descendant").exists()
    assert report["recipes"][0]["safe_checks"][0]["status"] == "error"
    assert report["recipes"][0]["safe_checks"][0]["isolated"] is False
    assert report["host_mutation"] is None


def test_safe_check_rejects_path_in_program_name(tmp_path: Path) -> None:
    from scripts.tools.bootstrap_recipe_check import _run_safe_checks

    for prog_candidate in ["/usr/bin/python3", "./python3", "../python3"]:
        recipe = _base_recipe()
        recipe["steps"] = [
            {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
            {
                "id": "path-probe",
                "phase": "probe",
                "argv": [prog_candidate, "-V"],
                "safe_check": True,
            },
        ]
        entry = _check_case(tmp_path / f"path_{abs(hash(prog_candidate))}", recipe)
        assert entry["status"] == "blocked"
        assert "unsafe_safe_check" in entry["reasons"]

        results = _run_safe_checks(
            recipe, temp_root=tmp_path / "probe_root", project_root=tmp_path, timeout=5
        )
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert results[0]["isolated"] is False
        assert "path_in_safe_check_program" in results[0]["error"]


def test_safe_check_rejects_protected_env_overrides(tmp_path: Path) -> None:
    import os

    from scripts.tools.bootstrap_recipe_check import _run_safe_checks

    protected_keys = ["PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"]
    for key in protected_keys:
        recipe = _base_recipe()
        recipe["steps"] = [
            {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
            {
                "id": f"env-{key}",
                "phase": "probe",
                "argv": ["python3", "-V"],
                "env": {key: "/tmp/custom_location"},
                "safe_check": True,
            },
        ]
        entry = _check_case(tmp_path / f"env_{key}", recipe)
        assert entry["status"] == "blocked"
        assert "unsafe_safe_check" in entry["reasons"]

        results = _run_safe_checks(
            recipe, temp_root=tmp_path / "probe_env_root", project_root=tmp_path, timeout=5
        )
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert results[0]["isolated"] is False
        assert f"protected_env_override: {key}" in results[0]["error"]

    # Verify that a recipe cannot substitute an untrusted binary via PATH
    custom_bin = tmp_path / "custom_bin"
    custom_bin.mkdir(parents=True, exist_ok=True)
    fake_python = custom_bin / "python3"
    marker_file = tmp_path / "pwned.txt"
    fake_python.write_text(f"#!/bin/sh\necho PWNED > {marker_file}\nexit 0\n")
    import stat

    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    hijack_recipe = {
        "steps": [
            {
                "id": "hijack-probe",
                "phase": "probe",
                "argv": ["python3", "-V"],
                "env": {"PATH": f"{custom_bin}:{os.environ.get('PATH', '')}"},
                "safe_check": True,
            }
        ]
    }
    results = _run_safe_checks(
        hijack_recipe, temp_root=tmp_path / "probe_hijack", project_root=tmp_path, timeout=5
    )
    assert results[0]["status"] == "error"
    assert results[0]["isolated"] is False
    assert not marker_file.exists()


def test_skipped_or_uncertain_probe_propagates_null_host_mutation_to_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    # 1. Probe skipped due to missing program propagates host_mutation: null to report
    gpu_recipe = _base_recipe()
    gpu_recipe["execution_class"] = "gpu_training"
    gpu_recipe["recipe_id"] = "gpu-training-skipped"
    gpu_recipe["verification_status"] = "unavailable"
    gpu_recipe["unavailable_reason"] = "no GPU hardware available in CI"
    gpu_recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {"id": "driver", "phase": "probe", "argv": ["nvidia-smi"], "safe_check": True},
    ]
    orig_which = shutil.which
    try:
        shutil.which = lambda prog: None if prog == "nvidia-smi" else orig_which(prog)
        report_skipped = check_recipes(
            recipes_path=_write_recipe(tmp_path / "skipped_prog", gpu_recipe),
            execute_safe_checks=True,
        )
        assert report_skipped["recipes"][0]["safe_checks"][0]["status"] == "skipped_missing_program"
        assert report_skipped["recipes"][0]["safe_checks"][0]["host_mutation"] is None
        assert report_skipped["host_mutation"] is None
    finally:
        shutil.which = orig_which

    # 2. Probe skipped due to private substitution propagates host_mutation: null to report
    sub_recipe = _base_recipe()
    sub_recipe["recipe_id"] = "private-sub-skipped"
    sub_recipe["private_substitutions"] = [
        {"placeholder": "$CUSTOM_IMAGE", "capability_class": "container"}
    ]
    sub_recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "inspect-custom",
            "phase": "probe",
            "argv": ["docker", "image", "inspect", "$CUSTOM_IMAGE"],
            "safe_check": True,
        },
    ]
    report_sub = check_recipes(
        recipes_path=_write_recipe(tmp_path / "skipped_sub", sub_recipe),
        execute_safe_checks=True,
    )
    assert report_sub["recipes"][0]["safe_checks"][0]["status"] == "skipped_private_substitution"
    assert report_sub["recipes"][0]["safe_checks"][0]["host_mutation"] is None
    assert report_sub["host_mutation"] is None

    # 3. Report with zero safe checks executed in execute_safe_checks=True mode emits host_mutation: null
    no_probe_recipe = _base_recipe()
    no_probe_recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {"id": "probe", "phase": "probe", "argv": ["python3", "-V"], "safe_check": False},
    ]
    report_zero = check_recipes(
        recipes_path=_write_recipe(tmp_path / "zero_safe", no_probe_recipe),
        execute_safe_checks=True,
    )
    assert report_zero["host_mutation"] is None

    # 4. Structural mode emits host_mutation: False
    report_structural = check_recipes(
        recipes_path=_write_recipe(tmp_path / "struct_mode", no_probe_recipe),
        execute_safe_checks=False,
    )
    assert report_structural["host_mutation"] is False


def test_lockfile_rejects_traversal_absolute_and_symlink_escapes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside_lock = outside_root / "target.lock"
    content = b"pinned dependencies lockfile content\n"
    outside_lock.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()

    # 1. Traversal: ../outside/target.lock is rejected even with matching digest
    recipe_traversal = _base_recipe()
    recipe_traversal["recipe_id"] = "lock-traversal"
    recipe_traversal["source_identity"]["lockfile"] = "../outside/target.lock"
    recipe_traversal["source_identity"]["lockfile_sha256"] = checksum
    report_traversal = check_recipes(
        recipes_path=_write_recipe(tmp_path / "case_trav", recipe_traversal),
        project_root=project_root,
    )
    assert report_traversal["recipes"][0]["status"] == "invalid"
    assert "lockfile_escape" in report_traversal["recipes"][0]["reasons"]
    assert report_traversal["verdict"] == "fail"

    # 2. Absolute path: /.../target.lock is rejected even with matching digest
    recipe_abs = _base_recipe()
    recipe_abs["recipe_id"] = "lock-abs"
    recipe_abs["source_identity"]["lockfile"] = str(outside_lock)
    recipe_abs["source_identity"]["lockfile_sha256"] = checksum
    report_abs = check_recipes(
        recipes_path=_write_recipe(tmp_path / "case_abs", recipe_abs),
        project_root=project_root,
    )
    assert report_abs["recipes"][0]["status"] == "invalid"
    assert "lockfile_escape" in report_abs["recipes"][0]["reasons"]
    assert report_abs["verdict"] == "fail"

    # 3. Project-root symlink targeting outside file is rejected even with matching digest
    escape_link = project_root / "escaped.lock"
    escape_link.symlink_to(outside_lock)
    recipe_symlink = _base_recipe()
    recipe_symlink["recipe_id"] = "lock-symlink-escape"
    recipe_symlink["source_identity"]["lockfile"] = "escaped.lock"
    recipe_symlink["source_identity"]["lockfile_sha256"] = checksum
    report_symlink = check_recipes(
        recipes_path=_write_recipe(tmp_path / "case_sym", recipe_symlink),
        project_root=project_root,
    )
    assert report_symlink["recipes"][0]["status"] == "invalid"
    assert "lockfile_escape" in report_symlink["recipes"][0]["reasons"]
    assert report_symlink["verdict"] == "fail"

    # 4. Valid in-root regular lockfile passes verification
    valid_lock = project_root / "uv.lock"
    valid_lock.write_bytes(content)
    recipe_valid = _base_recipe()
    recipe_valid["recipe_id"] = "lock-valid-in-root"
    recipe_valid["source_identity"]["lockfile"] = "uv.lock"
    recipe_valid["source_identity"]["lockfile_sha256"] = checksum
    report_valid = check_recipes(
        recipes_path=_write_recipe(tmp_path / "case_valid", recipe_valid),
        project_root=project_root,
    )
    assert report_valid["recipes"][0]["status"] == "verified"
    assert "lockfile_escape" not in report_valid["recipes"][0]["reasons"]


def test_safe_check_unresolved_private_placeholder_in_workdir_or_env(tmp_path: Path) -> None:
    # 1. Unresolved private placeholder in safe-step workdir fails closed without executing
    recipe_workdir = _base_recipe()
    recipe_workdir["recipe_id"] = "unresolved-workdir-sub"
    recipe_workdir["private_substitutions"] = [
        {"placeholder": "$CUSTOM_DIR", "capability_class": "storage"}
    ]
    recipe_workdir["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "probe-workdir",
            "phase": "probe",
            "argv": ["python3", "--version"],
            "workdir": "$RECIPE_ROOT/$CUSTOM_DIR",
            "safe_check": True,
        },
    ]
    report_workdir = check_recipes(
        recipes_path=_write_recipe(tmp_path / "case_workdir", recipe_workdir),
        execute_safe_checks=True,
        require_verified=True,
    )
    assert report_workdir["recipes"][0]["status"] == "unavailable"
    assert "unresolved_required_substitution" in report_workdir["recipes"][0]["reasons"]
    assert report_workdir["verdict"] == "fail"

    # 2. Unresolved private placeholder in safe-step env fails closed without executing
    recipe_env = _base_recipe()
    recipe_env["recipe_id"] = "unresolved-env-sub"
    recipe_env["private_substitutions"] = [
        {"placeholder": "$CUSTOM_SECRET", "capability_class": "credential"}
    ]
    recipe_env["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "probe-env",
            "phase": "probe",
            "argv": ["python3", "--version"],
            "workdir": "$RECIPE_ROOT",
            "env": {"STEP_SECRET": "$CUSTOM_SECRET"},
            "safe_check": True,
        },
    ]
    report_env = check_recipes(
        recipes_path=_write_recipe(tmp_path / "case_env", recipe_env),
        execute_safe_checks=True,
        require_verified=True,
    )
    assert report_env["recipes"][0]["status"] == "unavailable"
    assert "unresolved_required_substitution" in report_env["recipes"][0]["reasons"]
    assert report_env["verdict"] == "fail"

    # 3. Resolved private placeholders in workdir and env allow safe-checks to pass
    overlay_path = tmp_path / "valid_overlay.json"
    overlay_payload = {
        "schema": "bootstrap_recipe_private_overlay.v1",
        "placeholders": {
            "CUSTOM_DIR": {"capability_class": "storage", "value": "subfolder"},
            "CUSTOM_SECRET": {"capability_class": "credential", "value": "safe_val"},
        },
    }
    overlay_path.write_text(json.dumps(overlay_payload), encoding="utf-8")
    both_recipe = _base_recipe()
    both_recipe["recipe_id"] = "resolved-both-sub"
    both_recipe["private_substitutions"] = [
        {"placeholder": "$CUSTOM_DIR", "capability_class": "storage", "value": "subfolder"},
        {"placeholder": "$CUSTOM_SECRET", "capability_class": "credential", "value": "safe_val"},
    ]
    both_recipe["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "probe-both",
            "phase": "probe",
            "argv": ["python3", "--version"],
            "workdir": "$RECIPE_ROOT/$CUSTOM_DIR",
            "env": {"STEP_SECRET": "$CUSTOM_SECRET"},
            "safe_check": True,
        },
    ]
    report_both = check_recipes(
        recipes_path=_write_recipe(tmp_path / "case_both", both_recipe),
        overlay_path=overlay_path,
        execute_safe_checks=True,
    )
    assert report_both["recipes"][0]["status"] == "verified"
    assert report_both["verdict"] == "pass"


def test_safe_check_malformed_placeholders_in_workdir_or_env(tmp_path: Path) -> None:
    # 1. Lowercase / malformed placeholder in workdir returns non-verified before execution
    bad_workdir = _base_recipe()
    bad_workdir["recipe_id"] = "bad-workdir-ph"
    bad_workdir["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "probe-bad-workdir",
            "phase": "probe",
            "argv": ["python3", "--version"],
            "workdir": "$RECIPE_ROOT/$invalid_lower",
            "safe_check": True,
        },
    ]
    report_workdir = check_recipes(
        recipes_path=_write_recipe(tmp_path / "bad_workdir", bad_workdir),
        execute_safe_checks=True,
    )
    assert report_workdir["recipes"][0]["status"] in {"blocked", "invalid"}
    assert report_workdir["verdict"] in {"blocked", "fail"}

    # 2. Lowercase / malformed placeholder in env value returns non-verified before execution
    bad_env = _base_recipe()
    bad_env["recipe_id"] = "bad-env-ph"
    bad_env["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "probe-bad-env",
            "phase": "probe",
            "argv": ["python3", "--version"],
            "workdir": "$RECIPE_ROOT",
            "env": {"STEP_VAR": "$invalid_lower"},
            "safe_check": True,
        },
    ]
    report_env = check_recipes(
        recipes_path=_write_recipe(tmp_path / "bad_env", bad_env),
        execute_safe_checks=True,
    )
    assert report_env["recipes"][0]["status"] in {"blocked", "invalid"}
    assert report_env["verdict"] in {"blocked", "fail"}

    # 3. Invalid env key in safe check returns non-verified before execution
    bad_env_key = _base_recipe()
    bad_env_key["recipe_id"] = "bad-env-key"
    bad_env_key["steps"] = [
        {"id": "sync", "phase": "setup", "argv": ["python3", "-V"]},
        {
            "id": "probe-bad-key",
            "phase": "probe",
            "argv": ["python3", "--version"],
            "workdir": "$RECIPE_ROOT",
            "env": {"$INVALID_KEY": "val"},
            "safe_check": True,
        },
    ]
    report_key = check_recipes(
        recipes_path=_write_recipe(tmp_path / "bad_key", bad_env_key),
        execute_safe_checks=True,
    )
    assert report_key["recipes"][0]["status"] in {"blocked", "invalid"}
    assert report_key["verdict"] in {"blocked", "fail"}
