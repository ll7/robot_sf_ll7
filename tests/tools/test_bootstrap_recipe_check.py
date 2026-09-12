"""Tests for bootstrap_recipe_check (#8853): uv, module, container, driver, CARLA order,
overlay, mutable tag, unavailable recipe."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.tools.bootstrap_recipe_check import check_recipes, main, render_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_RECIPES = REPO_ROOT / "configs" / "bootstrap_recipes"
DIGEST = "sha256:" + "a" * 64


def _base_recipe() -> dict:
    return {
        "schema": "bootstrap_recipe.v1",
        "recipe_id": "cpu-batch-fixture-v1",
        "execution_class": "cpu_batch",
        "title": "Fixture CPU environment",
        "verification_status": "verified",
        "source_identity": {"repository": "ll7/robot_sf_ll7", "lockfile": "uv.lock"},
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
            "argv": ["python3", "-c", "raise SystemExit(3)"],
            "safe_check": True,
        },
    ]
    report = check_recipes(recipes_path=_write_recipe(tmp_path, failing), execute_safe_checks=True)
    assert report["verdict"] == "fail"
    assert "safe_check_failed" in report["recipes"][0]["reasons"]


def test_shipped_recipes_cover_all_execution_classes() -> None:
    report = check_recipes(
        recipes_path=SHIPPED_RECIPES,
        overlay_path=SHIPPED_RECIPES / "private_overlay.example.json",
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
