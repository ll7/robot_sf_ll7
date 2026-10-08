"""Actual evidence-writer and admission contracts for the bounded fit driver."""

import hashlib
import json
from pathlib import Path

import pytest

from robot_sf.evidence.writers import write_json
from scripts.validation.pedcontact_fit_10101 import freeze
from scripts.validation.render_pedcontact_round3_evidence import portable_paths


@pytest.mark.parametrize(
    "robot_extra",
    [
        {},
        {"gate_pass": False},
        {"gate_pass": True, "new_contacts": 1},
        {"gate_pass": True, "source_sha": "0" * 40},
    ],
)
def test_count_only_or_failed_receipt_cannot_admit_fit(tmp_path, robot_extra):
    write_json(tmp_path / "step4_comparison.json", {"fit_admitted": True})
    write_json(tmp_path / "robot_gate_summary.json", {"pairs": 8550, **robot_extra})
    root = tmp_path / "fit"
    root.mkdir()
    with pytest.raises(ValueError, match="qualification|robot gate"):
        freeze(root)
    assert not (root / "grid.json").exists()


def test_export_redacts_custody_identifiers_without_changing_measurements():
    """Nested receipt exports must preserve science without publishing private identities."""
    private_path = str(Path("/", "home", "test-user", "lanes", "test-lane", "bank.npz"))
    private_host = "example-" + "imech" + str(123)
    raw = {
        "host": private_host,
        "operator": "test-user",
        "driver_symbol": "frozen_" + private_host + "_packet",
        "files": {private_path: {"mean": 1.25, "source_sha256": "a" * 64}},
        "rows": [{"path": private_path, "seed": 1001}],
    }
    exported = portable_paths(raw)
    encoded = json.dumps(exported)
    assert str(Path("/", "home")) not in encoded
    assert private_host not in encoded
    assert "test-user" not in encoded
    assert len(exported["files"]) == 1
    assert next(iter(exported["files"].values())) == next(iter(raw["files"].values()))
    assert exported["rows"][0]["seed"] == 1001


@pytest.fixture
def qualified_fit(tmp_path, monkeypatch):
    """Hypothetical complete receipts test schema/custody, not empirical model validity."""
    import itertools
    import subprocess
    from copy import deepcopy

    from scripts.validation import pedcontact_fit_10101 as driver
    from scripts.validation.calfit_preflight_10074 import ideal_gate_records

    project = tmp_path / "source"
    project.mkdir()
    for name in driver.SOURCE_PATHS:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("hypothetical source fixture\n")
    axes = {
        "scenarios": [f"fixture_scenario_{i}" for i in range(57)],
        "arms": [f"fixture_arm_{i}" for i in range(5)],
        "variants": ["off", "contact", "joint"],
        "seeds": list(range(1001, 1011)),
    }
    write_json(
        project / "configs/benchmarks/pedcontact_robot_gate_v1.json",
        {"schema": "pedcontact.robot_contract.v1", **axes},
    )
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(
        ["git", "add", *driver.SOURCE_PATHS, "configs/benchmarks/pedcontact_robot_gate_v1.json"],
        cwd=project,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Receipt fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=project,
        check=True,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    monkeypatch.setattr(driver, "ROOT", project)
    sources = {
        name: hashlib.sha256((project / name).read_bytes()).hexdigest()
        for name in driver.SOURCE_PATHS
    }
    evidence = tmp_path / "receipts"
    evidence.mkdir()
    raw = evidence / "measured.json"
    raw.write_text('{"hypothetical":true}\n')
    reference = {"path": raw.name, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
    bank = []
    for seed in range(1001, 1031):
        for original in ideal_gate_records():
            row = deepcopy(original)
            row.update(
                seed=seed,
                raw_trajectory=raw.name,
                raw_trajectory_sha256=reference["sha256"],
                wall_penetration_ped_steps=0,
                step_runtime=[
                    {
                        "unresolved_count": 0,
                        "over_cap_samples": 0,
                        "fallback_count": 0,
                        "steps": 1,
                        "step_time_s": 0.01,
                        "maximum_projection_passes": 1,
                        "maximum_speed_m_s": 1.0,
                    }
                ],
            )
            bank.append(row)
    physical = {
        "schema": "pedcontact.physical_qualification.v1",
        "source_sha": sha,
        "source_files": sources,
        "rows": bank,
    }
    roster = {
        "schema": "pedcontact.robot_roster.v1",
        "source_sha": sha,
        "source_files": sources,
        **axes,
        "accepted_failure_disposition": "no_failures",
        "rows": [
            dict(
                zip(("scenario", "arm", "variant", "seed"), key, strict=True),
                status="PASS",
                fallback=False,
                degraded=False,
                new_contacts=0,
                evidence=reference,
            )
            for key in itertools.product(
                *(axes[k] for k in ("scenarios", "arms", "variants", "seeds"))
            )
        ],
    }

    def save():
        write_json(evidence / "physical.json", physical)
        write_json(evidence / "roster.json", roster)
        write_json(
            evidence / "step4_comparison.json",
            {
                "schema": "pedcontact.comparison.v1",
                "source_sha": sha,
                "fit_admitted": True,
                "qualification": {
                    "path": "physical.json",
                    "sha256": hashlib.sha256((evidence / "physical.json").read_bytes()).hexdigest(),
                },
            },
        )
        write_json(
            evidence / "robot_gate_summary.json",
            {
                "schema": "pedcontact.robot_gate.v1",
                "source_sha": sha,
                "gate_pass": True,
                "pairs": 8550,
                "qualification": {
                    "path": "roster.json",
                    "sha256": hashlib.sha256((evidence / "roster.json").read_bytes()).hexdigest(),
                },
            },
        )

    save()
    root = evidence / "fit"
    root.mkdir()
    return root, physical, roster, save


def test_freeze_writes_grid_from_complete_source_bound_receipts(qualified_fit):
    root, _, _, _ = qualified_fit
    freeze(root)
    grid = json.loads((root / "grid.json").read_bytes())
    assert len(grid["points"]) == len({point["id"] for point in grid["points"]}) == 108
    assert grid["seeds"] == [1001, 1002, 1003]
    assert set(grid["qualification_sha256"]) == {"physical.json", "roster.json", "measured.json"}


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("stale", "source mismatch"),
        ("physical", "qualification failed"),
        ("duplicate", "roster incomplete or duplicate"),
        ("failed", "author disposition"),
        ("missing", "evidence is missing"),
        ("tampered", "checksum mismatch"),
        ("wrong_roster", "differs from versioned contract"),
    ],
)
def test_freeze_refuses_stale_failed_missing_or_duplicate_evidence(qualified_fit, mutation, reason):
    root, physical, roster, save = qualified_fit
    if mutation == "stale":
        physical["source_sha"] = "0" * 40
    elif mutation == "physical":
        physical["rows"][0]["step_runtime"][0]["unresolved_count"] = 1
    elif mutation == "duplicate":
        roster["rows"][-1] = roster["rows"][0]
    elif mutation == "failed":
        roster["rows"][0]["new_contacts"] = 1
    elif mutation == "wrong_roster":
        roster["arms"][0] = "undeclared_arm"
    save()
    if mutation == "missing":
        (root.parent / "measured.json").unlink()
    elif mutation == "tampered":
        (root.parent / "measured.json").write_text("changed")
    with pytest.raises(ValueError, match=reason):
        freeze(root)
    assert not (root / "grid.json").exists()
