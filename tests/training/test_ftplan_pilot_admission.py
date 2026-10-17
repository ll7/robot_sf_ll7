"""Exercise selection admission in the real pilot preflight without loading a policy."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "SLURM/ftplan/released_ppo_pilot.slurm"


class AdmissionReached(Exception):
    """The release resolver marks successful admission, before any checkpoint work."""


@pytest.fixture
def admit_selection(tmp_path, monkeypatch):
    """Run the unchanged embedded preflight; replace only downstream model interfaces."""
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "receipt.txt"
    artifact.write_bytes(b"selection-admission-fixture")
    record = {
        "artifact_path": "receipt.txt",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "hub_artifact": "fixture-hub/receipt.txt",
        "wandb_artifact": "fixture/receipt:v0",
    }
    manifest = {
        "status": "reported",
        "worker_receipt_audit": "passed",
        "paired_master_reports": [
            {**record, "master_seed": seed} for seed in range(100000, 100003)
        ],
    }
    (tmp_path / "replication.json").write_text(json.dumps(manifest))
    protocol = tmp_path / "configs/training/ppo/ftplan/pilot_protocol_20261012.yaml"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("source:\n  model_id: ppo_release_robot_b1002_last_20261001\n")
    monkeypatch.setenv("FTPLAN_REPLICATION_MANIFEST", "replication.json")
    monkeypatch.setenv("FTPLAN_SELECTION_RECEIPT", "selection.json")
    monkeypatch.setenv("FTPLAN_INIT_SEED", "100001")
    monkeypatch.setenv("FTPLAN_SEGMENT", "1")

    def released_checkpoint(_model_id):
        raise AdmissionReached

    monkeypatch.setitem(
        sys.modules,
        "robot_sf.training.released_warm_start",
        SimpleNamespace(load_warm_start=None, released_checkpoint=released_checkpoint),
    )
    monkeypatch.setitem(
        sys.modules,
        "robot_sf.training.scenario_family_weights",
        SimpleNamespace(with_family_weights=None),
    )
    monkeypatch.setitem(
        sys.modules, "scripts.training.train_ppo", SimpleNamespace(load_expert_training_config=None)
    )
    blocks = SCRIPT.read_text().split("python - <<'PY'\n")[1:]
    code = next(block.split("\nPY\n", 1)[0] for block in blocks if "receipt = json.loads" in block)

    def admit(overrides, *, missing=()):
        receipt = {
            "status": "completed",
            "evaluation_seeds": list(range(1031, 1061)),
            "scenario_count": 48,
            "fallback_used": False,
            "execution_errors": 0,
            "completed_segment": 0,
            "results_artifact": record,
        }
        receipt.update(overrides)
        for field in missing:
            receipt.pop(field)
        (tmp_path / "selection.json").write_text(json.dumps(receipt))
        exec(compile(code, str(SCRIPT), "exec"), {"__name__": "__main__"})  # noqa: S102

    return admit


def test_completed_clean_selection_reaches_checkpoint_resolution(admit_selection):
    """A valid positive control gets past admission without loading or training a model."""
    with pytest.raises(AdmissionReached):
        admit_selection({})


@pytest.mark.parametrize("value", [None, "", 0, 0.0, [], {}])
def test_fallback_requires_explicit_boolean_false(admit_selection, value):
    """Unknown or coerced fallback flags cannot authorize a pilot segment."""
    with pytest.raises(ValueError, match="fallback_used must be explicitly false"):
        admit_selection({"fallback_used": value})


@pytest.mark.parametrize("value", [None, "", False, 0.0, [], {}])
def test_execution_errors_requires_integer_zero(admit_selection, value):
    """Only a validated zero error count authorizes checkpoint resolution."""
    with pytest.raises(ValueError, match="execution_errors must be integer zero"):
        admit_selection({"execution_errors": value})


@pytest.mark.parametrize("value", [None, "", "error", "failed", "pending"])
def test_selection_requires_completed_status(admit_selection, value):
    """A failed or incomplete selection stays blocked even with clean flag/count fields."""
    with pytest.raises(ValueError, match="Selection status must be completed"):
        admit_selection({"status": value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"fallback_used": True},
        {"fallback_used": "false"},
        {"execution_errors": -1},
        {"execution_errors": 1},
        {"execution_errors": "0"},
        {"execution_errors": "error"},
    ],
)
def test_known_invalid_flag_or_count_stays_rejected(admit_selection, overrides):
    """Negative controls retain rejection of truthy failure or malformed fields."""
    with pytest.raises(ValueError):
        admit_selection(overrides)


def test_missing_completion_status_cannot_authorize_selection(admit_selection):
    """Unreported status is unknown even when flags and counts claim clean execution."""
    with pytest.raises(ValueError, match="Selection status must be completed"):
        admit_selection({}, missing=("status",))
