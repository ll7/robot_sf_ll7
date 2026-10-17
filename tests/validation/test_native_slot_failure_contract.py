"""Native failure diagnostics retain the main reconciler's missing-field refusal."""

import json
from types import SimpleNamespace

import pytest

from scripts.validation import run_empty_world_sweep as native


def test_failure_without_scenario_id_raises(tmp_path):
    path = tmp_path / "runs/goal__differential_drive/summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"failures": [{"seed": 1001, "error": "runner failed"}]}))
    cfg = SimpleNamespace(
        planners=[SimpleNamespace(key="goal")], kinematics_matrix=["differential_drive"]
    )
    with pytest.raises(KeyError, match="scenario_id"):
        native.reconcile_execution(
            cfg, [{"name": "open", "seeds": [1001]}], [], tmp_path, {}, step_trace=False
        )
