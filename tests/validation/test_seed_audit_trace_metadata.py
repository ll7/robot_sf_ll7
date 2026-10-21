"""Provenance regressions that read source and trace fixtures without simulation."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import CodeType, FunctionType

ROOT = Path(__file__).resolve().parents[2]
CV_SCRIPT = ROOT / "scripts/benchmark/run_cv_forecast_eval.py"
DWA_SCRIPTS = [
    ROOT / f"scripts/benchmark/trace_dwa_{suffix}.py"
    for suffix in (
        "decisions_issue_5298",
        "route_rescue_issue_5319",
        "global_route_probe_issue_5331",
    )
]


def load_function(path, name, globals_dict):
    """Compile the real function in isolation; do not import the simulation stack."""
    tree = ast.parse(path.read_text())
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    compiled = compile(module, str(path), "exec")
    body = next(code for code in compiled.co_consts if isinstance(code, CodeType))
    defaults = tuple(ast.literal_eval(default) for default in node.args.defaults)
    return FunctionType(body, globals_dict, name, defaults or None)


def test_cv_evaluation_publishes_recorded_source_seed():
    """Run recorded-trace evaluation with the real loader, helpers and forecast metrics."""
    spec = importlib.util.spec_from_file_location("_seed_audit_cv_evaluator", CV_SCRIPT)
    assert spec is not None and spec.loader is not None
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    path = (
        ROOT
        / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_episode_0000.json"
    )
    recorded = json.loads(path.read_text())
    result = evaluator.evaluate_single_trace(
        {"path": str(path), "label": "recorded_probe", "family": "occluded_emergence", "seed": 1101}
    )
    assert result["status"] == "evaluated"
    assert result["seed"] == recorded["source"]["seed"]
    assert result["seed"] != 1101
    assert result["seed_provenance"] == "recorded_trace_source"
    assert result["episode_execution"] == "historical_trace_read_only"
    assert result["pedestrians_per_frame"] == 1
    assert result["actor_class_counts"] == {"pedestrian": len(recorded["frames"])}
    assert result["metadata_coverage"]["pedestrian_observations"] == len(recorded["frames"])
    assert result["has_motion"] is True
    assert result["dt_s"] == 0.1
    assert result["metrics"]["forecast_evaluable_samples"] > 0


def test_cv_candidates_do_not_relabel_repository_artifacts():
    """Durable candidates describe their artifact's source, not a replacement episode."""
    tree = ast.parse(CV_SCRIPT.read_text())
    node = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", None) == "TRACE_CANDIDATES"
    )
    for candidate in ast.literal_eval(node):
        source = json.loads((ROOT / candidate["path"]).read_text())["source"]
        assert "seed" not in candidate or candidate["seed"] == source["seed"], candidate["label"]


def test_changed_dwa_diagnostics_mark_prior_outcomes_superseded():
    """Both exported metadata and generated prose must disclose the changed seed."""
    for path in DWA_SCRIPTS:
        text = path.read_text()
        tree = ast.parse(text)
        node = next(
            (
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(getattr(target, "id", None) == "SEED_MIGRATION" for target in node.targets)
            ),
            None,
        )
        assert node is not None, "seed migration metadata missing"
        metadata = ast.literal_eval(node)
        assert metadata["development_seed"] == 1121
        assert metadata["prior_outcome_descriptions"] == "superseded"
        assert metadata["replacement_trace_recorded_by_seed_migration"] is False
        assert '"seed_migration": SEED_MIGRATION' in text
        assert "Prior outcome descriptions are superseded" in text
        assert "reproduces the #5262 canonical episode outcomes" not in text


def test_rescue_report_withholds_delta_for_different_seed_identity(tmp_path):
    """A new development trace cannot be compared as paired to an archived episode."""
    namespace = {
        "TARGET_EPISODES": (),
        "_fmt": str,
        "write_markdown_atomic": lambda path, text, **kwargs: path.write_text(text),
    }
    namespace["paired_comparisons"] = load_function(DWA_SCRIPTS[1], "paired_comparisons", namespace)
    write_readme = load_function(DWA_SCRIPTS[1], "_write_evidence_readme", namespace)
    episode = {
        "episode_id": "bottleneck_timeout",
        "scenario_id": "probe",
        "seed": 1121,
        "termination_reason": "goal",
        "steps": 7,
    }
    archived = {**episode, "seed": 1008, "steps": 11}
    output = tmp_path / "README.md"
    write_readme(
        output,
        summaries=[episode],
        baseline_summaries=[archived],
        trace_commit="synthetic",
        config_sha256="synthetic",
    )
    text = output.read_text()
    assert "Prior outcome descriptions are superseded" in text
    assert "baseline identity differs; paired delta withheld" in text
    assert "| bottleneck | steps |" not in text
