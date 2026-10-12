"""Provenance regressions that read source and trace fixtures without simulation."""

from __future__ import annotations

import ast
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


def test_cv_evaluation_publishes_recorded_source_seed(tmp_path):
    """A candidate label must never overwrite the historical trace identity."""
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"source": {"seed": 1008}, "frames": []}))
    namespace = {"REPO_ROOT": tmp_path, "_load_trace": lambda path: json.loads(path.read_text())}
    for helper in (
        "_pedestrian_count",
        "_get_actor_classes",
        "_actor_class_counts",
        "_trace_metadata_coverage",
        "_trace_has_motion",
    ):
        namespace[helper] = lambda trace: False
    evaluate = load_function(CV_SCRIPT, "evaluate_single_trace", namespace)
    result = evaluate({"path": "trace.json", "label": "probe", "family": "synthetic", "seed": 1101})
    assert result["seed"] == 1008
    assert result["seed_provenance"] == "recorded_trace_source"


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
