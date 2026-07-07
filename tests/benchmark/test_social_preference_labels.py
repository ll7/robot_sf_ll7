"""Tests for the social preference label annotation pipeline.

Covers the ``annotate_episode_social_preferences`` function, threshold-band
classification, CLI smoke paths, and summary building.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.social_preference_labels import (
    annotate_episode_social_preferences,
    annotate_episodes_social_preferences,
    build_label_summary,
    get_episode_metric_names,
    load_social_preference_label_config,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "diagnostics"
    / "social_preference_labels.yaml"
)


@pytest.fixture
def schema() -> dict:
    """Load the checked-in social preference label schema."""

    return load_social_preference_label_config(CONFIG_PATH)


def _make_episode(metrics: dict) -> dict:
    """Build a minimal episode record with the given metrics dict."""

    return {"episode_id": "test-ep", "metrics": metrics}


class TestAnnotateEpisodeClearanceMargin:
    """The clearance label is the first fully automated label."""

    def test_clearance_satisfied_when_high(self, schema: dict) -> None:
        """A clearance value well above the acceptable band is satisfied."""

        episode = _make_episode({"min_clearance": 1.2})
        result = annotate_episode_social_preferences(episode, schema=schema)
        clearance_annotation = next(a for a in result["labels"] if a["label_id"] == "clearance")

        assert clearance_annotation["annotation"] == "acceptable"
        assert clearance_annotation["method"] == "threshold_band"
        assert clearance_annotation["value"] == 1.2

    def test_clearance_violated_when_negative(self, schema: dict) -> None:
        """A negative clearance (collision) is violated."""

        episode = _make_episode({"min_clearance": -0.05})
        result = annotate_episode_social_preferences(episode, schema=schema)
        clearance_annotation = next(a for a in result["labels"] if a["label_id"] == "clearance")

        assert clearance_annotation["annotation"] == "poor"
        assert clearance_annotation["method"] == "threshold_band"

    def test_clearance_not_available_when_metric_missing(self, schema: dict) -> None:
        """Missing clearance metrics produce not_available."""

        episode = _make_episode({"other_metric": 0.5})
        result = annotate_episode_social_preferences(episode, schema=schema)
        clearance_annotation = next(a for a in result["labels"] if a["label_id"] == "clearance")

        assert clearance_annotation["annotation"] == "not_available"
        assert clearance_annotation["method"] == "not_available"


class TestAnnotateEpisodeTtcMargin:
    """TTC margin is automated when the metric is present."""

    def test_ttc_margin_violated_when_low(self, schema: dict) -> None:
        """A low TTC value is classified as poor."""

        episode = _make_episode({"near_misses_ttc": 0.3})
        result = annotate_episode_social_preferences(episode, schema=schema)
        ttc_annotation = next(a for a in result["labels"] if a["label_id"] == "ttc_margin")

        assert ttc_annotation["annotation"] == "poor"

    def test_ttc_margin_not_available_when_metric_missing(self, schema: dict) -> None:
        """Missing TTC metrics produce not_available."""

        episode = _make_episode({"min_clearance": 0.8})
        result = annotate_episode_social_preferences(episode, schema=schema)
        ttc_annotation = next(a for a in result["labels"] if a["label_id"] == "ttc_margin")

        assert ttc_annotation["annotation"] == "not_available"


class TestAnnotateEpisodePathBlocking:
    """Path blocking is not_available by design (no candidate metric keys)."""

    def test_path_blocking_always_not_available(self, schema: dict) -> None:
        """The path_blocking label has empty candidate_metric_keys, so it is always unavailable."""

        episode = _make_episode({"min_clearance": 0.8, "jerk_mean": 0.1})
        result = annotate_episode_social_preferences(episode, schema=schema)
        pb_annotation = next(a for a in result["labels"] if a["label_id"] == "path_blocking")

        assert pb_annotation["annotation"] == "not_available"
        assert pb_annotation["method"] == "not_available"


class TestAnnotationOutputStructure:
    """Ensure the annotation dict has the correct shape."""

    def test_output_includes_schema_version_and_claim_boundary(self, schema: dict) -> None:
        """The annotation output echoes schema version and claim boundary."""

        episode = _make_episode({"min_clearance": 0.8})
        result = annotate_episode_social_preferences(episode, schema=schema)

        assert result["schema_version"] == "social-preference-labels.v1"
        assert "diagnostic" in result["claim_boundary"].lower()
        assert "not a reward" in result["claim_boundary"].lower()
        assert "label" in result["episode_id"] or isinstance(result["episode_id"], str)

    def test_output_includes_all_labels(self, schema: dict) -> None:
        """Every label from the schema appears in the annotation."""

        episode = _make_episode({"min_clearance": 0.8})
        result = annotate_episode_social_preferences(episode, schema=schema)
        annotation_ids = {a["label_id"] for a in result["labels"]}

        config_label_ids = {label_entry["id"] for label_entry in schema["labels"]}
        assert annotation_ids == config_label_ids

    def test_each_annotation_has_required_fields(self, schema: dict) -> None:
        """Each annotation object carries the expected keys."""

        episode = _make_episode({"min_clearance": 0.8})
        result = annotate_episode_social_preferences(episode, schema=schema)

        required_keys = {
            "label_id",
            "display_name",
            "metric_family",
            "value",
            "annotation",
            "method",
            "reason",
            "evidence",
            "unit",
        }
        for annotation in result["labels"]:
            assert required_keys.issubset(annotation.keys()), (
                f"annotation {annotation.get('label_id', '?')} missing keys: "
                f"{required_keys - annotation.keys()}"
            )


class TestAnnotateMultipleEpisodes:
    """Batch annotation and summary generation."""

    def test_batch_annotations(self, schema: dict) -> None:
        """annotate_episodes_social_preferences returns one result per episode."""

        episodes = [
            _make_episode({"min_clearance": 1.2}),
            _make_episode({"min_clearance": -0.05}),
            _make_episode({"jerk_mean": 0.1}),
        ]
        results = annotate_episodes_social_preferences(episodes, schema=schema)

        assert len(results) == 3
        assert results[0]["labels"] is not None

    def test_build_label_summary_counts(self, schema: dict) -> None:
        """build_label_summary rolls up annotation counts per label."""

        episodes = [
            _make_episode({"min_clearance": 1.2, "jerk_mean": 0.05}),
            _make_episode({"min_clearance": -0.05, "jerk_mean": 0.3}),
            _make_episode({"min_clearance": 0.3}),
        ]
        results = annotate_episodes_social_preferences(episodes, schema=schema)
        summary = build_label_summary(results)

        assert summary["total_episodes"] == 3
        assert "clearance" in summary["labels"]
        assert "not_available_reasons" in summary


class TestTraceFieldAvailability:
    """Trace-field-based availability filters override metric presence."""

    def test_trace_fields_filter_clearance(self, schema: dict) -> None:
        """When required trace fields are missing, label is unavailable even with metrics."""

        episode = _make_episode({"min_clearance": 0.8})
        result = annotate_episode_social_preferences(
            episode,
            schema=schema,
            trace_fields={"robot_trajectory"},
        )
        clearance_annotation = next(a for a in result["labels"] if a["label_id"] == "clearance")

        assert clearance_annotation["annotation"] == "not_available"

    def test_trace_fields_allow_clearance(self, schema: dict) -> None:
        """When all required trace fields are present, label uses metric value."""

        episode = _make_episode({"min_clearance": 0.8})
        result = annotate_episode_social_preferences(
            episode,
            schema=schema,
            trace_fields={
                "robot_trajectory",
                "pedestrian_trajectories",
                "metric_parameters.threshold_profile",
            },
        )
        clearance_annotation = next(a for a in result["labels"] if a["label_id"] == "clearance")

        assert clearance_annotation["annotation"] == "acceptable"


class TestGetEpisodeMetricNames:
    """Metric name flattening for diagnostic tracing."""

    def test_flattens_nested_metrics(self) -> None:
        """Nested dicts produce dot-notation keys."""

        episode = {
            "metrics": {
                "min_clearance": 0.5,
                "social_acceptability": {
                    "social_proxemic_min_clearance_m": 0.6,
                },
            }
        }
        names = get_episode_metric_names(episode)

        assert "min_clearance" in names
        assert "social_acceptability.social_proxemic_min_clearance_m" in names

    def test_empty_metrics(self) -> None:
        """Missing metrics returns empty set."""

        episode = {"episode_id": "x"}
        names = get_episode_metric_names(episode)
        assert names == set()


class TestCliSmoke:
    """Smoke tests for the CLI script."""

    def test_annotate_cli_help(self) -> None:
        """The CLI --help flag succeeds."""

        import subprocess

        result = subprocess.run(
            ["python", "-m", "scripts.analysis.annotate_social_preference_labels", "--help"],
            capture_output=True,
            check=False,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert result.returncode == 0
        assert "social preference" in result.stdout.lower() or "social" in result.stdout.lower()

    def test_annotate_cli_writes_output(self, schema: dict) -> None:
        """The CLI reads JSONL stdin, writes annotated JSONL and summary."""

        episode = _make_episode({"min_clearance": 0.8})

        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as ep_file,
            tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as out_file,
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as summary_file,
        ):
            ep_file.write(json.dumps(episode) + "\n")
            ep_file.flush()

            import subprocess

            result = subprocess.run(
                [
                    "python",
                    str(
                        Path(__file__).resolve().parents[2]
                        / "scripts"
                        / "analysis"
                        / "annotate_social_preference_labels.py"
                    ),
                    "--episodes-jsonl",
                    ep_file.name,
                    "--schema",
                    str(CONFIG_PATH),
                    "--output-jsonl",
                    out_file.name,
                    "--summary-json",
                    summary_file.name,
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            with open(out_file.name) as f:
                output_lines = f.read().strip().split("\n")
                assert len(output_lines) == 1
                output = json.loads(output_lines[0])
                assert output["schema_version"] == "social-preference-labels.v1"
                clearance_label = next(a for a in output["labels"] if a["label_id"] == "clearance")
                assert clearance_label["annotation"] == "acceptable"

            with open(summary_file.name) as f:
                summary = json.load(f)
                assert summary["total_episodes"] == 1
                assert "clearance" in summary["labels"]


class TestNestedMetricLookup:
    """The candidate metric key ``metrics.social_acceptability.x`` is resolved correctly."""

    def test_nested_dot_path_cleared(self, schema: dict) -> None:
        """A nested social_proxemic_min_clearance_m metric is found by the annotation."""

        episode = _make_episode(
            {
                "social_acceptability": {
                    "social_proxemic_min_clearance_m": 0.7,
                }
            }
        )
        result = annotate_episode_social_preferences(episode, schema=schema)
        clearance_annotation = next(a for a in result["labels"] if a["label_id"] == "clearance")

        assert clearance_annotation["annotation"] == "acceptable"
        assert clearance_annotation["value"] == 0.7
