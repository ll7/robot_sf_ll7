"""Round-7 probes: incomplete checkpoint authority must never admit a window.

Refusal of this evidence class does not prove individual metadata fields are
validated. The unchanged control must also refuse until capture owns every
event criterion and provenance identity.
"""

# evidence-writer-exempt: temporary forged artifacts exercise the real API and
# are deleted with their context; no durable evidence is written here.

from __future__ import annotations

import gzip
import hashlib
import json
import math
import tempfile
from pathlib import Path

import pytest

from robot_sf.adversarial.counterexample_corpus import admit_window, export_replay_slice
from robot_sf.benchmark.scenario_generation.persistence_gate import (
    ScenarioPersistenceValidationError,
)

ROOT = Path(__file__).resolve().parents[2]
ADMISSION = ROOT / "docs/context/evidence/hardcase_positive_control/admission"
pytestmark = pytest.mark.new_feature_no_base(reason="base has no corpus admission API")

NATIVE_PROBES = [
    ("planner-speed", ("config", "max_linear_speed"), 0.25),
    ("planner-type", ("config", "max_linear_speed"), "2.0"),
    ("visibility-enable", ("observation_visibility", "enabled"), True),
    ("visibility-fov", ("observation_visibility", "fov_degrees"), 30.0),
    ("tracking-spec", ("tracking_precision", "spec", "enabled"), True),
    ("observation-mode", ("observation_level", "active_observation_mode"), "lidar"),
    ("runtime-dt", ("planner_runtime", "simulation_timestep", "seconds"), 0.2),
]
PROBES = [
    *[p[0] for p in NATIVE_PROBES],
    "planner-extra",
    "planner-remove",
    "planner-float-noise",
    "planner-integer",
    "manifest-config_hash",
    "manifest-repo_commit",
    "manifest-scenario_id",
    "source-robot-radius",
    "source-near-miss-threshold",
]


def _bind(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _mutate_config(probe: str, config: dict) -> None:
    """Cover numeric type/precision and complete dictionary shape contradictions."""
    if probe == "planner-extra":
        config["new_effective_option"] = True
    elif probe == "planner-remove":
        del config["max_linear_speed"]
    elif probe == "planner-float-noise":
        config["max_linear_speed"] = math.nextafter(2.0, math.inf)
    elif probe == "planner-integer":
        config["max_linear_speed"] = 2
    else:
        raise AssertionError(probe)


def _mutate(probe: str, source: dict, rows: list[dict], manifests: list[dict]) -> None:
    """Change one declaration, preserving every raw trajectory and input digest."""
    for label, keys, value in NATIVE_PROBES:
        if probe == label:
            target = rows[0]["algorithm_metadata"]
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            return
    if probe.startswith("planner-"):
        _mutate_config(probe, rows[0]["algorithm_metadata"]["config"])
    elif probe.startswith("manifest-"):
        key = probe.removeprefix("manifest-")
        values = {
            "config_hash": "0" * 16,
            "repo_commit": "0" * 40,
            "scenario_id": "another_scenario",
        }
        # No later reidentification may overwrite this forgery.
        manifests[0]["rows"][0][key] = values[key]
    elif probe == "source-robot-radius":
        source["effective_physics"]["clearance_definition"]["robot_radius_m"] = 1.2
    elif probe == "source-near-miss-threshold":
        source["metric_parameters"]["threshold_profile"]["near_miss_distance_m"] = 5.0
    else:
        assert probe == "unchanged-control"


def _assert_class_refused(probe: str) -> None:
    receipt = json.loads((ADMISSION / "window_receipt.json").read_text())
    with tempfile.TemporaryDirectory(prefix=".authority-negative-", dir=ROOT) as location:
        folder = Path(location)
        source = json.loads(
            gzip.decompress((ROOT / receipt["bindings"]["source_episode"]["path"]).read_bytes())
        )
        rows = [
            json.loads(gzip.decompress((ROOT / b["path"]).read_bytes()))
            for b in receipt["bindings"]["replays"]
        ]
        manifests = [
            json.loads((ROOT / b["path"]).read_text())
            for b in receipt["bindings"]["replay_manifests"]
        ]
        _mutate(probe, source, rows, manifests)
        source_path = folder / "source.json"
        source_path.write_text(json.dumps(source))
        receipt["bindings"]["source_episode"] = _bind(source_path)
        for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
            for key, value in (("replays", row), ("replay_manifests", manifest)):
                path = folder / f"{key}-{index}.json"
                path.write_text(json.dumps(value))
                receipt["bindings"][key][index] = _bind(path)
        receipt_path = folder / "receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        candidate = json.loads((ADMISSION / "candidate.json").read_text())
        candidate["window_receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        candidate_path = folder / "candidate.json"
        candidate_path.write_text(json.dumps(candidate))
        corpus = folder / "corpus"
        with pytest.raises(
            ScenarioPersistenceValidationError, match="checkpoint-owned admission authority"
        ):
            admit_window(candidate_path, receipt_path, corpus_dir=corpus, repo_root=ROOT)
        assert not corpus.exists(), "refusal must precede the lock, index and case writes"


@pytest.mark.parametrize("probe", PROBES)
def test_incomplete_authority_refuses_all_sixteen_reviewer_probes(probe: str) -> None:
    """All 16 were admitted before class refusal; this is not field-validation proof."""
    _assert_class_refused(probe)


def test_incomplete_authority_refuses_the_unchanged_positive_control() -> None:
    """Parity alone cannot certify criteria/identities absent from the checkpoint."""
    _assert_class_refused("unchanged-control")


def test_historical_admission_is_not_exported_as_currently_verified() -> None:
    """A case admitted by the superseded gate must not bypass today's refusal."""
    with pytest.raises(
        ScenarioPersistenceValidationError, match="checkpoint-owned admission authority"
    ):
        export_replay_slice(ADMISSION / "corpus")
