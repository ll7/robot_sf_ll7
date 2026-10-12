"""Real API forgeries against the retained checkpoint's effective-input contract."""

# evidence-writer-exempt: temporary adversarial fixtures never become durable evidence.

from __future__ import annotations

import gzip
import hashlib
import json
import math
import tempfile
from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.counterexample_corpus import admit_window
from robot_sf.benchmark.scenario_generation.persistence_gate import (
    ScenarioPersistenceValidationError,
)
from robot_sf.benchmark.utils import _config_hash

ROOT = Path(__file__).resolve().parents[2]
ADMISSION = ROOT / "docs/context/evidence/hardcase_positive_control/admission"
RECEIPT = ADMISSION / "window_receipt.json"
pytestmark = pytest.mark.new_feature_no_base(
    reason="base has no source-bound corpus input contract"
)


def _binding(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _reidentify(row: dict) -> None:
    digest = _config_hash(row["scenario_params"])
    row["config_hash"] = digest
    row["episode_id"] = f"{row['scenario_id']}--{row['seed']}--{digest}"
    row["result_provenance"]["config_hash"] = digest


def _attack(tmp_path: Path, mutate) -> None:
    receipt = json.loads(RECEIPT.read_text())
    packet = (ROOT / receipt["bindings"]["scenario"]["path"]).parent
    with (
        tempfile.TemporaryDirectory(prefix=".contract-negative-", dir=ROOT) as location,
        tempfile.NamedTemporaryFile(suffix=".yaml", dir=packet) as scenario_file,
    ):
        folder = Path(location)
        document = yaml.safe_load((ROOT / receipt["bindings"]["scenario"]["path"]).read_text())
        rows = [
            json.loads(gzip.decompress((ROOT / b["path"]).read_bytes()))
            for b in receipt["bindings"]["replays"]
        ]
        manifests = [
            json.loads((ROOT / b["path"]).read_text())
            for b in receipt["bindings"]["replay_manifests"]
        ]
        mutate(document["scenarios"][0], rows, manifests)
        scenario_path = Path(scenario_file.name)
        scenario_path.write_text(yaml.safe_dump(document))
        receipt["bindings"]["scenario"] = _binding(scenario_path)
        for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
            manifest["inputs"]["scenario_matrix"].update(
                receipt["bindings"]["scenario"],
                resolved_path=receipt["bindings"]["scenario"]["path"],
            )
            _reidentify(row)
            manifest["rows"][0].update(episode_id=row["episode_id"], config_hash=row["config_hash"])
            for key, value in (("replays", row), ("replay_manifests", manifest)):
                path = folder / f"{key}-{index}.json"
                path.write_text(json.dumps(value))
                receipt["bindings"][key][index] = _binding(path)
        path = tmp_path / "receipt.json"
        path.write_text(json.dumps(receipt))
        candidate = json.loads((ADMISSION / "candidate.json").read_text())
        candidate["window_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        candidate_path = tmp_path / "candidate.json"
        candidate_path.write_text(json.dumps(candidate))
        corpus = tmp_path / "corpus"
        with pytest.raises(ScenarioPersistenceValidationError, match="effective"):
            admit_window(candidate_path, path, corpus_dir=corpus, repo_root=ROOT)
        assert not corpus.exists()


@pytest.mark.parametrize(
    "probe", ["density_rebound", "timestep_rebound", "manifest_timestep", "latency_envelope"]
)
def test_refuse_four_round6_reviewer_forgeries(tmp_path: Path, probe: str) -> None:
    """Each attack was admitted at 03cd1c70 despite unchanged checkpoint/traces."""

    def mutate(scenario, rows, manifests):
        if probe == "density_rebound":
            scenario["simulation_config"]["ped_density"] = 0.0
            for row in rows:
                row["scenario_params"]["simulation_config"]["ped_density"] = 0.0
        elif probe == "timestep_rebound":
            for row in rows:
                row["scenario_params"]["run_dt"] = 0.2
                row["result_provenance"]["simulator_settings"]["dt"] = 0.2
            for manifest in manifests:
                manifest["rows"][0]["simulator_settings"]["dt"] = 0.2
        elif probe == "manifest_timestep":
            for manifest in manifests:
                manifest["rows"][0]["simulator_settings"]["dt"] = 0.2
        else:
            for row in rows:
                row["scenario_params"]["latency_stress_profile"] = {
                    "enabled": True,
                    "action_latency_steps": 1,
                }

    _attack(tmp_path, mutate)


def _leaves(value, path=()):
    if isinstance(value, dict) and value:
        for name, item in value.items():
            yield from _leaves(item, (*path, name))
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path


def _changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    if isinstance(value, int | float):
        return value + 1
    if isinstance(value, str):
        return value + "-forged"
    if isinstance(value, list):
        return ["forged"]
    return "forged"


def _profile():
    receipt = json.loads(RECEIPT.read_text())
    checkpoint = json.loads((ROOT / receipt["bindings"]["checkpoint"]["path"]).read_text())
    return checkpoint["effective_inputs"]


@pytest.mark.parametrize("carrier", ["metadata", "simulator", "manifest", "trace"])
@pytest.mark.parametrize("path", list(_leaves(_profile())), ids=lambda p: "/".join(map(str, p)))
def test_refuse_mutation_of_any_one_effective_input(
    tmp_path: Path, carrier: str, path: tuple
) -> None:
    """Every leaf of the whole captured object must agree in each independent carrier."""

    def mutate(_scenario, rows, manifests):
        row = rows[0]
        if carrier == "metadata":
            inputs = row["algorithm_metadata"]["generated_window_continuation"]["effective_inputs"]
        elif carrier == "simulator":
            inputs = row["result_provenance"]["simulator_settings"]["window_effective_inputs"]
        elif carrier == "manifest":
            inputs = manifests[0]["rows"][0]["simulator_settings"]["window_effective_inputs"]
        else:
            inputs = row["algorithm_metadata"]["simulation_step_trace"]["window_effective_inputs"]
        parent = inputs
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = _changed(parent[path[-1]])

    _attack(tmp_path, mutate)


@pytest.mark.parametrize("mutation", ["add_unknown", "remove_required"])
def test_whole_input_contract_refuses_added_or_removed_keys(tmp_path: Path, mutation: str) -> None:
    """Whole-object comparison must stay fail-closed as input fields evolve."""

    def mutate(_scenario, rows, _manifests):
        inputs = rows[0]["algorithm_metadata"]["generated_window_continuation"]["effective_inputs"]
        if mutation == "add_unknown":
            inputs["new_effective_option"] = True
        else:
            del inputs["configuration"]

    _attack(tmp_path, mutate)
