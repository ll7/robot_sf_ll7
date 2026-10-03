"""Post-freeze anchor wiring and fail-closed refusal for #10112.

Protect custody/source matching and pending-execution refusal. Swapping a source
or anchors is a credible regression. Existing calibration tests validate the
archive, but do not attach its output to the release runner. Synthetic freeze
spy below tests this wiring only; real acquisition remains separate evidence.
"""

import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.camera_ready._util import _config_hash_payload
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config, run_campaign
from robot_sf.benchmark.snqi import v2_calibration
from robot_sf.benchmark.snqi.v2_binding import bind_acquired_anchors, load_acquisition_binding
from robot_sf.evidence.writers import write_json
from tests.unit.benchmark.test_snqi_v2 import spec_files as _spec_files

spec_files = _spec_files
ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml"
)


def test_pending_campaign_cannot_execute():
    """No direct campaign route may silently publish an unscored SNQI-v2 campaign."""
    cfg = load_campaign_config(CONFIG)
    assert cfg.snqi_v2_binding is not None
    assert cfg.snqi_v2_spec is None
    with pytest.raises(ValueError, match="acquisition and anchors are required"):
        run_campaign(cfg)


@pytest.mark.parametrize("mutation", ["missing", "digest", "rule", "status"])
def test_binding_refuses_incomplete_or_changed_source_assets(tmp_path, mutation):
    """A typed pending contract rejects missing pins, changed bytes and fake frozen source."""
    raw = yaml.safe_load(CONFIG.read_bytes())["snqi_v2_spec"]
    if mutation == "missing":
        raw.pop("family_sha256")
    elif mutation == "digest":
        raw["family_sha256"] = "0" * 64
    elif mutation == "rule":
        raw["anchor_freeze"] = "caller-trusted"
    else:
        path = tmp_path / "anchors.json"
        write_json(path, {"status": "frozen"})
        raw["anchors_path"] = str(path)
        raw["anchors_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SNQI-v2"):
        load_acquisition_binding(raw, CONFIG)


@pytest.mark.parametrize("mutation", ["none", "source", "seeds", "anchors", "acquisition"])
def test_acquired_anchors_are_bound_before_scoring(tmp_path, monkeypatch, spec_files, mutation):
    """Actual strict scoring loader plus a freeze spy verifies source/custody attachment."""
    cfg = load_campaign_config(CONFIG)
    _weights, anchors, _family = spec_files
    document = json.loads(anchors.read_bytes())
    cal = document["calibration"]
    cal["seeds"] = [1001, 1002]
    grid = sorted(product(cal["arms"], cal["scenarios"], cal["seeds"]))
    cal["grid_sha256"] = hashlib.sha256(
        json.dumps(grid, separators=(",", ":")).encode()
    ).hexdigest()
    cal["split_id"] = f"snqi-v2-dev1001-1002-{cal['grid_sha256'][:12]}"
    write_json(anchors, document)
    document = json.loads(anchors.read_bytes())
    root = tmp_path / "acquisition"
    root.mkdir()
    write_json(root / "campaign_manifest.json", {"git": {"commit": "a" * 40}})
    calls = []

    def freeze(campaign_root, output_path, *, campaign_config):
        calls.append(campaign_root)
        assert campaign_config.seed_policy.seeds == (1001, 1002)
        result = json.loads(json.dumps(document))
        if mutation == "seeds":
            result["calibration"]["seeds"] = [101, 102]
        if mutation == "anchors":
            result["anchors"]["F"]["upper"] += 1
        return result

    monkeypatch.setattr(v2_calibration, "freeze_campaign_anchors", freeze)
    if mutation == "acquisition":
        cfg = replace(
            cfg, snqi_v2_binding={**cfg.snqi_v2_binding, "acquisition_config_sha256": "0" * 64}
        )
    kwargs = {
        "calibration_root": root,
        "anchors_path": anchors,
        "source_commit": "b" * 40 if mutation == "source" else "a" * 40,
        "diagnostic": True,
    }
    if mutation != "none":
        with pytest.raises(
            ValueError,
            match={
                "source": "source differs",
                "seeds": "development seeds",
                "anchors": "differ from acquired custody",
                "acquisition": "configuration changed",
            }[mutation],
        ):
            bind_acquired_anchors(cfg, **kwargs)
    else:
        scored = bind_acquired_anchors(cfg, **kwargs)
        assert scored.snqi_v2_spec.diagnostic
        assert scored.snqi_v2_spec.calibration_seeds == (1001, 1002)
        assert (
            scored.snqi_v2_spec.hashes["anchors"]
            == hashlib.sha256(anchors.read_bytes()).hexdigest()
        )
        assert calls == [root]


@pytest.mark.parametrize("mutation", ["none", "source", "configuration"])
def test_reviewed_artifact_route_still_requires_exact_acquisition_identity(
    monkeypatch, spec_files, mutation
):
    """Without raw custody, strict anchors must still bind this source and acquisition config."""
    cfg = load_campaign_config(CONFIG)
    _weights, anchors, _family = spec_files
    document = json.loads(anchors.read_bytes())
    cal = document["calibration"]
    cal["seeds"] = [1001, 1002]
    grid = sorted(product(cal["arms"], cal["scenarios"], cal["seeds"]))
    cal["grid_sha256"] = hashlib.sha256(
        json.dumps(grid, separators=(",", ":")).encode()
    ).hexdigest()
    cal["split_id"] = f"snqi-v2-dev1001-1002-{cal['grid_sha256'][:12]}"
    acquisition = load_campaign_config(cfg.snqi_v2_binding["acquisition_config_path"])
    cal["campaign_config_hash"] = hashlib.sha256(
        json.dumps(
            _config_hash_payload(acquisition), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if mutation == "configuration":
        cal["campaign_config_hash"] = "0" * 64
    write_json(anchors, document)

    def forbidden_freeze(*args, **kwargs):
        pytest.fail("artifact-only binding must not invent raw acquisition custody")

    monkeypatch.setattr(v2_calibration, "freeze_campaign_anchors", forbidden_freeze)
    kwargs = {
        "calibration_root": None,
        "anchors_path": anchors,
        "source_commit": "b" * 40 if mutation == "source" else "a" * 40,
        "diagnostic": False,
    }
    if mutation == "none":
        scored = bind_acquired_anchors(cfg, **kwargs)
        assert scored.snqi_v2_spec.calibration_seeds == (1001, 1002)
        assert not scored.snqi_v2_spec.diagnostic
    else:
        with pytest.raises(ValueError, match="source differs|acquisition configuration"):
            bind_acquired_anchors(cfg, **kwargs)
