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
from robot_sf.evidence.writers import write_json
from tests.unit.benchmark.test_snqi_v2 import spec_files as _spec_files

spec_files = _spec_files
ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml"
)


def _binding_api():
    """Collect on main, then refuse its missing authored contract before importing new APIs.

    Returns:
        Production binding API after proving the authored source prerequisite.
    """
    raw = yaml.safe_load(CONFIG.read_bytes())
    assert raw.get("snqi_v2_spec") is not None, "D-083 authored source has no SNQI-v2 binding"
    from robot_sf.benchmark.snqi import v2_binding

    return v2_binding


def test_pending_campaign_cannot_execute(monkeypatch):
    """No direct campaign route may silently publish an unscored SNQI-v2 campaign."""
    from robot_sf.benchmark.camera_ready import campaign

    cfg = load_campaign_config(CONFIG)

    def forbidden_execution(*args, **kwargs):
        pytest.fail("unscored authored campaign reached execution")

    monkeypatch.setattr(campaign, "_run_campaign_orchestrator", forbidden_execution)
    with pytest.raises(ValueError, match="acquisition and anchors are required"):
        run_campaign(cfg)


@pytest.mark.parametrize("mutation", ["missing", "digest", "rule", "status"])
def test_binding_refuses_incomplete_or_changed_source_assets(tmp_path, mutation):
    """A typed pending contract rejects missing pins, changed bytes and fake frozen source."""
    binding_api = _binding_api()
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
        binding_api.load_acquisition_binding(raw, CONFIG)


@pytest.mark.parametrize("mutation", ["none", "source", "seeds", "anchors", "acquisition"])
def test_acquired_anchors_are_bound_before_scoring(tmp_path, monkeypatch, spec_files, mutation):
    """Actual strict scoring loader plus a freeze spy verifies source/custody attachment."""
    binding_api = _binding_api()
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
            binding_api.bind_acquired_anchors(cfg, **kwargs)
    else:
        scored = binding_api.bind_acquired_anchors(cfg, **kwargs)
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
    binding_api = _binding_api()
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
        scored = binding_api.bind_acquired_anchors(cfg, **kwargs)
        assert scored.snqi_v2_spec.calibration_seeds == (1001, 1002)
        assert not scored.snqi_v2_spec.diagnostic
    else:
        with pytest.raises(ValueError, match="source differs|acquisition configuration"):
            binding_api.bind_acquired_anchors(cfg, **kwargs)


@pytest.mark.parametrize("allowed", [False, True])
def test_rehearsal_option_reaches_campaign_implementation(monkeypatch, allowed):
    """The compatibility entry must forward the explicit preparatory flag without admission.

    Forgetting the keyword raises before the native runner; earlier CLI tests
    replaced the facade and missed this seam. This uses the actual entry and
    its existing implementation hook, with no production test-only seam.
    """
    from robot_sf.benchmark import camera_ready_campaign as facade

    cfg = object()

    def implementation(observed, **kwargs):
        assert observed is cfg
        return kwargs

    monkeypatch.setattr(facade, "_run_campaign_impl", implementation)
    result = facade.run_campaign(cfg, allow_pending_snqi_v2=allowed)
    assert result["allow_pending_snqi_v2"] is allowed


@pytest.mark.parametrize("route", ["native", "facade"])
@pytest.mark.parametrize("identity", [None, "not-an-identity.json"])
def test_pending_permission_requires_rehearsal_identity_before_execution(
    monkeypatch, route, identity
):
    """The caller flag alone cannot unlock an unscored source-bound campaign."""
    from robot_sf.benchmark.camera_ready import campaign
    from robot_sf.benchmark.camera_ready_campaign import SeedPolicy

    cfg = replace(
        load_campaign_config(CONFIG), seed_policy=SeedPolicy(mode="fixed-list", seeds=(1001,))
    )

    def forbidden_execution(*args, **kwargs):
        pytest.fail("pending keyword bypassed identity admission")

    monkeypatch.setattr(campaign, "_run_campaign_orchestrator", forbidden_execution)
    execute = campaign.run_campaign if route == "native" else run_campaign
    kwargs = {"allow_pending_snqi_v2": True}
    if identity is not None:
        kwargs["pending_snqi_v2_identity"] = ROOT / identity
    with pytest.raises(
        (ValueError, FileNotFoundError), match="rehearsal identity|resolved release identity"
    ):
        execute(cfg, **kwargs)


@pytest.mark.parametrize(
    "seeds,drift",
    [((1001,), False), ((42,), False), ((1001, 42), False), ((1001,), True)],
    ids=["matched_dev", "non_dev", "mixed_seeds", "changed_config"],
)
def test_pending_permission_matches_verified_identity_config(monkeypatch, seeds, drift):
    """Even verified development identity input must match the actual runner config."""
    from types import SimpleNamespace

    from robot_sf.benchmark import release_protocol
    from robot_sf.benchmark.camera_ready import campaign
    from robot_sf.benchmark.camera_ready_campaign import SeedPolicy

    expected = replace(
        load_campaign_config(CONFIG), seed_policy=SeedPolicy(mode="fixed-list", seeds=(1001,))
    )
    cfg = replace(
        expected,
        seed_policy=SeedPolicy(mode="fixed-list", seeds=seeds),
        dt=0.2 if drift else expected.dt,
    )
    identity = SimpleNamespace(release_kind="development_rehearsal", resolved_seeds=(1001,))
    calls = []
    monkeypatch.setattr(release_protocol, "verify_resolved_release_identity", lambda path: identity)
    monkeypatch.setattr(release_protocol, "load_release_campaign_config", lambda manifest: expected)
    monkeypatch.setattr(
        campaign,
        "_run_campaign_orchestrator",
        lambda *a, **k: calls.append(a[0]) or {"diagnostic": True},
    )
    if seeds == (1001,) and not drift:
        assert campaign.run_campaign(
            cfg,
            allow_pending_snqi_v2=True,
            pending_snqi_v2_identity=ROOT / "output/rehearsal/identity.json",
        ) == {"diagnostic": True}
        assert calls == [cfg]
    else:
        with pytest.raises(ValueError, match="development seeds|identity config"):
            campaign.run_campaign(
                cfg,
                allow_pending_snqi_v2=True,
                pending_snqi_v2_identity=ROOT / "output/rehearsal/identity.json",
            )
        assert calls == []
