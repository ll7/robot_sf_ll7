"""Focused tests for the issue #5498 native-PPO exact-repeat slice runner.

These validate the single-host-achievable capability delivered under #5498:
the PPO-only bundle/manifest slicing helpers and the end-to-end
execute -> verify-host path on a small PPO subset. They do NOT run the full
60-target campaign and do NOT attempt the two-host cross-host matrix (which is
structurally impossible on one machine).
"""

from __future__ import annotations

import json
from pathlib import Path

import robot_sf.benchmark.exact_repeat_campaign as erc
from scripts.benchmark.run_issue_5498_native_ppo_slice import (
    TOTAL_PPO_TARGETS,
    _ppo_only_bundle,
    _ppo_only_manifest_slice,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "docs/context/evidence/issue_5263_exact_repeat"
BUNDLE_PATH = EVIDENCE_DIR / "resolved_definitions.json"
MANIFEST_PATH = EVIDENCE_DIR / "exact_repeat_manifest.json"


def _require_existing(path: Path) -> Path:
    assert path.is_file(), f"required evidence fixture missing: {path}"
    return path


def test_ppo_only_bundle_is_subset_and_rehashed() -> None:
    _require_existing(BUNDLE_PATH)
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    full_ppo = [t for t in bundle["targets"] if t["planner"] == "ppo"]

    slim = _ppo_only_bundle(None)
    assert len(slim["targets"]) == len(full_ppo) == TOTAL_PPO_TARGETS
    assert all(t["planner"] == "ppo" for t in slim["targets"])
    # A re-hashed slice must not equate the full bundle's hash.
    assert slim["bundle_sha256"] == erc.canonical_sha256(
        {k: v for k, v in slim.items() if k != "bundle_sha256"}
    )
    assert slim["bundle_sha256"] != bundle["bundle_sha256"]


def test_ppo_only_bundle_smoke_cap() -> None:
    _require_existing(BUNDLE_PATH)
    slim = _ppo_only_bundle(2)
    assert len(slim["targets"]) == 2
    assert all(t["planner"] == "ppo" for t in slim["targets"])


def test_ppo_only_manifest_slice_verifies_against_subset(tmp_path: Path) -> None:
    _require_existing(BUNDLE_PATH)
    _require_existing(MANIFEST_PATH)
    slim = _ppo_only_bundle(2)

    # Execute the tiny PPO subset and verify against the re-hashed manifest slice.
    host = erc.execute_campaign(slim, output_dir=tmp_path / "test_5498_slice")
    manifest_slice = _ppo_only_manifest_slice(
        [(r["scenario_id"], r["planner"], int(r["seed"])) for r in host["results"]]
    )
    aligned = dict(host)
    aligned["manifest_sha256"] = manifest_slice["manifest_sha256"]
    verified = erc.verify_host_report(manifest_slice, aligned)

    assert verified["summary"]["n_targets"] == 2
    assert verified["summary"]["n_runnable_targets"] == 2
    assert verified["summary"]["all_cells_bitwise_identical"] is True
    # Guard: the slice must not silently match the full 140-target manifest hash.
    full_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest_slice["manifest_sha256"] != full_manifest["manifest_sha256"]
