#!/usr/bin/env python3
"""Stdlib coverage for the no-submit campaign canary contract."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/tools/slurm_campaign_preflight.py"
SPEC = importlib.util.spec_from_file_location("slurm_campaign_preflight", SCRIPT)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def manifest(config: Path) -> dict:
    return {
        "campaign_id": "horizon-ablation",
        "expected_public_commit": "a" * 40,
        "public_commit": "a" * 40,
        "packet": {"config": str(config), "sha256": preflight._sha256(config)},
        "cells": [
            {
                "key": "h500",
                "native": True,
                "available": True,
                "ok": True,
                "declared_rows": 1728,
                "instantiated_rows": 1728,
                "output_root": "/results/h500",
                "artifact_contract": "stdout,manifest,summary",
            },
            {
                "key": "h600",
                "native": True,
                "available": True,
                "ok": True,
                "declared_rows": 1728,
                "instantiated_rows": 1728,
                "output_root": "/results/h600",
                "artifact_contract": "stdout,manifest,summary",
            },
        ],
        "paired_keys": ["h500", "h600"],
        "aggregate": {"status": "ok", "artifact_contract": "paired-summary.json"},
    }


def run_preflight(payload: dict, manifest_path: Path, **kwargs: object) -> dict:
    kwargs.setdefault("actual_public_commit", str(payload["public_commit"]))
    return preflight.preflight(payload, manifest_path=manifest_path, **kwargs)


class CampaignPreflightTests(unittest.TestCase):
    """Exercise safe full-campaign and canary admission boundaries."""

    def test_canary_passes_without_scheduler_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            report = run_preflight(
                manifest(config),
                root / "manifest.json",
                canary_key="h500",
            )
        self.assertTrue(report["canary_safe"])
        self.assertFalse(report["submit_safe"])
        self.assertFalse(report["full_campaign_safe"])
        self.assertTrue(report["no_submit"])
        self.assertEqual(report["status"], "canary_ready")
        self.assertEqual(report["canary_coverage"]["mode"], "canary")
        self.assertEqual(report["planner_keys"], ["h500"])
        self.assertEqual(report["canary_coverage"]["unvalidated_keys"], ["h600"])

    def test_canary_does_not_validate_unselected_cell_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            payload = manifest(config)
            payload["cells"][1]["ok"] = False
            report = run_preflight(
                payload,
                root / "manifest.json",
                canary_key="h500",
            )
        self.assertTrue(report["canary_safe"])
        self.assertFalse(report["submit_safe"])

    def test_declared_instantiated_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            payload = manifest(config)
            payload["cells"][0]["instantiated_rows"] = 1727
            report = run_preflight(payload, root / "manifest.json")
        self.assertFalse(report["submit_safe"])
        self.assertTrue(any("declared_rows" in item for item in report["blockers"]))

    def test_paired_key_and_native_status_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            payload = manifest(config)
            payload["paired_keys"] = ["h500", "h600", "h700"]
            payload["cells"][1]["ok"] = False
            report = run_preflight(payload, root / "manifest.json")
        self.assertFalse(report["submit_safe"])
        self.assertTrue(any("paired campaign cell" in item for item in report["blockers"]))
        self.assertTrue(any("missing status proof" in item for item in report["blockers"]))

    def test_placeholder_and_commit_mismatch_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            payload = manifest(config)
            payload["cells"][0]["output_root"] = "<campaign-root>"
            report = run_preflight(
                payload,
                root / "manifest.json",
                actual_public_commit="b" * 40,
            )
        self.assertFalse(report["submit_safe"])
        self.assertTrue(any("placeholder" in item for item in report["blockers"]))
        self.assertTrue(any("commit mismatch" in item for item in report["blockers"]))

    def test_non_empty_output_root_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            output = root / "output"
            output.mkdir()
            (output / "old.json").write_text("{}\n", encoding="utf-8")
            report = run_preflight(
                manifest(config),
                root / "manifest.json",
                output_root=output,
            )
        self.assertFalse(report["submit_safe"])
        self.assertTrue(any("output root" in item for item in report["blockers"]))

    def test_missing_packet_hash_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            payload = manifest(config)
            payload["packet"].pop("sha256")
            report = run_preflight(payload, root / "manifest.json")
        self.assertFalse(report["submit_safe"])
        self.assertIn("packet.sha256 is missing", report["blockers"])

    def test_relative_packet_config_must_be_a_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            payload = manifest(config)
            payload["packet"] = {
                "config": "missing/config.json",
                "sha256": "0" * 64,
            }
            report = run_preflight(
                payload,
                root / "manifest.json",
            )
        self.assertFalse(report["submit_safe"])
        self.assertTrue(any("packet config does not exist" in item for item in report["blockers"]))

    def test_full_commit_oids_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            report = run_preflight(
                manifest(config),
                root / "manifest.json",
                expected_public_commit="a" * 7,
                actual_public_commit="a" * 8,
            )
        self.assertFalse(report["submit_safe"])
        self.assertIn("expected public commit is missing or invalid", report["blockers"])
        self.assertIn("actual public commit is missing or invalid", report["blockers"])

    def test_manifest_commit_is_not_an_actual_commit_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            report = preflight.preflight(manifest(config), manifest_path=root / "manifest.json")
        self.assertFalse(report["submit_safe"])
        self.assertTrue(any("must be supplied explicitly" in item for item in report["blockers"]))

    def test_cells_cannot_share_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            payload = manifest(config)
            payload["cells"][1]["output_root"] = payload["cells"][0]["output_root"]
            report = run_preflight(payload, root / "manifest.json")
        self.assertFalse(report["submit_safe"])
        self.assertTrue(any("share output_root" in item for item in report["blockers"]))

    def test_output_root_aliases_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            output = root / "outputs" / "h500"
            payload = manifest(config)
            payload["cells"][0]["output_root"] = str(output)
            payload["cells"][1]["output_root"] = f"{output}/"
            report = run_preflight(payload, root / "manifest.json")
        self.assertFalse(report["submit_safe"])
        self.assertTrue(any("share output_root" in item for item in report["blockers"]))

    def test_each_cell_output_root_must_be_reservable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            output = root / "occupied"
            output.mkdir()
            (output / "old.json").write_text("{}\n", encoding="utf-8")
            payload = manifest(config)
            payload["cells"][0]["output_root"] = str(output)
            report = run_preflight(payload, root / "manifest.json")
        self.assertFalse(report["submit_safe"])
        self.assertTrue(any("cells[0].output_root" in item for item in report["blockers"]))

    def test_full_coverage_rejects_malformed_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            payload = manifest(config)
            payload["cells"].append("not-a-cell")
            report = run_preflight(payload, root / "manifest.json")
        self.assertFalse(report["submit_safe"])
        self.assertIn("cells[2] must be an object", report["blockers"])

    def test_json_cli_schema_is_stable_and_no_submit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest(config)), encoding="utf-8")
            self.assertEqual(
                preflight.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--actual-public-commit",
                        "a" * 40,
                        "--json",
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
