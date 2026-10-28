"""Regression tests for the ORCA attribution packet's offline tools."""

from __future__ import annotations

import csv
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent


class ExtractionTests(unittest.TestCase):
    """Exercise the extractor on a deliberately tied terminal window."""

    def test_tied_stalls_have_a_stable_alphabetical_winner(self):
        """A one-to-one tail tie must not depend on Python's hash seed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            contract = {
                "thresholds": {"terminal_window_steps": 100, "eps_mps": 0.05},
                "configurations": [
                    {"key": "hybrid", "algo": "hybrid", "kinematics": "differential_drive"}
                ],
            }
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            steps = []
            for index in range(100):
                # 98 moving steps, one plant stall, one command stall.
                moving = index < 98
                command = 0.1 if index == 98 else 0.0
                pre = {"x": 0.0, "y": 0.0, "heading": 0.0}
                post = {"x": 0.1 if moving else 0.0, "y": 0.0, "heading": 0.0}
                steps.append(
                    {
                        "step": index,
                        "planner": {
                            "selected_action": {"linear_velocity": command, "angular_velocity": 0.0}
                        },
                        "xorca": {"pre": pre, "post": post},
                    }
                )
            record = {
                "scenario_id": "fixture_cell",
                "seed": 1001,
                "status": "failure",
                "algorithm_metadata": {"simulation_step_trace": {"dt": 0.1, "steps": steps}},
            }
            (raw / "main__hybrid__fixture.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            for hash_seed in ("0", "1"):
                with self.subTest(hash_seed=hash_seed):
                    out = root / f"out-{hash_seed}"
                    subprocess.run(
                        [
                            sys.executable,
                            str(PACKET / "extract.py"),
                            "--raw-dir",
                            str(raw),
                            "--stage",
                            "main",
                            "--contract",
                            str(contract_path),
                            "--out-dir",
                            str(out),
                        ],
                        env={**os.environ, "PYTHONHASHSEED": hash_seed},
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    with (out / "episodes.csv").open(newline="", encoding="utf-8") as handle:
                        row = next(csv.DictReader(handle))
                    self.assertEqual(row["tail_moving"], "98")
                    self.assertEqual(row["tail_plant_zero"], "1")
                    self.assertEqual(row["tail_command_zero"], "1")
                    self.assertEqual(row["tail_dominant_stall"], "command_zero")


class ReconstructionTests(unittest.TestCase):
    """Exercise the comparator with both allowed and forbidden differences."""

    ORIGINAL = "a4a31e3575a130bbc0dfccec7fbc173ff114fa42"
    RERUN = "b" * 40

    def setUp(self):
        """Build two small raw and extracted datasets with different provenance."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract = self.root / "contract.json"
        self.contract.write_text(
            json.dumps(
                {
                    "thresholds": {"terminal_window_steps": 100, "eps_mps": 0.05},
                    "configurations": [
                        {"key": "hybrid", "algo": "hybrid", "kinematics": "differential_drive"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.paths = {}
        for name, commit, suffix in (
            ("reference", self.ORIGINAL, "batch"),
            ("rerun", self.RERUN, "one"),
        ):
            raw = self.root / f"{name}-raw"
            raw.mkdir()
            path = raw / f"main__hybrid__fixture__{suffix}.jsonl"
            pose = {"x": 0.0, "y": 0.0, "heading": 0.0}
            record = {
                "algo": "hybrid",
                "scenario_id": "fixture_cell",
                "seed": 1001,
                "git_hash": commit,
                "config_hash": suffix,
                "timestamps": {"start": suffix},
                "provenance": {"git_hash": commit},
                "result_provenance": {"repo_commit": commit},
                "event_ledger": {"software_commit": commit},
                "status": "failure",
                "termination_reason": "terminated",
                "metrics": {"deadlock": True},
                "metric_values": {},
                "interaction_exposure": {},
                "safety_predicates": {},
                "spawn_validity": {},
                "steps": 1,
                "horizon": 100,
                "outcome": {"timeout_event": True},
                "algorithm_metadata": {
                    "simulation_step_trace": {
                        "dt": 0.1,
                        "steps": [
                            {
                                "step": 0,
                                "planner": {
                                    "selected_action": {
                                        "linear_velocity": 0.0,
                                        "angular_velocity": 0.0,
                                    }
                                },
                                "xorca": {"pre": pose, "post": pose},
                            }
                        ],
                    }
                },
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            Path(f"{path}.provenance.json").write_text(
                json.dumps({"run": {"repo_commit": commit}}), encoding="utf-8"
            )
            output = self.root / f"{name}-extract"
            subprocess.run(
                [
                    sys.executable,
                    str(PACKET / "extract.py"),
                    "--raw-dir",
                    str(raw),
                    "--stage",
                    "main",
                    "--contract",
                    str(self.contract),
                    "--out-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.paths[name] = (raw, output, path)

    def compare(self):
        """Run the comparator without hiding expected failure statuses."""
        reference_raw, reference_extract, _ = self.paths["reference"]
        rerun_raw, rerun_extract, _ = self.paths["rerun"]
        return subprocess.run(
            [
                sys.executable,
                str(PACKET / "compare_reconstruction.py"),
                "--reference-raw",
                str(reference_raw),
                "--reference-extract",
                str(reference_extract),
                "--rerun-raw",
                str(rerun_raw),
                "--rerun-extract",
                str(rerun_extract),
                "--rerun-commit",
                self.RERUN,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_accepts_only_expected_provenance_differences(self):
        """Accept changed commit, filename, config hash, and timestamps when declared."""
        result = self.compare()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["compared_episode_count"], 1)
        self.assertEqual(report["compared_step_count"], 1)
        for field in ("source_file", "git_hash", "config_hash", "timestamps"):
            self.assertEqual(report["volatile_episode_difference_counts"][field], 1)
        self.assertEqual(report["mismatches"], [])

    def test_rejects_raw_trace_and_scientific_drift(self):
        """Detect independent changes to the raw trace and scientific metrics."""
        path = self.paths["rerun"][2]
        record = json.loads(path.read_text(encoding="utf-8"))
        record["metrics"]["deadlock"] = False
        record["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["xorca"]["post"]["x"] = (
            1.0
        )
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        result = self.compare()
        self.assertEqual(result.returncode, 1)
        mismatches = json.loads(result.stdout)["mismatches"]
        self.assertTrue(any(item.get("raw_trace") for item in mismatches))
        self.assertTrue(
            any("metrics" in item.get("raw_scientific_fields", []) for item in mismatches)
        )

    def test_rejects_mismatched_source_commit(self):
        """Reject an undeclared sidecar source identity."""
        path = self.paths["rerun"][2]
        sidecar = Path(f"{path}.provenance.json")
        sidecar.write_text(json.dumps({"run": {"repo_commit": self.ORIGINAL}}), encoding="utf-8")
        result = self.compare()
        self.assertEqual(result.returncode, 1)
        self.assertIn("sidecar commit mismatch", result.stdout)

    def test_rejects_contradictory_embedded_source_identity(self):
        """Reject embedded source identities that contradict the declared commit."""
        path = self.paths["rerun"][2]
        record = json.loads(path.read_text(encoding="utf-8"))
        record["provenance"]["git_hash"] = self.ORIGINAL
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        result = self.compare()
        self.assertEqual(result.returncode, 1)
        self.assertIn("provenance.git_hash mismatch", result.stdout)

    def test_rejects_changed_extracted_science(self):
        """Reject changes in an extracted scientific outcome."""
        path = self.paths["rerun"][1] / "episodes.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            assert fields is not None
            rows = list(reader)
        rows[0]["success"] = "1"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        result = self.compare()
        self.assertEqual(result.returncode, 1)
        mismatches = json.loads(result.stdout)["mismatches"]
        self.assertTrue(any("success" in item.get("episode_fields", []) for item in mismatches))

    def test_rejects_two_equally_truncated_step_tables(self):
        """Reject matching extracted tables that omit steps present in the raw trace."""
        for _, output, _ in self.paths.values():
            path = output / "steps.csv.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                header = handle.readline()
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(header)
        result = self.compare()
        self.assertEqual(result.returncode, 1)
        self.assertTrue(
            any("extracted_step_coverage" in m for m in json.loads(result.stdout)["mismatches"])
        )


if __name__ == "__main__":
    unittest.main()
