"""Actual seed-1001 corpus admission, refusal and append-only duplicate tests."""

# evidence-writer-exempt: pytest tmp_path fixtures intentionally corrupt parity
# receipts; they are never committed or presented as durable generated evidence.

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.counterexample_corpus import (
    admit_window,
    export_replay_slice,
    read_corpus,
)
from robot_sf.benchmark.scenario_generation.persistence_gate import (
    ScenarioPersistenceValidationError,
)
from robot_sf.evidence.writers import write_review_sidecar

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/context/evidence/hardcase_positive_control/admission"
CANDIDATE = EVIDENCE / "candidate.json"
RECEIPT = EVIDENCE / "window_receipt.json"
pytestmark = pytest.mark.new_feature_no_base(reason="base has no versioned corpus admission API")


def test_refuse_density_change_even_with_rebound_scenario_and_manifests(tmp_path: Path) -> None:
    """Rehashing a changed scenario cannot turn unchanged replay rows into evidence."""
    receipt = json.loads(RECEIPT.read_text())
    # Bound artifacts must live under the repository; this isolated temporary
    # directory exercises the real resolver without editing retained evidence.
    with ExitStack() as stack:
        location = stack.enter_context(
            tempfile.TemporaryDirectory(prefix=".hardcase-receipt-test-", dir=ROOT)
        )
        folder = Path(location)
        scenario_path = ROOT / receipt["bindings"]["scenario"]["path"]
        document = yaml.safe_load(scenario_path.read_text())
        scenario = document["scenarios"][0]
        assert scenario["simulation_config"]["ped_density"] == 0.08
        scenario["simulation_config"]["ped_density"] = 0.0
        temporary = stack.enter_context(
            tempfile.NamedTemporaryFile(
                prefix=".density-negative-", suffix=".yaml", dir=scenario_path.parent
            )
        )
        # Keep its directory and every other setting identical to the producer.
        changed = Path(temporary.name)
        changed.write_text(yaml.safe_dump(document))
        binding = {
            "path": changed.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(changed.read_bytes()).hexdigest(),
        }
        receipt["bindings"]["scenario"] = binding
        for index, original in enumerate(receipt["bindings"]["replay_manifests"]):
            manifest = json.loads((ROOT / original["path"]).read_text())
            manifest["inputs"]["scenario_matrix"]["sha256"] = binding["sha256"]
            manifest_path = folder / f"manifest-{index}.json"
            manifest_path.write_text(json.dumps(manifest))
            receipt["bindings"]["replay_manifests"][index] = {
                "path": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
        path = tmp_path / "receipt.json"
        path.write_text(json.dumps(receipt))
        candidate = json.loads(CANDIDATE.read_text())
        candidate["window_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        candidate_path = tmp_path / "candidate.json"
        candidate_path.write_text(json.dumps(candidate))
        corpus = tmp_path / "corpus"
        with pytest.raises(ScenarioPersistenceValidationError, match="effective configuration"):
            admit_window(candidate_path, path, corpus_dir=corpus, repo_root=ROOT)
        assert not corpus.exists()


def test_admit_real_seed1001_window_and_export_slice(tmp_path: Path) -> None:
    """The real command persists one verified case with complete pinned provenance."""
    corpus = tmp_path / "corpus"
    command = [
        sys.executable,
        str(ROOT / "scripts/tools/admit_counterexample_window.py"),
        "--candidate",
        str(CANDIDATE),
        "--window-receipt",
        str(RECEIPT),
        "--corpus",
        str(corpus),
        "--repo-root",
        str(ROOT),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["status"] == "admitted" and receipt["near_duplicates"] == []
    records = read_corpus(corpus)
    assert len(records) == 1 and records[0]["case_id"] == receipt["case_id"]
    record = records[0]
    assert record["identity"]["seed"] == 1001
    assert record["identity"]["source_commit"] == "53721be3248600e08641a9816d094b605e3c93e4"
    assert record["identity"]["source_config_hash"] == "f330dd29077077ac"
    assert (
        record["identity"]["model_digest"]
        == "fe4e5b99c0a9f51ada80cfccbf1361bb4f1872c0c523e2b35ce89e94e7b32958"
    )
    assert record["replay_verification"]["event"]["first_near_miss_step"] == 374
    assert (
        record["feasibility"]["status"] == record["perturbation_persistence"]["status"] == "unknown"
    )
    exported = export_replay_slice(corpus)
    assert len(exported["cases"]) == 1 and exported["cases"][0]["seed"] == 1001
    assert exported["cases"][0]["window"]["stop_kind"] == "window_censored"


def test_refuse_real_window_without_parity_before_any_write(tmp_path: Path) -> None:
    """Removing only parity from the otherwise valid real candidate refuses admission."""
    proof = json.loads(RECEIPT.read_text())
    del proof["parity_proof"]
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(proof))
    candidate = json.loads(CANDIDATE.read_text())
    candidate["window_receipt_sha256"] = hashlib.sha256(missing.read_bytes()).hexdigest()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    corpus = tmp_path / "corpus"
    with pytest.raises(ScenarioPersistenceValidationError, match="parity proof/schema invalid"):
        admit_window(path, missing, corpus_dir=corpus, repo_root=ROOT)
    assert not corpus.exists()


def test_idempotent_readmission_keeps_original_evidence_and_recovers_index(tmp_path: Path) -> None:
    """Exact re-admission preserves bytes/mtime and rebuilds an interrupted index."""
    corpus = tmp_path / "corpus"
    first = admit_window(CANDIDATE, RECEIPT, corpus_dir=corpus, repo_root=ROOT)
    path = corpus / "cases" / f"{first['case_id']}.json"
    original = path.read_bytes()
    timestamp = path.stat().st_mtime_ns
    index = (corpus / "corpus.json").read_bytes()
    write_review_sidecar(path, repo_root=tmp_path)
    second = admit_window(CANDIDATE, RECEIPT, corpus_dir=corpus, repo_root=ROOT)
    assert second["status"] == "already_admitted" and second["case_id"] == first["case_id"]
    assert path.read_bytes() == original and path.stat().st_mtime_ns == timestamp
    assert (corpus / "corpus.json").read_bytes() == index
    (corpus / "corpus.json").unlink()
    recovered = admit_window(CANDIDATE, RECEIPT, corpus_dir=corpus, repo_root=ROOT)
    assert recovered["status"] == "already_admitted"
    assert path.read_bytes() == original and (corpus / "corpus.json").read_bytes() == index
