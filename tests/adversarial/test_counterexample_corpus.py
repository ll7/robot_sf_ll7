"""Actual seed-1001 corpus admission, refusal and append-only duplicate tests."""

# evidence-writer-exempt: pytest tmp_path fixtures intentionally corrupt parity
# receipts; they are never committed or presented as durable generated evidence.

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

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
