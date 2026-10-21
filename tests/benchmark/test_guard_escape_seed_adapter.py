"""The frozen numeric analyzer can consume a new, explicitly declared seed block."""

import gzip
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml


def test_new_seed_block_preserves_every_numeric_output(tmp_path):
    """Relabeling acquisition seeds changes identity metadata, never estimates."""
    root = Path(__file__).resolve().parents[2]
    source = root / "docs/context/guard_escape_replication"
    spec = importlib.util.spec_from_file_location("seed_adapter", source / "reproduce.py")
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    packet = tmp_path / "packet"
    shutil.copytree(source, packet)
    study_path = packet / "inputs/study.yaml"
    study = yaml.safe_load(study_path.read_text())
    study["seeds"] = list(range(1061, 1091))
    study_path.write_text(yaml.safe_dump(study, sort_keys=False))
    episode_path = packet / "evidence/episodes.jsonl.gz"
    records = [json.loads(line) for line in gzip.decompress(episode_path.read_bytes()).splitlines()]
    for row in records:
        row["seed"] += 30
    episode_path.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in records).encode()))
    output = tmp_path / "output"
    adapter.reproduce(packet, output, acquisition_seed_start=1061)
    for name in ("figure_data.csv", "paired_effects.csv"):
        assert (output / name).read_bytes() == (source / "evidence" / name).read_bytes()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["development_seeds"] == {"first": 1061, "last": 1090}
    receipt = json.loads((output / "analysis-view.json").read_text())
    assert receipt["actual_seeds"] == {"first": 1061, "last": 1090}
    assert "- 60" in receipt["mapping"]
    with pytest.raises(AssertionError):
        adapter.reproduce(packet, tmp_path / "wrong-labels")
