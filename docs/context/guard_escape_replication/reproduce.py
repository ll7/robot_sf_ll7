"""Run the byte-identical analyzer on a reversible seed-label view; write honest metadata.

The temporary labels are row indices for the archived hard-coded analyzer only.
The compact evidence and producer-line identity always retain actual fresh seeds.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PACKET = Path(__file__).resolve().parent


def load_analyzer(packet):
    """Import the archived analyzer without changing its source or globals."""
    spec = importlib.util.spec_from_file_location("frozen_guardesc_analyzer", packet / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reproduce(packet, output, crosscheck=False):
    """Apply registered index labels and correct only output identity metadata."""
    packet, output = packet.resolve(), output.resolve()
    if output.is_relative_to(packet):
        raise ValueError("Write reproduction outside the immutable packet")
    analyzer = load_analyzer(packet)
    study = yaml.safe_load((packet / "inputs/study.yaml").read_text())
    assert study["seeds"] == list(range(1031, 1061))
    with gzip.open(packet / "evidence/episodes.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    assert len(records) == 2160
    aliases = []
    for row in records:
        assert type(row["seed"]) is int and row["seed"] in study["seeds"]
        alias = dict(row, seed=row["seed"] - 30, acquisition_seed=row["seed"])
        aliases.append(alias)
    payload = "".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in aliases
    ).encode()
    alias_gzip = gzip.compress(payload, compresslevel=9, mtime=0)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="guardesc-label-view-") as tmp:
        view = Path(tmp)
        (view / "inputs").mkdir()
        (view / "evidence").mkdir()
        shutil.copyfile(packet / "inputs/study.yaml", view / "inputs/study.yaml")
        (view / "evidence/episodes.jsonl.gz").write_bytes(alias_gzip)
        rates, effects, summary = analyzer.compute(view)
        analyzer.write_csv(output / "figure_data.csv", rates, analyzer.RATE_COLUMNS)
        analyzer.write_csv(output / "paired_effects.csv", effects, analyzer.EFFECT_COLUMNS)
        if crosscheck:
            # Execute the archived acquisition analyzer without editing a byte.
            (view / "configs").mkdir()
            (view / "out").mkdir()
            (view / "configs/groups.yaml").write_text(
                yaml.safe_dump(study["groups"], sort_keys=False)
            )
            shutil.copyfile(
                packet / "provenance/original_analyze.py.source.txt", view / "original.py"
            )
            (view / "aliased.jsonl").write_bytes(payload)
            subprocess.run(
                [sys.executable, str(view / "original.py"), "--input", str(view / "aliased.jsonl")],
                check=True,
                capture_output=True,
            )
            for name in ("figure_data.csv", "paired_effects.csv"):
                assert (view / "out" / name).read_bytes() == (output / name).read_bytes(), name
    summary["schema"] = "guard-escape-replication.summary.v1"
    summary["development_seeds"] = {"first": 1031, "last": 1060}
    summary["input_sha256"] = hashlib.sha256(
        (packet / "evidence/episodes.jsonl.gz").read_bytes()
    ).hexdigest()
    summary["review_verdict"] = "NUMERICALLY_VERIFIED; no independent replication review claimed"
    summary["registration_commit"] = json.loads(
        (packet / "provenance/registration.json").read_text()
    )["registration_commit"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    receipt = {
        "schema": "guard-escape-replication.analysis-view.v1",
        "actual_seeds": {"first": 1031, "last": 1060},
        "analysis_alias_seeds": {"first": 1001, "last": 1030},
        "mapping": "analysis_alias_seed = acquisition_seed - 30; one-to-one, order preserving",
        "rows": len(records),
        "alias_jsonl_sha256": hashlib.sha256(payload).hexdigest(),
        "alias_gzip_sha256": hashlib.sha256(alias_gzip).hexdigest(),
        "analyzer_sha256": hashlib.sha256((packet / "analyze.py").read_bytes()).hexdigest(),
        "analyzer_unchanged": True,
        "outcomes_changed": False,
        "output_adapter_changes": [
            "schema",
            "development_seeds",
            "input_sha256",
            "review_verdict",
            "registration_commit",
        ],
        "archived_acquisition_analyzer_csvs_byte_identical": crosscheck,
    }
    (output / "analysis-view.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return rates, effects, summary


def main():
    """Reproduce and optionally compare the frozen numerical outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--crosscheck-original", action="store_true")
    args = parser.parse_args()
    reproduce(args.packet, args.output, args.crosscheck_original)
    if args.verify:
        for name in ("summary.json", "paired_effects.csv", "figure_data.csv"):
            assert (args.output / name).read_bytes() == (
                args.packet / "evidence" / name
            ).read_bytes(), name
    print(
        "PASS: complete paired design, unchanged analyzer, registered label adapter"
        + (", frozen outputs byte-identical" if args.verify else "")
    )


if __name__ == "__main__":
    main()
