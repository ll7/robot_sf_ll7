"""Public source inputs exercise the comparator's real development projection.

Test value: a missed request overlay resolves sealed slots instead of dev slots;
existing comparator tests lack this opt-in. The real D-083 matrix/config and
pinned resolver compute identities; an environment sentinel prohibits execution.
"""

# seed-holdout: synthetic-fixture begin
import io
import json
import subprocess
from pathlib import Path

from scripts.analysis import _pinned_successor_runtime as pinned
from scripts.analysis import compare_release_0_0_7_to_0_0_8 as comparator
from scripts.analysis.compare_release_0_0_7_to_0_0_8 import V4_SLOT_REPLACEMENTS
from tests.benchmark.test_release_campaign_authority import no_execution

ROOT = Path(__file__).resolve().parents[2]


def test_pinned_runtime_projects_development_inventory_without_execution(monkeypatch, capsys):
    """The same pinned main function computes the full D-083 development grid."""
    import pysocialforce

    no_execution.__wrapped__(monkeypatch)
    monkeypatch.chdir(ROOT)
    # In production -I imports from the detached source. This in-process probe
    # identifies the identical checked-in physics source for the origin check.
    monkeypatch.setattr(
        pysocialforce, "__file__", str(ROOT / "fast-pysf/pysocialforce/__init__.py")
    )
    request = {
        "config_path": "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml",
        "versioned_keys": sorted(V4_SLOT_REPLACEMENTS.values()),
        "publication_identity": {
            "release_tag": "development-rehearsal-" + "a" * 40,
            "doi": "10.5281/zenodo.99000002",
        },
        "development_rehearsal_seeds": [1001, 1002, 1003],
        "rows": [],
    }
    monkeypatch.setattr(pinned.sys, "stdin", io.StringIO(json.dumps(request)))
    pinned.main()
    result = json.loads(capsys.readouterr().out)
    slots = result["expected_slots"]
    assert len(slots) == 2016
    assert {slot[3] for slot in slots} == {1001, 1002, 1003}
    assert len({slot[0] for slot in slots}) == 14
    assert len({slot[2] for slot in slots}) == 48


def test_detached_comparator_serializes_development_inventory():
    """Real detached-source execution must receive the explicit development tuple.

    A dropped request field silently resolves sealed slots; the existing paired
    comparator tests do not pass this field. This calls its public internal
    source resolver with real D-083 bytes and no episode rows to execute.
    """
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    _, _, _, _, slots = comparator._runtime_successor_identity(
        ROOT,
        source,
        "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml",
        {},
        publication_identity={
            "release_tag": "development-rehearsal-" + source,
            "doi": "10.5281/zenodo.99000002",
        },
        development_rehearsal_seeds=(1001, 1002, 1003),
    )
    assert len(slots) == 2016
    assert {slot[3] for slot in slots} == {1001, 1002, 1003}


# seed-holdout: synthetic-fixture end
