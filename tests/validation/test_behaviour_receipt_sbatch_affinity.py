"""Receipt pairs must not silently compare different scheduler hardware."""

from pathlib import Path

import pytest

from scripts.ci import make_behaviour_receipt as producer


def _args(tmp_path, *extra):
    return producer._parse_args(
        [
            "--head-sha",
            "a" * 40,
            "--receipt-id",
            "affinity-test",
            "--mode",
            "sbatch",
            "--work-dir",
            str(tmp_path),
            *extra,
        ]
    )


def test_sbatch_pair_requires_node_before_any_submission(tmp_path, monkeypatch):
    """An omitted hardware choice must not leave a partially submitted pair."""
    submissions = []
    monkeypatch.setattr(producer, "_ensure_worktree", lambda *args: tmp_path)
    monkeypatch.setattr(
        producer.subprocess,
        "check_output",
        lambda *args, **kwargs: submissions.append(args) or "123\n",
    )

    with pytest.raises(ValueError, match="--sbatch-node"):
        producer._prepare_sweeps(_args(tmp_path), tmp_path, "b" * 40)

    assert not submissions


def test_sbatch_pair_pins_same_single_node_and_records_runtime_hardware(tmp_path, monkeypatch):
    """Both submitted jobs carry the same literal node constraint and provenance."""
    scripts = []
    monkeypatch.setattr(producer, "_ensure_worktree", lambda *args: tmp_path)

    def submit(command, **kwargs):
        assert command[:2] == ["sbatch", "--parsable"]
        scripts.append(Path(command[2]).read_text())
        return f"{123 + len(scripts)}\n"

    monkeypatch.setattr(producer.subprocess, "check_output", submit)
    producer._prepare_sweeps(_args(tmp_path, "--sbatch-node", "compute-1"), tmp_path, "b" * 40)

    assert len(scripts) == 2
    for script in scripts:
        assert "#SBATCH --nodes=1\n" in script
        assert "#SBATCH --nodelist=compute-1\n" in script
        assert "#SBATCH --cpus-per-task=8\n" in script
        assert "hostname >" in script and "execution_node.txt" in script
        assert "lscpu --json >" in script and "execution_cpu.json" in script


@pytest.mark.parametrize("node", ["node1,node2", "node[1-2]", "node1\n#SBATCH --nodes=2"])
def test_sbatch_rejects_node_sets_and_directive_injection(tmp_path, monkeypatch, node):
    """The affinity option accepts one literal hostname, never a Slurm expression."""
    submissions = []
    monkeypatch.setattr(
        producer.subprocess,
        "check_output",
        lambda *args, **kwargs: submissions.append(args) or "123\n",
    )
    with pytest.raises(ValueError, match="single.*node"):
        producer._prepare_sweeps(_args(tmp_path, "--sbatch-node", node), tmp_path, "b" * 40)
    assert not submissions
