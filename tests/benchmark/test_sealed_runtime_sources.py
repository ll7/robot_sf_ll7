"""Fast static admission controls for copied runtime packages; no simulation."""
# seed-holdout: synthetic-fixture begin

import shutil
from types import SimpleNamespace

import pytest

from robot_sf.benchmark import release_protocol as protocol
from robot_sf.evidence.writers import write_text
from tests.benchmark.test_sealed_source_pins import (
    ROOT,
    bind_runtime_sources,
    copy_runtime_sources,
    git,
)


@pytest.fixture
def runtime_freeze(tmp_path, monkeypatch):
    """Use real tracked physics bytes, isolating runtime admission from config admission."""
    repo = tmp_path / "source"
    repo.mkdir()
    copy_runtime_sources(repo)
    bind_runtime_sources(repo, monkeypatch)
    config = (
        repo / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1.yaml"
    )
    matrix = repo / "configs/scenarios/francis2023_narrow_doorway_three_width_release_0_0_8_v1.yaml"
    for target in (config, matrix):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / target.relative_to(repo), target)
    write_text(repo / ".gitignore", "# AI-GENERATED NEEDS-REVIEW\noutput/\n")
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Runtime Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(
        repo,
        "add",
        "--",
        ".gitignore",
        "robot_sf/__init__.py",
        "fast-pysf/pysocialforce",
        "configs",
    )
    git(repo, "commit", "-qm", "freeze actual runtime bytes")
    # Scientific-input closure is independently covered by the native identity witnesses.
    monkeypatch.setattr(protocol, "_require_sealed_source_inputs", lambda *_a: None)
    manifest = SimpleNamespace(
        canonical_campaign_config_path=config,
        scenario_matrix_path=matrix,
        release_kind="benchmark-width-slice",
        release_id="three_width_doorway_0_0_8_v1",
        source_sha=git(repo, "rev-parse", "HEAD"),
        resolved_identity_path=repo / "output/identity.json",
        identity_template_path=config,
    )
    return repo, manifest, repo / "output/runtime/pysocialforce"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("different", "pysocialforce/forces.py bytes differ"),
        ("symlink-init", "pysocialforce/forces.py bytes differ"),
        ("extra", "pysocialforce/extra.py is an extra imported file"),
        ("missing", "pysocialforce/forces.py is missing"),
        ("foreign", "robot_sf is outside the checked repository"),
        ("empty-source", "frozen pysocialforce package has no tracked files"),
    ],
)
def test_sealed_runtime_guard_refuses_unfrozen_imports(
    runtime_freeze, monkeypatch, mutation, reason
):
    repo, manifest, installed = runtime_freeze
    if mutation in {"different", "symlink-init"}:
        target = installed / "forces.py"
        write_text(target, "# AI-GENERATED NEEDS-REVIEW\n" + target.read_text())
        if mutation == "symlink-init":
            (installed / "__init__.py").unlink()
            (installed / "__init__.py").symlink_to(repo / "fast-pysf/pysocialforce/__init__.py")
    elif mutation == "extra":
        write_text(installed / "extra.py", "# AI-GENERATED NEEDS-REVIEW\n")
    elif mutation == "missing":
        (installed / "forces.py").unlink()
    elif mutation == "foreign":
        import robot_sf

        monkeypatch.setattr(robot_sf, "__file__", str(repo.parent / "foreign/__init__.py"))
    else:
        git(repo, "rm", "-r", "fast-pysf/pysocialforce")
        git(repo, "commit", "-qm", "source without physics")
        manifest.source_sha = git(repo, "rev-parse", "HEAD")
    problem = protocol.sealed_seed_execution_problem(
        manifest, protocol.EVAL_SEEDS_0_0_8, repository_root=repo
    )
    assert problem is not None, f"unfrozen imported runtime admitted: {mutation}"
    assert reason in problem
    assert "uv sync --all-extras --reinstall-package robot-sf" in problem


def test_sealed_runtime_guard_accepts_matching_copy_and_ignores_caches(runtime_freeze):
    repo, manifest, installed = runtime_freeze
    (installed / "__pycache__").mkdir()
    for path in (installed / "unused.pyc", installed / "__pycache__/unused.py"):
        write_text(path, "# AI-GENERATED NEEDS-REVIEW\n")
    assert (
        protocol.sealed_seed_execution_problem(
            manifest, protocol.EVAL_SEEDS_0_0_8, repository_root=repo
        )
        is None
    )


# seed-holdout: synthetic-fixture end
