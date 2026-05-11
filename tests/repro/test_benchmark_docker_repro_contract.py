"""Contract tests for the pinned benchmark Docker reproduction path."""

from __future__ import annotations

import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "benchmark-repro.Dockerfile"
SMOKE_SCRIPT = ROOT / "scripts" / "repro" / "benchmark_bundle_smoke.sh"
WRAPPER_SCRIPT = ROOT / "scripts" / "repro" / "run_benchmark_docker_smoke.sh"
DOC = ROOT / "docs" / "benchmark_docker_repro.md"
CONTEXT_NOTE = ROOT / "docs" / "context" / "issue_1086_docker_reproduction_path.md"
DOCS_README = ROOT / "docs" / "README.md"
RELEASE_REPRO_DOC = ROOT / "docs" / "benchmark_release_reproducibility.md"


def test_benchmark_repro_dockerfile_is_pinned_and_uses_frozen_uv_sync() -> None:
    """The Docker recipe should pin the main runtime surfaces used by the smoke."""

    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM python:3.12.3-slim-bookworm" in text
    assert "ARG UV_VERSION=0.11.9" in text
    assert 'python -m pip install "uv==${UV_VERSION}"' in text
    assert "uv sync --all-extras --frozen --no-install-project" in text
    assert "uv sync --all-extras --frozen" in text
    assert 'ENTRYPOINT ["scripts/repro/benchmark_bundle_smoke.sh"]' in text


def test_benchmark_repro_scripts_define_one_command_smoke_contract() -> None:
    """The scripts should build the image and emit the documented benchmark artifacts."""

    smoke = SMOKE_SCRIPT.read_text(encoding="utf-8")
    wrapper = WRAPPER_SCRIPT.read_text(encoding="utf-8")

    assert SMOKE_SCRIPT.stat().st_mode & stat.S_IXUSR
    assert WRAPPER_SCRIPT.stat().st_mode & stat.S_IXUSR
    assert "configs/scenarios/planner_sanity_matrix_v1.yaml" in smoke
    assert "robot_sf_bench validate-config" in smoke
    assert "robot_sf_bench preview-scenarios" in smoke
    assert "robot_sf_bench --quiet run" in smoke
    assert "robot_sf_bench aggregate" in smoke
    assert "episodes.jsonl" in smoke
    assert "summary.json" in smoke
    assert "manifest.json" in smoke
    assert "if not path.is_file() or path.stat().st_size == 0:" in smoke
    assert "docker build" in wrapper
    assert "docker run --rm" in wrapper
    assert "docker/benchmark-repro.Dockerfile" in wrapper


def test_benchmark_repro_docs_are_discoverable_and_state_limits() -> None:
    """Documentation should expose the path, outputs, pinning, and determinism boundary."""

    doc = DOC.read_text(encoding="utf-8")
    context = CONTEXT_NOTE.read_text(encoding="utf-8")
    docs_readme = DOCS_README.read_text(encoding="utf-8")
    release_repro_doc = RELEASE_REPRO_DOC.read_text(encoding="utf-8")

    for fragment in [
        "scripts/repro/run_benchmark_docker_smoke.sh",
        "docker/benchmark-repro.Dockerfile",
        "output/docker_repro/benchmark_bundle_smoke/",
        "manifest.json",
        "python:3.12.3-slim-bookworm",
        "0.11.9",
        "does not claim GPU determinism",
        "not a full campaign runner",
    ]:
        assert fragment in doc

    assert "issue_1086_docker_reproduction_path.md" in docs_readme
    assert "benchmark_docker_repro.md" in docs_readme
    assert "Benchmark Docker Reproduction Path" in release_repro_doc
    assert "Dockerfile: `docker/benchmark-repro.Dockerfile`" in context
