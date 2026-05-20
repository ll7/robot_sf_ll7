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
WORKFLOW = ROOT / ".github" / "workflows" / "benchmark-docker-repro-smoke.yml"
PINNED_REPRO_MATRIX = "configs/scenarios/planner_sanity_matrix_v1.yaml"


def _read_text_file(path: Path) -> str:
    """Read a UTF-8 text file after verifying it is a regular file."""
    if not path.is_file():
        raise FileNotFoundError(f"Expected file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def test_benchmark_repro_dockerfile_is_pinned_and_uses_frozen_uv_sync() -> None:
    """The Docker recipe should pin the main runtime surfaces used by the smoke."""

    text = _read_text_file(DOCKERFILE)

    assert "FROM python:3.12.3-slim-bookworm" in text
    assert "ARG UV_VERSION=0.11.9" in text
    assert 'python -m pip install "uv==${UV_VERSION}"' in text
    assert "uv sync --all-extras --frozen --no-install-project" in text
    assert "uv sync --all-extras --frozen" in text
    assert "COPY third_party/python-rvo2 ./third_party/python-rvo2" in text
    assert "rm -rf third_party/python-rvo2/build" in text
    assert 'ENTRYPOINT ["scripts/repro/benchmark_bundle_smoke.sh"]' in text


def test_benchmark_repro_scripts_define_one_command_smoke_contract() -> None:
    """The scripts should build the image and emit the documented benchmark artifacts."""

    smoke = _read_text_file(SMOKE_SCRIPT)
    wrapper = _read_text_file(WRAPPER_SCRIPT)

    assert SMOKE_SCRIPT.stat().st_mode & stat.S_IXUSR
    assert WRAPPER_SCRIPT.stat().st_mode & stat.S_IXUSR
    assert PINNED_REPRO_MATRIX in smoke
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

    doc = _read_text_file(DOC)
    context = _read_text_file(CONTEXT_NOTE)
    docs_readme = _read_text_file(DOCS_README)
    release_repro_doc = _read_text_file(RELEASE_REPRO_DOC)

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


def test_benchmark_repro_workflow_qualifies_runner_before_smoke() -> None:
    """Docker CI proof should record runner capabilities before running the smoke."""

    workflow = _read_text_file(WORKFLOW)

    for fragment in [
        "workflow_dispatch:",
        "pull_request:",
        "paths:",
        PINNED_REPRO_MATRIX,
        "docker version --format",
        "docker info --format",
        "nvidia-smi",
        "docker run --rm --gpus all",
        "scripts/repro/run_benchmark_docker_smoke.sh",
        "runner_qualification.json",
        "image_inspect.json",
        "benchmark-docker-repro-smoke",
    ]:
        assert fragment in workflow

    assert "docker_daemon_available" in workflow
    assert "nvidia_docker_available" in workflow
    assert "configs/scenarios/**" not in workflow
