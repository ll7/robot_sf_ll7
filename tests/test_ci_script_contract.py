"""Guard CI shell wrappers against drift in shared script contracts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_DRIVER = ROOT / "scripts" / "dev" / "ci_driver.sh"
RUN_CI_LOCAL = ROOT / "scripts" / "dev" / "run_ci_local.sh"


def test_ci_driver_smoke_uses_runtime_schema_and_output_matrix_path() -> None:
    """Keep smoke preflight aligned with the runtime benchmark invocation."""

    script_text = CI_DRIVER.read_text(encoding="utf-8")

    assert 'local schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json"' in script_text
    assert '--schema "$schema_path"' in script_text
    assert 'local matrix_path="output/benchmarks/ci_smoke/matrix.yaml"' in script_text
    assert 'cat > "$matrix_path"' in script_text
    assert '--matrix "$matrix_path"' in script_text
    assert "cat > matrix.yaml" not in script_text


def test_ci_driver_test_phase_uses_shared_parallel_test_wrapper() -> None:
    """Preserve the repo's platform-specific pytest wrapper in the CI driver."""

    script_text = CI_DRIVER.read_text(encoding="utf-8")

    assert '"$SCRIPT_DIR/run_tests_parallel.sh" tests' in script_text
    assert "uv run pytest -q -n auto --max-worker-restart=0" not in script_text


def test_run_ci_local_loads_default_phases_from_ci_driver() -> None:
    """Avoid duplicating the canonical CI phase list in the local wrapper."""

    script_text = RUN_CI_LOCAL.read_text(encoding="utf-8")

    assert "scripts/dev/ci_driver.sh --list-phases" in script_text
    assert 'mapfile -t default_phases < <("$SCRIPT_DIR/ci_driver.sh" --list-phases)' in script_text
    assert "mapfile -t phases < <(load_default_phases)" in script_text
    assert 'phases=("lint" "typecheck" "test" "smoke" "artifact-policy")' not in script_text
