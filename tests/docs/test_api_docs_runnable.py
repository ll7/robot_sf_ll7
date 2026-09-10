"""Tests for the runnable Sphinx API documentation (docs/api/robot_sf.api.rst)."""

from __future__ import annotations

from pathlib import Path

import pytest

import robot_sf

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DOC_FILE = REPO_ROOT / "docs" / "api" / "robot_sf.api.rst"
PUBLIC_API_DOC_FILE = REPO_ROOT / "docs" / "public_api.md"
API_INDEX_FILE = REPO_ROOT / "docs" / "api" / "index.rst"


def _extract_testcode_blocks(rst_path: Path) -> list[str]:
    """Extract literal Python code blocks defined under `.. testcode::` directives."""
    text = rst_path.read_text(encoding="utf-8")
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == ".. testcode::":
            i += 1
            # Skip empty lines
            while i < len(lines) and not lines[i].strip():
                i += 1
            code_lines = []
            while i < len(lines):
                curr = lines[i]
                if not curr.strip():
                    code_lines.append("")
                    i += 1
                    continue
                # Block lines are indented by at least 3 spaces
                if curr.startswith("   "):
                    code_lines.append(curr[3:])
                    i += 1
                else:
                    break
            block_code = "\n".join(code_lines).strip()
            if block_code:
                blocks.append(block_code)
        else:
            i += 1
    return blocks


def test_api_doc_files_exist_and_cross_referenced() -> None:
    """Verify docs/api/robot_sf.api.rst exists and is linked from index and public_api.md."""
    assert API_DOC_FILE.is_file(), f"{API_DOC_FILE} must exist"
    assert PUBLIC_API_DOC_FILE.is_file(), f"{PUBLIC_API_DOC_FILE} must exist"
    assert API_INDEX_FILE.is_file(), f"{API_INDEX_FILE} must exist"

    index_text = API_INDEX_FILE.read_text(encoding="utf-8")
    assert "robot_sf.api" in index_text, "docs/api/index.rst must list robot_sf.api in toctree"

    public_api_text = PUBLIC_API_DOC_FILE.read_text(encoding="utf-8")
    assert "api/robot_sf.api.rst" in public_api_text, (
        "docs/public_api.md must reference the Sphinx API facade documentation"
    )


def test_api_doc_contains_expected_testcode_blocks() -> None:
    """Verify docs/api/robot_sf.api.rst contains at least 6 testcode blocks."""
    blocks = _extract_testcode_blocks(API_DOC_FILE)
    assert len(blocks) >= 6, (
        f"Expected at least 6 testcode blocks in {API_DOC_FILE}, found {len(blocks)}"
    )


def test_literal_code_blocks_execute_cleanly() -> None:
    """Execute all literal testcode blocks extracted directly from docs/api/robot_sf.api.rst."""
    blocks = _extract_testcode_blocks(API_DOC_FILE)
    assert blocks, "No testcode blocks found to execute"

    for idx, code in enumerate(blocks, start=1):
        global_scope = {"__builtins__": __builtins__}
        local_scope: dict[str, object] = {}
        try:
            exec(code, global_scope, local_scope)  # noqa: S102
        except Exception as exc:
            pytest.fail(
                f"Testcode block {idx} in {API_DOC_FILE} raised exception: {exc}\nCode:\n{code}"
            )


def test_example_1_environment_lifecycle_contract() -> None:
    """Direct test verifying environment creation and safe lifecycle close."""
    env = robot_sf.make_env(seed=42)
    try:
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert isinstance(info, dict)
    finally:
        env.close()


def test_example_2_scenario_loading_contract() -> None:
    """Direct test verifying scenario resolution."""
    scenario = robot_sf.load_scenario("francis2023_circular_crossing")
    assert scenario["name"] == "francis2023_circular_crossing"
    assert "map" in scenario or "map_file" in scenario


def test_example_3_episode_record_roundtrip(tmp_path: Path) -> None:
    """Direct test verifying episode rollout, persistence, and reload fidelity."""
    out_path = tmp_path / "test_episode.json"
    env = robot_sf.make_env(scenario="francis2023_circular_crossing", seed=42)
    try:
        record = robot_sf.run_episode(env, max_steps=3, seed=42)
        assert record.seed == 42
        assert record.horizon == 3
        assert record.metrics.get("steps") == 3.0

        record.save(out_path)
        assert out_path.is_file()

        loaded = robot_sf.EpisodeRecord.load(out_path)
        assert loaded.episode_id == record.episode_id
        assert loaded.seed == record.seed
        assert loaded.metrics.get("steps") == record.metrics.get("steps")
        assert loaded.metrics.to_dict() == record.metrics.to_dict()
    finally:
        env.close()


def test_error_handling_missing_scenario_contract() -> None:
    """Direct test verifying FileNotFoundError on missing scenario."""
    with pytest.raises(FileNotFoundError, match="could not be resolved"):
        robot_sf.load_scenario("non_existent_scenario_name_123")


def test_error_handling_invalid_planner_contract() -> None:
    """Direct test verifying TypeError on invalid planner object."""
    env = robot_sf.make_env(seed=42)
    try:
        with pytest.raises(
            TypeError, match="must provide a callable step\\(\\) method or be callable"
        ):
            robot_sf.run_episode(env, planner=object(), max_steps=1)
    finally:
        env.close()


def test_error_handling_invalid_max_steps_contract() -> None:
    """Direct test verifying ValueError on non-positive max_steps."""
    env = robot_sf.make_env(seed=42)
    try:
        with pytest.raises(ValueError, match="max_steps must be a positive integer"):
            robot_sf.run_episode(env, max_steps=-5)
    finally:
        env.close()
