"""Static release commands must carry the admitted anchors and absolute custody."""

import re
from pathlib import Path

import pytest

RUNBOOK = Path(__file__).resolve().parents[2] / "docs/release/0.0.8/runbook.md"


def _commands(section):
    return [block.replace("\\\n", " ") for block in re.findall(r"```bash\n(.*?)```", section, re.S)]


@pytest.mark.parametrize("track", ["main", "doorway", "wrapper"])
def test_sealed_campaign_examples_bind_anchors(track):
    text = RUNBOOK.read_text()
    section = text.split("## 4. Sealed campaign and fixed companion", 1)[1].split("## 5.", 1)[0]
    commands = _commands(section)
    if track == "wrapper":
        wrapper = next(block for block in commands if "submit_release_single_node.sbatch" in block)
        assert 'export ROBOT_SF_SNQI_V2_ANCHORS="$SNQI_ANCHORS"' in wrapper
        assert wrapper.index("export ROBOT_SF_SNQI_V2_ANCHORS=") < wrapper.index("sbatch ")
    else:
        runners = next(
            block for block in commands if "scripts/tools/run_benchmark_release.py" in block
        )
        runner = next(
            command
            for command in runners.split("uv run python ")
            if f"output/release-008/{track}/release_identity.resolved.json" in command
        )
        assert '--snqi-v2-anchors "$SNQI_ANCHORS"' in runner


def test_post_run_commands_pin_producer_custody_before_changing_checkout():
    text = RUNBOOK.read_text()
    commands = _commands(text)
    comparison = next(
        block for block in commands if "scripts/analysis/compare_release_distributions.py" in block
    )
    revalidation = next(
        block for block in commands if "scripts/tools/revalidate_benchmark_release.py" in block
    )
    anchors = 'export SNQI_ANCHORS="$SOURCE_ROOT/output/release-008/calibration/anchors.v2.0.acquired.json"'
    campaign = 'export CAMPAIGN_ROOT="$SOURCE_ROOT/output/benchmarks/camera_ready/$CAMPAIGN_ID"'
    setup = commands[0]
    assert "export SOURCE_ROOT=" in setup
    assert anchors in setup
    assert setup.index("export SOURCE_ROOT=") < setup.index(anchors)
    assert campaign in comparison
    assert comparison.index(campaign) < comparison.index('cd "$TOOLING_ROOT"')
    for command in (comparison, revalidation):
        assert '--snqi-v2-anchors "$SNQI_ANCHORS"' in command
    assert re.findall(r"--snqi-v2-anchors\s+(\S+)", "\n".join(commands)) == ['"$SNQI_ANCHORS"'] * 4


def test_revalidation_exports_its_inputs_in_a_fresh_shell():
    commands = _commands(RUNBOOK.read_text())
    revalidation = next(
        block for block in commands if "scripts/tools/revalidate_benchmark_release.py" in block
    )
    for variable in (
        "SOURCE_ROOT",
        "TOOLING_ROOT",
        "TOOLING_SHA",
        "SNQI_ANCHORS",
        "ARTIFACT_ROOT",
        "CAMPAIGN_ID",
    ):
        export = f"export {variable}="
        assert export in revalidation, f"Fresh-shell revalidation must export {variable}"
        assert revalidation.index(export) < revalidation.index('cd "$TOOLING_ROOT"')
    anchors = 'export SNQI_ANCHORS="$SOURCE_ROOT/output/release-008/calibration/anchors.v2.0.acquired.json"'
    assert anchors in revalidation
    assert revalidation.index("export SOURCE_ROOT=") < revalidation.index(anchors)
