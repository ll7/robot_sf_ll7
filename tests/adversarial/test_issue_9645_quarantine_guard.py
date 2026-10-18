"""Guard: tuning, calibration and campaign configs must not cite the quarantined 9645 bundle."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QUARANTINED_BUNDLE_NAME = "issue_9645_bounded_falsification_2026-09-24"
QUARANTINED_CONFIG_ROOTS = (
    "benchmarks",
    "snqi_v2",
    "policy_search",
    "releases",
    "calibration",
)


def find_quarantine_references(config_root: Path) -> list[str]:
    """Return config files under the tuning-relevant roots that name the quarantined bundle."""
    hits: list[str] = []
    for name in QUARANTINED_CONFIG_ROOTS:
        root = config_root / name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and QUARANTINED_BUNDLE_NAME in path.read_text(
                encoding="utf-8", errors="ignore"
            ):
                hits.append(path.relative_to(config_root).as_posix())
    return hits


def test_no_tuning_or_campaign_config_references_quarantined_bundle() -> None:
    """No benchmark, SNQI, policy-search, release or calibration config may cite the bundle."""
    assert find_quarantine_references(REPO_ROOT / "configs") == []


def test_quarantine_reference_finder_detects_a_citing_config(tmp_path: Path) -> None:
    """The guard reports a config that names the bundle and ignores one that does not."""
    bad = tmp_path / "snqi_v2" / "weights.yaml"
    bad.parent.mkdir(parents=True)
    bad.write_text(f"source: {QUARANTINED_BUNDLE_NAME}/payload\n", encoding="utf-8")
    good = tmp_path / "benchmarks" / "ok.yaml"
    good.parent.mkdir()
    good.write_text("source: unrelated_bundle\n", encoding="utf-8")

    assert find_quarantine_references(tmp_path) == ["snqi_v2/weights.yaml"]
