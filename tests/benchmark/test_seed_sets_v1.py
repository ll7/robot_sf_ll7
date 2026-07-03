"""Static contract tests for the benchmark seed schedules in ``seed_sets_v1.yaml``.

These tests are intentionally *static*: they load the YAML file and assert
structural invariants only. They run no benchmark campaign, submit no Slurm job,
and depend on no result store.

They lock the prefix-nesting property that keeps earlier seed budgets comparable
subsets of larger ones (S3 ``eval`` prefix of S5 prefix of S10 prefix of S20
prefix of S30) and pin the predeclared, deferred S30 schedule from issue #4304
(maintainer ruling 2026-07-03: S20 is the dissertation-draft tier; S30 is a
reversible future-escalation path only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SEED_SETS_PATH = ROOT / "configs/benchmarks/seed_sets_v1.yaml"

# Expected named paper-facing seed schedules and their exact seed counts.
EXPECTED_COUNTS = {
    "paper_eval_s5": 5,
    "paper_eval_s10": 10,
    "paper_eval_s20": 20,
    "paper_eval_s30": 30,
}

# The frozen contiguous schedule shared by every paper-facing tier.
EXPECTED_S30 = list(range(111, 141))  # 111..140 inclusive


@pytest.fixture(scope="module")
def seed_sets() -> dict[str, Any]:
    """Load the seed-set YAML once for the module."""
    assert SEED_SETS_PATH.is_file(), f"missing seed-set config: {SEED_SETS_PATH}"
    data = yaml.safe_load(SEED_SETS_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "seed_sets_v1.yaml must parse to a mapping"
    return data


def _seed_list(seed_sets: dict[str, Any], name: str) -> list[int]:
    """Return a named seed set, asserting it is present and list-shaped."""
    assert name in seed_sets, f"seed set {name!r} is missing"
    value = seed_sets[name]
    assert isinstance(value, list), f"seed set {name!r} must be a list"
    return value


def test_all_named_seed_sets_present(seed_sets: dict[str, Any]) -> None:
    """The base ``dev``/``eval`` sets and every paper tier must exist."""
    for name in ("dev", "eval", *EXPECTED_COUNTS):
        assert name in seed_sets, f"expected seed set {name!r} to be defined"


@pytest.mark.parametrize(("name", "count"), sorted(EXPECTED_COUNTS.items()))
def test_seed_set_counts(seed_sets: dict[str, Any], name: str, count: int) -> None:
    """Each paper tier has exactly the declared number of seeds."""
    assert len(_seed_list(seed_sets, name)) == count


@pytest.mark.parametrize("name", sorted(EXPECTED_COUNTS))
def test_seeds_are_unique_integers(seed_sets: dict[str, Any], name: str) -> None:
    """Seeds are plain ``int`` values, unique, and never boolean."""
    seeds = _seed_list(seed_sets, name)
    for seed in seeds:
        # ``bool`` is a subclass of ``int``; reject it explicitly so ``True``/
        # ``False`` can never masquerade as a seed value.
        assert type(seed) is int, f"{name} contains a non-int seed: {seed!r}"
    assert len(seeds) == len(set(seeds)), f"{name} contains duplicate seeds"


def test_prefix_nesting_holds_exactly(seed_sets: dict[str, Any]) -> None:
    """S3/S5/S10/S20 are exact prefixes of the next-larger schedule.

    Prefix nesting keeps earlier, smaller-budget campaigns valid subsets of
    larger ones, so a later escalation never invalidates prior comparisons.
    """
    eval_seeds = _seed_list(seed_sets, "eval")
    s5 = _seed_list(seed_sets, "paper_eval_s5")
    s10 = _seed_list(seed_sets, "paper_eval_s10")
    s20 = _seed_list(seed_sets, "paper_eval_s20")
    s30 = _seed_list(seed_sets, "paper_eval_s30")

    assert eval_seeds == s30[:3]
    assert s5 == s30[:5]
    assert s10 == s20[:10]
    assert s10 == s30[:10]
    assert s20 == s30[:20]


def test_s30_matches_predeclared_schedule(seed_sets: dict[str, Any]) -> None:
    """The deferred S30 schedule is the frozen contiguous block 111..140."""
    assert _seed_list(seed_sets, "paper_eval_s30") == EXPECTED_S30


def test_s30_does_not_reorder_earlier_tiers(seed_sets: dict[str, Any]) -> None:
    """Predeclaring S30 must not mutate or reorder already-used S10/S20 seeds.

    This is the reversibility guard: the escalation schedule only *appends*.
    """
    s10 = _seed_list(seed_sets, "paper_eval_s10")
    s20 = _seed_list(seed_sets, "paper_eval_s20")
    s30 = _seed_list(seed_sets, "paper_eval_s30")

    # Order-preserving append property, not just set containment.
    assert s30[:10] == s10
    assert s30[:20] == s20
    # The S30-only tail must be disjoint from the seeds S10/S20 already spent.
    tail = s30[20:]
    assert set(tail).isdisjoint(set(s20))
    assert len(tail) == 10
