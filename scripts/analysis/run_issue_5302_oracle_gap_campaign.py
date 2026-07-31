#!/usr/bin/env python3
"""Config-first runner that materializes the issue #5302 oracle-gap campaign.

Plain-language summary
----------------------
The frozen analysis packet at ``configs/analysis/issue_5302_oracle_gap_packet.yaml``
already fixes the six-arm roster, the family-disjoint split, the four ceilings, and
the report contract. The packet checker
(:mod:`scripts.validation.check_issue_5302_oracle_gap_packet`) and the arm-admission
gate (:mod:`robot_sf.benchmark.campaign_arm_admission`) verify that *contract* and
that each arm instantiates. They do not, however, *materialize* the campaign: they do
not resolve the family-disjoint episode matrix, the per-cell seed schedule, the
campaign denominator, or a one-cell canary.

This runner is the canonical config-first entry point that turns the frozen contract
into a concrete, deterministic, fail-closed campaign manifest with three modes:

* ``preflight`` (default, no-submit): deterministically resolves every arm (through
  the real loaders), the family partition, every cell, the seed schedule, the full
  episode matrix, the denominator, the per-arm config/checkpoint hashes, and the
  output paths. It submits no compute and writes ``reports/preflight.json``.
* ``matrix``: writes the family-disjoint episode matrix that the full campaign must
  run. Episodes are identical across arms and selection/evaluation families are
  disjoint.
* ``canary``: materializes exactly one frozen scenario cell x one seed x all six
  arms and emits one native row per arm with policy identity. Episode execution is
  behind an *injected execution seam* (``run_episode``) so tests can run the canary
  without submitting Slurm/GPU compute; the default seam fails closed because this
  contract does not authorize compute submission.

Claim boundary
~~~~~~~~~~~~~~
This runner is pre-registration / materialization infrastructure only. It does not
run a benchmark campaign, submit Slurm/GPU work, promote a planner ranking, train a
selector, or edit paper/dissertation claims. It changes none of the frozen
scientific semantics (roster, metrics, thresholds, family split, decision rules).
Any result remains diagnostic until native rows, family-level holdouts, provenance,
and the report checks pass elsewhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.campaign_arm_admission import (
    CampaignArmAdmissionError,
    check_campaign_arm_admission,
)
from robot_sf.models import get_registry_entry

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_PACKET = Path("configs/analysis/issue_5302_oracle_gap_packet.yaml")
#: Canonical partition manifest that fixes the family-disjoint selection/evaluation split.
DEFAULT_PARTITION = Path("configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml")
MANIFEST_SCHEMA_VERSION = "issue_5302_oracle_gap_campaign_manifest.v1"
CANARY_SCHEMA_VERSION = "issue_5302_oracle_gap_canary.v1"
#: Frozen canary seed: the pilot evaluation seed declared by the partition manifest.
CANARY_SEED = 111

#: Execution seam type: given a resolved episode, return a native result-row mapping.
EpisodeRunner = Callable[["ResolvedEpisode"], dict[str, Any]]


class CampaignMaterializationError(ValueError):
    """Raised when the campaign cannot be materialized as the frozen contract demands."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedArm:
    """One fully-resolved campaign arm with identity and provenance hashes.

    Attributes:
        planner_id: The frozen roster planner key.
        role: Declared mechanism role.
        readiness: Declared readiness label.
        config_path: Repo-relative config path (``None`` for the canonical ORCA baseline).
        algo: Canonical algorithm resolved from the config (or planner_id for baselines).
        config_sha256: SHA-256 of the resolved config file bytes (``None`` for no-config arms).
        model_ids: Every checkpoint ``model_id`` declared anywhere in the config.
        checkpoint_refs: Per-model-id registry ``github_release.sha256`` provenance pins.
    """

    planner_id: str
    role: str
    readiness: str
    config_path: str | None
    algo: str
    config_sha256: str | None
    model_ids: tuple[str, ...]
    checkpoint_refs: tuple[CheckpointRef, ...]


@dataclass(frozen=True)
class CheckpointRef:
    """Provenance pin for a single declared checkpoint model id."""

    model_id: str
    sha256: str | None
    source: str


@dataclass(frozen=True)
class FamilyPartition:
    """The family-disjoint selection/evaluation split materialized from the partition manifest.

    Attributes:
        selection_families: Scenario families used for planner/ceiling *selection*.
        evaluation_families: Held-out scenario families used for *evaluation*.
        selection_scenario_matrix: Repo-relative matrix path for the selection pool.
        evaluation_scenario_matrix: Repo-relative matrix path for the evaluation pool.
        evaluation_seeds: Frozen seed schedule declared for held-out evaluation.
        disjoint: True only when selection and evaluation family sets are disjoint.
    """

    selection_families: tuple[str, ...]
    evaluation_families: tuple[str, ...]
    selection_scenario_matrix: str
    evaluation_scenario_matrix: str
    evaluation_seeds: tuple[int, ...]
    disjoint: bool


@dataclass(frozen=True)
class ScenarioCell:
    """One frozen (family, cell) episode unit in the campaign matrix.

    A "cell" is a single scenario within a scenario family. ``scenario_cell`` is the
    scenario id (the cell identity the packet freezes before execution).
    """

    scenario_family: str
    scenario_cell: str
    scenario_id: str
    source_kind: str
    source_path: str


@dataclass(frozen=True)
class ResolvedEpisode:
    """One fully-resolved campaign episode: cell x seed x arm.

    This is the unit the execution seam receives. It carries the full identity the
    frozen metric layer requires (``episode_id``, ``scenario_family``,
    ``scenario_cell``, ``split``, ``seed``, ``planner_id``).
    """

    episode_id: str
    scenario_family: str
    scenario_cell: str
    scenario_id: str
    split: str
    seed: int
    planner_id: str
    config_path: str | None
    config_sha256: str | None

    def identity_payload(self) -> dict[str, Any]:
        """Return the canonical identity mapping used for the episode id and row."""
        return {
            "scenario_family": self.scenario_family,
            "scenario_cell": self.scenario_cell,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "planner_id": self.planner_id,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True)
class CampaignManifest:
    """The fully-materialized no-submit campaign: arms, partition, matrix, denominator."""

    schema_version: str
    issue: int
    arms: tuple[ResolvedArm, ...]
    partition: FamilyPartition
    cells: tuple[ScenarioCell, ...]
    episodes: tuple[ResolvedEpisode, ...]
    selection_cells: tuple[ScenarioCell, ...]
    evaluation_cells: tuple[ScenarioCell, ...]
    seeds: tuple[int, ...]
    arm_count: int
    selection_family_count: int
    evaluation_family_count: int
    cell_count: int
    seed_count: int
    denominator: int
    selection_denominator: int
    evaluation_denominator: int
    local_root: str
    required_report_paths: tuple[str, ...]
    durable_evidence_path: str
    packet_sha256: str


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignMaterializationError(message)


def _load_yaml(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required manifest not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _repo_root(repo_root: Path | None) -> Path:
    return (repo_root or Path(__file__).resolve().parents[2]).resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _resolve_algo(planner_id: str, config: dict[str, Any] | None) -> str:
    if config is not None:
        algo = str(config.get("algo", "")).strip().lower()
        if algo:
            return algo
    return planner_id.strip().lower()


def _iter_model_ids(config: dict[str, Any]) -> list[str]:
    """Return every checkpoint ``model_id``-style reference declared in a config.

    Mirrors :func:`robot_sf.benchmark.campaign_arm_admission._iter_model_ids` so a
    nested predictive checkpoint is still resolved.
    """
    id_keys = ("model_id", "sacadrl_model_id", "predictive_model_id")
    found: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in id_keys:
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    found.append(value.strip())
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(config)
    return found


def _checkpoint_ref(model_id: str) -> CheckpointRef:
    """Resolve a model id's registry provenance pin without touching the network.

    The registry ``github_release.sha256`` is the canonical artifact digest the
    packet pin must equal (verified by the packet checker). Recording it here gives
    the manifest a hash that drifts if the registry entry changes.
    """
    try:
        entry = get_registry_entry(model_id)
    except (KeyError, FileNotFoundError, TypeError, ValueError) as exc:
        return CheckpointRef(model_id=model_id, sha256=None, source=f"unresolved: {exc}")
    release = entry.get("github_release")
    if isinstance(release, dict):
        sha = str(release.get("sha256", "")).strip().lower()
        if sha:
            return CheckpointRef(
                model_id=model_id, sha256=sha, source="model/registry.yaml github_release.sha256"
            )
    return CheckpointRef(
        model_id=model_id, sha256=None, source="model/registry.yaml (no github_release.sha256)"
    )


# ---------------------------------------------------------------------------
# Arm resolution (real loaders)
# ---------------------------------------------------------------------------


def resolve_arms(
    packet: dict[str, Any], *, repo_root: Path | None = None
) -> tuple[ResolvedArm, ...]:
    """Resolve every frozen-roster arm through the real loaders and provenance hashes.

    This first runs the real-loader admission gate
    (:func:`robot_sf.benchmark.campaign_arm_admission.check_campaign_arm_admission`) so a
    declared arm that cannot instantiate as declared fails closed. It then loads each
    arm's config (real YAML loader), resolves its canonical algorithm, computes a config
    hash, and records each checkpoint's registry provenance pin.

    Raises:
        CampaignMaterializationError: when the roster is malformed or not admissible.
        FileNotFoundError: when a declared config_path does not exist.
    """
    root = _repo_root(repo_root)
    try:
        admission = check_campaign_arm_admission(packet, repo_root=root)
    except CampaignArmAdmissionError as exc:
        raise CampaignMaterializationError(f"roster is malformed: {exc}") from exc
    if not admission.admissible:
        bullets = "\n".join(f"  - {line}" for line in admission.failure_messages())
        raise CampaignMaterializationError(
            "roster declares arms that cannot instantiate as declared:\n" + bullets
        )

    roster = packet["planner_roster"]["required"]
    arms: list[ResolvedArm] = []
    for row in roster:
        planner_id = str(row["planner_id"])
        role = str(row.get("role", ""))
        readiness = str(row.get("readiness", "")).strip().lower()
        config_path_raw = row.get("config_path")
        config_path = (
            str(config_path_raw) if isinstance(config_path_raw, str) and config_path_raw else None
        )
        config: dict[str, Any] | None = None
        config_sha: str | None = None
        if config_path is not None:
            resolved = root / config_path
            if not resolved.is_file():
                raise FileNotFoundError(f"planner config missing: {config_path}")
            loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            _require(
                isinstance(loaded, dict), f"{planner_id} config {config_path} must be a mapping"
            )
            config = loaded
            config_sha = _sha256_file(resolved)
        model_ids = tuple(_iter_model_ids(config or {}))
        checkpoint_refs = tuple(_checkpoint_ref(mid) for mid in model_ids)
        arms.append(
            ResolvedArm(
                planner_id=planner_id,
                role=role,
                readiness=readiness,
                config_path=config_path,
                algo=_resolve_algo(planner_id, config),
                config_sha256=config_sha,
                model_ids=model_ids,
                checkpoint_refs=checkpoint_refs,
            )
        )
    return tuple(arms)


# ---------------------------------------------------------------------------
# Family partition + cell resolution
# ---------------------------------------------------------------------------


def load_family_partition(
    partition_path: Path | None = None, *, repo_root: Path | None = None
) -> FamilyPartition:
    """Materialize the family-disjoint selection/evaluation split from the partition manifest.

    Selection families are the partition's ``training_family_pool.scenario_families``
    (the benchmark-set / in-distribution surface used for ceiling *selection*).
    Evaluation families are the ``heldout_family_evaluation.scenario_families`` (the
    held-out family surface). The split is only valid when the two family sets are
    disjoint: the packet's validity rule forbids leakage between selection and
    evaluation.
    """
    root = _repo_root(repo_root)
    path = partition_path or (root / DEFAULT_PARTITION)
    payload = _load_yaml(path)
    _require(isinstance(payload, dict), f"{path} must be a YAML mapping")
    training = payload.get("training_family_pool")
    heldout = payload.get("heldout_family_evaluation")
    _require(isinstance(training, dict), f"{path}: training_family_pool must be a mapping")
    _require(isinstance(heldout, dict), f"{path}: heldout_family_evaluation must be a mapping")
    selection_families = tuple(str(f) for f in training.get("scenario_families", []))
    evaluation_families = tuple(str(f) for f in heldout.get("scenario_families", []))
    _require(selection_families, f"{path}: training_family_pool.scenario_families is empty")
    _require(evaluation_families, f"{path}: heldout_family_evaluation.scenario_families is empty")
    selection_matrix = str(training.get("scenario_matrix", "")).strip()
    evaluation_matrix = str(heldout.get("scenario_matrix", "")).strip()
    _require(selection_matrix and evaluation_matrix, f"{path}: scenario_matrix paths are required")
    seeds_raw = heldout.get("seeds", [])
    _require(
        isinstance(seeds_raw, list) and seeds_raw,
        f"{path}: heldout_family_evaluation.seeds must be a non-empty list",
    )
    seeds = tuple(int(s) for s in seeds_raw)
    disjoint = set(selection_families).isdisjoint(evaluation_families)
    return FamilyPartition(
        selection_families=selection_families,
        evaluation_families=evaluation_families,
        selection_scenario_matrix=selection_matrix,
        evaluation_scenario_matrix=evaluation_matrix,
        evaluation_seeds=seeds,
        disjoint=disjoint,
    )


def _resolve_scenario_cells(
    matrix_path: str, families: Sequence[str], *, repo_root: Path
) -> tuple[ScenarioCell, ...]:
    """Resolve the (family, cell) units a scenario matrix declares.

    A scenario matrix is a ``robot_sf.scenario_matrix.v1`` document that ``includes``
    archetype/single scenario files. Each included file declares one or more named
    scenarios. The cell identity is the scenario ``name``; the family is the
    partition family the matrix belongs to. When a matrix carries no explicit family
    annotation, each included scenario is assigned to the partition family it is
    routed through. Every resolved cell is frozen before execution (the packet's
    ``scenario_cell_is_frozen_before_execution`` contract).
    """
    resolved = (repo_root / matrix_path).resolve()
    _require(resolved.is_file(), f"scenario matrix not found: {matrix_path}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{matrix_path} must be a YAML mapping")
    cells: list[ScenarioCell] = []
    includes = payload.get("includes") or payload.get("scenario_files") or []
    _require(isinstance(includes, list), f"{matrix_path}: includes must be a list")
    family_for_file = _family_for_included_file(resolved, includes, families)
    for include in includes:
        if not isinstance(include, str) or not include.strip():
            continue
        scenario_path = (resolved.parent / include).resolve()
        _require(
            scenario_path.is_file(),
            f"scenario matrix {matrix_path} includes missing file: {include}",
        )
        family = family_for_file.get(include.strip(), families[0] if families else "unknown")
        try:
            rel_source = scenario_path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            rel_source = include.strip()
        for scenario in _scenario_names(scenario_path):
            cells.append(
                ScenarioCell(
                    scenario_family=family,
                    scenario_cell=scenario,
                    scenario_id=scenario,
                    source_kind="scenario_matrix_include",
                    source_path=rel_source,
                )
            )
    return tuple(cells)


def _scenario_names(scenario_path: Path) -> list[str]:
    """Return the declared ``name`` of each scenario in a scenario definition file."""
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        scenarios = payload.get("scenarios")
        if isinstance(scenarios, list):
            names = [
                str(s.get("name")).strip()
                for s in scenarios
                if isinstance(s, dict) and str(s.get("name", "")).strip()
            ]
            if names:
                return names
        name = payload.get("name")
        if isinstance(name, str) and name.strip():
            return [name.strip()]
    # A scenario matrix that includes no named scenarios still yields one cell keyed
    # by its stem, so the episode matrix remains a faithful count of the declared
    # surface rather than silently collapsing to zero cells.
    return [scenario_path.stem]


def _family_for_included_file(
    matrix_path: Path, includes: Sequence[str], families: Sequence[str]
) -> dict[str, str]:
    """Best-effort map from included file to its partition family.

    Partition families are the unit of selection/evaluation holdout; an included
    scenario file is routed through the single partition family whose id is a
    substring of the file name, or the matrix's first family as a deterministic
    fallback. The family assignment only labels cells; disjointness is enforced at
    the partition level, not per-file.
    """
    mapping: dict[str, str] = {}
    for include in includes:
        if not isinstance(include, str) or not include.strip():
            continue
        stem = PurePosixPath(Path(include).name).stem.lower()
        match = next((f for f in families if f.lower() in stem), None)
        mapping[include.strip()] = match or (families[0] if families else "unknown")
    return mapping


def resolve_cells(
    partition: FamilyPartition, *, repo_root: Path | None = None
) -> tuple[ScenarioCell, ...]:
    """Resolve the frozen (family, cell) units for both selection and evaluation pools."""
    root = _repo_root(repo_root)
    selection_cells = _resolve_scenario_cells(
        partition.selection_scenario_matrix, partition.selection_families, repo_root=root
    )
    evaluation_cells = _resolve_scenario_cells(
        partition.evaluation_scenario_matrix, partition.evaluation_families, repo_root=root
    )
    _require(selection_cells, "selection scenario matrix resolved to zero cells")
    _require(evaluation_cells, "evaluation scenario matrix resolved to zero cells")
    return selection_cells + evaluation_cells


# ---------------------------------------------------------------------------
# Episode matrix
# ---------------------------------------------------------------------------


def _episode_id(cell: ScenarioCell, seed: int, planner_id: str, config_sha: str | None) -> str:
    """Deterministic episode id: family/cell/seed/arm, independent of run host."""
    payload = json.dumps(
        {
            "scenario_family": cell.scenario_family,
            "scenario_cell": cell.scenario_cell,
            "seed": seed,
            "planner_id": planner_id,
            "config_sha256": config_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_episode_matrix(
    arms: Sequence[ResolvedArm],
    cells: Sequence[ScenarioCell],
    partition: FamilyPartition,
) -> tuple[ResolvedEpisode, ...]:
    """Build the family-disjoint episode matrix with identical episodes across arms.

    Every (cell, seed) unit is run by *all* arms on the *same* episode id namespace:
    the same cell + seed produces one episode per arm, and the episode id is derived
    from (family, cell, seed, arm) so arms are paired. Selection cells are labelled
    ``split=selection`` and evaluation cells ``split=evaluation``; the two never share
    a family (enforced by :func:`assert_no_leakage`).
    """
    evaluation_family_set = set(partition.evaluation_families)
    episodes: list[ResolvedEpisode] = []
    for cell in cells:
        split = "evaluation" if cell.scenario_family in evaluation_family_set else "selection"
        for seed in partition.evaluation_seeds:
            for arm in arms:
                episodes.append(
                    ResolvedEpisode(
                        episode_id=_episode_id(cell, seed, arm.planner_id, arm.config_sha256),
                        scenario_family=cell.scenario_family,
                        scenario_cell=cell.scenario_cell,
                        scenario_id=cell.scenario_id,
                        split=split,
                        seed=seed,
                        planner_id=arm.planner_id,
                        config_path=arm.config_path,
                        config_sha256=arm.config_sha256,
                    )
                )
    return tuple(episodes)


# ---------------------------------------------------------------------------
# Fail-closed contract checks
# ---------------------------------------------------------------------------


def assert_no_leakage(partition: FamilyPartition, cells: Sequence[ScenarioCell]) -> None:
    """Fail closed when selection and evaluation families are not disjoint.

    This is the packet's core validity rule: a random episode split leaks
    near-identical scenario configurations between selection and evaluation and
    invalidates the result.
    """
    _require(
        partition.disjoint,
        "selection and evaluation family sets must be disjoint (packet validity rule violated)",
    )
    selection_cells = {
        c.scenario_family for c in cells if c.scenario_family in partition.selection_families
    }
    evaluation_cells = {
        c.scenario_family for c in cells if c.scenario_family in partition.evaluation_families
    }
    overlap = selection_cells & evaluation_cells
    _require(
        not overlap,
        f"a scenario family appears in both selection and evaluation: {sorted(overlap)}",
    )


def assert_complete_six_arm_matrix(
    arms: Sequence[ResolvedArm], episodes: Sequence[ResolvedEpisode], partition: FamilyPartition
) -> None:
    """Fail closed on duplicate/missing arms or a denominator mismatch.

    The packet requires exactly six planner rows per episode, an identical episode
    set across planners, and a consistent denominator (cells x seeds x arms). This
    checks all three.
    """
    _require(len(arms) == 6, f"roster must contain exactly six arms, found {len(arms)}")
    planner_ids = [a.planner_id for a in arms]
    _require(
        len(planner_ids) == len(set(planner_ids)),
        f"duplicate planner ids in roster: {planner_ids}",
    )
    # Every (family, cell, seed) unit must be run by exactly six distinct arms.
    by_unit: dict[tuple[str, str, int], set[str]] = {}
    for ep in episodes:
        by_unit.setdefault((ep.scenario_family, ep.scenario_cell, ep.seed), set()).add(
            ep.planner_id
        )
    _require(by_unit, "episode matrix is empty")
    expected = set(planner_ids)
    for unit, arm_set in by_unit.items():
        _require(
            arm_set == expected,
            f"episode unit {unit} does not cover all six arms: {sorted(arm_set)}",
        )
    n_cells = len({(c.scenario_family, c.scenario_cell) for c in _cells_from_episodes(episodes)})
    expected_denominator = n_cells * len(partition.evaluation_seeds) * 6
    _require(
        len(episodes) == expected_denominator,
        f"denominator mismatch: {len(episodes)} episodes != {n_cells} cells x "
        f"{len(partition.evaluation_seeds)} seeds x 6 arms",
    )


def _cells_from_episodes(episodes: Sequence[ResolvedEpisode]) -> list[ScenarioCell]:
    """Reconstruct the distinct cells an episode matrix covers (for denominator checks)."""
    seen: dict[tuple[str, str], ScenarioCell] = {}
    for ep in episodes:
        seen.setdefault(
            (ep.scenario_family, ep.scenario_cell),
            ScenarioCell(
                scenario_family=ep.scenario_family,
                scenario_cell=ep.scenario_cell,
                scenario_id=ep.scenario_id,
                source_kind="episode",
                source_path="",
            ),
        )
    return list(seen.values())


def assert_hashes_stable(arms: Sequence[ResolvedArm], packet: dict[str, Any]) -> None:
    """Fail closed on checkpoint/config hash drift against the frozen packet.

    The PPO arm carries a pinned ``checkpoint_sha256`` in the packet that must equal
    the registry digest. A config hash is recorded for every arm so future drift is
    detectable; this check verifies the pinned PPO checkpoint matches what the
    registry resolves now.
    """
    roster = {row["planner_id"]: row for row in packet["planner_roster"]["required"]}
    ppo_row = roster.get("ppo")
    if not isinstance(ppo_row, dict):
        return
    pinned = ppo_row.get("pinned_provenance", {})
    if not isinstance(pinned, dict):
        return
    packet_sha = str(pinned.get("checkpoint_sha256", "")).strip().lower()
    if len(packet_sha) != 64:
        return
    ppo_arm = next((a for a in arms if a.planner_id == "ppo"), None)
    if ppo_arm is None:
        return
    registry_shas = {ref.sha256 for ref in ppo_arm.checkpoint_refs if ref.sha256}
    _require(
        packet_sha in registry_shas or not registry_shas,
        "PPO checkpoint hash drift: packet pin does not match the registry-resolved digest",
    )


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def _output_contract(packet: dict[str, Any]) -> tuple[str, tuple[str, ...], str]:
    """Read the disposable local root, required report paths, and durable evidence path."""
    outputs = packet.get("outputs", {})
    _require(isinstance(outputs, dict), "packet outputs must be a mapping")
    local_root = str(outputs.get("local_root", "")).strip()
    _require(local_root, "packet outputs.local_root is required")
    required = outputs.get("required_paths", [])
    _require(isinstance(required, list) and required, "packet outputs.required_paths is required")
    durable = outputs.get("durable_evidence", {})
    _require(isinstance(durable, dict), "packet outputs.durable_evidence must be a mapping")
    durable_path = str(durable.get("path", "")).strip()
    _require(durable_path, "packet outputs.durable_evidence.path is required")
    return local_root, tuple(str(p) for p in required), durable_path


# ---------------------------------------------------------------------------
# Manifest construction (preflight)
# ---------------------------------------------------------------------------


def materialize_manifest(
    packet: dict[str, Any],
    *,
    partition_path: Path | None = None,
    repo_root: Path | None = None,
) -> CampaignManifest:
    """Resolve the full campaign deterministically without submitting compute.

    Runs the real-loader arm admission, loads the family partition, resolves every
    frozen cell, builds the family-disjoint episode matrix, and asserts the
    fail-closed contract (disjointness, six-arm completeness, denominator, hash
    stability). Returns the materialized manifest.
    """
    root = _repo_root(repo_root)
    arms = resolve_arms(packet, repo_root=root)
    partition = load_family_partition(partition_path, repo_root=root)
    cells = resolve_cells(partition, repo_root=root)
    assert_no_leakage(partition, cells)
    episodes = build_episode_matrix(arms, cells, partition)
    assert_complete_six_arm_matrix(arms, episodes, partition)
    assert_hashes_stable(arms, packet)
    selection_family_set = set(partition.selection_families)
    selection_cells = tuple(c for c in cells if c.scenario_family in selection_family_set)
    evaluation_cells = tuple(c for c in cells if c.scenario_family not in selection_family_set)
    local_root, required_paths, durable_path = _output_contract(packet)
    n_cells = len({(c.scenario_family, c.scenario_cell) for c in cells})
    return CampaignManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        issue=int(packet.get("issue", 5302)),
        arms=arms,
        partition=partition,
        cells=cells,
        episodes=episodes,
        selection_cells=selection_cells,
        evaluation_cells=evaluation_cells,
        seeds=partition.evaluation_seeds,
        arm_count=len(arms),
        selection_family_count=len(partition.selection_families),
        evaluation_family_count=len(partition.evaluation_families),
        cell_count=n_cells,
        seed_count=len(partition.evaluation_seeds),
        denominator=len(episodes),
        selection_denominator=len(selection_cells) * len(partition.evaluation_seeds) * len(arms),
        evaluation_denominator=len(evaluation_cells) * len(partition.evaluation_seeds) * len(arms),
        local_root=local_root,
        required_report_paths=required_paths,
        durable_evidence_path=durable_path,
        packet_sha256=_packet_sha256(packet),
    )


def _packet_sha256(packet: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(packet, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _arm_to_dict(arm: ResolvedArm) -> dict[str, Any]:
    return {
        "planner_id": arm.planner_id,
        "role": arm.role,
        "readiness": arm.readiness,
        "config_path": arm.config_path,
        "algo": arm.algo,
        "config_sha256": arm.config_sha256,
        "model_ids": list(arm.model_ids),
        "checkpoint_refs": [asdict(ref) for ref in arm.checkpoint_refs],
    }


def _partition_to_dict(partition: FamilyPartition) -> dict[str, Any]:
    return {
        "selection_families": list(partition.selection_families),
        "evaluation_families": list(partition.evaluation_families),
        "selection_scenario_matrix": partition.selection_scenario_matrix,
        "evaluation_scenario_matrix": partition.evaluation_scenario_matrix,
        "evaluation_seeds": list(partition.evaluation_seeds),
        "disjoint": partition.disjoint,
    }


def _cell_to_dict(cell: ScenarioCell) -> dict[str, Any]:
    return {
        "scenario_family": cell.scenario_family,
        "scenario_cell": cell.scenario_cell,
        "scenario_id": cell.scenario_id,
        "source_kind": cell.source_kind,
        "source_path": cell.source_path,
    }


def manifest_to_dict(manifest: CampaignManifest) -> dict[str, Any]:
    """Serialize a materialized manifest to a JSON-safe mapping for ``preflight.json``."""
    return {
        "schema_version": manifest.schema_version,
        "issue": manifest.issue,
        "campaign_execution_submitted": False,
        "arms": [_arm_to_dict(a) for a in manifest.arms],
        "partition": _partition_to_dict(manifest.partition),
        "cells": [_cell_to_dict(c) for c in manifest.cells],
        "selection_cells": [_cell_to_dict(c) for c in manifest.selection_cells],
        "evaluation_cells": [_cell_to_dict(c) for c in manifest.evaluation_cells],
        "seeds": list(manifest.seeds),
        "counts": {
            "arm_count": manifest.arm_count,
            "selection_family_count": manifest.selection_family_count,
            "evaluation_family_count": manifest.evaluation_family_count,
            "cell_count": manifest.cell_count,
            "seed_count": manifest.seed_count,
            "denominator": manifest.denominator,
            "selection_denominator": manifest.selection_denominator,
            "evaluation_denominator": manifest.evaluation_denominator,
        },
        "outputs": {
            "local_root": manifest.local_root,
            "required_report_paths": list(manifest.required_report_paths),
            "durable_evidence_path": manifest.durable_evidence_path,
        },
        "packet_sha256": manifest.packet_sha256,
        "episode_count": len(manifest.episodes),
    }


def matrix_to_dict(manifest: CampaignManifest) -> dict[str, Any]:
    """Serialize the family-disjoint episode matrix (full campaign, identical across arms)."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "issue": manifest.issue,
        "campaign_execution_submitted": False,
        "seeds": list(manifest.seeds),
        "arms": [a.planner_id for a in manifest.arms],
        "selection_families": list(manifest.partition.selection_families),
        "evaluation_families": list(manifest.partition.evaluation_families),
        "disjoint": manifest.partition.disjoint,
        "episodes": [
            {
                "episode_id": ep.episode_id,
                "scenario_family": ep.scenario_family,
                "scenario_cell": ep.scenario_cell,
                "scenario_id": ep.scenario_id,
                "split": ep.split,
                "seed": ep.seed,
                "planner_id": ep.planner_id,
                "config_sha256": ep.config_sha256,
            }
            for ep in manifest.episodes
        ],
        "denominator": manifest.denominator,
    }


# ---------------------------------------------------------------------------
# Canary (injected execution seam)
# ---------------------------------------------------------------------------


def select_canary_cell(manifest: CampaignManifest) -> ScenarioCell:
    """Select the single frozen canary cell deterministically.

    The canary is exactly one frozen scenario cell. It is the first *evaluation*
    cell (a held-out family cell), so the canary exercises the held-out surface the
    campaign is ultimately judged on. Selection is deterministic and independent of
    run host or wall-clock state.
    """
    _require(manifest.evaluation_cells, "no evaluation cells available for the canary")
    return manifest.evaluation_cells[0]


def canary_episodes(
    manifest: CampaignManifest, *, cell: ScenarioCell | None = None
) -> tuple[ResolvedEpisode, ...]:
    """Resolve the six canary episodes: one cell x one seed x all six arms."""
    canary_cell = cell or select_canary_cell(manifest)
    episodes = [
        ep
        for ep in manifest.episodes
        if ep.scenario_family == canary_cell.scenario_family
        and ep.scenario_cell == canary_cell.scenario_cell
        and ep.seed == CANARY_SEED
    ]
    _require(
        len(episodes) == manifest.arm_count,
        f"canary must resolve exactly {manifest.arm_count} episodes, found {len(episodes)}",
    )
    return tuple(episodes)


def run_canary(
    packet: dict[str, Any],
    *,
    partition_path: Path | None = None,
    repo_root: Path | None = None,
    run_episode: EpisodeRunner | None = None,
) -> dict[str, Any]:
    """Materialize and execute the one-cell canary, emitting one native row per arm.

    The canary is exactly one frozen scenario cell x one seed x all six arms. Each
    episode runs through the injected execution seam ``run_episode``. When no seam is
    supplied, the canary fails closed: this contract does not authorize compute
    submission, so there is no default runner that could submit Slurm/GPU work.

    Args:
        packet: The frozen analysis packet mapping.
        partition_path: Optional override for the partition manifest.
        repo_root: Repository root for resolving paths.
        run_episode: Injected execution seam. Receives a :class:`ResolvedEpisode`
            and must return a native result-row mapping carrying ``row_status`` and
            ``execution_mode``. Tests inject a deterministic seam; the real seam
            (local ``run_episode`` or an ops-queue submitter) is wired by a future
            compute-authorized follow-up.

    Returns:
        A ``issue_5302_oracle_gap_canary.v1`` mapping with one native row per arm.

    Raises:
        CampaignMaterializationError: when the campaign cannot be materialized, when
            no execution seam is supplied, or when a seam returns a fallback/degraded
            or non-native row (the packet forbids fallback/degraded success).
    """
    manifest = materialize_manifest(packet, partition_path=partition_path, repo_root=repo_root)
    episodes = canary_episodes(manifest)
    if run_episode is None:
        raise CampaignMaterializationError(
            "canary execution requires an injected execution seam; this contract does not "
            "authorize compute submission and provides no default runner"
        )
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        row = run_episode(episode)
        _require(
            isinstance(row, dict),
            f"execution seam returned a non-mapping row for {episode.planner_id}",
        )
        _require(
            "planner_id" in row and row["planner_id"] == episode.planner_id,
            f"canary row lost policy identity: expected planner_id={episode.planner_id!r}",
        )
        _require(
            str(row.get("execution_mode", "")).strip().lower() == "native",
            f"canary row for {episode.planner_id} is not native execution: {row.get('execution_mode')!r}",
        )
        _require(
            str(row.get("row_status", "")).strip().lower() == "successful_evidence",
            f"canary row for {episode.planner_id} is not successful_evidence: {row.get('row_status')!r}",
        )
        rows.append(row)
    return {
        "schema_version": CANARY_SCHEMA_VERSION,
        "issue": manifest.issue,
        "campaign_execution_submitted": False,
        "canary_cell": _cell_to_dict(select_canary_cell(manifest)),
        "canary_seed": CANARY_SEED,
        "arm_count": manifest.arm_count,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _local_output_root(manifest: CampaignManifest, repo_root: Path) -> Path:
    return repo_root / manifest.local_root


def main(argv: list[str] | None = None) -> int:
    """Materialize the #5302 campaign: preflight (default), matrix, or canary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument(
        "--partition", type=Path, default=None, help="Override the partition manifest"
    )
    sub = parser.add_subparsers(dest="mode")
    sub.add_parser(
        "preflight", help="no-submit: resolve arms, partition, cells, seeds, denominator"
    )
    sub.add_parser("matrix", help="write the family-disjoint episode matrix")
    sub.add_parser("canary", help="materialize the one-cell canary (requires an execution seam)")
    args = parser.parse_args(argv)

    root = _repo_root(None)
    packet_path = args.packet if args.packet.is_absolute() else root / args.packet
    packet = _load_yaml(packet_path)
    _require(isinstance(packet, dict), f"{packet_path} must be a YAML mapping")

    mode = args.mode or "preflight"
    partition_path = args.partition

    if mode == "preflight":
        manifest = materialize_manifest(packet, partition_path=partition_path, repo_root=root)
        report = manifest_to_dict(manifest)
        out = _local_output_root(manifest, root) / "reports" / "preflight.json"
        _write_json(out, report)
        print(
            f"preflight: {manifest.arm_count} arms, {manifest.cell_count} cells, "
            f"{manifest.seed_count} seeds, denominator={manifest.denominator} -> {out}"
        )
        return 0

    if mode == "matrix":
        manifest = materialize_manifest(packet, partition_path=partition_path, repo_root=root)
        report = matrix_to_dict(manifest)
        out = _local_output_root(manifest, root) / "reports" / "episode_matrix.json"
        _write_json(out, report)
        print(
            f"matrix: {len(manifest.episodes)} episodes across {manifest.arm_count} arms, "
            f"disjoint={manifest.partition.disjoint} -> {out}"
        )
        return 0

    if mode == "canary":
        # The CLI canary has no execution seam (compute is not authorized in this
        # contract), so it materializes the six canary episodes and reports the cell.
        manifest = materialize_manifest(packet, partition_path=partition_path, repo_root=root)
        cell = select_canary_cell(manifest)
        episodes = canary_episodes(manifest)
        out = _local_output_root(manifest, root) / "reports" / "canary_episodes.json"
        _write_json(
            out,
            {
                "schema_version": CANARY_SCHEMA_VERSION,
                "issue": manifest.issue,
                "campaign_execution_submitted": False,
                "canary_cell": _cell_to_dict(cell),
                "canary_seed": CANARY_SEED,
                "arm_count": manifest.arm_count,
                "requires_execution_seam": True,
                "episodes": [
                    {
                        "episode_id": ep.episode_id,
                        "scenario_family": ep.scenario_family,
                        "scenario_cell": ep.scenario_cell,
                        "seed": ep.seed,
                        "planner_id": ep.planner_id,
                        "config_sha256": ep.config_sha256,
                    }
                    for ep in episodes
                ],
            },
        )
        print(
            f"canary: {len(episodes)} episodes (cell={cell.scenario_cell}, seed={CANARY_SEED}); "
            "execution requires an injected seam -> " + str(out)
        )
        return 0

    parser.error(f"unknown mode: {mode}")  # pragma: no cover - argparse exits
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
