"""Bounded behaviour-cloning loader/overfit smoke for the issue #1496 oracle dataset.

Issue #1496 is the downstream oracle-imitation warm-start step that consumes the durable
``expert_traj_v1`` dataset registered by job 13520 (PR #5914). The full comparison -- a BC
warm-start policy trained on a large balanced collection and benchmarked against the RL-only
baseline -- is gated on SLURM/GPU compute and a larger collection, and is explicitly out of
scope for shared-PC work. The maintainer disposition after the materialization blocker was
resolved is the bounded *loader/overfit smoke* delivered here:

    "This is enough to validate loading, schema compatibility, a tiny overfit, and
     end-to-end BC smoke execution. It is not a credible dataset for the predeclared
     BC-warm-start-versus-RL sample-efficiency or final-performance comparison."

This module owns exactly that smoke. It:

* loads the ``expert_traj_v1.npz`` artifact (the NPZ schema validated by
  :class:`robot_sf.benchmark.validation.trajectory_dataset.TrajectoryDatasetValidator`),
* partitions episodes by split and enforces the split/leakage contract named in the issue
  acceptance gates (no episode id shared across splits, disjoint train seeds, evaluation
  held out from training),
* flattens the structured per-step observation/action pairs for the training split into
  (feature, target) tensors (skipping the large occupancy grid -- see
  :func:`flatten_observation_action_pairs`),
* trains a tiny MLP for a few epochs on CPU and records the initial-to-final loss reduction
  as a memorization/overfit probe,
* writes a checkpoint, a metrics file, and a fail-closed smoke manifest.

Claim boundary: a checkpoint produced by this smoke is ``smoke evidence`` -- it proves the
loader, the split/leakage gate, and a BC training step execute end to end and that the tiny
training split is *memorable*. It is NOT benchmark evidence, navigation-quality evidence, a
warm-start-quality result, or a paper/dissertation claim. Running against the real registered
artifact requires the authorized private artifact root; without it the loader fails closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_sf.benchmark.artifact_catalog import sha256_file
from robot_sf.common.atomic_io import atomic_write_json
from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "issue_1496.oracle_imitation_bc_smoke.v1"

# Evidence tiers mirror robot_sf/planner/learned_short_horizon_trainer.py: a smoke checkpoint
# is never benchmark/quality evidence regardless of the data source.
EVIDENCE_TIER = "smoke"
REAL_TRAJECTORY_EVIDENCE_TIER = "real-trajectory-smoke"
CLAIM_BOUNDARY = (
    "BC loader/overfit smoke; proves the expert_traj_v1 NPZ loads, the split/leakage gate "
    "holds, and a tiny BC training step reduces loss on the training split. Smoke evidence "
    "only -- not benchmark, navigation-quality, warm-start-quality, or paper-facing evidence. "
    "The full BC-warm-start-vs-RL comparison remains gated on a larger collection and "
    "SLURM/GPU compute."
)

# Required NPZ arrays for the expert_traj_v1 schema (expert-trajectory-dataset-manifest.v1 /
# trajectory_dataset.v2.decision_transformer_preflight). Mirrors the arrays recorded in the
# job 13520 registration and TrajectoryDatasetValidator.DECISION_TRANSFORMER_REQUIRED_ARRAYS.
_REQUIRED_ARRAYS: tuple[str, ...] = (
    "actions",
    "observations",
    "positions",
    "rewards",
    "return_to_go",
    "terminated",
    "truncated",
    "episode_ids",
    "scenario_ids",
    "seeds",
    "splits",
)
_REQUIRED_SPLITS: tuple[str, ...] = ("train", "validation", "evaluation")
_TRAIN_ONLY_SPLITS: tuple[str, ...] = ("train",)

# Structured observation fields flattened into the BC feature vector. The large ``occupancy_grid``
# (3x160x160 per step) is intentionally excluded: the smoke probes the loader and a tiny MLP
# overfit, not a CNN policy, so a compact state vector is the correct and reproducible choice.
# Field order is fixed so the feature vector is deterministic; the manifest records it.
_FEATURE_FIELDS: tuple[str, ...] = (
    "robot_position",
    "robot_heading",
    "robot_speed",
    "robot_velocity_xy",
    "robot_angular_velocity",
    "robot_radius",
    "goal_current",
    "goal_next",
    "pedestrians_positions",
    "pedestrians_velocities",
    "pedestrians_radius",
    "pedestrians_count",
    "map_size",
    "sim_timestep",
)


class OracleImitationBcSmokeError(RobotSfError, ValueError):
    """Raised when the BC loader/overfit smoke fails a fail-closed invariant."""


@dataclass(frozen=True, slots=True)
class BCSmokeConfig:
    """Configuration for the bounded BC loader/overfit smoke.

    Attributes:
        dataset_path: Path to the ``expert_traj_v1.npz`` artifact. Required -- the smoke does
            not fabricate a dataset; it fails closed when this path is absent.
        output_dir: Directory to receive the checkpoint, metrics, and smoke manifest.
        epochs: Number of full-batch gradient steps. Small by design (this is an overfit probe).
        learning_rate: Adam learning rate.
        hidden_dim: MLP hidden width.
        seed: Torch/numpy RNG seed for reproducibility.
        device: Torch device; CPU is the intended lane.
    """

    dataset_path: str
    output_dir: str = "output/smoke/issue_1496/oracle_imitation_bc_smoke"
    epochs: int = 200
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    seed: int = 1496
    device: str = "cpu"

    def feature_fields(self) -> tuple[str, ...]:
        """Return the ordered observation feature fields flattened into the BC input."""
        return _FEATURE_FIELDS


@dataclass(frozen=True, slots=True)
class BCSmokeResult:
    """Outcome of the BC loader/overfit smoke.

    Attributes:
        checkpoint_path: Trained tiny-MLP checkpoint path.
        manifest_path: Smoke manifest path.
        metrics_path: Training-metrics path.
        initial_loss: Mean-squared action error before any gradient step.
        final_loss: Mean-squared action error after the final gradient step.
        loss_reduction: ``initial_loss - final_loss`` (must be strictly positive).
        num_train_steps: Number of (observation, action) pairs used for training.
        episode_counts: Per-split episode counts actually loaded from the artifact.
        evidence_tier: Evidence-tier label recorded in the manifest.
    """

    checkpoint_path: str
    manifest_path: str
    metrics_path: str
    initial_loss: float
    final_loss: float
    loss_reduction: float
    num_train_steps: int
    episode_counts: dict[str, int]
    evidence_tier: str


def _require_array(npz: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    """Return a required array from the NPZ or raise a fail-closed error."""
    if name not in npz.files:
        raise OracleImitationBcSmokeError(
            f"expert_traj_v1 NPZ is missing required array {name!r}; found {sorted(npz.files)}"
        )
    return npz[name]


def _observations_array(array: np.ndarray) -> np.ndarray:
    """Return the per-step observation array, validated as 2-D (episodes, steps).

    The expert_traj_v1 schema pads episodes to a common max-step width, so observations are
    stored as a 2-D object array of structured observation dicts (or ``None`` padding).
    """
    data = np.asarray(array)
    if data.ndim != 2:
        raise OracleImitationBcSmokeError(
            f"NPZ array 'observations' expected 2-D (episodes, steps), got shape {data.shape}"
        )
    return data


def _actions_array(array: np.ndarray) -> np.ndarray:
    """Return the per-step action array, validated as 2-D or 3-D.

    Actions are stored as ``(episodes, steps)`` of action vectors (object dtype) or as
    ``(episodes, steps, action_dim)``. Both layouts index identically as ``actions[e, s]``.
    """
    data = np.asarray(array)
    if data.ndim not in (2, 3):
        raise OracleImitationBcSmokeError(
            f"NPZ array 'actions' expected 2-D or 3-D (episodes, steps[, action_dim]), "
            f"got shape {data.shape}"
        )
    return data


def _episode_identity(array: np.ndarray, *, name: str) -> list[str]:
    """Read a per-episode identity array as a list of non-empty strings.

    Returns:
        Per-episode string identities in storage order.
    """
    data = np.asarray(array)
    if data.ndim == 2 and data.shape[1] == 1:
        values = [data[i, 0] for i in range(data.shape[0])]
    elif data.ndim == 1:
        values = list(data)
    else:
        raise OracleImitationBcSmokeError(
            f"NPZ identity array {name!r} has unexpected shape {data.shape}"
        )

    identities: list[str] = []
    for index, value in enumerate(values):
        try:
            scalar = np.asarray(value).item()
        except (TypeError, ValueError) as exc:
            raise OracleImitationBcSmokeError(
                f"NPZ identity array {name!r} entry {index} is not scalar"
            ) from exc
        if not isinstance(scalar, str) or not scalar.strip():
            raise OracleImitationBcSmokeError(
                f"NPZ identity array {name!r} entry {index} must be a non-empty string"
            )
        identities.append(scalar.strip())
    return identities


def _episode_seeds(array: np.ndarray, *, name: str) -> list[int]:
    """Read a per-episode seed array as a list of integer, non-boolean seeds.

    Returns:
        Per-episode integer seeds in storage order.
    """
    data = np.asarray(array)
    if data.ndim == 2 and data.shape[1] == 1:
        values = [data[i, 0] for i in range(data.shape[0])]
    elif data.ndim == 1:
        values = list(data)
    else:
        raise OracleImitationBcSmokeError(
            f"NPZ seed array {name!r} has unexpected shape {data.shape}"
        )

    seeds: list[int] = []
    for index, value in enumerate(values):
        try:
            scalar = np.asarray(value).item()
        except (TypeError, ValueError) as exc:
            raise OracleImitationBcSmokeError(
                f"NPZ seed array {name!r} entry {index} is not scalar"
            ) from exc
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, (int, np.integer)):
            raise OracleImitationBcSmokeError(
                f"NPZ seed array {name!r} entry {index} must be an integer, not boolean"
            )
        seeds.append(int(scalar))
    return seeds


def _splits_from_metadata(npz: np.lib.npyio.NpzFile) -> dict[str, list[str]]:
    """Build the expected split -> episode-id mapping from the artifact metadata.

    The NPZ carries both a per-step ``splits`` array and a ``metadata`` mapping. The metadata
    ``splits`` block is the authoritative per-split episode-id mapping; we fall back to the
    per-episode ``splits``/``episode_ids`` arrays when metadata is absent.

    Returns:
        Mapping of split name to the list of episode ids belonging to that split.
    """
    metadata = (
        npz["metadata"].item() if "metadata" in npz.files and npz["metadata"].ndim == 0 else {}
    )
    meta_splits = metadata.get("splits") if isinstance(metadata, dict) else None
    if isinstance(meta_splits, dict) and meta_splits:
        out: dict[str, list[str]] = {}
        for split, payload in meta_splits.items():
            ids = payload.get("episode_ids") if isinstance(payload, dict) else None
            if isinstance(ids, list) and ids:
                out[str(split)] = [str(value) for value in ids]
        if out:
            return out
    episode_ids = _episode_identity(npz["episode_ids"], name="episode_ids")
    split_tags = _episode_identity(npz["splits"], name="splits")
    out = {split: [] for split in _REQUIRED_SPLITS}
    for episode_id, split in zip(episode_ids, split_tags, strict=True):
        out.setdefault(str(split), []).append(episode_id)
    return out


def _validate_dataset_arrays(arrays: dict[str, np.ndarray]) -> int:
    """Validate array alignment and provenance fields.

    Returns:
        Number of episodes represented by the aligned observation/action arrays.
    """
    observations = _observations_array(arrays["observations"])
    actions = _actions_array(arrays["actions"])
    if actions.shape[0] != observations.shape[0]:
        raise OracleImitationBcSmokeError(
            "actions and observations disagree on episode count: "
            f"{actions.shape[0]} vs {observations.shape[0]}"
        )
    if actions.shape[1] != observations.shape[1]:
        raise OracleImitationBcSmokeError(
            "actions and observations disagree on padded step width: "
            f"{actions.shape[1]} vs {observations.shape[1]}"
        )

    episode_count = observations.shape[0]
    for name in _REQUIRED_ARRAYS:
        array = np.asarray(arrays[name])
        if array.ndim == 0 or len(array) != episode_count:
            raise OracleImitationBcSmokeError(
                f"NPZ array {name!r} has {len(array) if array.ndim else 0} episodes; "
                f"expected {episode_count}"
            )
    _validate_declared_episode_count(arrays, episode_count)

    # Parse required provenance fields while loading so malformed values cannot reach the
    # split/leakage gate as stringified placeholders or silently coerced seeds.
    _episode_identity(arrays["episode_ids"], name="episode_ids")
    _episode_identity(arrays["scenario_ids"], name="scenario_ids")
    _episode_identity(arrays["splits"], name="splits")
    _episode_seeds(arrays["seeds"], name="seeds")
    return episode_count


def _validate_declared_episode_count(arrays: dict[str, np.ndarray], episode_count: int) -> None:
    """Ensure an optional scalar episode-count field agrees with the arrays."""
    if "episode_count" not in arrays:
        return
    try:
        declared_count = int(np.asarray(arrays["episode_count"]).item())
    except (TypeError, ValueError) as exc:
        raise OracleImitationBcSmokeError("NPZ episode_count must be a scalar integer") from exc
    if declared_count != episode_count:
        raise OracleImitationBcSmokeError(
            f"NPZ episode_count declares {declared_count}; arrays contain {episode_count}"
        )


def load_expert_trajectory_dataset(dataset_path: str | Path) -> dict[str, Any]:
    """Load and structurally validate an ``expert_traj_v1.npz`` artifact.

    Args:
        dataset_path: Path to the NPZ artifact.

    Returns:
        A mapping of validated arrays plus the per-split episode mapping and metadata.

    Raises:
        OracleImitationBcSmokeError: If the file is missing, malformed, or does not carry the
            required arrays.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise OracleImitationBcSmokeError(f"expert_traj_v1 dataset not found at {path}")
    if path.suffix.lower() != ".npz":
        raise OracleImitationBcSmokeError(
            f"expert_traj_v1 dataset must be an .npz artifact, got {path.name}"
        )
    with np.load(path, allow_pickle=True) as npz:
        arrays = {name: _require_array(npz, name) for name in _REQUIRED_ARRAYS}
        if "metadata" in npz.files:
            arrays["metadata"] = npz["metadata"]
        if "episode_count" in npz.files:
            arrays["episode_count"] = npz["episode_count"]
        splits = _splits_from_metadata(npz)

    episode_count = _validate_dataset_arrays(arrays)

    return {
        "dataset_path": path,
        "arrays": arrays,
        "splits": splits,
        "episode_count": episode_count,
    }


def _validate_split_metadata_coverage(splits: dict[str, list[str]], all_ids: list[str]) -> None:
    """Require split metadata to cover exactly the NPZ episode identities."""
    metadata_ids = [episode_id for ids in splits.values() for episode_id in ids]
    if len(metadata_ids) != len(all_ids) or set(metadata_ids) != set(all_ids):
        missing_ids = sorted(set(all_ids) - set(metadata_ids))
        unexpected_ids = sorted(set(metadata_ids) - set(all_ids))
        raise OracleImitationBcSmokeError(
            "split/leakage violation: split metadata must cover exactly the NPZ episode ids; "
            f"missing={missing_ids}, unexpected={unexpected_ids}"
        )


def _validate_split_identity_uniqueness(episode_ids_by_split: dict[str, list[str]]) -> None:
    """Reject episode identities repeated within or across split metadata."""
    seen: dict[str, str] = {}
    for split, episode_ids in episode_ids_by_split.items():
        for episode_id in episode_ids:
            if episode_id in seen:
                raise OracleImitationBcSmokeError(
                    f"split/leakage violation: episode {episode_id!r} appears in both "
                    f"{seen[episode_id]!r} and {split!r}"
                )
            seen[episode_id] = split


def _validate_split_tags(
    all_ids: list[str], split_tags: list[str], episode_ids_by_split: dict[str, list[str]]
) -> None:
    """Require each NPZ row's split tag to agree with the metadata mapping."""
    ids_by_split = {split: set(ids) for split, ids in episode_ids_by_split.items()}
    for episode_id, split_tag in zip(all_ids, split_tags, strict=True):
        if split_tag not in _REQUIRED_SPLITS:
            raise OracleImitationBcSmokeError(
                f"split/leakage violation: episode {episode_id!r} has unknown split {split_tag!r}"
            )
        if episode_id not in ids_by_split[split_tag]:
            raise OracleImitationBcSmokeError(
                f"split/leakage violation: episode {episode_id!r} metadata split does not "
                f"match its NPZ row tag {split_tag!r}"
            )


def validate_split_leakage_contract(dataset: dict[str, Any]) -> dict[str, Any]:
    """Enforce the issue #1496 split/leakage contract.

    The contract mirrors the issue's acceptance gate "Split/leakage validation passes before
    training starts": every required split is present, no episode id appears in more than one
    split, no train seed is reused in validation/evaluation, and the evaluation split is held
    out from training.

    Args:
        dataset: The mapping returned by :func:`load_expert_trajectory_dataset`.

    Returns:
        A compact report with per-split episode counts, the cross-split leakage check, and the
        seed-disjoint check.

    Raises:
        OracleImitationBcSmokeError: If any split/leakage invariant is violated.
    """
    splits = dataset["splits"]
    arrays = dataset["arrays"]
    missing = [split for split in _REQUIRED_SPLITS if not splits.get(split)]
    if missing:
        raise OracleImitationBcSmokeError(
            f"expert_traj_v1 dataset is missing required split(s): {missing}; found {sorted(splits)}"
        )

    episode_ids_by_split: dict[str, list[str]] = {
        split: list(splits[split]) for split in _REQUIRED_SPLITS
    }
    all_ids = _episode_identity(arrays["episode_ids"], name="episode_ids")
    split_tags = _episode_identity(arrays["splits"], name="splits")
    _validate_split_metadata_coverage(splits, all_ids)
    _validate_split_identity_uniqueness(episode_ids_by_split)

    # Seed disjointness: train seeds must be disjoint from validation/evaluation seeds.
    all_seeds = _episode_seeds(arrays["seeds"], name="seeds")
    id_to_seed = dict(zip(all_ids, all_seeds, strict=True))
    train_seeds = {id_to_seed[episode_id] for episode_id in episode_ids_by_split["train"]}
    holdout_seeds = set()
    for split in ("validation", "evaluation"):
        holdout_seeds.update(id_to_seed[episode_id] for episode_id in episode_ids_by_split[split])
    shared_seeds = sorted(train_seeds & holdout_seeds)
    if shared_seeds:
        raise OracleImitationBcSmokeError(
            f"split/leakage violation: train seeds reused in validation/evaluation: {shared_seeds}"
        )

    # Per-episode NPZ split tags must match the metadata mapping (defensive consistency check).
    _validate_split_tags(all_ids, split_tags, episode_ids_by_split)

    return {
        "episode_ids_by_split": episode_ids_by_split,
        "episode_counts": {split: len(ids) for split, ids in episode_ids_by_split.items()},
        "train_seeds_disjoint_from_holdout": True,
        "evaluation_held_out_from_training": True,
    }


def _flatten_observation(observation: Any, *, feature_fields: tuple[str, ...]) -> np.ndarray:
    """Flatten one structured observation dict into a 1-D float32 feature vector.

    Each requested field is ravelled and concatenated in field order; variable-width fields
    (e.g. pedestrian arrays with differing live counts) keep their own width here and are
    right-padded to the batch's maximum feature width later by :func:`_stack_fixed_width`.
    The occupancy grid is excluded by the field list.

    Returns:
        Concatenated float32 feature vector for the observation.
    """
    if not isinstance(observation, dict):
        raise OracleImitationBcSmokeError(
            f"expected structured observation dict, got {type(observation).__name__}"
        )
    chunks: list[np.ndarray] = []
    for field_name in feature_fields:
        if field_name not in observation:
            raise OracleImitationBcSmokeError(
                f"observation is missing feature field {field_name!r}; "
                f"present keys: {sorted(observation)}"
            )
        value = np.asarray(observation[field_name], dtype=np.float32).reshape(-1)
        chunks.append(value)
    return np.concatenate(chunks).astype(np.float32)


def flatten_observation_action_pairs(
    dataset: dict[str, Any], *, splits: tuple[str, ...] = _TRAIN_ONLY_SPLITS
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Flatten (observation, action) pairs for the requested splits into feature/target arrays.

    Args:
        dataset: The mapping returned by :func:`load_expert_trajectory_dataset`.
        splits: Splits to flatten. Defaults to the training split only (BC trains on train).

    Returns:
        ``(features, targets, provenance)`` where ``features`` is shape ``(n_steps, feat_dim)``
        float32, ``targets`` is shape ``(n_steps, action_dim)`` float32, and ``provenance``
        records per-split step counts.
    """
    arrays = dataset["arrays"]
    observations = _observations_array(arrays["observations"])
    actions = _actions_array(arrays["actions"])
    episode_ids = _episode_identity(arrays["episode_ids"], name="episode_ids")
    split_tags = _episode_identity(arrays["splits"], name="splits")
    wanted = set(splits)

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    per_split_steps: dict[str, int] = dict.fromkeys(splits, 0)
    for episode in range(observations.shape[0]):
        if split_tags[episode] not in wanted:
            continue
        split = split_tags[episode]
        for step in range(observations.shape[1]):
            observation = observations[episode, step]
            if observation is None:
                continue
            action = actions[episode, step]
            features.append(_flatten_observation(observation, feature_fields=_FEATURE_FIELDS))
            targets.append(np.asarray(action, dtype=np.float32).reshape(-1))
            per_split_steps[split] = per_split_steps.get(split, 0) + 1

    if not features:
        raise OracleImitationBcSmokeError(
            f"no (observation, action) pairs found for splits={list(splits)} "
            f"(episode_ids={episode_ids})"
        )
    feature_matrix = _stack_fixed_width(features)
    target_matrix = _stack_fixed_width(targets)
    if feature_matrix.shape[0] != target_matrix.shape[0]:
        raise OracleImitationBcSmokeError(
            "feature/target step count mismatch after flatten: "
            f"{feature_matrix.shape[0]} vs {target_matrix.shape[0]}"
        )
    return feature_matrix, target_matrix, per_split_steps


def _stack_fixed_width(vectors: list[np.ndarray]) -> np.ndarray:
    """Right-pad 1-D float vectors to a common width so they stack into a 2-D matrix.

    Returns:
        2-D float32 matrix of shape ``(len(vectors), max_width)`` with each vector left-aligned.
    """
    width = max(int(vec.shape[0]) for vec in vectors)
    matrix = np.zeros((len(vectors), width), dtype=np.float32)
    for row, vec in enumerate(vectors):
        matrix[row, : vec.shape[0]] = vec
    return matrix


class _BCPolicy(torch.nn.Module):
    """Tiny MLP that maps a flattened observation vector to a continuous action."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(features)


def run_bc_overfit_smoke(config: BCSmokeConfig) -> BCSmokeResult:
    """Run the bounded BC loader/overfit smoke and write checkpoint + manifest artifacts.

    The smoke loads the artifact, enforces the split/leakage contract, flattens the training
    split, trains a tiny MLP to memorize it, and asserts the loss strictly decreased. It fails
    closed if the artifact is absent or any invariant is violated.

    Args:
        config: Smoke configuration. ``dataset_path`` must point at a real ``expert_traj_v1.npz``.

    Returns:
        The :class:`BCSmokeResult` with the recorded loss reduction and artifact paths.

    Raises:
        OracleImitationBcSmokeError: If the loss did not decrease (the overfit probe failed), the
            artifact is missing/malformed, or a split/leakage invariant is violated.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = load_expert_trajectory_dataset(config.dataset_path)
    split_report = validate_split_leakage_contract(dataset)
    features_np, targets_np, per_split_steps = flatten_observation_action_pairs(dataset)
    num_train_steps = int(features_np.shape[0])
    if num_train_steps == 0:
        raise OracleImitationBcSmokeError("flattened training split has zero steps")

    feature_tensor = torch.as_tensor(features_np, dtype=torch.float32, device=config.device)
    target_tensor = torch.as_tensor(targets_np, dtype=torch.float32, device=config.device)
    input_dim = int(feature_tensor.shape[1])
    output_dim = int(target_tensor.shape[1])
    module = _BCPolicy(input_dim=input_dim, hidden_dim=config.hidden_dim, output_dim=output_dim)
    module.to(config.device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=config.learning_rate)

    module.train()
    with torch.no_grad():
        initial_loss = float(loss_fn(module(feature_tensor), target_tensor).item())
    final_loss = initial_loss
    for _ in range(config.epochs):
        optimizer.zero_grad()
        loss = loss_fn(module(feature_tensor), target_tensor)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(loss_fn(module(feature_tensor), target_tensor).item())

    loss_reduction = float(initial_loss - final_loss)
    if not np.isfinite(loss_reduction) or loss_reduction <= 0.0:
        raise OracleImitationBcSmokeError(
            f"BC overfit smoke failed: loss did not decrease (initial={initial_loss:.6g}, "
            f"final={final_loss:.6g}, reduction={loss_reduction:.6g}); the loader/loop is broken "
            "or the training split has no learnable structure."
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "bc_smoke_policy.pt"
    metrics_path = output_dir / "training_metrics.json"
    manifest_path = output_dir / "bc_smoke_manifest.json"

    module.eval()
    state_dict = {key: value.detach().cpu() for key, value in module.state_dict().items()}
    dataset_path = Path(config.dataset_path)
    dataset_sha256 = sha256_file(dataset_path)
    evidence_tier = REAL_TRAJECTORY_EVIDENCE_TIER
    torch.save(
        {
            "state_dict": state_dict,
            "schema_version": SCHEMA_VERSION,
            "evidence_tier": evidence_tier,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "feature_fields": list(config.feature_fields()),
        },
        checkpoint_path,
    )

    metrics = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "epochs": config.epochs,
        "num_train_steps": num_train_steps,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "evidence_tier": evidence_tier,
    }
    atomic_write_json(metrics_path, metrics)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": 1496,
        "evidence_tier": evidence_tier,
        "claim_boundary": CLAIM_BOUNDARY,
        "task": "bounded BC loader/overfit smoke on the expert_traj_v1 training split",
        "smoke_not_quality": True,
        "training_performed": True,
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "size_bytes": dataset_path.stat().st_size,
        },
        "split_contract": split_report,
        "per_split_steps": per_split_steps,
        "trainer_config": asdict(config),
        "architecture": {
            "model_type": "mlp",
            "input_dim": input_dim,
            "hidden_dim": config.hidden_dim,
            "output_dim": output_dim,
            "activation": "tanh",
            "feature_fields": list(config.feature_fields()),
        },
        "metrics": metrics,
        "checkpoint": checkpoint_path.name,
        "not_full_warm_start_comparison": True,
    }
    atomic_write_json(manifest_path, manifest)

    return BCSmokeResult(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(manifest_path),
        metrics_path=str(metrics_path),
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_reduction=loss_reduction,
        num_train_steps=num_train_steps,
        episode_counts=split_report["episode_counts"],
        evidence_tier=evidence_tier,
    )
