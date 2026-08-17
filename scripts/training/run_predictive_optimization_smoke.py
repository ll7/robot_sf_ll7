#!/usr/bin/env python3
"""Run the bounded GPU optimization smoke comparison from issue #7254."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from robot_sf.planner.obstacle_features import infer_predictive_feature_schema

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_SCHEMA = "predictive-optimization-smoke.v1"
_RESULT_SCHEMA = "predictive-optimization-smoke-result.v1"
_METRICS = ("train_loss", "val_loss", "val_ade", "val_fde")
_TIMING_FIELDS = ("train_runtime_sec", "val_runtime_sec")
_COUNT_FIELDS = ("train_steps", "val_steps", "train_examples", "val_examples")


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    """Read a YAML mapping and fail closed on a different top-level shape."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a YAML mapping: {path}")
    return dict(payload)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return a nested mapping with a useful contract error."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _int(value: Any, *, label: str, minimum: int | None = None) -> int:
    """Parse an integer without accepting booleans or floats."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}")
    return int(value)


def _float(value: Any, *, label: str, minimum: float | None = None) -> float:
    """Parse a finite floating-point configuration value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite, got {value!r}")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {parsed}")
    return parsed


def _bool(value: Any, *, label: str) -> bool:
    """Parse a strict YAML boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean, got {value!r}")
    return value


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    """Return the exact repository commit used for every subprocess arm."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()


def _validate_fixture(fixture: dict[str, Any]) -> None:
    """Validate the deterministic fixture dimensions."""
    dataset_id = fixture.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("fixture.dataset_id must be a non-empty string")
    _int(fixture.get("seed"), label="fixture.seed", minimum=0)
    _int(fixture.get("samples"), label="fixture.samples", minimum=64)
    _int(fixture.get("max_agents"), label="fixture.max_agents", minimum=1)
    if _int(fixture.get("input_dim"), label="fixture.input_dim", minimum=4) != 4:
        raise ValueError("fixture.input_dim must be 4 for the legacy predictive schema")
    _int(fixture.get("horizon_steps"), label="fixture.horizon_steps", minimum=2)


def _validate_training(training: dict[str, Any]) -> int:
    """Validate common trainer settings and return the epoch count."""
    epochs = _int(training.get("epochs"), label="training.epochs", minimum=2)
    _int(training.get("batch_size"), label="training.batch_size", minimum=1)
    _int(training.get("seed"), label="training.seed", minimum=0)
    _int(training.get("hidden_dim"), label="training.hidden_dim", minimum=1)
    _int(training.get("message_passing_steps"), label="training.message_passing_steps", minimum=0)
    _float(training.get("lr"), label="training.lr", minimum=0.0)
    _float(training.get("weight_decay"), label="training.weight_decay", minimum=0.0)
    val_split = _float(training.get("val_split"), label="training.val_split")
    if not 0.0 < val_split < 1.0:
        raise ValueError("training.val_split must be between 0 and 1")
    return epochs


def _validate_comparison(comparison: dict[str, Any], *, epochs: int) -> None:
    """Validate the predeclared warm-up, dispersion, and metric rules."""
    warmup = _int(comparison.get("warmup_epochs"), label="comparison.warmup_epochs", minimum=0)
    if warmup >= epochs:
        raise ValueError("comparison.warmup_epochs must be less than training.epochs")
    _float(
        comparison.get("control_dispersion_multiplier"),
        label="comparison.control_dispersion_multiplier",
        minimum=0.0,
    )
    _float(
        comparison.get("minimum_abs_tolerance"),
        label="comparison.minimum_abs_tolerance",
        minimum=0.0,
    )
    metrics = comparison.get("metrics")
    if tuple(metrics or ()) != _METRICS:
        raise ValueError(f"comparison.metrics must be exactly {_METRICS!r}")


def _validate_arm(arm: dict[str, Any]) -> None:
    """Validate one optimization arm's permitted trainer flags."""
    arm_id = str(arm.get("id"))
    _int(arm.get("repeats"), label=f"arms[{arm_id}].repeats", minimum=1)
    _int(arm.get("num_workers"), label=f"arms[{arm_id}].num_workers", minimum=0)
    _bool(arm.get("pin_memory"), label=f"arms[{arm_id}].pin_memory")
    _bool(arm.get("persistent_workers"), label=f"arms[{arm_id}].persistent_workers")
    _bool(arm.get("amp"), label=f"arms[{arm_id}].amp")
    _int(arm.get("prefetch_factor"), label=f"arms[{arm_id}].prefetch_factor", minimum=1)


def _validate_arm_semantics(arm: dict[str, Any]) -> None:
    """Ensure each named arm changes only the flags declared by the contract."""
    expected = {
        "fp32_control": {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "amp": False,
        },
        "fp32_loader": {
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "amp": False,
        },
        "amp_loader": {
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "amp": True,
        },
    }
    arm_id = str(arm["id"])
    for field, expected_value in expected[arm_id].items():
        if arm[field] != expected_value:
            raise ValueError(
                f"arms[{arm_id}].{field} must be {expected_value!r}, got {arm[field]!r}"
            )


def _expected_loader_manifest(arm: dict[str, Any]) -> dict[str, Any]:
    """Return the effective CUDA loader manifest required for one arm."""
    num_workers = int(arm["num_workers"])
    return {
        "num_workers": num_workers,
        "pin_memory": bool(arm["pin_memory"]),
        "persistent_workers": bool(arm["persistent_workers"] and num_workers > 0),
        "prefetch_factor": int(arm["prefetch_factor"]) if num_workers > 0 else None,
        "non_blocking": bool(arm["pin_memory"]),
        "device": "cuda",
    }


def _expected_amp_manifest(arm: dict[str, Any]) -> dict[str, Any]:
    """Return the effective AMP manifest required for one arm."""
    enabled = bool(arm["amp"])
    return {"enabled": enabled, "requested": enabled, "dtype": "float16" if enabled else None}


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the frozen three-arm comparison contract."""
    if config.get("schema_version") != _CONFIG_SCHEMA:
        raise ValueError(f"expected schema_version={_CONFIG_SCHEMA!r}")
    if _int(config.get("issue"), label="issue", minimum=1) != 7254:
        raise ValueError("this runner is reserved for issue #7254")
    _validate_fixture(_mapping(config.get("fixture"), label="fixture"))
    epochs = _validate_training(_mapping(config.get("training"), label="training"))
    _validate_comparison(_mapping(config.get("comparison"), label="comparison"), epochs=epochs)

    arms = config.get("arms")
    if not isinstance(arms, list) or not arms:
        raise TypeError("arms must be a non-empty list")
    arm_ids = {str(_mapping(arm, label="arm").get("id")) for arm in arms}
    if arm_ids != {"fp32_control", "fp32_loader", "amp_loader"}:
        raise ValueError("arms must contain exactly fp32_control, fp32_loader, and amp_loader")
    for arm_raw in arms:
        arm = _mapping(arm_raw, label="arm")
        _validate_arm(arm)
        _validate_arm_semantics(arm)
    if not any(_bool(_mapping(arm, label="arm").get("amp"), label="arm.amp") for arm in arms):
        raise ValueError("one arm must request AMP")


def _validate_summary_identity(
    *, summary: dict[str, Any], arm_id: str, dataset: dict[str, Any], git_commit: str
) -> None:
    """Validate commit, device, and dataset provenance in one trainer summary."""
    if str(summary.get("git_commit")) != git_commit:
        raise RuntimeError(f"{arm_id} recorded a different git commit: {summary.get('git_commit')}")
    if str(summary.get("device")) != "cuda":
        raise RuntimeError(f"{arm_id} did not run on CUDA: {summary.get('device')!r}")
    dataset_text = str(summary.get("dataset", "")).strip()
    if not dataset_text or Path(dataset_text).resolve() != Path(str(dataset["path"])).resolve():
        raise RuntimeError(f"{arm_id} recorded a different dataset: {summary.get('dataset')!r}")
    expected_dataset_ids = [f"prediction_planner:{dataset['dataset_id']}"]
    if summary.get("source_dataset_ids") != expected_dataset_ids:
        raise RuntimeError(
            f"{arm_id} recorded different dataset ids: {summary.get('source_dataset_ids')!r}"
        )


def _validate_summary_runtime_flags(*, summary: dict[str, Any], arm: dict[str, Any]) -> None:
    """Validate effective loader and AMP settings in one trainer summary."""
    arm_id = str(arm["id"])
    if summary.get("data_loader") != _expected_loader_manifest(arm):
        raise RuntimeError(f"{arm_id} loader manifest drifted: {summary.get('data_loader')!r}")
    if summary.get("amp") != _expected_amp_manifest(arm):
        raise RuntimeError(f"{arm_id} AMP manifest drifted: {summary.get('amp')!r}")


def _validate_summary_training_contract(
    *, summary: dict[str, Any], config: dict[str, Any], arm_id: str
) -> None:
    """Validate common trainer and model settings against the frozen config."""
    training = _mapping(config["training"], label="training")
    expected_scalars = {
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["lr"]),
        "weight_decay": float(training["weight_decay"]),
        "seed": int(training["seed"]),
    }
    for field, expected in expected_scalars.items():
        if summary.get(field) != expected:
            raise RuntimeError(
                f"{arm_id} recorded {field}={summary.get(field)!r}, expected {expected!r}"
            )

    model_config = _mapping(summary.get("config"), label=f"{arm_id}.config")
    fixture = _mapping(config["fixture"], label="fixture")
    expected_model_fields = {
        "max_agents": int(fixture["max_agents"]),
        "horizon_steps": int(fixture["horizon_steps"]),
        "input_dim": int(fixture["input_dim"]),
        "hidden_dim": int(training["hidden_dim"]),
        "message_passing_steps": int(training["message_passing_steps"]),
    }
    for field, expected in expected_model_fields.items():
        if model_config.get(field) != expected:
            raise RuntimeError(
                f"{arm_id} recorded model config {field}={model_config.get(field)!r}, "
                f"expected {expected!r}"
            )


def _validate_arm_summary(
    *,
    summary: dict[str, Any],
    config: dict[str, Any],
    arm: dict[str, Any],
    dataset: dict[str, Any],
    git_commit: str,
) -> None:
    """Fail closed when a trainer summary drifts from the frozen arm contract."""
    arm_id = str(arm["id"])
    _validate_summary_identity(
        summary=summary, arm_id=arm_id, dataset=dataset, git_commit=git_commit
    )
    _validate_summary_runtime_flags(summary=summary, arm=arm)
    _validate_summary_training_contract(summary=summary, config=config, arm_id=arm_id)


def _write_fixture(*, fixture: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Generate the deterministic non-degenerate predictive training fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = _int(fixture["seed"], label="fixture.seed", minimum=0)
    samples = _int(fixture["samples"], label="fixture.samples", minimum=64)
    max_agents = _int(fixture["max_agents"], label="fixture.max_agents", minimum=1)
    horizon_steps = _int(fixture["horizon_steps"], label="fixture.horizon_steps", minimum=2)
    rng = np.random.default_rng(seed)
    positions = rng.normal(0.0, 1.0, size=(samples, max_agents, 2)).astype(np.float32)
    velocities = rng.normal(0.0, 0.25, size=(samples, max_agents, 2)).astype(np.float32)
    state = np.concatenate((positions, velocities), axis=2)
    times = np.arange(1, horizon_steps + 1, dtype=np.float32)[None, None, :, None]
    noise = rng.normal(0.0, 0.01, size=(samples, max_agents, horizon_steps, 2)).astype(np.float32)
    target = positions[:, :, None, :] + velocities[:, :, None, :] * times + noise
    mask = np.ones((samples, max_agents), dtype=np.float32)
    target_mask = np.ones((samples, max_agents, horizon_steps), dtype=np.float32)

    dataset_path = output_dir / "predictive_fixture.npz"
    np.savez_compressed(
        dataset_path, state=state, target=target, mask=mask, target_mask=target_mask
    )
    manifest = {
        "schema_version": "predictive-training-fixture.v1",
        "dataset_id": str(fixture["dataset_id"]),
        "generator": "scripts/training/run_predictive_optimization_smoke.py",
        "seed": seed,
        "arrays": {
            "state": list(state.shape),
            "target": list(target.shape),
            "mask": list(mask.shape),
            "target_mask": list(target_mask.shape),
        },
        "feature_schema": infer_predictive_feature_schema(int(state.shape[2])),
        "dataset_sha256": _sha256(dataset_path),
    }
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"path": str(dataset_path), "manifest_path": str(manifest_path), **manifest}


def _host_metadata() -> dict[str, Any]:
    """Capture compact host/runtime provenance without storing raw environment dumps."""
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
        payload["gpu_total_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        payload["nvidia_smi"] = None
    else:
        payload["nvidia_smi"] = result.stdout.strip()
    return payload


def _arm_training_args(*, config: dict[str, Any], arm: dict[str, Any]) -> list[str]:
    """Build the canonical trainer CLI for one frozen arm."""
    training = _mapping(config["training"], label="training")
    args = [
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
        "--lr",
        str(training["lr"]),
        "--weight-decay",
        str(training["weight_decay"]),
        "--val-split",
        str(training["val_split"]),
        "--seed",
        str(training["seed"]),
        "--hidden-dim",
        str(training["hidden_dim"]),
        "--message-passing-steps",
        str(training["message_passing_steps"]),
        "--num-workers",
        str(arm["num_workers"]),
        "--prefetch-factor",
        str(arm["prefetch_factor"]),
        "--max-val-ade",
        "1e9",
        "--max-val-fde",
        "1e9",
    ]
    if bool(arm["pin_memory"]):
        args.append("--pin-memory")
    if bool(arm["persistent_workers"]):
        args.append("--persistent-workers")
    if bool(arm["amp"]):
        args.append("--amp")
    return args


def _finite_history(summary: dict[str, Any]) -> bool:
    """Return whether all required metric/timing fields are finite."""
    history = summary.get("history")
    required_fields = (*_METRICS, *_TIMING_FIELDS, "train_steps", "val_steps")
    if (
        not isinstance(history, list)
        or not history
        or any(not isinstance(row, dict) for row in history)
    ):
        return False
    return all(
        all(field in row and math.isfinite(float(row[field])) for field in required_fields)
        for row in history
    )


def _throughput(summary: dict[str, Any], *, warmup_epochs: int) -> dict[str, float | int]:
    """Compute train throughput after the predeclared warm-up epochs."""
    history = [dict(row) for row in summary["history"]][warmup_epochs:]
    train_seconds = sum(float(row["train_runtime_sec"]) for row in history)
    train_examples = sum(float(row["train_examples"]) for row in history)
    train_steps = sum(float(row["train_steps"]) for row in history)
    if train_seconds <= 0.0:
        raise ValueError("training runtime is zero; cannot compute throughput")
    return {
        "warmup_epochs": int(warmup_epochs),
        "measured_epochs": len(history),
        "train_examples": train_examples,
        "train_steps": train_steps,
        "train_runtime_sec": train_seconds,
        "examples_per_sec": train_examples / train_seconds,
        "steps_per_sec": train_steps / train_seconds,
    }


def _run_arm(
    *,
    config: dict[str, Any],
    arm: dict[str, Any],
    repeat: int,
    dataset: dict[str, Any],
    output_root: Path,
    git_commit: str,
) -> dict[str, Any]:
    """Run one canonical training subprocess and collect compact provenance."""
    arm_id = str(arm["id"])
    run_dir = output_root / "runs" / arm_id / f"repeat_{repeat:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "trainer.log"
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts/training/train_predictive_planner.py"),
        "--dataset",
        str(dataset["path"]),
        "--output-dir",
        str(run_dir),
        *_arm_training_args(config=config, arm=arm),
    ]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            env=dict(os.environ),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    wall_runtime_sec = float(time.perf_counter() - started)
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(f"{arm_id} repeat {repeat} failed ({completed.returncode}):\n{tail}")

    summary_path = run_dir / "training_summary.json"
    summary = _read_mapping(summary_path, label="training summary")
    _validate_arm_summary(
        summary=summary,
        config=config,
        arm=arm,
        dataset=dataset,
        git_commit=git_commit,
    )
    if not _finite_history(summary):
        raise RuntimeError(f"{arm_id} produced missing or non-finite metric history")
    checkpoint_path = Path(str(summary.get("checkpoint", "")))
    if not checkpoint_path.exists():
        raise RuntimeError(f"{arm_id} did not produce a readable checkpoint path")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if state_dict is None and isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"{arm_id} checkpoint has no supported state-dict payload")
    throughput = _throughput(
        summary,
        warmup_epochs=_int(
            _mapping(config["comparison"], label="comparison")["warmup_epochs"],
            label="comparison.warmup_epochs",
            minimum=0,
        ),
    )
    return {
        "arm": arm_id,
        "repeat": int(repeat),
        "run_dir": str(run_dir),
        "command": command,
        "wall_runtime_sec": wall_runtime_sec,
        "dataset_sha256": dataset["dataset_sha256"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "device": summary["device"],
        "data_loader": summary["data_loader"],
        "amp": summary["amp"],
        "peak_cuda_memory_allocated_bytes": summary.get("peak_cuda_memory_allocated_bytes"),
        "peak_cuda_memory_reserved_bytes": summary.get("peak_cuda_memory_reserved_bytes"),
        "history": summary["history"],
        "throughput": throughput,
    }


def _count_signature(history: list[dict[str, Any]]) -> tuple[tuple[int, ...], ...]:
    """Return validated per-epoch update/example counts for cross-arm comparison."""
    signature: list[tuple[int, ...]] = []
    for epoch, row in enumerate(history, start=1):
        counts: list[int] = []
        for field in _COUNT_FIELDS:
            value = float(row[field])
            if not math.isfinite(value) or value <= 0.0 or not value.is_integer():
                raise ValueError(f"epoch {epoch} has invalid {field}: {row[field]!r}")
            counts.append(int(value))
        signature.append(tuple(counts))
    return tuple(signature)


def _compare(*, results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Compare optimized arms against repeated FP32 control curves."""
    comparison = _mapping(config["comparison"], label="comparison")
    warmup = _int(comparison["warmup_epochs"], label="comparison.warmup_epochs", minimum=0)
    multiplier = _float(
        comparison["control_dispersion_multiplier"],
        label="comparison.control_dispersion_multiplier",
        minimum=0.0,
    )
    minimum_tolerance = _float(
        comparison["minimum_abs_tolerance"], label="comparison.minimum_abs_tolerance", minimum=0.0
    )
    controls = [result for result in results if result["arm"] == "fp32_control"]
    if len(controls) < 2:
        raise ValueError("at least two FP32 control repeats are required")

    configured_arms = {
        str(_mapping(arm, label="arm")["id"]): _mapping(arm, label="arm") for arm in config["arms"]
    }
    for arm_id, arm in configured_arms.items():
        observed = [result for result in results if str(result["arm"]) == arm_id]
        expected_repeats = _int(arm["repeats"], label=f"arms[{arm_id}].repeats", minimum=1)
        if len(observed) != expected_repeats:
            raise ValueError(f"{arm_id} has {len(observed)} repeats; expected {expected_repeats}")

    dataset_digests = {str(result.get("dataset_sha256", "")) for result in results}
    if len(dataset_digests) != 1 or not next(iter(dataset_digests), ""):
        raise ValueError("all arms must use one non-empty dataset digest")
    count_signatures = {
        _count_signature([dict(row) for row in result["history"]]) for result in results
    }
    if len(count_signatures) != 1:
        raise ValueError("all arms must process identical data identities and update counts")

    control_histories = [[dict(row) for row in result["history"]][warmup:] for result in controls]
    control_reference: dict[str, list[float]] = {}
    control_dispersion: dict[str, float] = {}
    envelopes: dict[str, float] = {}
    for metric in _METRICS:
        values = [[float(row[metric]) for row in history] for history in control_histories]
        control_reference[metric] = [
            float(np.mean(epoch_values)) for epoch_values in zip(*values, strict=True)
        ]
        max_pairwise_delta = max(
            abs(left - right) for left, right in zip(values[0], values[1], strict=True)
        )
        control_dispersion[metric] = float(max_pairwise_delta)
        envelopes[metric] = max(minimum_tolerance, multiplier * max_pairwise_delta)

    arms: dict[str, Any] = {}
    all_equivalent = True
    for arm_id in sorted({str(result["arm"]) for result in results}):
        arm_results = [result for result in results if result["arm"] == arm_id]
        arm_payload: dict[str, Any] = {
            "repeats": len(arm_results),
            "throughput": {
                "examples_per_sec": float(
                    np.mean([result["throughput"]["examples_per_sec"] for result in arm_results])
                ),
                "steps_per_sec": float(
                    np.mean([result["throughput"]["steps_per_sec"] for result in arm_results])
                ),
            },
            "peak_cuda_memory_allocated_bytes": [
                result["peak_cuda_memory_allocated_bytes"] for result in arm_results
            ],
        }
        if arm_id != "fp32_control":
            deltas: dict[str, float] = {}
            arm_equivalent = True
            for metric in _METRICS:
                delta = max(
                    abs(float(row[metric]) - reference)
                    for result in arm_results
                    for row, reference in zip(
                        [dict(row) for row in result["history"]][warmup:],
                        control_reference[metric],
                        strict=True,
                    )
                )
                deltas[metric] = float(delta)
                arm_equivalent = arm_equivalent and delta <= envelopes[metric]
            arm_payload["max_abs_delta_vs_control_mean"] = deltas
            arm_payload["equivalent_to_control"] = bool(arm_equivalent)
            all_equivalent = all_equivalent and arm_equivalent
        arms[arm_id] = arm_payload

    return {
        "warmup_epochs_excluded": warmup,
        "metrics": list(_METRICS),
        "control_reference": control_reference,
        "control_max_pairwise_abs_delta": control_dispersion,
        "equivalence_envelope": envelopes,
        "arms": arms,
        "result_classification": "equivalent_smoke" if all_equivalent else "numerically_divergent",
    }


def _markdown(result: dict[str, Any]) -> str:
    """Render a compact human-readable diagnostic handoff."""
    comparison = result["comparison"]
    lines = [
        "# Issue #7254 predictive optimization smoke",
        "",
        f"- Result classification: `{comparison['result_classification']}`",
        f"- Git commit: `{result['git_commit']}`",
        f"- Dataset SHA-256: `{result['dataset']['dataset_sha256']}`",
        f"- Host: `{result['host'].get('gpu_name', 'unavailable')}`; CUDA `{result['host'].get('torch_cuda')}`",
        f"- Warm-up excluded: `{comparison['warmup_epochs_excluded']}` epoch(s)",
        "",
        "| Arm | Repeats | Examples/s after warm-up | Peak allocated bytes | Equivalent to FP32 control |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for arm_id, payload in comparison["arms"].items():
        equivalent = payload.get("equivalent_to_control", "control")
        memory = ", ".join(str(value) for value in payload["peak_cuda_memory_allocated_bytes"])
        lines.append(
            f"| `{arm_id}` | {payload['repeats']} | {payload['throughput']['examples_per_sec']:.2f} | "
            f"{memory} | `{equivalent}` |"
        )
    lines.extend(
        [
            "",
            "This is diagnostic implementation/smoke evidence only. It does not establish policy "
            "equivalence, benchmark improvement, general GPU speedup, or a paper claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runner CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/predictive/predictive_optimization_smoke_issue_7254.yaml"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print the compact result JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the frozen comparison and write compact diagnostic artifacts."""
    args = parse_args(argv)
    config_path = args.config if args.config.is_absolute() else (_REPO_ROOT / args.config).resolve()
    config = _read_mapping(config_path, label="config")
    _validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("issue #7254 requires a CUDA-capable host; refusing a CPU fallback")
    output_root_raw = args.output_root or _mapping(config["output"], label="output").get("root")
    output_root = Path(str(output_root_raw))
    if not output_root.is_absolute():
        output_root = (_REPO_ROOT / output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)

    git_commit = _git_commit()
    fixture = _write_fixture(
        fixture=_mapping(config["fixture"], label="fixture"),
        output_dir=output_root / "dataset",
    )
    arms = [_mapping(arm, label="arm") for arm in config["arms"]]
    results: list[dict[str, Any]] = []
    for arm in arms:
        for repeat in range(1, _int(arm["repeats"], label="arm.repeats", minimum=1) + 1):
            results.append(
                _run_arm(
                    config=config,
                    arm=arm,
                    repeat=repeat,
                    dataset=fixture,
                    output_root=output_root,
                    git_commit=git_commit,
                )
            )
    result = {
        "schema_version": _RESULT_SCHEMA,
        "issue": 7254,
        "config": str(config_path),
        "git_commit": git_commit,
        "host": _host_metadata(),
        "dataset": fixture,
        "runs": results,
        "comparison": _compare(results=results, config=config),
    }
    result_path = output_root / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "comparison.md").write_text(_markdown(result), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {"status": "ok", "result": str(result_path), **result["comparison"]}, indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
