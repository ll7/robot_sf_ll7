"""Pre-registered SNQI-v2 anchor derivation from the complete development split.

This module never chooses weights or evaluates planner rankings. It checks the
14-arm by 48-scenario by two-seed grid, selects F by the declared correlation rule,
and computes linear-interpolated episode p95 anchors without zero imputation.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import spearmanr

from robot_sf.benchmark.fallback_policy import (
    summarize_benchmark_availability,
)
from robot_sf.benchmark.snqi.v2_reports import read_episode_files, validate_episode_execution
from robot_sf.benchmark.snqi.v2_spec import (
    PP_EQUIV_FORCE,
    SIMULATED_FORCE,
    finite_nonnegative,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def derive_calibration_anchors(
    episodes: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str],
    scenarios: Sequence[str],
    run_id: str,
    source_commit: str,
    episodes_sha256: str,
) -> dict[str, Any]:
    """Derive the frozen asset only after complete source and split validation.

    Returns:
        A JSON-serializable anchor document ready for review and commit.
    """
    expected = set(product(arms, scenarios, (101, 102)))
    if len(set(arms)) != 14 or len(set(scenarios)) != 48 or len(episodes) != 1344:
        raise ValueError("SNQI-v2 calibration requires 14 arms x48 scenarios x2 seeds =1344")
    observed = set()
    force, exposure, fractions = [], [], []
    for episode in episodes:
        identity = (episode.get("planner_key"), episode.get("scenario_id"), episode.get("seed"))
        if identity in observed or identity not in expected:
            raise ValueError(f"SNQI-v2 calibration duplicate or out-of-split identity: {identity}")
        observed.add(identity)
        validate_episode_execution(episode)
        if episode.get("status") not in {"success", "collision", "failure"}:
            raise ValueError(
                f"SNQI-v2 calibration rejects failed/fallback/degraded row: {identity}"
            )
        metrics = episode["metrics"]
        steps = finite_nonnegative(episode.get("steps"), "executed steps")
        near = finite_nonnegative(metrics.get("near_misses"), "near_misses")
        if steps < 1 or not steps.is_integer() or not near.is_integer() or near > steps:
            raise ValueError("SNQI-v2 calibration invalid close-clearance step coverage")
        force.append(finite_nonnegative(metrics.get(SIMULATED_FORCE), SIMULATED_FORCE))
        fractions.append(near / steps)
        exposure.append(min(near / steps / 0.25, 1.0))
    if observed != expected:
        raise ValueError("SNQI-v2 calibration grid incomplete")
    if len(set(force)) < 2 or len(set(exposure)) < 2:
        raise ValueError("SNQI-v2 F/N calibration correlation is undefined")
    rho = float(spearmanr(force, exposure).statistic)
    source = PP_EQUIV_FORCE if abs(rho) >= 0.90 else SIMULATED_FORCE
    anchors = {
        "T": {"lower": 0, "upper": 3, "type": "normative"},
        "N": {"lower": 0, "upper": 0.25, "type": "normative"},
    }
    for term, metric in (("F", source), ("J", "jerk_mean"), ("K", "curvature_mean")):
        values = [finite_nonnegative(ep["metrics"].get(metric), metric) for ep in episodes]
        upper = float(np.percentile(values, 95, method="linear"))
        if upper <= 0:
            raise ValueError(f"SNQI-v2 {term} calibration p95 is not positive")
        anchors[term] = {"lower": 0, "upper": upper, "type": "calibration_p95"}
    grid_hash = hashlib.sha256(
        json.dumps(sorted(expected), separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "version": "SNQI-v2.0",
        "status": "frozen",
        "anchors": anchors,
        "force_decision": {
            "source": source,
            "spearman_rho_F_N": rho,
            "spearman_rho_F_exposure_fraction": float(spearmanr(force, fractions).statistic),
            "threshold_absolute_rho": 0.90,
            "N": "clip(near_misses/steps/0.25)",
            "selected_source_coverage": len(episodes),
        },
        "calibration": {
            "split_id": f"snqi-v2-dev101-102-{grid_hash[:12]}",
            "run_id": run_id,
            "source_commit": source_commit,
            "episodes_sha256": episodes_sha256,
            "grid_sha256": grid_hash,
            "seeds": [101, 102],
            "episode_count": 1344,
            "arms": sorted(arms),
            "scenarios": sorted(scenarios),
            "execution_mode": "native",
            "quantile_method": "linear",
        },
    }


def freeze_campaign_anchors(campaign_root: Path, output_path: Path) -> dict[str, Any]:
    """Validate a completed development campaign and atomically write its anchor asset.

    Returns:
        Frozen anchor document, including hashes of every source episode file.
    """
    manifest = json.loads((campaign_root / "campaign_manifest.json").read_text())
    summary = json.loads((campaign_root / "reports/campaign_summary.json").read_text())
    preview = json.loads((campaign_root / "preflight/preview_scenarios.json").read_text())
    if preview.get("truncated") or manifest["seed_policy"]["resolved_seeds"] != [101, 102]:
        raise ValueError("SNQI-v2 calibration requires complete preview and development seeds")
    arms = [arm["key"] for arm in manifest["planners"] if arm["enabled"]]
    scenarios = [scenario["name"] for scenario in preview["scenarios"]]
    records, hashes = [], {}
    for entry in summary["runs"]:
        availability = summarize_benchmark_availability(entry.get("summary"))
        if entry.get("status") != "ok" or not availability.benchmark_success:
            raise ValueError("SNQI-v2 calibration run has incomplete/fallback/degraded execution")
        source_path = Path(entry["episodes_path"])
        # Campaign archives preserve the runs/ tree while their producer's absolute
        # workspace may no longer exist. Resolve only that confined archive suffix.
        if "runs" not in source_path.parts:
            raise ValueError("SNQI-v2 calibration source path must be under campaign runs/")
        suffix = source_path.parts[source_path.parts.index("runs") :]
        path = (campaign_root.joinpath(*suffix)).resolve()
        if not path.is_relative_to(campaign_root.resolve()):
            raise ValueError("SNQI-v2 calibration source escapes campaign root")
        key = str(path.relative_to(campaign_root.resolve()))
        if key in hashes:
            raise ValueError("SNQI-v2 calibration duplicate episode source")
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        for record in read_episode_files([path]):
            if record.get("git_hash") != manifest["git"]["commit"]:
                raise ValueError("SNQI-v2 calibration record source commit mismatch")
            records.append({**record, "planner_key": entry["planner"]["key"]})
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    document = derive_calibration_anchors(
        records,
        arms=arms,
        scenarios=scenarios,
        run_id=manifest["campaign_id"],
        source_commit=manifest["git"]["commit"],
        episodes_sha256=digest,
    )
    document["calibration"]["episode_files_sha256"] = hashes
    document["calibration"]["episodes_hash_rule"] = (
        "sha256(sorted compact JSON relative-path-to-file-sha256 map)"
    )
    document["calibration"]["campaign_manifest_sha256"] = hashlib.sha256(
        (campaign_root / "campaign_manifest.json").read_bytes()
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output_path)
    return document
