"""Read-only census of predictive horizon bindings in all benchmark campaign matrices."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml

from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec, SeedPolicy
from robot_sf.benchmark.camera_ready._run_state import _resolve_path
from robot_sf.benchmark.campaign.campaign_checkpoint_preflight import (
    CampaignCheckpointPreflightError,
)
from robot_sf.benchmark.campaign.predictive_horizon_preflight import (
    check_campaign_predictive_horizons_preflight,
)


def scan_predictive_horizons(
    root: Path,
    *,
    registry_path: Path | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Inspect every matrix's planner bindings without resolving seed policies or scenarios.

    Historical campaign YAML may no longer satisfy current campaign admission rules. Use
    the canonical arm path resolver rather than admitting/executing the whole campaign.
    Historical planner-group labels do not affect the checkpoint binding being audited.
    Unavailable checkpoints remain unverified; they never count as compatible.

    Returns:
        File/matrix census and per-binding compatibility or explicit verification failure.
    """
    files = sorted(p for p in root.rglob("*") if p.suffix in {".yaml", ".yml"})
    records = []
    matrices = 0
    other_formats = []
    for path in files:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("planners"), list):
            continue
        matrices += 1
        if any(not isinstance(a, dict) for a in payload["planners"]):
            other_formats.append(str(path))
            if any("predict" in str(a) for a in payload["planners"]):
                records.append(
                    {
                        "matrix": str(path),
                        "status": "unverified",
                        "detail": "Predictive planner-name roster has no campaign algo_config bindings",
                    }
                )
            continue
        # The compute-feasibility format uses inline configs instead of campaign paths.
        # Inspect its roster explicitly; an inline predictive arm cannot be admitted here.
        if any(isinstance(a.get("algo_config"), dict) for a in payload["planners"]):
            other_formats.append(str(path))
            if any("predict" in str(a.get("algo", "")) for a in payload["planners"]):
                records.append(
                    {
                        "matrix": str(path),
                        "status": "unverified",
                        "detail": "Inline predictive planner config requires its matrix loader",
                    }
                )
            continue
        cfg = CampaignConfig(
            name=path.stem,
            scenario_matrix_path=path,
            seed_policy=SeedPolicy(),
            planners=tuple(
                PlannerSpec(
                    key=str(a.get("key") or a["algo"]),
                    algo=str(a["algo"]),
                    algo_config_path=_resolve_path(a.get("algo_config"), base_dir=path.parent),
                    enabled=bool(a.get("enabled", True)),
                )
                for a in payload["planners"]
            ),
        )
        for arm in cfg.planners:
            try:
                bindings = check_campaign_predictive_horizons_preflight(
                    replace(cfg, planners=(arm,)),
                    registry_path=registry_path,
                    cache_dir=cache_dir,
                )
            except (CampaignCheckpointPreflightError, OSError, ValueError, TypeError) as exc:
                message = str(exc)
                records.append(
                    {
                        "matrix": str(path),
                        "planner_key": arm.key,
                        "algo_config_path": str(arm.algo_config_path),
                        "status": "incompatible"
                        if "required_horizon_steps=" in message
                        else "unverified",
                        "detail": message,
                    }
                )
            else:
                records.extend({"matrix": str(path), **binding} for binding in bindings)
    return {
        "schema_version": "predictive-horizon-scan.v1",
        "yaml_files_scanned": len(files),
        "campaign_matrices_scanned": matrices,
        "other_matrix_formats_inspected": other_formats,
        "compatible": sum(r["status"] == "compatible" for r in records),
        "incompatible": sum(r["status"] == "incompatible" for r in records),
        "unverified": sum(r["status"] == "unverified" for r in records),
        "bindings": records,
    }


def main() -> int:
    """Emit the complete census, returning failure for incompatible or unverified bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("configs/benchmarks"))
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = scan_predictive_horizons(
        args.root,
        registry_path=args.registry_path,
        cache_dir=args.cache_dir,
    )
    text = json.dumps(report, indent=2)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return int(bool(report["incompatible"] or report["unverified"]))


if __name__ == "__main__":
    raise SystemExit(main())
