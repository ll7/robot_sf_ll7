"""Bind post-freeze acquisition to unchanged source bytes, without scientific admission."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.snqi.v2_spec import load_snqi_v2_spec, parse_v2_json
from robot_sf.common.artifact_paths import get_repository_root

ASSET_NAMES = ("weights", "anchors", "family", "acquisition_config")


def load_acquisition_binding(raw: Any, config_path: Path) -> dict[str, Any] | None:
    """Validate explicit pending assets; historical immediate specifications stay strict.

    Returns:
        Pending acquisition pins, or None for historical immediate specifications."""
    if not isinstance(raw, dict) or "anchor_freeze" not in raw:
        return None
    required = {f"{name}_{suffix}" for name in ASSET_NAMES for suffix in ("path", "sha256")}
    if set(raw) != required | {"anchor_freeze"} or raw["anchor_freeze"] != "same-source-custody":
        raise ValueError(
            "SNQI-v2 acquisition binding requires explicit assets and same-source-custody"
        )
    root = next(
        (parent for parent in config_path.parents if (parent / "configs").is_dir()),
        get_repository_root(),
    )
    binding: dict[str, Any] = {"anchor_freeze": raw["anchor_freeze"]}
    for name in ASSET_NAMES:
        path = Path(raw[f"{name}_path"])
        if not path.is_absolute():
            local = config_path.parent / path
            path = local if local.exists() else root / path
        path = path.resolve()
        if sha256_file(path) != raw[f"{name}_sha256"]:
            raise ValueError(f"SNQI-v2 {name} acquisition asset digest mismatch")
        binding[f"{name}_path"] = path
        binding[f"{name}_sha256"] = raw[f"{name}_sha256"]
    anchors = parse_v2_json(binding["anchors_path"].read_bytes())
    if anchors.get("status") != "pending_calibration":
        raise ValueError("SNQI-v2 same-source acquisition requires pending source anchors")
    return binding


def bind_acquired_anchors(
    cfg: Any,
    *,
    calibration_root: Path | None,
    anchors_path: Path,
    source_commit: str,
    diagnostic: bool,
) -> Any:
    """Re-derive custody-bound anchors before scoring; caller receipts confer no trust.

    When raw custody is supplied, the complete producer grid, raw row/sidecar
    identity and input hashes are revalidated by the canonical freeze tool.
    Otherwise the strict frozen loader and source/config bindings apply; trusted
    anchor bytes must additionally be admitted by the independent private mint.
    The source must equal this run's identity. An independent reviewer still owns
    production scientific admission.

    Returns:
        Campaign with a custody-validated scoring spec and the explicit diagnostic flag.
    """
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config  # noqa: PLC0415
    from robot_sf.benchmark.snqi.v2_calibration import freeze_campaign_anchors  # noqa: PLC0415

    binding = cfg.snqi_v2_binding
    if binding is None:
        raise ValueError("campaign has no source-bound SNQI-v2 acquisition binding")
    acquisition = load_campaign_config(binding["acquisition_config_path"])
    if acquisition.source_config_sha256 != binding["acquisition_config_sha256"]:
        raise ValueError("SNQI-v2 acquisition configuration changed after binding")
    anchors_bytes = anchors_path.read_bytes()
    derived = parse_v2_json(anchors_bytes)
    if calibration_root is not None:
        manifest = parse_v2_json((calibration_root / "campaign_manifest.json").read_bytes())
        if manifest.get("git", {}).get("commit") != source_commit:
            raise ValueError("SNQI-v2 calibration source differs from release identity")
        with tempfile.TemporaryDirectory(
            prefix="anchor-custody-", dir=anchors_path.parent
        ) as scratch:
            derived = freeze_campaign_anchors(
                calibration_root, Path(scratch) / "anchors.json", campaign_config=acquisition
            )
    _validate_acquisition_identity(
        derived, acquisition, source_commit, require_config_identity=calibration_root is None
    )
    if derived.get("calibration", {}).get("seeds") != [1001, 1002]:
        raise ValueError("SNQI-v2 acquisition requires development seeds 1001/1002")
    if derived != parse_v2_json(anchors_bytes):
        raise ValueError("SNQI-v2 supplied anchors differ from acquired custody")
    spec = load_snqi_v2_spec(binding["weights_path"], anchors_path, binding["family_path"])
    if any(spec.hashes[name] != binding[f"{name}_sha256"] for name in ("weights", "family")):
        raise ValueError("SNQI-v2 scoring assets changed after source binding")
    if sha256_file(anchors_path) != spec.hashes["anchors"]:
        raise ValueError("SNQI-v2 anchors changed during acquisition binding")
    return replace(cfg, snqi_v2_spec=replace(spec, diagnostic=diagnostic))


def _validate_acquisition_identity(
    derived: dict[str, Any], acquisition: Any, source_commit: str, *, require_config_identity: bool
) -> None:
    """Match frozen source and configuration; raw rederivation already validates the latter."""
    calibration = derived.get("calibration", {})
    if calibration.get("source_commit") != source_commit:
        raise ValueError("SNQI-v2 calibration source differs from release identity")
    if require_config_identity:
        import hashlib  # noqa: PLC0415
        import json  # noqa: PLC0415

        from robot_sf.benchmark.camera_ready._util import _config_hash_payload  # noqa: PLC0415

        expected = hashlib.sha256(
            json.dumps(
                _config_hash_payload(acquisition), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if calibration.get("campaign_config_hash") != expected:
            raise ValueError("SNQI-v2 anchors do not bind the frozen acquisition configuration")
