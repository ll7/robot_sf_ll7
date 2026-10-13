"""Install fresh-campaign secondary seal capture using the runtime manifest writer.

Install before calling the sweep entry point. This does not repair historical
artifacts: the native primary writer must succeed in the same recorder call.
Only stdlib imports occur until installation, so pinned tooling can embed this
module while selecting the runtime package from its source-first PYTHONPATH.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class ProducerCaptureError(ValueError):
    """A fresh campaign cannot produce trustworthy secondary evidence."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ProducerCaptureError(f"producer capture refused: {reason}")


def _json(raw: bytes, *, legacy_raw: bool = False) -> Any:
    """Reject duplicate keys and numeric overflow; retain legacy raw constants."""

    def pairs(entries):
        result = {}
        for key, value in entries:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def number(token):
        value = float(token)
        _require(math.isfinite(value), "JSON numeric overflow")
        return value

    def constant(token):
        _require(legacy_raw, "nonfinite provenance JSON")
        return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[token]

    return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)


def _capture(out_path: Path, summary: dict, validate_raw_outcomes: Callable) -> None:
    """Validate the newly written primary seal before creating a secondary seal."""
    from robot_sf.benchmark.manifest import manifest_path_for, save_manifest
    from robot_sf.benchmark.result_provenance import (
        manifest_path_for_result_jsonl,
        validate_result_provenance_manifest,
    )

    primary = manifest_path_for_result_jsonl(out_path)
    secondary = manifest_path_for(out_path)
    _require(not secondary.exists() and not secondary.is_symlink(), "secondary manifest exists")
    provenance = summary.get("provenance")
    _require(
        isinstance(provenance, dict) and provenance.get("result_manifest_status") == "available",
        "native primary writer did not succeed",
    )
    _require(
        out_path.is_file()
        and not out_path.is_symlink()
        and primary.is_file()
        and not primary.is_symlink(),
        "raw or primary manifest is not a regular file",
    )
    raw_bytes, primary_bytes = out_path.read_bytes(), primary.read_bytes()
    payload = _json(primary_bytes)
    _require(isinstance(payload, dict), "primary manifest is not an object")
    validate_result_provenance_manifest(payload)
    _require(payload["completeness"].get("status") == "complete", "primary seal is incomplete")
    rows = [_json(line, legacy_raw=True) for line in raw_bytes.splitlines() if line.strip()]
    _require(bool(rows), "no emitted rows")
    for row in rows:
        _require(isinstance(row, dict), "raw row is not an object")
        validate_raw_outcomes(row)
    campaign = payload["campaign_identity"]
    _require(
        all(
            type(campaign.get(key)) is int and campaign[key] == len(rows)
            for key in ("written", "total_jobs")
        ),
        "campaign counts differ from emitted rows",
    )
    matches = [
        entry
        for entry in payload["raw_artifacts"]
        if isinstance(entry, dict)
        and entry.get("kind") == "episodes_jsonl"
        and entry.get("path") == str(out_path)
    ]
    _require(len(matches) == 1, "primary seal lacks unique raw artifact")
    _require(
        matches[0].get("artifact_status") == "available"
        and matches[0].get("sha256") == hashlib.sha256(raw_bytes).hexdigest(),
        "primary raw digest differs from collected bytes",
    )
    ids = [row.get("episode_id") for row in rows]
    _require(
        all(isinstance(value, str) and bool(value) for value in ids) and len(set(ids)) == len(rows),
        "emitted episode identities are missing or duplicated",
    )
    links = payload.get("rows")
    _require(isinstance(links, list) and len(links) == len(rows), "primary row coverage differs")
    for index, (row, link) in enumerate(zip(rows, links, strict=True)):
        _require(
            isinstance(link, dict)
            and type(link.get("jsonl_line")) is int
            and link["jsonl_line"] == index
            and link.get("episode_id") == row["episode_id"]
            and link.get("raw_artifact") == str(out_path),
            "primary row link differs",
        )
    save_manifest(out_path, ids)
    _require(
        out_path.read_bytes() == raw_bytes and primary.read_bytes() == primary_bytes,
        "original producer bytes changed during capture",
    )


def install_capture(
    *, validate_raw_outcomes: Callable[[dict[str, Any]], Any]
) -> Callable[[], None]:
    """Wrap the selected runtime recorder; require the base-owned outcome reader.

    Returns:
        A function restoring the original recorder after a bounded campaign.
    """
    _require(callable(validate_raw_outcomes), "canonical outcome validator is required")
    runner = importlib.import_module("robot_sf.benchmark.map_runner.map_runner")
    original = runner._record_result_provenance_manifest
    _require(
        not getattr(original, "_receipt_capture_installed", False), "capture already installed"
    )

    @functools.wraps(original)
    def record(**kwargs):
        original(**kwargs)
        _capture(Path(kwargs["out_path"]), kwargs["summary"], validate_raw_outcomes)

    record.__dict__["_receipt_capture_installed"] = True
    runner.__dict__["_record_result_provenance_manifest"] = record

    def restore():
        _require(
            runner._record_result_provenance_manifest is record, "recorder changed after install"
        )
        runner.__dict__["_record_result_provenance_manifest"] = original

    return restore
