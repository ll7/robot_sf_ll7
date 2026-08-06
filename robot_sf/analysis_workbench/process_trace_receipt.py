"""Injective strict-JSON receipts for process-trace source exports."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_sf.analysis_workbench.simulation_trace_export import SimulationTraceExport

SIMULATION_TRACE_RECEIPT_SCHEMA_VERSION = "simulation_trace_export_receipt.v1"
_NONFINITE_VALUES = frozenset({"nan", "inf", "-inf"})


def build_simulation_trace_receipt(trace: SimulationTraceExport) -> dict[str, Any]:
    """Encode a source export as an injective strict-JSON receipt.

    Returns:
        Receipt envelope with strict content and a sorted nonfinite-value ledger.
    """

    raw_contract = {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": trace.trace_id,
        "source": {
            "scenario_id": trace.source.scenario_id,
            "seed": trace.source.seed,
            "planner_id": trace.source.planner_id,
            "episode_id": trace.source.episode_id,
            "generated_by": trace.source.generated_by,
        },
        "evidence_boundary": trace.evidence_boundary,
        "coordinate_frame": trace.coordinate_frame,
        "units": trace.units,
        "frames": [
            {
                "step": frame.step,
                "time_s": frame.time_s,
                "robot": frame.robot,
                "pedestrians": list(frame.pedestrians),
                "planner": frame.planner,
            }
            for frame in trace.frames
        ],
    }
    ledger: list[dict[str, str]] = []
    content_contract = _strict_json_value(raw_contract, path="", ledger=ledger)
    ledger.sort(key=lambda item: item["path"])
    return {
        "schema_version": SIMULATION_TRACE_RECEIPT_SCHEMA_VERSION,
        "content_contract": content_contract,
        "nonfinite_numbers": ledger,
    }


def simulation_trace_receipt_sha256(receipt: object) -> str:
    """Return the canonical internal digest of a strict source receipt."""

    _assert_exact_json_value(receipt)
    encoded = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decode_simulation_trace_receipt(receipt: object) -> dict[str, Any]:  # noqa: C901
    """Decode only nonfinite values explicitly declared by a valid receipt ledger.

    Returns:
        Reconstructed simulation-trace mapping for strict source validation.
    """

    _assert_exact_json_value(receipt)
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version",
        "content_contract",
        "nonfinite_numbers",
    }:
        raise ValueError("invalid simulation trace receipt envelope")
    if receipt.get("schema_version") != SIMULATION_TRACE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("invalid simulation trace receipt schema version")
    contract = receipt.get("content_contract")
    ledger = receipt.get("nonfinite_numbers")
    if not isinstance(contract, Mapping) or not isinstance(ledger, list):
        raise ValueError("invalid simulation trace receipt content or ledger")
    restored = deepcopy(dict(contract))
    parsed_entries: list[tuple[tuple[str, ...], str, str]] = []
    for entry in ledger:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "value"}:
            raise ValueError("invalid nonfinite ledger entry")
        path = entry.get("path")
        value = entry.get("value")
        if (
            not isinstance(path, str)
            or not isinstance(value, str)
            or value not in _NONFINITE_VALUES
        ):
            raise ValueError("invalid nonfinite ledger path or value")
        parsed_entries.append((_parse_json_pointer(path), path, value))
    paths = [path for _, path, _ in parsed_entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("nonfinite ledger paths must be sorted and unique")
    token_paths = [tokens for tokens, _, _ in parsed_entries]
    for index, tokens in enumerate(token_paths):
        for other in token_paths[index + 1 :]:
            if len(tokens) < len(other) and other[: len(tokens)] == tokens:
                raise ValueError("nonfinite ledger paths cannot overlap by prefix")
    for tokens, _path, value in parsed_entries:
        parent, key = _resolve_pointer_parent(restored, tokens)
        target = parent[key]  # type: ignore[index]
        if target is not None:
            raise ValueError("nonfinite ledger target must be null")
        parent[key] = _nonfinite_float(value)  # type: ignore[index]
    return restored


def _strict_json_value(
    value: Any,
    *,
    path: str,
    ledger: list[dict[str, str]],
) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        ledger.append({"path": path, "value": _nonfinite_label(value)})
        return None
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"receipt mapping key at {path or '/'} must be a string")
            result[key] = _strict_json_value(
                item,
                path=f"{path}/{_escape_json_pointer_token(key)}",
                ledger=ledger,
            )
        return result
    if type(value) is list:
        return [
            _strict_json_value(item, path=f"{path}/{index}", ledger=ledger)
            for index, item in enumerate(value)
        ]
    raise TypeError(f"unsupported receipt value at {path or '/'}: {type(value).__name__}")


def _assert_exact_json_value(value: object, *, path: str = "") -> None:
    if value is None or type(value) in {bool, str, int}:
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise TypeError(f"nonfinite JSON number at {path or '/'}")
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_exact_json_value(item, path=f"{path}/{index}")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"non-string JSON key at {path or '/'}")
            _assert_exact_json_value(item, path=f"{path}/{_escape_json_pointer_token(key)}")
        return
    raise TypeError(f"non-JSON receipt value at {path or '/'}: {type(value).__name__}")


def _escape_json_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _parse_json_pointer(path: str) -> tuple[str, ...]:
    if not path.startswith("/"):
        raise ValueError("nonfinite ledger path must be a non-root JSON pointer")
    raw_tokens = path[1:].split("/")
    tokens: list[str] = []
    for raw_token in raw_tokens:
        decoded: list[str] = []
        index = 0
        while index < len(raw_token):
            if raw_token[index] != "~":
                decoded.append(raw_token[index])
                index += 1
                continue
            if index + 1 >= len(raw_token) or raw_token[index + 1] not in {"0", "1"}:
                raise ValueError("invalid RFC6901 escape in nonfinite ledger path")
            decoded.append("~" if raw_token[index + 1] == "0" else "/")
            index += 2
        tokens.append("".join(decoded))
    return tuple(tokens)


def _resolve_pointer_parent(
    document: dict[str, Any],
    tokens: tuple[str, ...],
) -> tuple[dict[str, Any] | list[Any], str | int]:
    if not tokens:
        raise ValueError("nonfinite ledger cannot target the document root")
    cursor: object = document
    for token in tokens[:-1]:
        cursor = _pointer_child(cursor, token)
    final = tokens[-1]
    if isinstance(cursor, dict):
        if final not in cursor:
            raise ValueError("nonfinite ledger path does not exist")
        return cursor, final
    if isinstance(cursor, list):
        index = _canonical_array_index(final)
        if index >= len(cursor):
            raise ValueError("nonfinite ledger array index is out of range")
        return cursor, index
    raise ValueError("nonfinite ledger traverses a scalar")


def _pointer_child(value: object, token: str) -> object:
    if isinstance(value, dict):
        if token not in value:
            raise ValueError("nonfinite ledger path does not exist")
        return value[token]
    if isinstance(value, list):
        index = _canonical_array_index(token)
        if index >= len(value):
            raise ValueError("nonfinite ledger array index is out of range")
        return value[index]
    raise ValueError("nonfinite ledger traverses a scalar")


def _canonical_array_index(token: str) -> int:
    if not token.isascii() or not token.isdigit() or (len(token) > 1 and token.startswith("0")):
        raise ValueError("nonfinite ledger array index must be canonical")
    return int(token)


def _nonfinite_label(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _nonfinite_float(value: str) -> float:
    if value == "nan":
        return math.nan
    if value == "inf":
        return math.inf
    return -math.inf
