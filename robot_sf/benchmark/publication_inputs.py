"""Strict content parsing and provenance reconciliation for future publications.

These checks are publication tooling. They do not change the simulator or scorer.
No historical scoring version exempts a bundle from input custody.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

from robot_sf.benchmark.snqi.v2_spec import parse_v2_json, parse_v2_yaml

_ROLES = ("weights", "anchors", "family")
_DIAGNOSTICS_SCHEMA = "snqi-v2-diagnostics.v1"
_METHOD_PINS = {
    "metric_definitions_sha256",
    "snqi_v2_evaluation_seeds_sha256",
    "snqi_v2_calibration_split_id",
    "snqi_v2_force_source_contract",
    "snqi_v2_definitions_binding",
}
_BOMS = (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff")
_SCORING_FIELD = re.compile(r"\b(?:snqi_v2_[\w]+|force_source|force_decision)\b", re.I)
_CSV_FIELD = re.compile(r"[A-Za-z_][\w.-]*\Z")


def publication_values(value: Any) -> Iterator[Any]:
    """Yield every nested value of an already validated document."""
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from publication_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from publication_values(item)


def _validate_tree(value: Any, ancestors: frozenset[int] = frozenset()) -> None:  # noqa: C901
    """Refuse non-JSON containers, recursive aliases and ambiguous object keys."""
    if isinstance(value, (dict, list)):
        if id(value) in ancestors:
            raise ValueError("recursive publication container")
        ancestors = ancestors | {id(value)}
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or "\ufeff" in key:
                raise ValueError("publication object keys must be strings without a BOM")
            _validate_tree(item, ancestors)
    elif isinstance(value, list):
        for item in value:
            _validate_tree(item, ancestors)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("publication value is not a finite number")
    elif value is not None and not isinstance(value, (str, int, bool, date)):
        raise ValueError("unknown publication container or scalar type")


def _object(value: Any) -> dict[str, Any]:
    """Require object roots and the declared types of scoring containers.

    Returns:
        The validated mapping.
    """
    if not isinstance(value, dict):
        raise ValueError(
            "expected a JSON object or YAML mapping; list-wrapped/scalar roots are unsupported"
        )
    _validate_tree(value)
    for item in publication_values(value):
        if isinstance(item, dict):
            _object_containers(item)
    return value


def _object_containers(item: dict[str, Any]) -> None:
    """Validate object/nullable field schemas independently of the file grammar."""
    if "schema_version" in item and (
        isinstance(item["schema_version"], bool)
        or not isinstance(item["schema_version"], (str, int))
    ):
        raise ValueError("schema_version must be a scalar identifier")
    for key in (
        "provenance",
        "anchors",
        "force_decision",
        "calibration",
        "weights",
        "snqi_v2_terms",
        "snqi_v2_force_source_contract",
    ):
        if key in item and not isinstance(item[key], dict):
            raise ValueError(f"{key} must be an object")
    if "event_ledger" in item and not isinstance(item["event_ledger"], dict):
        raise ValueError("episode row is missing object event_ledger")
    if "metrics" in item:
        metrics = item["metrics"]
        # Data summaries may have nullable tables or rows of metric descriptors;
        # scoring records have an explicit, stricter container schema.
        schema = str(item.get("schema_version", ""))
        episode = "event_ledger" in item or "episode_id" in item
        campaign = schema.startswith("benchmark-camera-ready-campaign")
        diagnostic = schema.startswith("snqi-v2-diagnostics")
        names = isinstance(metrics, list) and all(isinstance(name, str) for name in metrics)
        if not isinstance(metrics, dict) and (episode or diagnostic or (campaign and not names)):
            raise ValueError("metrics must be an object or a declared metric-name list")


def _without_bom(raw: str | bytes) -> str:
    """Decode UTF-8 without permitting an encoding marker to conceal a document.

    Returns:
        UTF-8 text without a BOM.
    """
    if isinstance(raw, bytes):
        if raw.lstrip().startswith(_BOMS):
            raise ValueError("publication BOM is forbidden")
        raw = raw.decode("utf-8")
    if raw.lstrip().startswith("\ufeff"):
        raise ValueError("publication BOM is forbidden")
    return raw


def parse_publication_json(raw: str | bytes) -> dict[str, Any]:
    """Parse a strict JSON object, preserving the canonical duplicate-key refusal.

    Returns:
        The validated object.
    """
    return _object(parse_v2_json(_without_bom(raw)))


def _csv_records(text: str) -> Iterator[dict[str, Any]]:
    """Parse CSV without last-column-wins or truncated/overfull rows."""
    rows = csv.reader(io.StringIO(text), strict=True)
    header = [name.strip() for name in next(rows)]
    if not header or any(not name for name in header) or len(set(header)) != len(header):
        raise ValueError("duplicate or empty CSV headers")
    for row in rows:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError("CSV row width differs from its header")
        # Flat provenance columns are the same declarations as JSON/YAML fields.
        document: dict[str, Any] = {}
        for name, value in zip(header, row, strict=True):
            container = document
            parts = name.split(".")
            for part in parts[:-1]:
                existing = container.setdefault(part, {})
                if not isinstance(existing, dict):
                    raise ValueError("conflicting CSV column containers")
                container = existing
            if parts[-1] in container:
                raise ValueError("duplicate CSV column declaration")
            container[parts[-1]] = value
        yield _object(document)


def _records(raw: bytes, *, required: bool) -> Iterator[dict[str, Any]]:  # noqa: C901
    """Select a supported content grammar, never using the file extension."""
    text = _without_bom(raw)
    stripped = text.lstrip()
    if not stripped:
        if required:
            raise ValueError("empty publication input")
        return
    if (
        stripped[0] in '{["'
        or stripped in {"null", "true", "false", "NaN", "Infinity", "-Infinity"}
        or re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", stripped)
    ):
        try:
            decoded = parse_v2_json(text)
        except ValueError as exc:
            # A sequence of complete JSON object records is JSONL. No fallback
            # to another grammar can turn malformed JSON into accepted content.
            if getattr(exc, "msg", None) != "Extra data":
                raise
            for line in text.splitlines():
                if line.strip():
                    yield parse_publication_json(line)
        else:
            yield _object(decoded)
        return
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    header = next(csv.reader([first_line], strict=True))
    csv_shape = len(header) > 1 and all(
        not name.strip() or _CSV_FIELD.fullmatch(name.strip()) for name in header
    )
    if csv_shape or (
        len(header) == 1 and _SCORING_FIELD.fullmatch(header[0].strip().split(".")[-1])
    ):
        yield from _csv_records(text)
        return
    decoded = parse_v2_yaml(text)
    if isinstance(decoded, (dict, list)) or required:
        yield _object(decoded)
    elif decoded == "pending_calibration" or _SCORING_FIELD.search(text):
        raise ValueError("unknown scoring content schema; expected an object or CSV records")


def publication_documents(
    root: Path, *, required: bool = False
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield every structured object/record, independent of basename or extension.

    Opaque binary figures/media have no structured provenance schema. A declared
    input must have a supported object/record grammar instead.
    """
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        if not path.is_file():
            continue
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("diagnostic input refuses symlink metadata")
        raw = path.read_bytes()
        try:
            yield from ((path, document) for document in _records(raw, required=required))
        except UnicodeDecodeError as exc:
            opaque = (
                raw.startswith(
                    (
                        b"%PDF-",
                        b"\x89PNG\r\n\x1a\n",
                        b"\xff\xd8\xff",
                        b"GIF87a",
                        b"GIF89a",
                        b"PK\x03\x04",
                        b"\x1f\x8b",
                    )
                )
                or raw[4:8] == b"ftyp"
            )
            if (
                required
                or not opaque
                or _SCORING_FIELD.search(raw.decode("utf-8", errors="ignore"))
            ):
                raise ValueError(f"diagnostic input invalid UTF-8 content: {path.name}") from exc
        except yaml.YAMLError as exc:
            # Human report text is opaque; machine-shaped/scoring text cannot
            # opt out of parsing by introducing a syntax error.
            text = raw.decode("utf-8")
            if (
                required
                or _SCORING_FIELD.search(text)
                or any(ord(character) < 32 and character not in "\n\r\t" for character in text)
                or re.search(r"(?m)^\s*[\w.-]+\s*:", text)
            ):
                raise ValueError(f"diagnostic input invalid YAML content: {path.name}") from exc
        except (ValueError, TypeError, csv.Error) as exc:
            raise ValueError(
                f"diagnostic input invalid structured content: {path.name}: {exc}"
            ) from exc


def read_publication_object(path: Path) -> dict[str, Any]:
    """Read exactly one strict object from a declared metadata/input file.

    Returns:
        The single validated object.
    """
    records = publication_documents(path, required=True)
    try:
        _, document = next(records)
    except StopIteration as exc:
        raise ValueError("diagnostic input requires a publication object") from exc
    if next(records, None) is not None:
        raise ValueError("diagnostic input expected one object, not a record sequence")
    return document


def _input_path(raw: Any) -> Path:
    """Validate a source-relative custody path.

    Returns:
        The safe relative path.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("diagnostic input path must be a nonempty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or "\\" in raw or path == Path("."):
        raise ValueError("diagnostic input path must stay within the source root")
    return path


def _input_digest(raw: Any) -> str:
    """Validate a custody digest without forgiving a missing/wrongly typed pin.

    Returns:
        The normalized SHA-256 value.
    """
    if not isinstance(raw, str) or re.fullmatch(r"[0-9a-fA-F]{64}", raw) is None:
        raise ValueError("diagnostic input requires a valid SHA-256 digest")
    return raw.lower()


def _contains_v2_name(value: Any) -> bool:
    """Recognize the explicit v2 name in declared metric lists.

    Returns:
        Whether a v2 name is declared.
    """
    return any(
        isinstance(item, str) and re.search(r"(?:^|[/._ -])snqi[-_]v2(?:$|[/._ -])", item, re.I)
        for item in publication_values(value)
    )


def _campaign_metric_lists(value: dict[str, Any], metrics: Any) -> None:
    """Refuse malformed or contradictory declared scoring metric lists."""
    for declaration in (value, metrics):
        if not isinstance(declaration, dict):
            continue
        for key in (
            "metrics",
            "metric_list",
            "metric_names",
            "required_metrics",
            "scored_metrics",
        ):
            names = declaration.get(key)
            if key not in declaration or isinstance(names, dict):
                continue
            if (
                not isinstance(names, list)
                or not names
                or not all(isinstance(name, str) for name in names)
            ):
                raise ValueError(
                    "diagnostic input v2 campaign requires a valid declared metric list"
                )
            if not _contains_v2_name(names):
                raise ValueError("diagnostic input v2 campaign has a conflicting metric list")


@dataclass
class PublicationInputCustody:
    """One accumulator for all document, retained-input and episode declarations."""

    references: dict[Path, str] = field(default_factory=dict)
    role_pins: dict[str, Path | str] = field(default_factory=dict)
    anchors_candidates: set[Path] = field(default_factory=set)
    weights_candidates: set[Path] = field(default_factory=set)
    force_sources: set[str] = field(default_factory=set)
    versions: set[str] = field(default_factory=set)
    method_pins: dict[str, Any] = field(default_factory=dict)
    metric_schema_version: str | None = None
    horizon_path: Path | None = None
    horizons: dict[str, int] | None = None
    diagnostics_seen: bool = False

    @property
    def v2_assets(self) -> dict[str, Path]:
        """Return the reconciled role paths after mandatory-custody validation."""
        assets = {}
        for role in _ROLES:
            path = self.role_pins.get(f"snqi_v2_{role}_path")
            if not isinstance(path, Path):
                raise ValueError(f"diagnostic input requires a v2 {role} path")
            assets[role] = path
        return assets

    def add_reference(self, raw_path: Any, raw_digest: Any) -> Path:
        """Register one generic path/digest pin without replacing conflicting bytes.

        Returns:
            The registered relative path.
        """
        path, digest = _input_path(raw_path), _input_digest(raw_digest)
        if path in self.references and self.references[path] != digest:
            raise ValueError("diagnostic input has contradictory digests")
        self.references[path] = digest
        return path

    def _asset_pins(self, value: dict[str, Any]) -> None:
        for role in _ROLES:
            for suffix in ("path", "sha256"):
                key = f"snqi_v2_{role}_{suffix}"
                if key not in value:
                    continue
                pin = _input_path(value[key]) if suffix == "path" else _input_digest(value[key])
                if key in self.role_pins and self.role_pins[key] != pin:
                    raise ValueError(f"diagnostic input has conflicting v2 {role} pins")
                self.role_pins[key] = pin
            path = self.role_pins.get(f"snqi_v2_{role}_path")
            digest = self.role_pins.get(f"snqi_v2_{role}_sha256")
            if path is not None and digest is not None:
                self.add_reference(str(path), digest)

    def _scoring_copies(self, value: dict[str, Any]) -> None:  # noqa: C901
        context = (
            str(value.get("schema_version", "")).startswith("snqi-v2-diagnostics")
            or "anchors" in value
            or "force_decision" in value
            or any(key.startswith("snqi_v2_") for key in value)
        )
        for key in ("snqi_v2_version", "snqi_version", "snqi_score_version", "version"):
            if key not in value:
                continue
            version = value[key]
            if (
                key == "version"
                and not context
                and not (isinstance(version, str) and version.startswith("SNQI-"))
            ):
                continue
            if not isinstance(version, str):
                raise ValueError("diagnostic input has invalid scoring version")
            self.versions.add("SNQI-v2" if version == "SNQI-v2.0" else version)
        if self.versions and self.versions != {"SNQI-v2"}:
            raise ValueError("diagnostic input has conflicting v2 version copies")
        sources = [value[key] for key in ("snqi_v2_force_source", "force_source") if key in value]
        if isinstance(value.get("force_decision"), dict) and "source" in value["force_decision"]:
            sources.append(value["force_decision"]["source"])
        for source in sources:
            if not isinstance(source, str):
                raise ValueError("diagnostic input has invalid force source")
            self.force_sources.add(source)
        if len(self.force_sources) > 1:
            raise ValueError("diagnostic input has conflicting force source copies")
        for key in _METHOD_PINS.intersection(value):
            pin = value[key]
            if key.endswith("_sha256"):
                pin = _input_digest(pin)
            if key in {"snqi_v2_calibration_split_id", "snqi_v2_definitions_binding"} and (
                not isinstance(pin, str) or not pin
            ):
                raise ValueError("diagnostic input has invalid scoring method pin")
            if key in self.method_pins and self.method_pins[key] != pin:
                raise ValueError(f"diagnostic input has conflicting scoring provenance pin: {key}")
            self.method_pins[key] = pin
        schema = value.get("metric_schema_version")
        if schema is not None:
            if not isinstance(schema, str) or self.metric_schema_version not in (None, schema):
                raise ValueError("diagnostic input has conflicting v2 metric schema declarations")
            self.metric_schema_version = schema

    def _diagnostics(self, value: dict[str, Any]) -> None:
        schema = value.get("schema_version")
        if not isinstance(schema, str) or not schema.startswith("snqi-v2-diagnostics"):
            return
        if schema != _DIAGNOSTICS_SCHEMA:
            raise ValueError("diagnostic input has unknown diagnostics schema")
        provenance = value.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("diagnostic input v2 diagnostics require asset provenance object")
        if type(value.get("episode_count")) is not int or value["episode_count"] < 0:
            raise ValueError("diagnostic input episode_count must be a nonnegative integer")
        self._asset_pins(provenance)
        for role in _ROLES:
            self.add_reference(
                provenance.get(f"snqi_v2_{role}_path"), provenance.get(f"snqi_v2_{role}_sha256")
            )
        # Every declared diagnostic input needs a digest, including extra roles.
        for item in publication_values(value):
            if isinstance(item, dict):
                for key, raw_path in item.items():
                    if key.endswith(("_path", "_file")):
                        self.add_reference(raw_path, item.get(key[:-5] + "_sha256"))
        self.diagnostics_seen = True

    def campaign(self, value: dict[str, Any]) -> None:
        """Require explicit campaign scoring provenance and evaluation schedule."""
        metrics = value.get("metrics")
        declarations = [
            item
            for item in (value, metrics)
            if isinstance(item, dict) and any(key.startswith("snqi_v2_") for key in item)
        ]
        if not declarations or any(
            item.get("snqi_v2_version") != "SNQI-v2" for item in declarations
        ):
            raise ValueError(
                "diagnostic input v2 campaign requires an explicit consistent SNQI-v2 version"
            )
        for declaration in declarations:
            for role in _ROLES:
                self.add_reference(
                    declaration.get(f"snqi_v2_{role}_path"),
                    declaration.get(f"snqi_v2_{role}_sha256"),
                )
            self._asset_pins(declaration)
        if "snqi_v2" in value and value["snqi_v2"] != "SNQI-v2":
            raise ValueError("diagnostic input v2 campaign has a conflicting scoring state")
        _campaign_metric_lists(value, metrics)
        schedule = value.get("scenario_horizons")
        if not isinstance(schedule, dict) or not schedule:
            raise ValueError("diagnostic input v2 campaign requires its evaluation schedule")
        if "path" in schedule:
            path = self.add_reference(schedule["path"], schedule.get("sha256"))
            if self.horizon_path not in (None, path):
                raise ValueError("diagnostic input has conflicting campaign schedule pins")
            self.horizon_path = path
        else:
            if not all(
                isinstance(name, str) and type(horizon) is int and horizon > 0
                for name, horizon in schedule.items()
            ):
                raise ValueError(
                    "diagnostic input campaign schedule requires positive integer horizons"
                )
            if self.horizons not in (None, schedule):
                raise ValueError("diagnostic input has conflicting campaign schedules")
            self.horizons = schedule

    def observe(self, path: Path, document: dict[str, Any]) -> None:
        """Reconcile every nested declaration, including every episode provenance pin."""
        schema = document.get("schema_version", "")
        if isinstance(schema, str) and schema.startswith("benchmark-camera-ready-campaign"):
            self.campaign(document)
        for value in publication_values(document):
            if isinstance(value, str) and value == "pending_calibration":
                raise ValueError(
                    "diagnostic input pending_calibration refuses publication; calibration anchors are not frozen"
                )
            if not isinstance(value, dict):
                continue
            self._scoring_copies(value)
            self._asset_pins(value)
            self._diagnostics(value)
            if "anchors" in value or "force_decision" in value:
                self.anchors_candidates.add(path)
            if value.get("version") == "SNQI-v2.0" and isinstance(value.get("weights"), dict):
                self.weights_candidates.add(path)
            if "path" in value and "sha256" in value:
                self.add_reference(value["path"], value["sha256"])
            for key, raw_path in value.items():
                if key.endswith(("_path", "_file")) and key[:-5] + "_sha256" in value:
                    self.add_reference(raw_path, value[key[:-5] + "_sha256"])

    def require_custody(self) -> None:
        """Refuse every publication lacking diagnostics and the three scoring input pins."""
        if not self.diagnostics_seen:
            raise ValueError(
                "diagnostic input v2 campaign requires diagnostics and complete custody; no historical exemption"
            )
        for role in _ROLES:
            self.add_reference(
                str(self.v2_assets[role]), self.role_pins.get(f"snqi_v2_{role}_sha256")
            )
