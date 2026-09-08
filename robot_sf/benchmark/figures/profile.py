"""Strict, portable rendering profiles for publication figure producers.

A profile records target dimensions and typography independently from a renderer's
scientific inputs. Generated bundles hash and copy the normalized profile so a
consumer can distinguish byte transport, source admission, and presentation intent.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA = "robot-sf-figure-profile.v1"
_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _number(value: Any) -> bool:
    """Return whether *value* is a finite real number but not a boolean."""
    return type(value) in (int, float) and math.isfinite(value)


def _object(path: Path) -> dict[str, Any]:
    """Read one strict JSON object, rejecting duplicate keys and non-finite values."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"non-finite JSON literal: {value}")

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid,
    )
    if not isinstance(value, dict):
        raise ValueError("figure profile must be a JSON object")
    return value


@dataclass(frozen=True)
class FigureProfile:
    """Validated final-size and typography contract for one figure family."""

    profile_id: str
    target_width_in: float
    height_ratio: float
    font_family: tuple[str, ...]
    language: Literal["en", "de"] = "en"
    font_size_pt: float = 10.0
    axes_label_size_pt: float = 10.0
    axes_title_size_pt: float = 10.0
    legend_size_pt: float = 8.5
    tick_label_size_pt: float = 9.0
    annotation_size_pt: float = 8.0
    line_width_pt: float = 1.5
    marker_size_pt: float = 4.0
    dpi: int = 300

    def __post_init__(self) -> None:
        """Fail closed on ambiguous IDs, implausible dimensions, and invalid typography."""
        if not isinstance(self.profile_id, str) or _PROFILE_ID.fullmatch(self.profile_id) is None:
            raise ValueError("profile_id must be a stable ASCII identifier")
        if self.language not in {"en", "de"}:
            raise ValueError("language must be en or de")
        if not _number(self.target_width_in) or not 1.0 <= self.target_width_in <= 20.0:
            raise ValueError("target_width_in must be finite and in [1, 20]")
        if not _number(self.height_ratio) or not 0.3 <= self.height_ratio <= 2.5:
            raise ValueError("height_ratio must be finite and in [0.3, 2.5]")
        if (
            not isinstance(self.font_family, tuple)
            or not 1 <= len(self.font_family) <= 8
            or any(
                not isinstance(name, str)
                or not name.strip()
                or len(name) > 128
                or any(ord(char) < 32 for char in name)
                for name in self.font_family
            )
        ):
            raise ValueError("font_family must contain 1-8 nonempty font family names")
        for name in (
            "font_size_pt",
            "axes_label_size_pt",
            "axes_title_size_pt",
            "legend_size_pt",
            "tick_label_size_pt",
            "annotation_size_pt",
        ):
            value = getattr(self, name)
            if not _number(value) or not 4.0 <= value <= 36.0:
                raise ValueError(f"{name} must be finite and in [4, 36]")
        if not _number(self.line_width_pt) or not 0.1 <= self.line_width_pt <= 10.0:
            raise ValueError("line_width_pt must be finite and in [0.1, 10]")
        if not _number(self.marker_size_pt) or not 0.1 <= self.marker_size_pt <= 30.0:
            raise ValueError("marker_size_pt must be finite and in [0.1, 30]")
        if type(self.dpi) is not int or not 72 <= self.dpi <= 1200:
            raise ValueError("dpi must be an integer in [72, 1200]")

    @classmethod
    def from_file(cls, path: Path) -> FigureProfile:
        """Load a profile without silently accepting misspelled or future fields."""
        values = _object(path)
        if values.pop("schema_version", None) != SCHEMA:
            raise ValueError(f"figure profile schema_version must be {SCHEMA}")
        if set(values) - set(cls.__dataclass_fields__):
            raise ValueError("unknown figure profile fields")
        family = values.get("font_family")
        if not isinstance(family, list):
            raise ValueError("font_family must be a JSON array")
        values["font_family"] = tuple(family)
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("figure profile fields are incomplete or have invalid names") from exc

    @classmethod
    def builtin(cls, size: Literal["single", "double"]) -> FigureProfile:
        """Preserve the legacy scenario-pack dimensions when no external profile is supplied."""
        if size == "single":
            width, height = 3.4, 6.2
        elif size == "double":
            width, height = 7.0, 5.6
        else:
            raise ValueError("size must be single or double")
        return cls(
            profile_id=f"builtin-{size}",
            target_width_in=width,
            height_ratio=height / width,
            font_family=("Times New Roman", "DejaVu Serif", "serif"),
            axes_label_size_pt=11.0,
            legend_size_pt=9.0,
        )

    def payload(self) -> dict[str, Any]:
        """Return the portable normalized profile payload."""
        payload = asdict(self)
        payload["font_family"] = list(self.font_family)
        return {"schema_version": SCHEMA, **payload}

    def canonical_json(self) -> str:
        """Serialize the profile deterministically for copying and hashing."""
        return json.dumps(self.payload(), sort_keys=True, indent=2, allow_nan=False) + "\n"

    def sha256(self) -> str:
        """Return the digest of the normalized profile, not its local source path."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def figure_size(self) -> tuple[float, float]:
        """Return exact final-size dimensions in inches."""
        return self.target_width_in, self.target_width_in * self.height_ratio

    def rc_params(self) -> dict[str, Any]:
        """Return Matplotlib overrides that implement the declared profile."""
        return {
            "font.family": "serif",
            "font.serif": list(self.font_family),
            "font.size": self.font_size_pt,
            "axes.labelsize": self.axes_label_size_pt,
            "axes.titlesize": self.axes_title_size_pt,
            "legend.fontsize": self.legend_size_pt,
            "xtick.labelsize": self.tick_label_size_pt,
            "ytick.labelsize": self.tick_label_size_pt,
            "figure.figsize": self.figure_size(),
            "lines.linewidth": self.line_width_pt,
            "lines.markersize": self.marker_size_pt,
            "savefig.dpi": self.dpi,
        }

    def resolve_font_family(self) -> str:
        """Return the actual family Matplotlib resolves without exposing a machine-local path."""
        font_manager = importlib.import_module("matplotlib.font_manager")
        properties = font_manager.FontProperties(family=list(self.font_family))
        path = font_manager.findfont(properties, fallback_to_default=True)
        return str(font_manager.FontProperties(fname=path).get_name())
