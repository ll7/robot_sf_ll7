"""Minimal DotMap implementation for vendored SocNavBench modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DotMap(dict):
    """Dictionary with attribute-style access used by SocNavBench."""

    def __init__(self, mapping: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        if mapping:
            for key, value in mapping.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
