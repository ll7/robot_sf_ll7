"""Lazy proxy for constructing ``SimulationView`` only when rendering is used."""

from __future__ import annotations

import importlib
from typing import Any


class LazySimulationView:
    """Defer pygame-backed ``SimulationView`` construction until first visual use."""

    _DEFAULT_WIDTH = 1280
    _DEFAULT_HEIGHT = 720

    def __init__(self, **view_kwargs: Any) -> None:
        """Store view construction arguments without importing pygame."""
        object.__setattr__(self, "_view_kwargs", dict(view_kwargs))
        object.__setattr__(self, "_pending_attrs", {})
        object.__setattr__(self, "_view", None)
        object.__setattr__(self, "record_video", bool(view_kwargs.get("record_video", False)))
        object.__setattr__(self, "video_path", view_kwargs.get("video_path"))
        object.__setattr__(self, "video_fps", view_kwargs.get("video_fps"))
        object.__setattr__(self, "width", view_kwargs.get("width", self._DEFAULT_WIDTH))
        object.__setattr__(self, "height", view_kwargs.get("height", self._DEFAULT_HEIGHT))

    @property
    def materialized(self) -> bool:
        """Return whether the underlying ``SimulationView`` has been created."""
        return self._view is not None

    def _ensure_view(self) -> Any:
        """Create and return the underlying pygame-backed view.

        Returns:
            Any: The materialized ``SimulationView`` instance.
        """
        view = self._view
        if view is not None:
            return view

        sim_view_module = importlib.import_module("robot_sf.render.sim_view")
        view = sim_view_module.SimulationView(**self._view_kwargs)
        for name, value in self._pending_attrs.items():
            setattr(view, name, value)
        object.__setattr__(self, "_pending_attrs", {})
        object.__setattr__(self, "_view", view)
        return view

    def __bool__(self) -> bool:
        """A configured lazy view should behave like a present ``sim_ui`` handle.

        Returns:
            bool: Always ``True`` for configured lazy views.
        """
        return True

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the materialized view.

        Returns:
            Any: Attribute value from the materialized view.
        """
        return getattr(self._ensure_view(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Store pre-materialization mutations and replay them on first visual use."""
        if name.startswith("_") or name in {
            "record_video",
            "video_path",
            "video_fps",
            "width",
            "height",
        }:
            object.__setattr__(self, name, value)
            return
        view = self._view
        if view is None:
            self._pending_attrs[name] = value
            return
        setattr(view, name, value)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Materialize the view and render a frame.

        Returns:
            Any: Result returned by ``SimulationView.render``.
        """
        return self._ensure_view().render(*args, **kwargs)

    def exit_simulation(self, *args: Any, **kwargs: Any) -> Any:
        """Close the materialized view, or no-op if rendering never started.

        Returns:
            Any: Result returned by ``SimulationView.exit_simulation`` when materialized.
        """
        if self._view is None:
            return None
        return self._view.exit_simulation(*args, **kwargs)
