"""TensorBoard mirroring helpers for telemetry snapshots."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.telemetry.models import TelemetrySnapshot

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Iterable, Iterator

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except ImportError:  # pragma: no cover - optional dependency fallback
    try:
        from tensorboardX import SummaryWriter as _SummaryWriter  # type: ignore
    except ImportError:  # pragma: no cover - tensorboard unavailable
        _SummaryWriter = None  # type: ignore

__all__ = ["TensorBoardAdapter", "iter_telemetry_snapshots"]

_SNAPSHOT_FIELDS = tuple(field.name for field in fields(TelemetrySnapshot))


@dataclass(slots=True)
class TensorBoardAdapter:
    """Mirror telemetry snapshots into TensorBoard event files."""

    log_dir: Path
    tag_prefix: str = "telemetry"

    def __post_init__(self) -> None:
        """TODO docstring. Document this function."""
        self.log_dir = Path(self.log_dir)
        self._writer_cls = _SummaryWriter
        self._writer: _SummaryWriter | None = None
        self._samples = 0

    @property
    def is_available(self) -> bool:
        """Return ``True`` when a SummaryWriter backend is importable."""

        return self._writer_cls is not None

    def start(self) -> None:
        """Initialize the SummaryWriter if TensorBoard support exists."""

        if not self.is_available:
            raise RuntimeError(
                "TensorBoard SummaryWriter is unavailable; install torch or tensorboardX"
            )
        if self._writer is not None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = self._writer_cls(log_dir=str(self.log_dir))  # type: ignore[misc]
        logger.debug("TensorBoard adapter writing to {}", self.log_dir)

    def close(self) -> None:
        """Flush and close the underlying SummaryWriter."""

        if self._writer is None:
            return
        with contextlib.suppress(Exception):
            self._writer.flush()
            self._writer.close()
        self._writer = None

    def consume_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        """Send a telemetry snapshot to TensorBoard (no-op if unavailable)."""

        if not self.is_available:
            return
        if self._writer is None:
            self.start()
        assert self._writer is not None  # narrow type for static checkers
        self._samples += 1
        step = self._samples
        for tag, value in _iter_scalar_values(snapshot, prefix=self.tag_prefix):
            if value is not None:
                self._writer.add_scalar(tag, value, global_step=step)

    def mirror_file(self, telemetry_file: Path) -> int:
        """Stream a telemetry JSONL file into TensorBoard.

        Returns:
            int: The number of telemetry snapshots mirrored into TensorBoard.
        """

        count = 0
        for snapshot in iter_telemetry_snapshots(telemetry_file):
            self.consume_snapshot(snapshot)
            count += 1
        self.close()
        return count


def iter_telemetry_snapshots(path: Path) -> Iterator[TelemetrySnapshot]:
    """Yield :class:`TelemetrySnapshot` entries from a JSONL file."""

    if not path.exists():  # pragma: no cover - defensive guard
        return
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:  # pragma: no cover - defensive guard
                logger.warning("Skipping invalid telemetry JSON line: {}", line[:80])
                continue
            if not isinstance(payload, dict):
                continue
            snapshot_kwargs = {field: payload.get(field) for field in _SNAPSHOT_FIELDS}
            yield TelemetrySnapshot(**snapshot_kwargs)


def _iter_scalar_values(
    snapshot: TelemetrySnapshot,
    *,
    prefix: str,
) -> Iterable[tuple[str, float | None]]:
    """Yield TensorBoard tag/value pairs derived from a snapshot.

    Args:
        snapshot: Telemetry snapshot to serialize into scalar metrics.
        prefix: Tag prefix (e.g. run identifier) prepended to each emitted key.

    Yields:
        tuple[str, float | None]: ``(tag, value)`` pairs suitable for writer.add_scalar.
    """
    yield f"{prefix}/steps_per_sec", snapshot.steps_per_sec
    yield f"{prefix}/cpu_process_percent", snapshot.cpu_percent_process
    yield f"{prefix}/cpu_system_percent", snapshot.cpu_percent_system
    yield f"{prefix}/memory_rss_mb", snapshot.memory_rss_mb
    yield f"{prefix}/gpu_util_percent", snapshot.gpu_util_percent
    yield f"{prefix}/gpu_mem_used_mb", snapshot.gpu_mem_used_mb
