"""Durable atomic JSON writes shared across the benchmark package.

Previously this ``mkstemp -> fsync -> os.replace`` helper was copy-pasted in
``benchmark/imitation_manifest.py`` and ``benchmark/full_classic/io_utils.py``
(see issue #3386). Centralizing it keeps the durable-write contract in one place.

The write is atomic: data is fully written and fsync'd to a temporary file in the
destination directory, then ``os.replace`` swaps it into place in a single
filesystem operation, so a reader never observes a partially written file.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write ``payload`` as pretty-printed, key-sorted JSON to ``path``.

    The parent directory is created if needed. The payload is flushed and fsync'd
    to a temporary file in the same directory before an atomic ``os.replace``, and
    the temporary file is cleaned up if anything fails before the replace.

    Args:
        path: Destination file path.
        payload: JSON-serializable mapping to write.
    """
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    owns_tmp_fd = True
    try:
        try:
            handle = os.fdopen(tmp_fd, "w", encoding="utf-8")
        except (OSError, ValueError):
            try:
                os.close(tmp_fd)
            except OSError:  # pragma: no cover - defensive cleanup
                pass
            owns_tmp_fd = False
            raise
        owns_tmp_fd = False
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if owns_tmp_fd:
            try:
                os.close(tmp_fd)
            except OSError:  # pragma: no cover - defensive cleanup
                pass
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:  # pragma: no cover - defensive cleanup
                pass


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text to ``path``.

    The temporary file is created beside the destination so ``os.replace`` is
    atomic on the same filesystem.  The content is flushed and fsync'd before
    replacement, and failed writes clean up their temporary file.
    """
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    owns_tmp_fd = True
    try:
        try:
            handle = os.fdopen(tmp_fd, "w", encoding="utf-8")
        except (OSError, ValueError):
            try:
                os.close(tmp_fd)
            except OSError:  # pragma: no cover - defensive cleanup
                pass
            owns_tmp_fd = False
            raise
        owns_tmp_fd = False
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if owns_tmp_fd:
            try:
                os.close(tmp_fd)
            except OSError:  # pragma: no cover - defensive cleanup
                pass
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:  # pragma: no cover - defensive cleanup
                pass
