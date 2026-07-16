"""Regression tests for Loguru sink lifecycle safety under xdist teardown.

Issue #5849: benchmark workers emitted ``ValueError: I/O operation on closed
file`` during ``pytest-xdist`` teardown because test-owned logging sinks were
bound to a transient capture stream (the ``capsys``/``capfd``-swapped
``sys.stdout``/``sys.stderr``) that pytest closes at teardown. A late log
record flushed after the stream closed raised instead of being dropped.

These tests pin the smallest faithful lifecycle contract:

* A sink wrapped via :func:`robot_sf.common.logging.safe_sink` must swallow
  writes to an already-closed stream (the benign teardown case) instead of
  propagating ``ValueError``.
* The same wrapper must still propagate real ``OSError``/``ValueError`` from a
  *live* stream and must still deliver ordinary records, so unrelated logging
  failures are never hidden.
* A sink that binds to a captured ``sys.stdout`` (the exact ``capsys``
  ownership pattern) must survive teardown without raising, matching the real
  benchmark-worker diagnostic.
"""

from __future__ import annotations

import io

from loguru import logger

from robot_sf.common.logging import safe_sink


def test_safe_sink_drops_write_to_closed_stream() -> None:
    """A closed-stream write must not raise; the record is dropped (issue #5849)."""
    buf = io.StringIO()
    sink = safe_sink(buf)
    sink("before close\n")
    buf.close()  # Simulate pytest closing the capture stream at teardown.
    # Must not raise ValueError: I/O operation on closed file.
    sink("after close\n")


def test_unguarded_write_to_closed_stream_raises_original_diagnostic() -> None:
    """Negative control: the pre-fix pattern still raises the issue #5849 error.

    This pins the failure mode the regression is defending against, so the suite
    fails for the right reason if ``safe_sink`` ever stops guarding the closed
    stream. A bare ``io.StringIO`` write after ``close()`` raises exactly the
    ``ValueError: I/O operation on closed file`` the benchmark workers emitted.
    """
    buf = io.StringIO()
    buf.close()
    raised: Exception | None = None
    try:
        buf.write("late write\n")
    except ValueError as exc:
        raised = exc
    assert raised is not None
    assert "I/O operation on closed file" in str(raised)


def test_safe_sink_propagates_live_stream_writes() -> None:
    """Ordinary records are delivered and flushed like Loguru stream sinks."""

    class _FlushTrackingStream(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.flush_calls = 0

        def flush(self) -> None:
            self.flush_calls += 1
            super().flush()

    buf = _FlushTrackingStream()
    sink = safe_sink(buf)
    sink("alpha\n")
    sink("beta\n")
    assert buf.getvalue() == "alpha\nbeta\n"
    assert buf.flush_calls == 2


def test_safe_sink_still_raises_real_os_failure_on_live_stream() -> None:
    """A genuine failure on a live stream is propagated, not masked.

    ``safe_sink`` only swallows the benign closed-stream ``ValueError``; any
    other error (here a forced ``OSError``) must escape so real logging
    breakage is never hidden.
    """

    class _FailingStream:
        closed = False

        def write(self, message: str) -> int:
            raise OSError("disk full")

    sink = safe_sink(_FailingStream())
    raised: Exception | None = None
    try:
        sink("boom\n")
    except OSError as exc:
        raised = exc
    assert isinstance(raised, OSError)


def test_safe_sink_does_not_raise_for_captured_stdout_teardown(capsys) -> None:
    """Bind to the live captured stdout, then a late write must not raise.

    Reproduces the exact benchmark-worker pattern: ``logger.add`` is given the
    ``sys.stdout`` object that ``capsys`` has swapped in. Under xdist the
    worker's capture stream can close before the handler is removed; wrapping the
    sink keeps teardown clean.
    """
    import sys

    sink_id = logger.add(safe_sink(sys.stdout), format="{message}")
    try:
        logger.info("write through captured stdout")
        assert capsys.readouterr().out
    finally:
        try:
            logger.remove(sink_id)
        except ValueError:
            pass
