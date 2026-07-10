"""Guard against silent truncation in bounded `gh ... list --limit N` calls.

GitHub CLI list commands such as `gh pr list` and `gh issue list` accept a numeric
`--limit N` and return at most ``N`` rows. When exactly ``N`` rows come back, the
caller cannot tell whether the page was complete or silently capped. This module
makes that ambiguity loud instead of silent.

Two complementary entry points:

- :func:`assert_not_truncated` raises :class:`GhListTruncated` for callers that
  want to fail closed.
- :func:`is_likely_truncated` returns a boolean for callers that prefer to record
  a ``truncated: true`` marker in their structured snapshot JSON.

The motivation and contract are tracked in issue #4991.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sized


class GhListTruncated(RuntimeError):
    """Raised when a bounded ``gh ... list --limit N`` result may be truncated.

    A result containing exactly ``limit`` rows is treated as potentially
    truncated because a full page and a capped page are indistinguishable.
    """


def is_likely_truncated(row_count: int, *, limit: int) -> bool:
    """Return ``True`` when a bounded gh list result may be truncated.

    A non-positive ``limit`` never reports truncation: there is no cap to hit
    when pagination is effectively unbounded.
    """
    return limit > 0 and row_count >= limit


def assert_not_truncated(
    rows: Sized,
    *,
    limit: int,
    context: str = "",
) -> None:
    """Raise :class:`GhListTruncated` when ``rows`` may be truncated by the cap.

    Parameters
    ----------
    rows:
        The sized container of rows returned by ``gh ... list --limit``.
    limit:
        The numeric ``--limit`` value passed to ``gh``.
    context:
        Optional short label (for example, the gh subcommand and search) that is
        included in the error message to aid diagnosis.

    Raises
    ------
    GhListTruncated
        If ``len(rows) >= limit`` and ``limit > 0``.
    """
    count = len(rows)
    if is_likely_truncated(count, limit=limit):
        detail = (
            f"gh list may be truncated: got {count} rows at --limit {limit}; "
            "raise --limit or paginate"
        )
        if context:
            detail = f"{detail} ({context})"
        raise GhListTruncated(detail)
