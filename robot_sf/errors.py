"""Shared exception hierarchy for the ``robot_sf`` package.

This module defines :class:`RobotSfError`, the common base for the ad-hoc
``*Error`` / ``*Exception`` classes scattered across subpackages. Migrating
those onto a shallow, catchable hierarchy lets callers (and the broad-except
narrowing work in #4880) target ``except RobotSfError`` instead of a bare
``except Exception``.

Behavior-preservation principle
-------------------------------
Re-parenting an exception onto :class:`RobotSfError` is behavior-preserving
only when the MRO still makes every existing ``except`` clause match. A class
that previously inherited a builtin (for example ``ValueError``) becomes
``class FooError(RobotSfError, ValueError)`` so that existing
``except ValueError`` and ``except FooError`` sites keep catching it. Any
re-parent that would stop an existing ``except`` from matching is out of scope
for this slice.

See issue #4993.
"""

from __future__ import annotations

__all__ = ["RobotSfError"]


class RobotSfError(Exception):
    """Base class for all ``robot_sf`` exception types.

    Subpackages should re-parent their ad-hoc errors onto this base while
    preserving any existing builtin ancestry (e.g.
    ``class FooError(RobotSfError, ValueError)``) so that existing ``except``
    clauses keep matching.
    """
