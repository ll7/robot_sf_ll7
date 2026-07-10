"""Canonical optional-dependency import guard (see issue #4990).

Optional-dependency import guards in ``robot_sf`` had drifted into ~40 distinct
spellings. This module provides the single blessed spelling for the *common*
case so that reviews can focus scrutiny on the (few) guards that legitimately
catch more than ``ImportError``.

The common case
---------------
Most optional imports are of the form "import a module if it is available,
otherwise bind the name to ``None`` and degrade gracefully". Use
:func:`try_import`::

    from robot_sf.common.optional_import import try_import

    yaml = try_import("yaml")
    if yaml is None:
        ...  # degrade gracefully

:func:`try_import` catches **only** :class:`ImportError` (which includes
:class:`ModuleNotFoundError`). It never swallows broader failures, so a buggy
optional dependency cannot hide behind an "optional import".

When is a broader ``except`` legitimate?
----------------------------------------
Some optional imports intentionally catch more than ``ImportError``. Do **not**
use :func:`try_import` for these; keep the explicit broad ``except`` and add a
``# pragma: no cover`` plus a one-line justification:

* **Native/extension libraries** (OMPL, NVML/pynvml, pygame) may raise
  ``RuntimeError`` or ``OSError`` during a partially-broken install even when
  the package is importable by name.
* **Matplotlib backend selection** can raise ``RuntimeError``.
* **Compiled-backend entry points** (shapely/scipy) may raise ``AttributeError``
  when a native wheel component is absent.

The AST inventory ratchet in
``tests/test_optional_import_guard_inventory.py`` must be updated to bless any
new (justified) broad spelling; otherwise the test fails and blocks the change
in review.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def try_import(name: str) -> ModuleType | None:
    """Import ``name`` if available, otherwise return ``None``.

    This is the canonical spelling for the common optional-dependency case. It
    catches only :class:`ImportError` (which includes
    :class:`ModuleNotFoundError`) and never swallows broader failures such as
    ``RuntimeError``/``OSError``/``AttributeError`` -- those indicate a broken
    install and should surface, not be masked as "missing dependency".

    Parameters
    ----------
    name:
        Dotted module name to import, e.g. ``"yaml"`` or
        ``"shapely.geometry"``.

    Returns
    -------
    ModuleType | None
        The imported module, or ``None`` if it is not installed.

    Examples
    --------
    >>> yaml = try_import("yaml")
    >>> if yaml is None:
    ...     pass  # degrade gracefully without the YAML backend
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None
