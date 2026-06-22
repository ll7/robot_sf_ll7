"""Shared RFC6901 JSON-pointer rendering for schema-validation error paths.

This helper was previously copy-pasted in eight modules across ``robot_sf``
(see issue #3386). Centralizing it removes the latent correctness hazard of a
schema-escaping fix diverging between copies.

The function escapes each path element per RFC6901 (``~`` -> ``~0``,
``/`` -> ``~1``) and renders the root path (an empty element sequence) as the
empty string ``""``, which is the RFC6901 representation of "the whole
document".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def json_pointer(path_elems: Iterable[Any]) -> str:
    """Render an RFC6901-style JSON pointer from a jsonschema error path.

    Args:
        path_elems: Iterable of keys/indices describing a JSON path (for
            example a :class:`jsonschema.exceptions.ValidationError`'s
            ``absolute_path``).

    Returns:
        The JSON pointer string, or ``""`` for the root path.
    """
    parts = [str(part).replace("~", "~0").replace("/", "~1") for part in path_elems]
    return "/" + "/".join(parts) if parts else ""
