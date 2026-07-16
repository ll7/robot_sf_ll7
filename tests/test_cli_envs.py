"""Tests for the ``robot-sf envs`` list/describe UX (issue #5801)."""

from __future__ import annotations

import importlib

import pytest

from robot_sf import cli_envs
from robot_sf.cli import _HANDLERS, main
from robot_sf.cli_envs import describe_env_payload
from robot_sf.gym_env.env_registry import (
    BETA,
    STABLE,
    EnvEntry,
    env_ids,
    get_env,
    list_envs,
    register_env,
)


def test_list_envs_includes_public_factories() -> None:
    """The default catalog must expose the five public make_* environments."""
    ids = set(env_ids())
    assert {"robot", "robot-image", "pedestrian", "crowd-sim", "multi-robot"} <= ids


def test_list_envs_ordered_by_stability_then_id() -> None:
    """Stable entries sort before beta/experimental; ties break by id."""
    rows = list_envs()
    stabilities = [e.stability for e in rows]
    assert (
        stabilities.index(STABLE) < stabilities.index(BETA)
        if STABLE in stabilities and BETA in stabilities
        else True
    )
    # Ids within the same stability level are lexicographic.
    stable_ids = [e.env_id for e in rows if e.stability == STABLE]
    assert stable_ids == sorted(stable_ids)


def test_default_registry_dotted_symbols_resolve() -> None:
    """Every advertised factory, environment class, and config must be importable."""
    for entry in list_envs():
        for dotted_path in (entry.factory, entry.env_class, entry.default_config):
            module_name, symbol_name = dotted_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            assert hasattr(module, symbol_name), dotted_path


def test_describe_returns_full_entry_dict() -> None:
    """describe_env_payload returns every documented field for a known id."""
    payload = describe_env_payload("robot")
    assert payload["env_id"] == "robot"
    assert payload["factory"].endswith("make_robot_env")
    assert payload["stability"] == STABLE
    for key in (
        "display_name",
        "summary",
        "env_class",
        "agent_count",
        "default_config",
        "notes",
    ):
        assert key in payload


def test_describe_unknown_id_raises_keyerror() -> None:
    """Describing an unregistered id raises KeyError naming known ids."""
    with pytest.raises(KeyError, match="unknown-env"):
        get_env("unknown-env")
    with pytest.raises(KeyError):
        describe_env_payload("unknown-env")


def test_register_env_rejects_empty_id_and_bad_stability() -> None:
    """Registration guards against empty ids and unknown stability labels."""
    with pytest.raises(ValueError):
        register_env(
            EnvEntry(
                env_id="",
                display_name="x",
                summary="x",
                factory="f",
                env_class="c",
                agent_count="single",
                default_config="d",
            )
        )
    with pytest.raises(ValueError):
        register_env(
            EnvEntry(
                env_id="bad-stability",
                display_name="x",
                summary="x",
                factory="f",
                env_class="c",
                agent_count="single",
                default_config="d",
                stability="not-a-level",
            )
        )


def test_register_env_duplicate_requires_override() -> None:
    """Re-registering an id fails unless override=True."""
    entry = EnvEntry(
        env_id="dup",
        display_name="d",
        summary="s",
        factory="f",
        env_class="c",
        agent_count="single",
        default_config="dc",
    )
    register_env(entry)
    with pytest.raises(KeyError):
        register_env(entry)
    register_env(entry, override=True)
    assert get_env("dup").display_name == "d"


def test_cli_envs_list_friendly_output(capsys) -> None:
    """``envs list`` prints every env id in friendly mode and exits 0."""
    import argparse

    rc = cli_envs._handle_envs_list(argparse.Namespace(format="friendly"))
    out = capsys.readouterr().out
    assert rc == 0
    for env_id in ("robot", "crowd-sim"):
        assert env_id in out


def test_cli_envs_list_json_output(capsys) -> None:
    """``envs list --format json`` emits valid JSON list of dicts."""
    import argparse
    import json

    rc = cli_envs._handle_envs_list(argparse.Namespace(format="json"))
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert isinstance(data, list)
    assert any(d["env_id"] == "robot" for d in data)


def test_cli_envs_describe_friendly_output(capsys) -> None:
    """``envs describe <id>`` prints the factory and exits 0."""
    import argparse

    rc = cli_envs._handle_envs_describe(argparse.Namespace(env_id="crowd-sim", format="friendly"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "CrowdSimEnv" in out
    assert "make_crowd_sim_env" in out


def test_cli_envs_describe_unknown_exits_2(capsys) -> None:
    """``envs describe <unknown>`` exits 2 and writes to stderr."""
    import argparse

    rc = cli_envs._handle_envs_describe(argparse.Namespace(env_id="no-such-env", format="friendly"))
    err = capsys.readouterr().err
    assert rc == 2
    assert "no-such-env" in err


def test_top_level_cli_envs_list_and_describe(capsys) -> None:
    """The top-level ``robot-sf envs`` dispatch reaches the handlers."""
    assert main(["envs", "list"]) == 0
    assert main(["envs", "describe", "robot"]) == 0
    assert main(["envs", "describe", "missing"]) == 2


def test_envs_registration_preserves_main_cli_handlers(capsys) -> None:
    """Adding env discovery must not shadow command handlers already on main."""
    assert "gallery" in _HANDLERS
    assert main(["envs", "list", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
