"""Tests for the ``robot-sf examples`` discovery CLI (issue #5794).

These tests exercise the catalog logic in :mod:`robot_sf.examples_cli` against
the real ``examples/examples_manifest.yaml``. They assert the acceptance
criteria from the issue:

* every manifest entry is listable (and filterable by tag);
* an example resolves by id, full path, or unique filename stem;
* unknown ids fail clearly with closest matches;
* at least one example actually runs headless.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from robot_sf.examples.manifest_loader import load_manifest
from robot_sf.examples_cli import (
    ExampleIdentityError,
    example_id,
    examples_cli_main,
    find_example,
    format_examples_table,
    run_example,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = load_manifest(validate_paths=True)


def test_every_manifest_entry_has_a_listable_id() -> None:
    """Every example produces a stable, unique, .py-free discovery id."""

    ids = [example_id(example) for example in _MANIFEST.examples]
    assert len(ids) == len(set(ids)), "Discovery ids are not unique."
    for identifier in ids:
        assert not identifier.endswith(".py"), identifier
        assert "/" not in identifier or identifier.count("/") <= 2


def test_every_manifest_entry_is_listable_in_full_table() -> None:
    """The unfiltered table lists every manifest entry by id."""

    table = format_examples_table(_MANIFEST)
    for example in _MANIFEST.examples:
        identifier = example_id(example)
        assert identifier in table
        assert example.name in table
    assert f"{len(_MANIFEST.examples)} example(s) listed." in table


@pytest.mark.parametrize("example", _MANIFEST.examples, ids=example_id)
def test_each_example_id_resolves_back_to_itself(example) -> None:
    """find_example resolves each entry's own id back to itself."""

    resolved = find_example(_MANIFEST, example_id(example))
    assert resolved is example or resolved.path == example.path


def test_find_example_accepts_full_path_and_extension() -> None:
    """Resolution accepts the full manifest path (with .py)."""

    first = _MANIFEST.examples[0]
    resolved = find_example(_MANIFEST, first.path.as_posix())
    assert resolved.path == first.path


def test_find_example_accepts_unique_filename_stem() -> None:
    """Resolution accepts the bare filename stem when it is unique."""

    # quickstart/01_basic_robot.py has a globally unique stem.
    resolved = find_example(_MANIFEST, "01_basic_robot")
    assert resolved.path.as_posix() == "quickstart/01_basic_robot.py"


def test_find_example_is_case_insensitive() -> None:
    """Resolution is case-insensitive for convenience."""

    resolved = find_example(_MANIFEST, "Quickstart/01_Basic_Robot")
    assert resolved.path.as_posix() == "quickstart/01_basic_robot.py"


def test_unknown_id_reports_closest_matches() -> None:
    """An unknown id raises an error naming the closest known ids."""

    with pytest.raises(ExampleIdentityError) as exc_info:
        find_example(_MANIFEST, "01-basic-robot")
    message = str(exc_info.value)
    assert "Unknown example id" in message
    assert "quickstart/01_basic_robot" in message
    assert "Closest matches" in message


def test_unknown_id_with_no_neighbors_still_lists_hint() -> None:
    """An id with no close neighbors still gives actionable guidance."""

    with pytest.raises(ExampleIdentityError) as exc_info:
        find_example(_MANIFEST, "zzz-nothing-like-this-xyz")
    assert "Unknown example id" in str(exc_info.value)


def test_tag_filter_returns_only_matching_examples() -> None:
    """--tag filters the table to examples carrying that tag."""

    table = format_examples_table(_MANIFEST, tag="maps")
    matching = {
        example_id(example)
        for example in _MANIFEST.examples
        if "maps" in {t.lower() for t in example.tags}
    }
    assert matching, "Precondition: at least one example is tagged 'maps'."
    for identifier in matching:
        assert identifier in table
    # An example known NOT to carry the 'maps' tag should be absent.
    non_maps = next(
        example_id(example)
        for example in _MANIFEST.examples
        if "maps" not in {t.lower() for t in example.tags}
    )
    assert non_maps not in table


def test_tag_filter_is_case_insensitive() -> None:
    """Tag filtering ignores case."""

    lower = format_examples_table(_MANIFEST, tag="quickstart")
    upper = format_examples_table(_MANIFEST, tag="QUICKSTART")
    assert lower == upper


def test_tag_filter_with_no_matches_reports_empty() -> None:
    """A tag nobody carries produces a clear empty message."""

    table = format_examples_table(_MANIFEST, tag="nonexistent-tag-xyz")
    assert "No examples match" in table


def test_category_filter_restricts_to_category() -> None:
    """--category restricts the table to one category slug."""

    table = format_examples_table(_MANIFEST, category="quickstart")
    matching = {example_id(example) for example in _MANIFEST.examples_for_category("quickstart")}
    assert matching
    for identifier in matching:
        assert identifier in table


def test_expected_runtime_column_renders_when_present() -> None:
    """An example with expected_runtime shows it in the RUNTIME column."""

    with_runtime = next(
        (example for example in _MANIFEST.examples if example.expected_runtime), None
    )
    assert with_runtime is not None, "Precondition: at least one example declares expected_runtime."
    table = format_examples_table(_MANIFEST, tag=with_runtime.tags[0])
    assert with_runtime.expected_runtime in table


def test_list_subcommand_prints_table_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`robot-sf examples list` exits 0 and prints every id."""

    exit_code = examples_cli_main(["list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    for example in _MANIFEST.examples:
        assert example_id(example) in captured.out


def test_list_subcommand_supports_tag_filter(capsys: pytest.CaptureFixture[str]) -> None:
    """`robot-sf examples list --tag` filters the output."""

    exit_code = examples_cli_main(["list", "--tag", "quickstart"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "quickstart/01_basic_robot" in captured.out


def test_run_unknown_id_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    """`robot-sf examples run <unknown>` exits non-zero with a helpful message."""

    exit_code = examples_cli_main(["run", "definitely-not-an-example"])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Unknown example id" in captured.err


def test_run_uses_fast_env_vars_and_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run --fast` sets ROBOT_SF_FAST_DEMO and launches the resolved script."""

    captured: dict[str, object] = {}

    class _FakeResult:
        returncode = 0

    def fake_runner(command, env, cwd, timeout):
        captured["command"] = command
        captured["env"] = env
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        return _FakeResult()

    monkeypatch.setenv("HOME", str(_REPO_ROOT))
    exit_code = run_example(
        _MANIFEST,
        "quickstart/01_basic_robot",
        fast=True,
        runner=fake_runner,
        timeout=30.0,
    )
    assert exit_code == 0
    command = captured["command"]
    assert command[1].endswith("examples/quickstart/01_basic_robot.py")
    env = captured["env"]
    assert env["ROBOT_SF_FAST_DEMO"] == "1"
    assert env["ROBOT_SF_EXAMPLES_MAX_STEPS"] == "64"
    assert env["MPLBACKEND"] == "Agg"
    assert env["SDL_VIDEODRIVER"] == "dummy"
    assert str(captured["cwd"]) == str(_REPO_ROOT)


def test_run_fast_overrides_inherited_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--fast`` must win over inherited flags, step caps, and GUI backends."""
    captured: dict[str, object] = {}

    class _FakeResult:
        returncode = 0

    def fake_runner(command, env, cwd, timeout):
        captured["env"] = env
        return _FakeResult()

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    monkeypatch.setenv("SDL_VIDEODRIVER", "x11")
    monkeypatch.setenv("ROBOT_SF_FAST_DEMO", "0")
    monkeypatch.setenv("ROBOT_SF_EXAMPLES_MAX_STEPS", "4096")

    assert run_example(_MANIFEST, "quickstart/01_basic_robot", fast=True, runner=fake_runner) == 0

    env = captured["env"]
    assert env["DISPLAY"] == ""
    assert env["MPLBACKEND"] == "Agg"
    assert env["SDL_VIDEODRIVER"] == "dummy"
    assert env["ROBOT_SF_FAST_DEMO"] == "1"
    assert env["ROBOT_SF_EXAMPLES_MAX_STEPS"] == "64"


def test_run_passes_extra_args_to_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra args after the id are forwarded to the example script."""

    captured: dict[str, object] = {}

    class _FakeResult:
        returncode = 0

    def fake_runner(command, env, cwd, timeout):
        captured["command"] = command
        return _FakeResult()

    exit_code = run_example(
        _MANIFEST,
        "quickstart/01_basic_robot",
        fast=False,
        extra_args=["--foo", "bar"],
        runner=fake_runner,
    )
    assert exit_code == 0
    assert captured["command"][-2:] == ["--foo", "bar"]


@pytest.mark.slow
def test_at_least_one_example_runs_headless_fast() -> None:
    """At least one example executes end-to-end in headless fast mode.

    This is the acceptance criterion: a real example runs headless via the CLI.
    Uses quickstart/01_basic_robot, which honours ROBOT_SF_FAST_DEMO.
    """

    exit_code = run_example(
        _MANIFEST,
        "quickstart/01_basic_robot",
        fast=True,
        timeout=180.0,
    )
    assert exit_code == 0, "quickstart/01_basic_robot should run headless in fast mode"


def test_top_level_cli_dispatches_examples(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The umbrella `robot-sf` CLI forwards `examples` to the examples CLI."""

    from robot_sf.cli import main as umbrella_main

    exit_code = umbrella_main(["examples", "list", "--tag", "maps"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "maps" in captured.out.lower() or "example(s) listed." in captured.out


def test_top_level_cli_keeps_doctor_subcommand() -> None:
    """Adding the examples subcommand did not remove the doctor subcommand."""

    from robot_sf.cli import _build_parser

    parser = _build_parser()
    choices: set[str] = set()
    for action in parser._actions:
        choice_map = getattr(action, "choices", None)
        if isinstance(choice_map, dict):
            choices.update(choice_map.keys())
    assert "doctor" in choices
    assert "examples" in choices


def test_load_manifest_parses_expected_runtime_field(tmp_path: Path) -> None:
    """The manifest loader surfaces the optional expected_runtime field."""

    manifest_text = dedent(
        """
        version: 0.1.0
        categories:
          - slug: quickstart
            title: "Quickstart"
            description: "d"
            order: 1
            ci_default: true
        examples:
          - path: quickstart/demo.py
            name: "Demo"
            summary: "A demo."
            category_slug: quickstart
            ci_enabled: true
            ci_reason: null
            doc_reference: null
            tags: [demo]
            expected_runtime: "~5s"
        """
    ).strip()
    examples_dir = tmp_path / "examples" / "quickstart"
    examples_dir.mkdir(parents=True)
    (examples_dir / "demo.py").write_text('"""A demo."""\n', encoding="utf-8")
    manifest_path = tmp_path / "examples" / "examples_manifest.yaml"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    manifest = load_manifest(manifest_path, validate_paths=True)
    example = manifest.examples[0]
    assert example.expected_runtime == "~5s"


def test_examples_table_handles_entries_without_runtime(tmp_path: Path) -> None:
    """Entries without expected_runtime render a dash in the RUNTIME column."""

    manifest_text = dedent(
        """
        version: 0.1.0
        categories:
          - slug: quickstart
            title: "Quickstart"
            description: "d"
            order: 1
            ci_default: true
        examples:
          - path: quickstart/demo.py
            name: "Demo"
            summary: "A demo."
            category_slug: quickstart
            ci_enabled: true
            ci_reason: null
            doc_reference: null
            tags: [demo]
        """
    ).strip()
    examples_dir = tmp_path / "examples" / "quickstart"
    examples_dir.mkdir(parents=True)
    (examples_dir / "demo.py").write_text('"""A demo."""\n', encoding="utf-8")
    manifest_path = tmp_path / "examples" / "examples_manifest.yaml"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    manifest = load_manifest(manifest_path, validate_paths=True)
    table = format_examples_table(manifest)
    assert "RUNTIME" in table
    assert "-" in table
