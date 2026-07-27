"""Deterministic unit coverage for the multi-extractor artifact path helpers.

These tests lock the filesystem contracts exposed by
``robot_sf.training.multi_extractor_paths`` (issue #6341):

* ``resolve_base_output_root`` honors the ``ROBOT_SF_MULTI_EXTRACTOR_TMP``
  override (``~`` expansion plus symlink / relative-path resolution) and
  otherwise delegates to the shared artifact resolver with the canonical
  default root.
* ``make_run_directory`` creates ``<stamp>-<run_id>`` under the resolved base,
  materializes the base root, creates the ``extractors/`` sibling subdirectory,
  and rejects empty run ids.
* ``make_extractor_directory`` creates the per-extractor directory under
  ``extractors/`` using a filesystem-safe normalized name and rejects names
  that contain no alphanumeric characters.
* ``load_configuration`` rejects extractor names that normalize or case-fold to
  the same artifact directory before a training run can overwrite one profile
  with another.
* ``summary_paths`` returns the canonical ``summary.json`` / ``summary.md``
  locations for a run.

Determinism / isolation rules:

* All filesystem effects stay inside the pytest ``tmp_path`` fixture via
  explicit environment dictionaries that point ``ROBOT_SF_MULTI_EXTRACTOR_TMP``
  at ``tmp_path``; no repository ``output/`` or ``tmp/`` artifact is created.
* The generated-timestamp case asserts the directory-name pattern and the
  run-id suffix instead of a wall-clock value.
* The default-delegation path mocks the shared artifact resolver
  (``robot_sf.common.artifact_paths.resolve_artifact_path``) so no
  repository-rooted path is materialized.

Extractor names are normalized to a single portable directory component: unsafe
characters (including path separators and whitespace) become underscores, and
leading/trailing punctuation is removed. This prevents traversal-like names
from escaping the run's ``extractors/`` directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import pytest

from robot_sf.training.multi_extractor_paths import (
    DEFAULT_TMP_ROOT,
    ENV_TMP_OVERRIDE,
    make_extractor_directory,
    make_run_directory,
    resolve_base_output_root,
    summary_paths,
)
from scripts.multi_extractor_training import load_configuration

# %Y%m%d-%H%M%S as produced by ``datetime.now(UTC).strftime`` in make_run_directory.
RUN_TIMESTAMP_PATTERN = re.compile(r"^\d{8}-\d{6}$")


def _override_env(target: Path) -> dict[str, str]:
    """Return an env dict that redirects the base output root to ``target``."""

    return {ENV_TMP_OVERRIDE: str(target)}


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_module_constants_lock_override_key_and_default_root() -> None:
    """The override env key and default root relpath are part of the contract."""

    assert ENV_TMP_OVERRIDE == "ROBOT_SF_MULTI_EXTRACTOR_TMP"
    assert DEFAULT_TMP_ROOT == Path("tmp/multi_extractor_training")


# ---------------------------------------------------------------------------
# resolve_base_output_root
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env", [None, {}, {"UNRELATED": "value"}])
def test_resolve_base_output_root_default_delegates_to_artifact_resolver(
    env: dict[str, str] | None,
) -> None:
    """Without the override key the helper delegates to the shared resolver."""

    sentinel = Path("/nonexistent/sentinel-multi-extractor")
    with mock.patch(
        "robot_sf.training.multi_extractor_paths.resolve_artifact_path",
        return_value=sentinel,
    ) as resolver:
        result = resolve_base_output_root(env)

    assert result == sentinel
    resolver.assert_called_once_with(DEFAULT_TMP_ROOT)


def test_resolve_base_output_root_override_expands_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leading ``~`` is expanded against ``HOME`` deterministically."""

    monkeypatch.setenv("HOME", str(tmp_path))

    result = resolve_base_output_root({ENV_TMP_OVERRIDE: "~/multi"})

    assert result == (tmp_path / "multi").resolve()


def test_resolve_base_output_root_override_resolves_symlink(tmp_path: Path) -> None:
    """The override is resolved, so a symlink target is returned, not the link."""

    real_target = tmp_path / "real_target"
    real_target.mkdir()
    link = tmp_path / "link_to_real"
    link.symlink_to(real_target, target_is_directory=True)

    result = resolve_base_output_root({ENV_TMP_OVERRIDE: str(link)})

    assert result == real_target.resolve()
    assert result.name == "real_target"


def test_resolve_base_output_root_override_relative_resolves_to_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative override is resolved to an absolute path under the CWD."""

    monkeypatch.chdir(tmp_path)

    result = resolve_base_output_root({ENV_TMP_OVERRIDE: "nested/rel"})

    assert result == (tmp_path / "nested" / "rel").resolve()
    assert result.is_absolute()


def test_resolve_base_output_root_override_absolute_path_returned_resolved(
    tmp_path: Path,
) -> None:
    """An absolute override is returned resolved (and absolute)."""

    target = tmp_path / "abs_target"

    result = resolve_base_output_root({ENV_TMP_OVERRIDE: str(target)})

    assert result == target.resolve()
    assert result.is_absolute()


# ---------------------------------------------------------------------------
# make_run_directory
# ---------------------------------------------------------------------------


def test_make_run_directory_explicit_timestamp_names_and_creates_layout(
    tmp_path: Path,
) -> None:
    """An explicit timestamp yields an exact dir name plus the extractors dir."""

    base = tmp_path / "mx_base"

    run_dir = make_run_directory(
        "alpha",
        env=_override_env(base),
        timestamp="20240101-120000",
    )

    assert run_dir == base / "20240101-120000-alpha"
    assert run_dir.is_dir()
    assert (run_dir / "extractors").is_dir()
    # make_run_directory materializes the base root itself.
    assert base.is_dir()


def test_make_run_directory_generated_timestamp_uses_pattern_and_run_id_suffix(
    tmp_path: Path,
) -> None:
    """A generated timestamp matches the stamp pattern and keeps the run id."""

    base = tmp_path / "mx_base"

    run_dir = make_run_directory("beta", env=_override_env(base))

    assert run_dir.parent == base
    name = run_dir.name
    assert name.endswith("-beta")
    stamp = name[: -len("-beta")]
    assert RUN_TIMESTAMP_PATTERN.match(stamp), name
    assert (run_dir / "extractors").is_dir()


def test_make_run_directory_empty_run_id_raises(tmp_path: Path) -> None:
    """An empty run id is rejected before any filesystem work happens."""

    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        make_run_directory(
            "",
            env=_override_env(tmp_path / "mx_base"),
            timestamp="20240101-000000",
        )


# ---------------------------------------------------------------------------
# make_extractor_directory
# ---------------------------------------------------------------------------


def test_make_extractor_directory_creates_dir_under_extractors(tmp_path: Path) -> None:
    """Each extractor gets its own directory under ``extractors/``."""

    base = tmp_path / "mx_base"
    run_dir = make_run_directory(
        "run1",
        env=_override_env(base),
        timestamp="20240101-000000",
    )

    first = make_extractor_directory(run_dir, "cnn")
    second = make_extractor_directory(run_dir, "transformer")

    assert first == run_dir / "extractors" / "cnn"
    assert first.is_dir()
    assert second == run_dir / "extractors" / "transformer"
    assert second.is_dir()
    # Adding a second extractor leaves the first intact.
    assert first.is_dir()


@pytest.mark.parametrize(
    ("extractor_name", "expected_name"),
    [
        ("My_CNN.v2", "My_CNN.v2"),
        ("My CNN/v2", "My_CNN_v2"),
        ("../outside", "outside"),
    ],
)
def test_make_extractor_directory_normalizes_to_safe_single_component(
    tmp_path: Path,
    extractor_name: str,
    expected_name: str,
) -> None:
    """Unsafe names cannot introduce nested or traversal directory components."""

    base = tmp_path / "mx_base"
    run_dir = make_run_directory(
        "run1",
        env=_override_env(base),
        timestamp="20240101-000000",
    )

    extractor_dir = make_extractor_directory(run_dir, extractor_name)

    assert extractor_dir == run_dir / "extractors" / expected_name
    assert extractor_dir.is_dir()
    assert extractor_dir.parent == run_dir / "extractors"


@pytest.mark.parametrize(
    ("first_name", "second_name", "normalized_name"),
    [
        ("alpha/beta", "alpha beta", "alpha_beta"),
        ("CNN", "cnn", "cnn"),
    ],
)
def test_load_configuration_rejects_colliding_normalized_extractor_names(
    tmp_path: Path,
    first_name: str,
    second_name: str,
    normalized_name: str,
) -> None:
    """Reject names that would otherwise write into one artifact directory."""

    config_path = tmp_path / "colliding_extractors.yaml"
    config_path.write_text(
        f"extractors:\n  - name: {first_name}\n  - name: {second_name}\n",
        encoding="utf-8",
    )

    collision_message = (
        rf"{re.escape(first_name)}.*{re.escape(second_name)}.*{re.escape(normalized_name)}"
    )
    with pytest.raises(ValueError, match=collision_message):
        load_configuration(config_path)


@pytest.mark.parametrize("extractor_name", ["", "   ", "../", "---"])
def test_make_extractor_directory_name_without_alphanumeric_characters_raises(
    tmp_path: Path, extractor_name: str
) -> None:
    """Names that normalize to no usable directory component are rejected."""

    base = tmp_path / "mx_base"
    run_dir = make_run_directory(
        "run1",
        env=_override_env(base),
        timestamp="20240101-000000",
    )

    with pytest.raises(ValueError, match="extractor_name must contain"):
        make_extractor_directory(run_dir, extractor_name)


# ---------------------------------------------------------------------------
# summary_paths
# ---------------------------------------------------------------------------


def test_summary_paths_returns_json_and_markdown_locations(tmp_path: Path) -> None:
    """summary_paths returns the canonical run-summary filenames only."""

    run_dir = tmp_path / "run_x"

    paths = summary_paths(run_dir)

    assert set(paths) == {"json", "markdown"}
    assert paths["json"] == run_dir / "summary.json"
    assert paths["markdown"] == run_dir / "summary.md"
    # The helper computes locations only; it must not create the files.
    assert not paths["json"].exists()
    assert not paths["markdown"].exists()
