"""Tests for manual-control mode configuration and CLI selection."""

import json
import subprocess
import sys

import pytest

from robot_sf.manual_control.config import ManualControlRuntimeConfig


def test_runtime_config_records_mode_versions_and_overlay_labels():
    """Runtime config should expose artifact metadata for selected modes."""
    config = ManualControlRuntimeConfig.from_strings(
        control_mode="keyboard_cruise",
        view_mode="fixed_map",
    )

    payload = config.to_json_dict()

    assert payload["control_mode"] == "keyboard_cruise"
    assert payload["view_mode"] == "fixed_map"
    assert payload["input_mapping_version"] == "keyboard_cruise_diff_drive_v1"
    assert "persistent target velocity" in payload["control_overlay_label"]


@pytest.mark.parametrize("view_mode", ["ego_up", "robot_static"])
def test_runtime_config_accepts_camera_transform_view_modes(view_mode):
    """Camera-transform views are selectable once the renderer exposes the hook."""
    config = ManualControlRuntimeConfig.from_strings(
        control_mode="keyboard_cruise",
        view_mode=view_mode,
    )

    assert config.view_mode == view_mode
    assert "robot-centered" in config.overlay_metadata()["view_overlay_label"]


def test_validate_modes_cli_prints_selected_mode_metadata():
    """CLI switch surface should expose mode selection metadata for runners."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/manual_control/validate_modes.py",
            "--control-mode",
            "mouse_target",
            "--view-mode",
            "fixed_map",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["control_mode"] == "mouse_target"
    assert payload["view_mode"] == "fixed_map"
    assert payload["input_mapping_version"] == "mouse_target_diff_drive_v1"
