"""Tests for safe pickle integration in playback_recording module."""

import os
import pickle
import tempfile
from pathlib import Path

import pytest

from robot_sf.common.safe_pickle import UnsafePickleError
from robot_sf.render.playback_recording import load_states

# Resolve repo-relative fixtures from this file, not the process cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPlaybackRecordingSafePickle:
    """Tests for safe pickle in playback_recording.load_states."""

    def test_real_example_recording_loads(self) -> None:
        """Verify that a real example recording loads unchanged."""
        example_dir = _REPO_ROOT / "examples" / "recordings"
        pkl_files = list(example_dir.glob("*.pkl"))

        if not pkl_files:
            pytest.skip("No example recordings available")

        smallest = min(pkl_files, key=lambda p: p.stat().st_size)
        loaded_states, _loaded_map_def = load_states(str(smallest))
        assert len(loaded_states) > 0

    def test_malicious_pickle_rejected(self) -> None:
        """Verify that malicious pickle is rejected before execution."""

        class Evil:
            def __reduce__(self):
                return (os.system, ("echo should-not-run",))

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = Path(f.name)
            pickle.dump(Evil(), f)

        try:
            with pytest.raises(UnsafePickleError) as exc_info:
                load_states(str(temp_path))

            assert "posix.system" in str(exc_info.value) or "os.system" in str(exc_info.value)
            assert "playback recording" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_empty_file_error(self) -> None:
        """Verify that empty file raises appropriate error."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="empty"):
                load_states(str(temp_path))
        finally:
            temp_path.unlink()

    def test_wrong_type_error(self) -> None:
        """Verify that wrong tuple type raises TypeError."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = Path(f.name)
            pickle.dump({"wrong": "structure"}, f)

        try:
            with pytest.raises((TypeError, ValueError)):
                load_states(str(temp_path))
        finally:
            temp_path.unlink()
