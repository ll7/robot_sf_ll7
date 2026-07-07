"""Tests for restricted unpickler security and functionality."""

import io
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from robot_sf.common.safe_pickle import (
    PLAYBACK_RECORDING_ALLOWED_GLOBALS,
    SOCNAVBENCH_TRAVERSIBLE_ALLOWED_GLOBALS,
    UnsafePickleError,
    restricted_pickle_load,
    restricted_pickle_load_path,
)


class TestRestrictedUnpickler:
    """Tests for the RestrictedUnpickler class."""

    def test_allowed_global_is_accepted(self) -> None:
        """Verify that an allowed global is accepted."""
        data = {"key": "value"}
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)

        result = restricted_pickle_load(
            buffer,
            allowed_globals=frozenset({("builtins", "dict")}),
            label="test",
        )
        assert result == data

    def test_disallowed_global_is_rejected(self) -> None:
        """Verify that a disallowed global raises UnsafePickleError."""
        data = {"key": np.array([1, 2, 3])}
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)

        # Use an empty allowlist to force rejection of numpy globals
        with pytest.raises(UnsafePickleError, match="Unsafe pickle global rejected"):
            restricted_pickle_load(
                buffer,
                allowed_globals=frozenset(),
                label="test",
            )

    def test_malicious_pickle_os_system_rejected(self) -> None:
        """Verify that malicious pickle with os.system is rejected."""

        class Evil:
            def __reduce__(self):
                return (os.system, ("echo should-not-run",))

        buffer = io.BytesIO()
        pickle.dump(Evil(), buffer)
        buffer.seek(0)

        with pytest.raises(UnsafePickleError) as exc_info:
            restricted_pickle_load(
                buffer,
                allowed_globals=PLAYBACK_RECORDING_ALLOWED_GLOBALS,
                label="malicious test payload",
            )

        # Verify the error message contains the rejected symbol
        assert "posix.system" in str(exc_info.value) or "os.system" in str(exc_info.value)
        assert "malicious test payload" in str(exc_info.value)

    def test_malicious_pickle_subprocess_rejected(self) -> None:
        """Verify that malicious pickle with subprocess is rejected."""

        class Evil:
            def __reduce__(self):
                return (subprocess.call, (["echo", "should-not-run"],))

        import subprocess

        buffer = io.BytesIO()
        pickle.dump(Evil(), buffer)
        buffer.seek(0)

        with pytest.raises(UnsafePickleError):
            restricted_pickle_load(
                buffer,
                allowed_globals=PLAYBACK_RECORDING_ALLOWED_GLOBALS,
                label="malicious subprocess payload",
            )

    def test_malicious_pickle_exec_rejected(self) -> None:
        """Verify that malicious pickle with exec is rejected."""

        class Evil:
            def __reduce__(self):
                return (eval, ("__import__('os').system('echo should-not-run')",))  # noqa: S307

        buffer = io.BytesIO()
        pickle.dump(Evil(), buffer)
        buffer.seek(0)

        with pytest.raises(UnsafePickleError):
            restricted_pickle_load(
                buffer,
                allowed_globals=PLAYBACK_RECORDING_ALLOWED_GLOBALS,
                label="malicious eval payload",
            )

    def test_error_message_includes_rejected_symbol(self) -> None:
        """Verify error message includes the rejected module.name and label."""
        data = {"key": np.array([1, 2, 3])}
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)

        with pytest.raises(UnsafePickleError) as exc_info:
            restricted_pickle_load(
                buffer,
                allowed_globals=frozenset(),
                label="test_label",
            )

        assert "test_label" in str(exc_info.value)
        assert "numpy" in str(exc_info.value)


class TestPlaybackRecordingAllowlist:
    """Tests for the PLAYBACK_RECORDING_ALLOWED_GLOBALS allowlist."""

    def test_numpy_array_reconstruction(self) -> None:
        """Verify NumPy arrays can be reconstructed."""
        data = {"array": np.array([1, 2, 3]), "scalar": np.float64(3.14)}
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)

        result = restricted_pickle_load(
            buffer,
            allowed_globals=PLAYBACK_RECORDING_ALLOWED_GLOBALS,
            label="numpy test",
        )
        assert np.array_equal(result["array"], data["array"])
        assert result["scalar"] == data["scalar"]

    def test_nested_dict_with_arrays(self) -> None:
        """Verify nested dicts with arrays work."""
        data = {
            "outer": {
                "inner": np.array([[1, 2], [3, 4]]),
                "value": 42,
            }
        }
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)

        result = restricted_pickle_load(
            buffer,
            allowed_globals=PLAYBACK_RECORDING_ALLOWED_GLOBALS,
            label="nested test",
        )
        assert np.array_equal(result["outer"]["inner"], data["outer"]["inner"])


class TestSocNavBenchAllowlist:
    """Tests for the SOCNAVBENCH_TRAVERSIBLE_ALLOWED_GLOBALS allowlist."""

    def test_dict_with_numpy_array(self) -> None:
        """Verify dict with NumPy array (typical traversible payload) loads."""
        data = {
            "resolution": 0.05,
            "traversible": np.zeros((100, 100), dtype=np.float32),
        }
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)

        result = restricted_pickle_load(
            buffer,
            allowed_globals=SOCNAVBENCH_TRAVERSIBLE_ALLOWED_GLOBALS,
            label="SocNavBench traversible test",
        )
        assert result["resolution"] == 0.05
        assert result["traversible"].shape == (100, 100)

    def test_nested_structure(self) -> None:
        """Verify nested dict/list/tuple with arrays works."""
        data = {
            "resolution": 0.1,
            "metadata": {"name": "test", "values": [1, 2, 3]},
            "traversible": np.ones((50, 50), dtype=np.float64),
        }
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)

        result = restricted_pickle_load(
            buffer,
            allowed_globals=SOCNAVBENCH_TRAVERSIBLE_ALLOWED_GLOBALS,
            label="SocNavBench nested test",
        )
        assert result["resolution"] == 0.1
        assert result["metadata"]["name"] == "test"
        assert result["traversible"].shape == (50, 50)


class TestRestrictedPickleLoadPath:
    """Tests for the restricted_pickle_load_path function."""

    def test_load_from_path(self) -> None:
        """Verify loading from a file path works."""
        data = {"test": "value", "array": np.array([1, 2, 3])}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = Path(f.name)
            pickle.dump(data, f)

        try:
            result = restricted_pickle_load_path(
                temp_path,
                allowed_globals=PLAYBACK_RECORDING_ALLOWED_GLOBALS,
                label="path test",
            )
            assert result["test"] == data["test"]
            assert np.array_equal(result["array"], data["array"])
        finally:
            temp_path.unlink()

    def test_load_from_path_rejects_malicious(self) -> None:
        """Verify that malicious file is rejected when loading from path."""

        class Evil:
            def __reduce__(self):
                return (os.system, ("echo should-not-run",))

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = Path(f.name)
            pickle.dump(Evil(), f)

        try:
            with pytest.raises(UnsafePickleError):
                restricted_pickle_load_path(
                    temp_path,
                    allowed_globals=PLAYBACK_RECORDING_ALLOWED_GLOBALS,
                    label="malicious path test",
                )
        finally:
            temp_path.unlink()
