"""Test the ensure_directory utility function."""

import tempfile
from pathlib import Path

from robot_sf.benchmark.utils import ensure_directory


def test_ensure_directory_creates_directory():
    """Test that ensure_directory creates a directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test_dir"

        result = ensure_directory(test_path)

        assert result == test_path
        assert test_path.exists()
        assert test_path.is_dir()


def test_ensure_directory_creates_nested_directories():
    """Test that ensure_directory creates nested directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "level1" / "level2" / "level3"

        result = ensure_directory(test_path)

        assert result == test_path
        assert test_path.exists()
        assert test_path.is_dir()


def test_ensure_directory_with_file_path():
    """Test that ensure_directory creates parent directory for file paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "results" / "data.json"

        result = ensure_directory(file_path)

        expected_dir = file_path.parent
        assert result == expected_dir
        assert expected_dir.exists()
        assert expected_dir.is_dir()


def test_ensure_directory_already_exists():
    """Test that ensure_directory works when directory already exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "existing_dir"
        test_path.mkdir()  # Create it first

        result = ensure_directory(test_path)

        assert result == test_path
        assert test_path.exists()
        assert test_path.is_dir()


def test_ensure_directory_with_string_path():
    """Test that ensure_directory works with string paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path_str = str(Path(temp_dir) / "string_dir")

        result = ensure_directory(test_path_str)

        assert result == Path(test_path_str)
        assert result.exists()
        assert result.is_dir()


def test_ensure_directory_nested_file_path():
    """Test ensure_directory with deeply nested file path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "a" / "b" / "c" / "data.txt"

        result = ensure_directory(file_path)

        expected_dir = file_path.parent
        assert result == expected_dir
        assert expected_dir.exists()
        assert expected_dir.is_dir()

        # File itself should not exist
        assert not file_path.exists()
