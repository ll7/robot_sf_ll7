"""Tests for PR-gate worktree lease management."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.dev.pr_gate_lease import (
    PRGateLease,
    create_lease,
    heartbeat,
    is_active,
    lease_path,
    load_lease,
    release_lease,
    status,
)


@pytest.fixture
def mock_git_dirs(tmp_path: Path) -> None:
    """Mock git common dir to use a temp directory."""
    git_common = tmp_path / ".git"
    git_common.mkdir()

    with (
        patch("scripts.dev.pr_gate_lease._git_common_dir", return_value=git_common),
        patch("scripts.dev.pr_gate_lease._repo_root", return_value=tmp_path),
    ):
        yield


class TestPRGateLeaseCreation:
    """Tests for lease creation."""

    def test_create_lease_basic(self, mock_git_dirs: None) -> None:
        """Creating a lease should write a valid lease file."""
        lease = create_lease(pr_number=5727, gate_id="test-gate", owner="test-user")

        assert lease.schema == "pr_gate_lease.v1"
        assert lease.pr_number == 5727
        assert lease.gate_id == "test-gate"
        assert lease.owner == "test-user"
        assert lease.created_at is not None
        assert lease.expires_at is not None
        assert lease.last_heartbeat is not None

        # Verify file was written
        lease_file = lease_path()
        assert lease_file.exists()

        data = json.loads(lease_file.read_text())
        assert data["schema"] == "pr_gate_lease.v1"
        assert data["pr_number"] == 5727

    def test_create_lease_default_ttl(self, mock_git_dirs: None) -> None:
        """Lease should expire after default TTL."""
        before = datetime.now(UTC)
        lease = create_lease()
        after = datetime.now(UTC)

        created = datetime.fromisoformat(lease.created_at)
        expires = datetime.fromisoformat(lease.expires_at)

        assert before <= created <= after
        # Default TTL is 2 hours
        expected_expiry = created + timedelta(hours=2)
        assert abs((expires - expected_expiry).total_seconds()) < 1

    def test_create_lease_custom_ttl(self, mock_git_dirs: None) -> None:
        """Lease should respect custom TTL."""
        lease = create_lease(ttl_hours=0.5)

        created = datetime.fromisoformat(lease.created_at)
        expires = datetime.fromisoformat(lease.expires_at)

        expected_expiry = created + timedelta(hours=0.5)
        assert abs((expires - expected_expiry).total_seconds()) < 1


class TestPRGateLeaseLoading:
    """Tests for lease loading."""

    def test_load_lease_exists(self, mock_git_dirs: None) -> None:
        """Loading should return the lease when it exists."""
        create_lease(pr_number=1234)
        lease = load_lease()

        assert lease is not None
        assert lease.pr_number == 1234

    def test_load_lease_not_exists(self, mock_git_dirs: None) -> None:
        """Loading should return None when no lease file exists."""
        lease = load_lease()
        assert lease is None


class TestPRGateLeaseHeartbeat:
    """Tests for lease heartbeat."""

    def test_heartbeat_refreshes_timestamp(self, mock_git_dirs: None) -> None:
        """Heartbeat should update last_heartbeat."""
        create_lease(pr_number=5727)
        time.sleep(0.1)

        before_heartbeat = datetime.now(UTC)
        lease = heartbeat()
        after_heartbeat = datetime.now(UTC)

        heartbeat_time = datetime.fromisoformat(lease.last_heartbeat)
        assert before_heartbeat <= heartbeat_time <= after_heartbeat

    def test_heartbeat_no_extend(self, mock_git_dirs: None) -> None:
        """Heartbeat without extend should keep original expiry."""
        lease = create_lease(pr_number=5727, ttl_hours=2)
        original_expiry = lease.expires_at

        time.sleep(0.1)
        refreshed = heartbeat()

        assert refreshed.expires_at == original_expiry

    def test_heartbeat_with_extend(self, mock_git_dirs: None) -> None:
        """Heartbeat with extend should update expiry."""
        create_lease(pr_number=5727, ttl_hours=0.1)
        before = datetime.now(UTC)

        lease = heartbeat(extend_hours=1.0)

        expected_expiry = before + timedelta(hours=1.0)
        actual_expiry = datetime.fromisoformat(lease.expires_at)

        # Allow some tolerance for execution time
        assert abs((actual_expiry - expected_expiry).total_seconds()) < 1

    def test_heartbeat_no_lease_raises(self, mock_git_dirs: None) -> None:
        """Heartbeat should raise when no lease exists."""
        with pytest.raises(RuntimeError, match="No active lease"):
            heartbeat()


class TestPRGateLeaseRelease:
    """Tests for lease release."""

    def test_release_removes_file(self, mock_git_dirs: None) -> None:
        """Release should delete the lease file."""
        create_lease(pr_number=5727)
        assert lease_path().exists()

        result = release_lease()

        assert result is True
        assert not lease_path().exists()

    def test_release_no_lease_returns_false(self, mock_git_dirs: None) -> None:
        """Release should return False when no lease exists."""
        result = release_lease()
        assert result is False


class TestPRGateLeaseExpiration:
    """Tests for lease expiration checking."""

    def test_is_active_true(self, mock_git_dirs: None) -> None:
        """is_active should return True for valid lease."""
        create_lease(pr_number=5727, ttl_hours=1.0)
        assert is_active() is True

    def test_is_active_false_no_lease(self, mock_git_dirs: None) -> None:
        """is_active should return False when no lease exists."""
        assert is_active() is False

    def test_is_active_false_expired(self, mock_git_dirs: None) -> None:
        """is_active should return False for expired lease."""
        # Create an already-expired lease by manipulating the file directly
        lease_file = lease_path()

        expired_time = datetime.now(UTC) - timedelta(hours=1)
        lease_data = {
            "schema": "pr_gate_lease.v1",
            "created_at": (expired_time - timedelta(hours=1)).isoformat(),
            "expires_at": expired_time.isoformat(),
            "pr_number": 5727,
            "gate_id": "test",
            "owner": "test",
            "last_heartbeat": expired_time.isoformat(),
        }
        lease_file.write_text(json.dumps(lease_data) + "\n")

        assert is_active() is False


class TestPRGateLeaseStatus:
    """Tests for lease status."""

    def test_status_no_lease(self, mock_git_dirs: None) -> None:
        """Status should report no lease when none exists."""
        stat = status()
        assert stat["active"] is False
        assert stat["reason"] == "no_lease"

    def test_status_active_lease(self, mock_git_dirs: None) -> None:
        """Status should report active lease details."""
        create_lease(pr_number=5727, gate_id="gate-1", owner="user-1")

        stat = status()
        assert stat["active"] is True
        assert "seconds_until_expiry" in stat
        assert stat["seconds_until_expiry"] > 0

        lease_data = stat["lease"]
        assert lease_data["pr_number"] == 5727
        assert lease_data["gate_id"] == "gate-1"
        assert lease_data["owner"] == "user-1"

    def test_status_expired_lease(self, mock_git_dirs: None) -> None:
        """Status should report expired lease."""
        lease_file = lease_path()

        expired_time = datetime.now(UTC) - timedelta(hours=1)
        lease_data = {
            "schema": "pr_gate_lease.v1",
            "created_at": (expired_time - timedelta(hours=1)).isoformat(),
            "expires_at": expired_time.isoformat(),
            "pr_number": 5727,
            "gate_id": "test",
            "owner": "test",
            "last_heartbeat": expired_time.isoformat(),
        }
        lease_file.write_text(json.dumps(lease_data) + "\n")

        stat = status()
        assert stat["active"] is False
        assert stat["reason"] == "expired"
        assert "expired_at" in stat


class TestPRGateLeaseSchema:
    """Tests for lease schema validation."""

    def test_lease_is_expired_method(self) -> None:
        """The is_expired method should work correctly."""
        now = datetime.now(UTC)

        # Future expiry - not expired
        lease_future = PRGateLease(
            schema="pr_gate_lease.v1",
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=1)).isoformat(),
            pr_number=5727,
            gate_id="test",
            owner="test",
            last_heartbeat=now.isoformat(),
        )
        assert lease_future.is_expired() is False

        # Past expiry - expired
        lease_past = PRGateLease(
            schema="pr_gate_lease.v1",
            created_at=now.isoformat(),
            expires_at=(now - timedelta(hours=1)).isoformat(),
            pr_number=5727,
            gate_id="test",
            owner="test",
            last_heartbeat=now.isoformat(),
        )
        assert lease_past.is_expired() is True

    def test_time_until_expiry_seconds(self) -> None:
        """time_until_expiry_seconds should return correct value."""
        now = datetime.now(UTC)

        lease = PRGateLease(
            schema="pr_gate_lease.v1",
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=1)).isoformat(),
            pr_number=5727,
            gate_id="test",
            owner="test",
            last_heartbeat=now.isoformat(),
        )

        seconds = lease.time_until_expiry_seconds()
        # Should be approximately 3600 seconds (1 hour)
        assert 3599 < seconds < 3601


class TestPRGateLeaseRobustnessAndIsolation:
    """Additional tests for invalid durations, atomic writes, robust parsing, and concurrent isolation."""

    def test_invalid_durations_rejected(self, mock_git_dirs: None) -> None:
        """Reject non-positive and non-finite lease durations."""

        for invalid_val in [0, -1, float("inf"), float("-inf"), float("nan"), "string", True]:
            with pytest.raises(ValueError, match="ttl_hours|extend_hours"):
                create_lease(ttl_hours=invalid_val)

        # Create a valid lease first
        create_lease(pr_number=5727, ttl_hours=2)
        for invalid_val in [0, -1, float("inf"), float("-inf"), float("nan"), "string", True]:
            with pytest.raises(ValueError, match="extend_hours|ttl_hours"):
                heartbeat(extend_hours=invalid_val)

    def test_robust_load_extra_or_missing_fields(self, mock_git_dirs: None) -> None:
        """Loading a lease file with extra fields should succeed, but missing fields should raise error."""
        lease_file = lease_path()

        # Extra fields should be ignored gracefully
        data = {
            "schema": "pr_gate_lease.v1",
            "created_at": datetime.now(UTC).isoformat(),
            "expires_at": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
            "pr_number": 5727,
            "gate_id": "test",
            "owner": "test",
            "last_heartbeat": datetime.now(UTC).isoformat(),
            "extra_field_123": "ignore_me",
        }
        lease_file.write_text(json.dumps(data) + "\n")
        loaded = load_lease()
        assert loaded is not None
        assert loaded.pr_number == 5727

        # Missing required fields should raise RuntimeError
        bad_data = {
            "schema": "pr_gate_lease.v1",
            "pr_number": 5727,
        }
        lease_file.write_text(json.dumps(bad_data) + "\n")
        with pytest.raises(RuntimeError, match="Invalid lease file"):
            load_lease()

    def test_atomic_lease_write(self, mock_git_dirs: None, monkeypatch) -> None:
        """Verify that saving a lease replaces the file atomically."""
        replaced_called = False
        original_replace = Path.replace

        def mock_replace(self, target):
            nonlocal replaced_called
            replaced_called = True
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", mock_replace)
        create_lease(pr_number=5727)
        assert replaced_called is True

    def test_concurrent_worktree_isolation(self, tmp_path: Path) -> None:
        """Linked worktrees should map to different lease paths based on worktree directory."""
        git_common = tmp_path / ".git"
        git_common.mkdir()

        wt1 = tmp_path / "wt1"
        wt2 = tmp_path / "wt2"
        wt1.mkdir()
        wt2.mkdir()

        with patch("scripts.dev.pr_gate_lease._git_common_dir", return_value=git_common):
            path1 = lease_path(wt1)
            path2 = lease_path(wt2)

            assert path1 != path2
            assert path1.parent == git_common
            assert path2.parent == git_common
            assert path1.name.startswith(".pr-gate-lease-")
            assert path2.name.startswith(".pr-gate-lease-")
