"""Additional branch coverage for release-doctor admission checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark import release_doctor


def _result(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    """Build a deterministic subprocess result fixture."""
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


@pytest.mark.parametrize(
    ("head", "status", "expected", "match"),
    [
        ("a" * 40, "", "a" * 40, "pass"),
        ("b" * 40, "", "a" * 40, "HEAD differs"),
        ("a" * 40, " M file.py", "a" * 40, "dirty"),
    ],
)
def test_git_check_requires_exact_clean_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    head: str,
    status: str,
    expected: str,
    match: str,
) -> None:
    """Doctor Git admission reports both source drift and dirty state."""
    calls = iter(
        [
            _result(["git", "rev-parse", "HEAD"], stdout=head),
            _result(["git", "status", "--porcelain"], stdout=status),
        ]
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: next(calls))
    check = release_doctor._git_check(tmp_path, expected)
    assert (check.status == "pass") is (match == "pass")
    if match != "pass":
        assert match in check.summary


def test_git_check_fails_when_git_commands_are_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A diagnostic command failure is not silently treated as clean state."""
    monkeypatch.setattr(
        release_doctor,
        "_run",
        lambda *args: _result([], returncode=1),
    )
    check = release_doctor._git_check(tmp_path, "a" * 40)
    assert check.status == "fail"
    assert "could not be inspected" in check.summary


@pytest.mark.parametrize(
    ("result", "expected_status", "summary"),
    [
        (_result([], returncode=1), "fail", "unavailable"),
        (_result([], stdout="not-json"), "fail", "missing"),
        (
            _result(
                [],
                stdout=json.dumps(
                    [
                        {
                            "headSha": "b" * 40,
                            "status": "completed",
                            "conclusion": "success",
                            "workflowName": "CI",
                        },
                        {
                            "headSha": "b" * 40,
                            "status": "completed",
                            "conclusion": "success",
                            "workflowName": "CodeQL",
                        },
                    ]
                ),
            ),
            "fail",
            "missing",
        ),
        (
            _result(
                [],
                stdout=json.dumps(
                    [
                        {
                            "headSha": "a" * 40,
                            "status": "completed",
                            "conclusion": "success",
                            "workflowName": "CI",
                        },
                        {
                            "headSha": "a" * 40,
                            "status": "completed",
                            "conclusion": "success",
                            "workflowName": "CodeQL",
                        },
                    ]
                ),
            ),
            "pass",
            "green",
        ),
    ],
)
def test_ci_check_requires_completed_success_for_exact_sha(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result: subprocess.CompletedProcess[str],
    expected_status: str,
    summary: str,
) -> None:
    """Only a completed successful CI run for the requested SHA admits release."""
    monkeypatch.setattr(release_doctor, "_run", lambda *args: result)
    check = release_doctor._ci_check(tmp_path, "a" * 40)
    assert check.status == expected_status
    assert summary in check.summary


@pytest.mark.parametrize(
    ("local_code", "remote_ref_code", "release_code", "expected"),
    [(0, 2, 1, "fail"), (1, 0, 1, "fail"), (1, 2, 0, "fail"), (1, 2, 1, "fail")],
)
def test_tag_check_rejects_local_or_remote_collisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    local_code: int,
    remote_ref_code: int,
    release_code: int,
    expected: str,
) -> None:
    """A planned release tag must be unused in both Git and GitHub."""
    calls = iter(
        [
            _result(["git"], returncode=local_code),
            _result(["git", "ls-remote"], returncode=remote_ref_code),
            _result(["gh"], returncode=release_code),
        ]
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: next(calls))
    check = release_doctor._tag_check(tmp_path, "tag")
    assert check.status == expected


def test_tag_check_accepts_explicit_github_release_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only an explicit GitHub not-found response means no release collision."""
    calls = iter(
        [
            _result(["git"], returncode=1),
            _result(["git", "ls-remote"], returncode=2),
            _result(["gh"], returncode=1, stderr="release not found"),
        ]
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: next(calls))
    check = release_doctor._tag_check(tmp_path, "tag")
    assert check.status == "pass"


@pytest.mark.parametrize(
    ("command_index", "returncode", "stderr", "summary"),
    [
        (0, 1, "fatal: not a git repository", "local tag state is unavailable"),
        (1, 2, "fatal: could not read from remote", "remote tag state is unavailable"),
        (2, 1, "authentication failed", "GitHub release state is unavailable"),
        (2, 1, "HTTP 404: Not Found", "GitHub release state is unavailable"),
    ],
)
def test_tag_check_fails_closed_on_ambiguous_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command_index: int,
    returncode: int,
    stderr: str,
    summary: str,
) -> None:
    """An error that resembles absence must not be treated as an unused tag."""
    results = [
        _result(["git"], returncode=1),
        _result(["git", "ls-remote"], returncode=2),
        _result(["gh"], returncode=1, stderr="release not found"),
    ]
    results[command_index] = _result(
        results[command_index].args, returncode=returncode, stderr=stderr
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: results.pop(0))
    check = release_doctor._tag_check(tmp_path, "tag")
    assert check.status == "fail"
    assert summary in check.summary


def test_manifest_check_reports_bad_path_and_wrong_cell_count(tmp_path: Path) -> None:
    """Manifest diagnostics fail safely for malformed paths and cardinality drift."""
    missing, manifest, cfg = release_doctor._manifest_check(tmp_path / "missing.yaml", 1)
    assert missing.status == "fail"
    assert manifest is None
    assert cfg is None

    check, manifest, cfg = release_doctor._manifest_check(
        Path(
            "configs/benchmarks/releases/"
            "paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml"
        ),
        1,
    )
    assert check.status == "fail"
    assert "20160" in check.summary
    assert manifest is not None
    assert cfg is not None


def test_checkpoint_check_reports_missing_and_validator_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Checkpoint admission remains blocked until a valid receipt is present."""
    missing = release_doctor._checkpoint_check(None, None, None)
    assert missing.status == "fail"
    assert "missing" in missing.summary

    manifest = SimpleNamespace(canonical_campaign_config_path=tmp_path / "campaign.yaml")
    monkeypatch.setattr(
        release_doctor,
        "validate_checkpoint_staging_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            release_doctor.CheckpointStagingReceiptError("receipt mismatch")
        ),
    )
    rejected = release_doctor._checkpoint_check(object(), manifest, tmp_path / "receipt.json")
    assert rejected.status == "fail"
    assert rejected.summary == "receipt mismatch"

    monkeypatch.setattr(
        release_doctor,
        "validate_checkpoint_staging_receipt",
        lambda *args, **kwargs: {"arms": [{"planner_key": "ppo"}]},
    )
    admitted = release_doctor._checkpoint_check(object(), manifest, tmp_path / "receipt.json")
    assert admitted.status == "pass"
    assert "1 checkpoint" in admitted.summary


def test_release_identity_check_reports_each_mismatch() -> None:
    """The final manifest must match schema, latest-main base, and tag."""
    rejected = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.1",
            latest_main_base_commit="b" * 40,
            release_tag="other",
        ),
        "a" * 40,
        "expected",
    )
    assert rejected.status == "fail"
    assert "v0.2" in rejected.summary
    assert "base commit" in rejected.summary
    assert "release tag" in rejected.summary

    admitted = release_doctor._release_identity_check(
        SimpleNamespace(
            schema_version="benchmark-release-manifest.v0.2",
            latest_main_base_commit="a" * 40,
            release_tag="expected",
        ),
        "a" * 40,
        "expected",
    )
    assert admitted.status == "pass"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_load_mapping_supports_json_and_yaml(tmp_path: Path, suffix: str) -> None:
    """Private launch packets may use either supported serialization."""
    path = tmp_path / f"packet{suffix}"
    if suffix == ".json":
        path.write_text('{"ok": true}\n', encoding="utf-8")
    else:
        path.write_text("ok: true\n", encoding="utf-8")
    assert release_doctor._load_mapping(path) == {"ok": True}


def test_load_mapping_rejects_non_mapping(tmp_path: Path) -> None:
    """Launch packets must deserialize to mappings."""
    path = tmp_path / "packet.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected mapping"):
        release_doctor._load_mapping(path)


@pytest.mark.parametrize(
    ("payload", "expected_status", "summary"),
    [
        ({"admission": {"status": "admitted"}, "dispatchable": True}, "fail", "source SHA"),
        ({"admission": {"status": "pending"}, "dispatchable": True}, "fail", "not admitted"),
    ],
)
def test_cluster_check_rejects_invalid_admission_or_identity(
    tmp_path: Path,
    payload: dict[str, object],
    expected_status: str,
    summary: str,
) -> None:
    """A packet without admission and source binding cannot dispatch."""
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    check = release_doctor._cluster_check(path, "a" * 40)
    assert check.status == expected_status
    assert summary in check.summary


def test_cluster_check_reports_missing_and_invalid_packet(tmp_path: Path) -> None:
    """Missing and malformed private packets fail without exposing contents."""
    missing = release_doctor._cluster_check(tmp_path / "missing.json", "a" * 40)
    assert missing.status == "fail"
    assert "missing" in missing.summary
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not-json", encoding="utf-8")
    invalid = release_doctor._cluster_check(invalid_path, "a" * 40)
    assert invalid.status == "fail"
    assert "invalid" in invalid.summary


@pytest.mark.parametrize("minimum", [0.0, 10_000_000.0])
def test_disk_check_applies_free_space_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, minimum: float
) -> None:
    """Artifact capacity admission compares free bytes with the configured threshold."""
    monkeypatch.setattr(
        release_doctor.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=2 * 1024**3),
    )
    check = release_doctor._disk_check(tmp_path, minimum)
    assert check.status == ("pass" if minimum <= 2.0 else "fail")
    assert "GiB free" in check.summary


class _AuthResponse:
    """Minimal successful auth response."""

    def raise_for_status(self) -> None:
        """Accept the request."""


class _AuthSession:
    """Minimal authenticated Zenodo session."""

    def get(self, *args, **kwargs) -> _AuthResponse:
        """Return a successful response."""
        return _AuthResponse()


@pytest.mark.parametrize(
    ("hook_result", "require_disabled", "expected_status", "summary"),
    [
        (_result([], returncode=1), False, "fail", "unavailable"),
        (_result([], stdout="not-json"), False, "fail", "invalid"),
        (_result([], stdout="[]"), False, "pass", "not found"),
        (_result([], stdout="[]"), True, "fail", "not found"),
        (
            _result([], stdout=json.dumps([{"active": False, "config": {"url": "zenodo"}}])),
            True,
            "pass",
            "disabled",
        ),
    ],
)
def test_zenodo_check_sanitizes_auth_and_hook_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hook_result: subprocess.CompletedProcess[str],
    require_disabled: bool,
    expected_status: str,
    summary: str,
) -> None:
    """Doctor reports hook/auth state without returning private hook configuration."""
    monkeypatch.setattr(release_doctor, "read_token_file", lambda path: "secret")
    monkeypatch.setattr(release_doctor, "build_session", lambda path: _AuthSession())
    monkeypatch.setattr(release_doctor, "_run", lambda *args: hook_result)
    checks = release_doctor._zenodo_check(
        tmp_path,
        tmp_path / "token",
        require_hook_disabled=require_disabled,
    )
    hook_check = checks[-1]
    assert hook_check.status == expected_status
    assert summary in hook_check.summary
    assert "secret" not in json.dumps([check.summary for check in checks])


def test_zenodo_check_reports_auth_failure_without_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unavailable credentials are summarized, never echoed."""
    monkeypatch.setattr(
        release_doctor,
        "read_token_file",
        lambda path: (_ for _ in ()).throw(RuntimeError("private token")),
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: _result([], stdout="[]"))
    checks = release_doctor._zenodo_check(tmp_path, tmp_path / "token", require_hook_disabled=False)
    assert checks[0].status == "fail"
    assert "private token" not in json.dumps([check.summary for check in checks])


def _make_dissertation(path: Path, *, stale: bool = False) -> None:
    """Create the minimum dissertation release path fixture."""
    for relative in (
        "diss/robot_sf_release.tex",
        "docs/context/evidence_pins.yaml",
        "spine/evidence_release.yaml",
    ):
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "/Users/lennart/git/robot_sf_ll7\n" if stale else "healthy\n",
            encoding="utf-8",
        )


def test_dissertation_check_reports_missing_healthy_and_stale_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dissertation health checks cover required files and configurable paths."""
    missing = release_doctor._dissertation_check(None)
    assert missing.status == "fail"

    healthy = tmp_path / "healthy"
    _make_dissertation(healthy)
    monkeypatch.setattr(release_doctor, "_run", lambda *args: _result([], returncode=1))
    assert release_doctor._dissertation_check(healthy).status == "pass"

    stale = tmp_path / "stale"
    _make_dissertation(stale, stale=True)
    monkeypatch.setattr(
        release_doctor,
        "_run",
        lambda *args: _result(["rg"], stdout="./diss/robot_sf_release.tex"),
    )
    rejected = release_doctor._dissertation_check(stale)
    assert rejected.status == "fail"
    assert "hard-coded" in rejected.summary


def test_dissertation_check_allows_repository_urls_and_relative_paths(tmp_path: Path) -> None:
    """The path health check rejects local checkouts, not public URLs or names."""
    healthy = tmp_path / "healthy"
    _make_dissertation(healthy)
    (healthy / "docs" / "links.md").write_text(
        "See https://github.com/ll7/robot_sf_ll7 and robot_sf_ll7/configs.\n",
        encoding="utf-8",
    )
    assert release_doctor._dissertation_check(healthy).status == "pass"

    stale = tmp_path / "stale"
    _make_dissertation(stale)
    (stale / "docs" / "local.md").write_text(
        "\n\t/scratch/luttkule/projects/robot_sf_ll7/configs\n",
        encoding="utf-8",
    )
    assert release_doctor._dissertation_check(stale).status == "fail"


def test_ci_check_rejects_pending_codeql_even_when_ci_is_green(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pending required workflow cannot be hidden by a green aggregate CI run."""
    result = _result(
        [],
        stdout=json.dumps(
            [
                {
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "workflowName": "CI",
                },
                {
                    "headSha": "a" * 40,
                    "status": "in_progress",
                    "conclusion": "",
                    "workflowName": "CodeQL",
                },
            ]
        ),
    )
    monkeypatch.setattr(release_doctor, "_run", lambda *args: result)
    check = release_doctor._ci_check(tmp_path, "a" * 40)
    assert check.status == "fail"
    assert "CodeQL" in check.summary
