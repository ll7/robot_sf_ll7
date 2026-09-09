"""Offline tests for the REST-only label helper (issue #6266)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.gh_pr_label_rest import (
    LABEL_PAGE_CEILING,
    LABEL_PAGE_SIZE,
    _get_label_names,
    add_label,
    check_merge_ready_carriers,
    get_label_names,
    main,
    remove_label,
    validate_result_envelope,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENTRYPOINT = _REPO_ROOT / "scripts/dev/gh_pr_label_rest.py"
_LABEL_GET_ARGS = [
    "api",
    "repos/ll7/robot_sf_ll7/issues/5220/labels?per_page=100&page=1",
]


@pytest.fixture
def offline_cli(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Expose only a read-only fake gh, without credentials or ambient Python paths."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "gh-calls.jsonl"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['LABEL_FAKE_LOG'], 'a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:] != json.loads(os.environ['LABEL_FAKE_EXPECTED_ARGS']):\n"
        "    sys.exit('unexpected transport request; only the fixture GET is allowed')\n"
        "print(os.environ['LABEL_FAKE_RESPONSE'])\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    return {
        "PATH": str(fake_bin),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LABEL_FAKE_LOG": str(log),
        "LABEL_FAKE_EXPECTED_ARGS": json.dumps(_LABEL_GET_ARGS),
        "LABEL_FAKE_RESPONSE": json.dumps([{"name": "state:ready"}, {"name": "technical-debt"}]),
    }, log


def _run_cli(
    args: list[str], *, cwd: Path, env: dict[str, str], module: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run the actual entrypoint without site packages; isolate direct execution fully."""
    entry = ["-m", "scripts.dev.gh_pr_label_rest"] if module else ["-I", str(_ENTRYPOINT)]
    return subprocess.run(
        [sys.executable, "-S", "-B", *entry, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


@pytest.mark.parametrize("from_root", [True, False])
def test_isolated_direct_help_makes_no_transport_call(
    tmp_path: Path, offline_cli: tuple[dict[str, str], Path], from_root: bool
) -> None:
    """The documented script starts without editable imports from either directory."""
    env, log = offline_cli
    result = _run_cli(["--help"], cwd=_REPO_ROOT if from_root else tmp_path, env=env)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not result.stderr
    assert not log.exists()


@pytest.mark.parametrize("response", [None, "not-json", '{"name":"not-a-list"}'])
def test_isolated_direct_and_module_list_parity(
    tmp_path: Path, offline_cli: tuple[dict[str, str], Path], response: str | None
) -> None:
    """Real direct/module startup shares read-only JSON and fail-closed transport semantics."""
    env, log = offline_cli
    if response is not None:
        env["LABEL_FAKE_RESPONSE"] = response
    results = [
        _run_cli(["list", "5220"], cwd=_REPO_ROOT, env=env),
        _run_cli(["list", "5220"], cwd=tmp_path, env=env),
        _run_cli(["list", "5220"], cwd=_REPO_ROOT, env=env, module=True),
    ]
    expected_status = 0 if response is None else 1
    for result in results:
        assert result.returncode == expected_status, result.stderr
        assert (result.stdout, result.stderr) == (results[0].stdout, results[0].stderr)
        payload = json.loads(result.stdout if expected_status == 0 else result.stderr)
        if response is None:
            assert payload == {
                "status": "ok",
                "number": 5220,
                "action": "list",
                "repo": "ll7/robot_sf_ll7",
                "labels": ["state:ready", "technical-debt"],
            }
        else:
            assert payload["status"] == "error"
            assert "labels" not in payload
            assert not result.stdout
    assert [json.loads(line) for line in log.read_text().splitlines()] == [_LABEL_GET_ARGS] * 3


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("args", [["add", "5220"], ["list", "not-a-number"]])
def test_isolated_cli_invalid_arguments_never_reach_transport(
    offline_cli: tuple[dict[str, str], Path], module: bool, args: list[str]
) -> None:
    """Both entrypoints reject invalid arguments before any GitHub operation."""
    env, log = offline_cli
    result = _run_cli(args, cwd=_REPO_ROOT, env=env, module=module)
    assert result.returncode == 2, result.stderr
    assert "error:" in result.stderr
    assert not result.stdout
    assert not log.exists()


@pytest.mark.parametrize("via_symlink", [False, True])
def test_direct_entrypoint_rejects_hostile_checkout_shadowing(
    tmp_path: Path, offline_cli: tuple[dict[str, str], Path], via_symlink: bool
) -> None:
    """The resolved invoking checkout wins even when already later on PYTHONPATH."""
    env, log = offline_cli
    shadow = tmp_path / "hostile-checkout"
    package = shadow / "scripts" / "dev"
    package.mkdir(parents=True)
    (shadow / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in ("_gh_rest", "github_transport_policy", "pr_carrier_gate", "pr_write_guard"):
        (package / f"{name}.py").write_text(
            "raise RuntimeError('hostile checkout imported')\n", encoding="utf-8"
        )
    env["PYTHONPATH"] = os.pathsep.join((str(shadow), str(_REPO_ROOT)))
    env["PYTHONSAFEPATH"] = "1"
    entrypoint = _ENTRYPOINT
    if via_symlink:
        entrypoint = tmp_path / "label-helper.py"
        entrypoint.symlink_to(_ENTRYPOINT)
    result = subprocess.run(
        [sys.executable, "-S", "-B", str(entrypoint), "list", "5220"],
        cwd=shadow,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["labels"] == ["state:ready", "technical-debt"]
    assert not result.stderr
    assert [json.loads(line) for line in log.read_text().splitlines()] == [_LABEL_GET_ARGS]


def test_isolated_merge_ready_missing_carrier_dependency_prevents_post(
    tmp_path: Path, offline_cli: tuple[dict[str, str], Path]
) -> None:
    """A real no-site invocation cannot write when the canonical carrier import fails."""
    env, log = offline_cli
    head, base = "a" * 40, "b" * 40
    read_args = ["api", "repos/ll7/robot_sf_ll7/pulls/5220"]
    env["LABEL_FAKE_EXPECTED_ARGS"] = json.dumps(read_args)
    env["LABEL_FAKE_RESPONSE"] = json.dumps(
        {"state": "open", "head": {"sha": head}, "base": {"sha": base}, "merged_at": None}
    )
    env["ROBOT_SF_PR_WRITE_LOCK_DIR"] = str(tmp_path / "write-locks")
    result = _run_cli(
        [
            "add",
            "5220",
            "--label",
            "merge-ready",
            "--expected-head-sha",
            head,
            "--expected-base-sha",
            base,
        ],
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 1, result.stderr
    assert not result.stdout
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert "carrier" in payload["error"]
    assert "No module named 'yaml'" in payload["error"]
    assert [json.loads(line) for line in log.read_text().splitlines()] == [read_args]


@pytest.mark.parametrize("status", ["ok", "error"])
def test_lazy_carrier_checker_forwards_exact_arguments_and_result(status: str) -> None:
    """Delay only import timing; retain the canonical guard's inputs and verdict verbatim."""
    verdict = {"status": status, "reason": "canonical verdict"}
    with patch(
        "scripts.dev.pr_carrier_gate.check_merge_ready_carriers", return_value=verdict
    ) as checker:
        result = check_merge_ready_carriers(5220, repo="owner/repo", live_head="a", live_base="b")
    assert result is verdict
    checker.assert_called_once_with(5220, repo="owner/repo", live_head="a", live_base="b")


def test_module_import_preserves_sys_path_without_loading_carrier_dependency(
    offline_cli: tuple[dict[str, str], Path],
) -> None:
    """Normal imports must not run the direct-only bootstrap or load write-only dependencies."""
    env, log = offline_cli
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            "-c",
            "import sys; before = list(sys.path); "
            "import scripts.dev.gh_pr_label_rest; "
            "assert sys.path == before; "
            "assert 'scripts.dev.pr_carrier_gate' not in sys.modules",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert not log.exists()


def _proc(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake ``subprocess.CompletedProcess`` for ``gh api``."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def _mock_labels_payload(*names: str) -> str:
    """Build a JSON labels-array payload from label names."""
    return json.dumps([{"name": n} for n in names])


def _page_path(page: int) -> str:
    return f"repos/ll7/robot_sf_ll7/issues/5220/labels?per_page=100&page={page}"


def test_validate_result_envelope_rejects_untrusted_success_payloads() -> None:
    """Orchestrators must not trust a zero exit code without the exact result envelope."""
    valid_list = {
        "status": "ok",
        "action": "list",
        "number": 5220,
        "repo": "ll7/robot_sf_ll7",
        "labels": ["state:ready"],
    }
    invalid_results = [
        {},
        {**valid_list, "action": "remove"},
        {**valid_list, "number": 5221},
        {**valid_list, "repo": "other/repo"},
        {**valid_list, "labels": ["state:ready", "state:ready"]},
        {**valid_list, "labels": [""]},
        {**valid_list, "labels": ["state:ready\nfoo"]},
    ]

    for result in invalid_results:
        with pytest.raises(ValueError):
            validate_result_envelope(
                result,
                action="list",
                number=5220,
                repo="ll7/robot_sf_ll7",
            )

    with pytest.raises(ValueError):
        validate_result_envelope(
            {"status": "ok", "action": "remove", "number": 5220, "repo": "ll7/robot_sf_ll7"},
            action="remove",
            number=5220,
            repo="ll7/robot_sf_ll7",
            label="state:ready",
        )

    with pytest.raises(ValueError, match="printable"):
        validate_result_envelope(
            {
                "status": "ok",
                "action": "remove",
                "number": 5220,
                "repo": "ll7/robot_sf_ll7",
                "label": "state:ready\nfoo",
            },
            action="remove",
            number=5220,
            repo="ll7/robot_sf_ll7",
            label="state:ready\nfoo",
        )


class TestLabelRead:
    """Tests for complete and strict paginated label reads."""

    def test_finds_label_on_second_page(self) -> None:
        page_one = [{"name": f"label-{index}"} for index in range(LABEL_PAGE_SIZE)]
        with patch("scripts.dev.gh_pr_label_rest._gh_api_get") as mock_get:
            mock_get.side_effect = [
                _proc(stdout=json.dumps(page_one)),
                _proc(stdout=_mock_labels_payload("target")),
            ]
            result = _get_label_names(5220, repo="ll7/robot_sf_ll7")

        assert result == {
            "status": "ok",
            "labels": [f"label-{i}" for i in range(100)] + ["target"],
        }
        assert [call.args[0] for call in mock_get.call_args_list] == [
            _page_path(1),
            _page_path(2),
        ]

    def test_rejects_malformed_page_and_row(self) -> None:
        for payload in (
            {"name": "not-a-page"},
            [{"name": ""}],
            [{"name": None}],
            [{"name": 42}],
            [{"name": "state:ready\nfoo"}],
            ["not-a-row"],
        ):
            with patch(
                "scripts.dev.gh_pr_label_rest._gh_api_get",
                return_value=_proc(stdout=json.dumps(payload)),
            ):
                result = _get_label_names(5220)

            assert result["status"] == "error"
            assert "page 1" in result["error"]
            assert "labels" not in result

    def test_fails_closed_when_page_fetch_fails(self) -> None:
        page_one = [{"name": f"label-{index}"} for index in range(LABEL_PAGE_SIZE)]
        with patch(
            "scripts.dev.gh_pr_label_rest._gh_api_get",
        ) as mock_get:
            mock_get.side_effect = [
                _proc(stdout=json.dumps(page_one)),
                _proc(returncode=1, stderr="HTTP 503: unavailable"),
            ]
            result = _get_label_names(5220)

        assert result == {
            "status": "error",
            "error": "could not read labels page 2: HTTP 503: unavailable",
        }

    def test_fails_closed_at_page_ceiling(self) -> None:
        full_pages = [
            _mock_labels_payload(
                *[f"label-{page * LABEL_PAGE_SIZE + index}" for index in range(LABEL_PAGE_SIZE)]
            )
            for page in range(LABEL_PAGE_CEILING)
        ]
        with patch(
            "scripts.dev.gh_pr_label_rest._gh_api_get",
            side_effect=[_proc(stdout=page) for page in full_pages],
        ) as mock_get:
            result = _get_label_names(5220)

        assert result["status"] == "error"
        assert str(LABEL_PAGE_CEILING) in result["error"]
        assert "labels" not in result
        assert mock_get.call_count == LABEL_PAGE_CEILING

    def test_public_read_api_rejects_nonpositive_number(self) -> None:
        with patch("scripts.dev.gh_pr_label_rest._gh_api_get") as mock_get:
            result = get_label_names(0)

        assert result["status"] == "error"
        assert "must be positive" in result["error"]
        mock_get.assert_not_called()


class TestAddLabel:
    """Tests for the add_label helper function."""

    def test_merge_ready_requires_matching_open_head(self) -> None:
        """The merge-ready write performs the exact-head preflight first."""
        head_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
        base_sha = "b1c2d3e4f5061728394a5b6c7d8e9f0011121314"
        with (
            patch(
                "scripts.dev.gh_pr_label_rest.guard_pr_write",
                return_value={
                    "status": "ok",
                    "observed_head_sha": head_sha,
                    "observed_base_sha": base_sha,
                },
            ) as mock_guard,
            patch(
                "scripts.dev.gh_pr_label_rest.check_merge_ready_carriers",
                return_value={"status": "ok"},
            ) as mock_carriers,
            patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _proc(stdout=json.dumps({"name": "merge-ready"})),
                _proc(stdout=_mock_labels_payload("merge-ready")),
            ]
            result = add_label(
                5220,
                "merge-ready",
                repo="ll7/robot_sf_ll7",
                expected_head_sha=head_sha,
                expected_base_sha=base_sha,
            )

        assert result["status"] == "ok"
        mock_guard.assert_called_once_with(
            5220,
            repo="ll7/robot_sf_ll7",
            expected_head_sha=head_sha,
            expected_base_sha=base_sha,
            operation="merge_ready_label",
        )
        mock_carriers.assert_called_once_with(
            5220,
            repo="ll7/robot_sf_ll7",
            live_head=head_sha,
            live_base=base_sha,
        )

    def test_merge_ready_withholds_write_when_carrier_gate_fails(self) -> None:
        """A carrier blocker, including an active review worker, blocks the label write."""
        head_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
        with (
            patch(
                "scripts.dev.gh_pr_label_rest.guard_pr_write",
                return_value={
                    "status": "ok",
                    "observed_head_sha": head_sha,
                    "observed_base_sha": "b1c2d3e4f5061728394a5b6c7d8e9f0011121314",
                },
            ),
            patch(
                "scripts.dev.gh_pr_label_rest.check_merge_ready_carriers",
                return_value={
                    "status": "error",
                    "error": "active exact-head review claim 'lane-a' covers the live head; "
                    "review worker is still running and merge-ready must be withheld",
                },
            ),
            patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post,
        ):
            result = add_label(
                5220,
                "merge-ready",
                expected_head_sha=head_sha,
            )

        assert result["status"] == "error"
        assert "review worker is still running" in result["error"]
        mock_post.assert_not_called()

    def test_merge_ready_stale_state_skips_post(self) -> None:
        """A merged or moved PR must not receive a merge-ready label write."""
        stale = {
            "status": "review_skipped_stale_state",
            "reason": "pr_not_open",
            "observed_state": "MERGED",
        }
        with (
            patch("scripts.dev.gh_pr_label_rest.guard_pr_write", return_value=stale),
            patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post,
        ):
            result = add_label(
                5220,
                "merge-ready",
                expected_head_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9001020304",
            )

        assert result == stale
        mock_post.assert_not_called()

    def test_adds_label_via_rest_endpoint_and_verifies(self) -> None:
        """The helper must POST JSON labels[] and verify via re-read."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=json.dumps({"name": "state:running"})),
                _proc(stdout=_mock_labels_payload("state:running", "bug")),
            ]
            result = add_label(5220, "state:running", repo="ll7/robot_sf_ll7")

        assert result == {
            "status": "ok",
            "number": 5220,
            "label": "state:running",
            "action": "add",
            "repo": "ll7/robot_sf_ll7",
        }
        # First call: POST to add the label
        assert mock_run.call_args_list[0].args[0] == [
            "gh",
            "api",
            "--method",
            "POST",
            "repos/ll7/robot_sf_ll7/issues/5220/labels",
            "--input",
            "-",
        ]
        assert json.loads(mock_run.call_args_list[0].kwargs["input"]) == {
            "labels": ["state:running"]
        }
        # Second call: GET to verify
        assert mock_run.call_args_list[1].args[0] == [
            "gh",
            "api",
            _page_path(1),
        ]

    def test_fails_closed_on_authentication_error(self) -> None:
        """Auth failures must be surfaced in the result."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post:
            mock_post.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
            result = add_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "Bad credentials" in result["error"]

    def test_fails_closed_on_timeout(self) -> None:
        """A timeout must remain a structured error rather than escaping."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["gh", "api"], timeout=30)
            result = add_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "timed out" in result["error"]
        assert "not verified" in result["error"]

    def test_fails_closed_when_post_write_verification_fails(self) -> None:
        """A successful POST is insufficient when the re-read lacks the label."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=json.dumps({"name": "cheap-lane"})),
                _proc(stdout=_mock_labels_payload("bug")),
            ]
            result = add_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "was not found in labels after add" in result["error"]

    def test_fails_closed_when_post_write_readback_has_duplicate_labels(self) -> None:
        """A duplicate label row makes add verification indeterminate."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=json.dumps({"name": "cheap-lane"})),
                _proc(stdout=_mock_labels_payload("cheap-lane", "cheap-lane")),
            ]
            result = add_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "duplicate label" in result["error"]

    def test_fails_closed_for_negative_number(self) -> None:
        """Zero or negative numbers must be rejected without network calls."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post:
            result = add_label(0, "cheap-lane")

        assert result["status"] == "error"
        assert "must be positive" in result["error"]
        mock_post.assert_not_called()

    def test_fails_closed_for_empty_label(self) -> None:
        """Empty label strings must be rejected."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post:
            result = add_label(5220, "")

        assert result["status"] == "error"
        assert "non-empty" in result["error"]
        mock_post.assert_not_called()

    @pytest.mark.parametrize("label", ["state:ready\nfoo", "state:\tready", "state:\x00ready"])
    def test_fails_closed_for_non_printable_label(self, label: str) -> None:
        """Control characters must be rejected before an add request is sent."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post:
            result = add_label(5220, label)

        assert result["status"] == "error"
        assert "printable" in result["error"]
        mock_post.assert_not_called()


class TestRemoveLabel:
    """Tests for the remove_label helper function."""

    def test_removes_label_via_rest_endpoint_and_verifies(self) -> None:
        """The helper must DELETE the label endpoint and verify via re-read."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=""),
                _proc(stdout=_mock_labels_payload("bug")),
            ]
            result = remove_label(5220, "state:running", repo="ll7/robot_sf_ll7")

        assert result == {
            "status": "ok",
            "number": 5220,
            "label": "state:running",
            "action": "remove",
            "repo": "ll7/robot_sf_ll7",
        }
        # First call: DELETE the label
        assert mock_run.call_args_list[0].args[0] == [
            "gh",
            "api",
            "--method",
            "DELETE",
            "repos/ll7/robot_sf_ll7/issues/5220/labels/state%3Arunning",
        ]
        # Second call: GET to verify
        assert mock_run.call_args_list[1].args[0] == [
            "gh",
            "api",
            _page_path(1),
        ]

    def test_fails_closed_on_authentication_error(self) -> None:
        """Auth failures must be surfaced in the result."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_delete") as mock_del:
            mock_del.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
            result = remove_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "Bad credentials" in result["error"]

    def test_treats_concurrent_absent_label_delete_as_idempotent(self) -> None:
        """A verified already-absent label is a successful remove outcome."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(returncode=1, stderr="gh: Label does not exist (HTTP 404)"),
                _proc(stdout=_mock_labels_payload("bug")),
            ]
            result = remove_label(5220, "state:running", repo="ll7/robot_sf_ll7")

        assert result == {
            "status": "ok",
            "number": 5220,
            "label": "state:running",
            "action": "remove",
            "repo": "ll7/robot_sf_ll7",
            "idempotent": True,
        }

    def test_fails_closed_when_absent_delete_readback_still_has_label(self) -> None:
        """The narrow 404 is not success when authoritative readback disagrees."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(returncode=1, stderr="gh: Label does not exist (HTTP 404)"),
                _proc(stdout=_mock_labels_payload("state:running", "bug")),
            ]
            result = remove_label(5220, "state:running")

        assert result["status"] == "error"
        assert "was still found" in result["error"]

    def test_fails_closed_on_unrelated_not_found_error(self) -> None:
        """A generic 404 must not be mistaken for the absent-label race."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_delete") as mock_del:
            mock_del.return_value = _proc(returncode=1, stderr="gh: Not Found (HTTP 404)")
            result = remove_label(5220, "state:running")

        assert result["status"] == "error"
        assert "Not Found" in result["error"]

    def test_fails_closed_when_absent_delete_readback_fails(self) -> None:
        """An idempotent response still requires a successful labels readback."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(returncode=1, stderr="gh: Label does not exist (HTTP 404)"),
                _proc(returncode=1, stderr="HTTP 401: Bad credentials"),
            ]
            result = remove_label(5220, "state:running")

        assert result["status"] == "error"
        assert "could not read labels" in result["error"]

    def test_fails_closed_when_post_write_verification_fails(self) -> None:
        """A successful DELETE is insufficient when the re-read still has the label."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=""),
                _proc(stdout=_mock_labels_payload("cheap-lane", "bug")),
            ]
            result = remove_label(5220, "cheap-lane")

        assert result["status"] == "error"
        assert "was still found" in result["error"]

    def test_fails_closed_when_remove_readback_has_duplicate_labels(self) -> None:
        """A duplicate label row makes remove verification indeterminate."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=""),
                _proc(stdout=_mock_labels_payload("bug", "bug")),
            ]
            result = remove_label(5220, "state:running")

        assert result["status"] == "error"
        assert "duplicate label" in result["error"]

    def test_fails_closed_for_negative_number(self) -> None:
        """Zero or negative numbers must be rejected without network calls."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_delete") as mock_del:
            result = remove_label(0, "cheap-lane")

        assert result["status"] == "error"
        assert "must be positive" in result["error"]
        mock_del.assert_not_called()

    @pytest.mark.parametrize("label", ["state:ready\nfoo", "state:\tready", "state:\x00ready"])
    def test_fails_closed_for_non_printable_label(self, label: str) -> None:
        """Control characters must be rejected before a remove request is sent."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_delete") as mock_delete:
            result = remove_label(5220, label)

        assert result["status"] == "error"
        assert "printable" in result["error"]
        mock_delete.assert_not_called()


class TestCli:
    """Tests for the CLI entry point."""

    def test_cli_list_prints_compact_label_inventory(self, capsys) -> None:
        """The list command exposes the strict REST read for shell workflows."""
        with patch(
            "scripts.dev.gh_pr_label_rest._gh_api_get",
            return_value=_proc(stdout=_mock_labels_payload("state:ready", "bug")),
        ):
            rc = main(["list", "5220", "--repo", "ll7/robot_sf_ll7"])

        captured = capsys.readouterr()
        assert rc == 0
        assert json.loads(captured.out) == {
            "action": "list",
            "labels": ["state:ready", "bug"],
            "number": 5220,
            "repo": "ll7/robot_sf_ll7",
            "status": "ok",
        }

    def test_cli_list_prints_read_error_to_stderr(self, capsys) -> None:
        """A failed label read remains an observable nonzero CLI result."""
        with patch(
            "scripts.dev.gh_pr_label_rest._gh_api_get",
            return_value=_proc(returncode=1, stderr="HTTP 403: forbidden"),
        ):
            rc = main(["list", "5220"])

        captured = capsys.readouterr()
        assert rc == 1
        payload = json.loads(captured.err)
        assert payload["status"] == "error"
        assert "forbidden" in payload["error"]

    def test_cli_add_prints_compact_success_json(self) -> None:
        """The command-line contract is a single machine-readable success result."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=json.dumps({"name": "cheap-lane"})),
                _proc(stdout=_mock_labels_payload("cheap-lane", "bug")),
            ]
            rc = main(["add", "5220", "--label", "cheap-lane", "--repo", "ll7/robot_sf_ll7"])

        assert rc == 0

    def test_cli_add_prints_error_json_to_stderr_on_failure(self, capsys) -> None:
        """A failed add must print JSON to stderr and exit 1."""
        with patch("scripts.dev.gh_pr_label_rest._gh_api_post") as mock_post:
            mock_post.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
            rc = main(["add", "5220", "--label", "cheap-lane"])

        captured = capsys.readouterr()
        assert rc == 1
        payload = json.loads(captured.err)
        assert payload["status"] == "error"
        assert "Bad credentials" in payload["error"]

    def test_cli_remove_prints_compact_success_json(self) -> None:
        """The CLI must also succeed for remove."""
        with patch("scripts.dev.gh_pr_label_rest.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=""),
                _proc(stdout=_mock_labels_payload("bug")),
            ]
            rc = main(["remove", "5220", "--label", "cheap-lane", "--repo", "ll7/robot_sf_ll7"])

        assert rc == 0
