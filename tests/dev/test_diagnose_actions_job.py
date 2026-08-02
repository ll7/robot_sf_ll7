"""Tests for the missing-GitHub-Actions-log annotation fallback."""

from __future__ import annotations

import json
import subprocess

from scripts.dev import diagnose_actions_job


def _result(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    """Build a compact mocked ``gh`` result."""
    return subprocess.CompletedProcess(
        args=["gh"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _include_page(body: str, *, next_url: str | None = None) -> str:
    """Build a mocked ``gh api --include`` response mirroring real gh output.

    gh prints the HTTP status line with a trailing LF, the header lines and the
    blank header/body separator with CRLF, then the JSON body.
    """
    header_lines = ["HTTP/2.0 200 OK"]
    if next_url is not None:
        header_lines.append(f'Link: <{next_url}>; rel="next"')
    header_lines.append("Content-Type: application/json; charset=utf-8")
    status_line = header_lines[0] + "\n"
    rest_headers = "".join(line + "\r\n" for line in header_lines[1:])
    return status_line + rest_headers + "\r\n" + body


def test_split_include_output_separates_headers_and_body() -> None:
    """The status/headers and JSON body are split at the blank line."""
    stdout = _include_page('[{"message": "x"}]')
    headers, body = diagnose_actions_job._split_include_output(stdout)
    assert "HTTP/2.0 200 OK" in headers
    assert body == '[{"message": "x"}]'


def test_next_link_extracts_rel_next_url() -> None:
    """Only the ``rel="next"`` entry of a Link header is followed."""
    headers_block = (
        'Link: <https://api.github.com/r/c/1/annotations?page=2>; rel="next", '
        '<https://api.github.com/r/c/1/annotations?page=3>; rel="last"'
    )
    assert (
        diagnose_actions_job._next_link(headers_block)
        == "https://api.github.com/r/c/1/annotations?page=2"
    )


def test_next_link_returns_none_without_a_next_page() -> None:
    """A Link header with only a ``rel="last"`` (or no Link) has no next URL."""
    headers_block = 'Link: <https://api.github.com/r/c/1/annotations?page=3>; rel="last"'
    assert diagnose_actions_job._next_link(headers_block) is None
    assert diagnose_actions_job._next_link("") is None


def test_annotations_path_requires_a_github_check_run_url() -> None:
    """Only GitHub API check-run URLs may become annotation endpoints."""
    assert (
        diagnose_actions_job._annotations_path(
            "https://api.github.com/repos/ll7/robot_sf_ll7/check-runs/123",
        )
        == "repos/ll7/robot_sf_ll7/check-runs/123/annotations?per_page=100"
    )
    assert diagnose_actions_job._annotations_path("https://example.test/check-runs/123") is None
    assert diagnose_actions_job._annotations_path(None) is None


def test_main_prints_normal_logs_without_requesting_annotations(monkeypatch, capsys) -> None:
    """A usable normal log remains the preferred diagnostic output."""
    calls: list[list[str]] = []
    results = iter(
        [
            _result(0, json.dumps({"run_id": 456, "check_run_url": "unused"})),
            _result(0, "unit test output\n"),
        ]
    )

    def fake_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return next(results)

    monkeypatch.setattr(diagnose_actions_job, "_gh", fake_gh)

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 0
    assert capsys.readouterr().out == "unit test output\n"
    assert len(calls) == 2
    assert calls[1][:3] == ["run", "view", "456"]


def test_main_falls_back_to_check_run_annotations_when_logs_are_absent(monkeypatch, capsys) -> None:
    """An unavailable job log should expose GitHub's retained error annotation."""
    calls: list[list[str]] = []
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        "run_id": 456,
                        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
                    }
                ),
            ),
            _result(1, stderr="HTTP 404: Not Found"),
            _result(0, _include_page(json.dumps([{"message": "No space left on device"}]))),
        ]
    )

    def fake_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return next(results)

    monkeypatch.setattr(diagnose_actions_job, "_gh", fake_gh)

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == [{"message": "No space left on device"}]
    assert "Normal log retrieval unavailable" in captured.err
    assert "Falling back to check-run annotations." in captured.err
    # the annotation request is a single --include page (one JSON array)
    assert calls[2] == [
        "api",
        "--include",
        "repos/owner/repo/check-runs/789/annotations?per_page=100",
    ]


def test_main_falls_back_across_multiple_annotation_pages(monkeypatch, capsys) -> None:
    """The fallback follows the ``rel="next"`` Link URL across pages."""
    calls: list[list[str]] = []
    next_url = (
        "https://api.github.com/repos/owner/repo/check-runs/789/annotations?per_page=100&page=2"
    )
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        "run_id": 456,
                        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
                    }
                ),
            ),
            _result(1, stderr="HTTP 404: Not Found"),
            _result(0, _include_page(json.dumps([{"message": "anno one"}]), next_url=next_url)),
            _result(0, _include_page(json.dumps([{"message": "anno two"}]))),
        ]
    )

    def fake_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return next(results)

    monkeypatch.setattr(diagnose_actions_job, "_gh", fake_gh)

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert [annotation["message"] for annotation in payload] == ["anno one", "anno two"]
    # the first annotation request targets the check-run path, then the
    # follow-up request uses the rel="next" Link URL verbatim
    assert calls[2] == [
        "api",
        "--include",
        "repos/owner/repo/check-runs/789/annotations?per_page=100",
    ]
    assert calls[3] == ["api", "--include", next_url]


def test_collect_annotations_accepts_a_terminal_page_at_the_guard(monkeypatch, capsys) -> None:
    """A terminal page at the request cap is complete, not an exhaustion error."""
    monkeypatch.setattr(diagnose_actions_job, "MAX_ANNOTATION_PAGES", 1)
    monkeypatch.setattr(
        diagnose_actions_job,
        "_gh",
        lambda _args: _result(0, _include_page(json.dumps([{"message": "final"}]))),
    )

    assert diagnose_actions_job._collect_annotations(
        "repos/owner/repo/check-runs/789/annotations"
    ) == [{"message": "final"}]
    assert capsys.readouterr().err == ""


def test_collect_annotations_rejects_a_next_page_beyond_the_guard(monkeypatch, capsys) -> None:
    """A rel=next link at the request cap fails closed without fetching it."""
    monkeypatch.setattr(diagnose_actions_job, "MAX_ANNOTATION_PAGES", 1)
    monkeypatch.setattr(
        diagnose_actions_job,
        "_gh",
        lambda _args: _result(
            0,
            _include_page(
                json.dumps([{"message": "first"}]),
                next_url="https://api.github.com/repos/owner/repo/check-runs/789/annotations?page=2",
            ),
        ),
    )

    assert (
        diagnose_actions_job._collect_annotations("repos/owner/repo/check-runs/789/annotations")
        is None
    )
    assert "pagination exceeded the page guard" in capsys.readouterr().err


def test_main_fails_closed_when_annotation_fallback_is_unavailable(monkeypatch, capsys) -> None:
    """Missing logs are not treated as diagnosed when annotations also fail."""
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        "run_id": 456,
                        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
                    }
                ),
            ),
            _result(1, stderr="HTTP 404: Not Found"),
            _result(1, stderr="HTTP 403: Forbidden"),
        ]
    )

    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 1
    assert "Could not recover check-run annotations: HTTP 403: Forbidden" in capsys.readouterr().err


def test_main_fails_closed_when_annotations_are_empty(monkeypatch, capsys) -> None:
    """An empty annotations response is not a successful diagnosis."""
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        "run_id": 456,
                        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
                    }
                ),
            ),
            _result(1, stderr="HTTP 404: Not Found"),
            _result(0, _include_page(json.dumps([]))),
        ]
    )

    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 1
    assert "the endpoint returned no annotations" in capsys.readouterr().err


def test_main_fails_closed_when_annotation_json_is_malformed(monkeypatch, capsys) -> None:
    """Malformed JSON on an annotation page fails closed rather than succeeding."""
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        "run_id": 456,
                        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
                    }
                ),
            ),
            _result(1, stderr="HTTP 404: Not Found"),
            _result(0, _include_page("{not valid json")),
        ]
    )

    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 1
    assert "Could not parse check-run annotations JSON" in capsys.readouterr().err


def test_main_fails_closed_when_annotation_items_are_not_objects(monkeypatch, capsys) -> None:
    """A JSON array containing a scalar is not a valid annotation page."""
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        "run_id": 456,
                        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
                    }
                ),
            ),
            _result(1, stderr="HTTP 404: Not Found"),
            _result(0, _include_page(json.dumps([{"message": "valid"}, "not an object"]))),
        ]
    )

    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 1
    assert "expected annotation objects" in capsys.readouterr().err


def test_main_fails_closed_when_include_headers_are_missing(monkeypatch, capsys) -> None:
    """A valid JSON body without ``--include`` headers cannot prove pagination ended."""
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        "run_id": 456,
                        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
                    }
                ),
            ),
            _result(1, stderr="HTTP 404: Not Found"),
            _result(0, json.dumps([{"message": "could be page one"}])),
        ]
    )

    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 1
    assert "expected HTTP headers from gh api --include" in capsys.readouterr().err


def test_main_fails_closed_when_gh_is_missing(monkeypatch, capsys) -> None:
    """A missing gh binary fails closed at the first request instead of crashing."""

    def raise_missing(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gh")

    monkeypatch.setattr(diagnose_actions_job.subprocess, "run", raise_missing)

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 1
    assert "gh CLI not found" in capsys.readouterr().err
