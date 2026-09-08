"""Tests for the missing-GitHub-Actions-log annotation fallback."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.dev import diagnose_actions_job


def _job() -> dict[str, object]:
    """Sanitized exact-job metadata; no run-level latest-attempt lookup needed."""
    return {
        "id": 123,
        "run_id": 456,
        "run_attempt": 1,
        "head_sha": "a" * 40,
        "status": "completed",
        "conclusion": "failure",
        "url": "https://api.github.com/repos/owner/repo/actions/jobs/123",
        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
    }


# Sanitized error record from issue #8636's three attempt-1 incidents:
# 34248777227/102137512695, 34250375088/102143335065, 34247287993/102133036580.
# Timestamp retained as a representative log prefix; no raw surrounding logs or credentials.
FINALIZATION_403 = (
    "Failed to FinalizeArtifact: Received non-retryable error: Failed request: (403) Forbidden: "
    'Error from intermediary with HTTP status code 403 "Forbidden"'
)


def test_json_classifies_exact_job_error_without_changing_failed_conclusion(monkeypatch, capsys):
    """A diagnostic classification is not a successful job or confirmed publication."""
    calls = []
    responses = iter(
        [
            _result(0, json.dumps(_job())),
            _result(
                0, f"tests failed too\n2026-09-08T16:09:24.6765116Z ##[error]{FINALIZATION_403}\n"
            ),
        ]
    )

    def fake_gh(args):
        calls.append(args)
        return next(responses)

    monkeypatch.setattr(diagnose_actions_job, "_gh", fake_gh)
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "actions_job_diagnostic.v1"
    assert payload["diagnostic_status"] == "matched"
    assert payload["classification"] == "artifact_finalization_403"
    assert payload["job_status"] == "completed"
    assert payload["job_conclusion"] == "failure"
    assert payload["artifact_publication"] == "unconfirmed"
    assert [
        payload[key] for key in ("repository", "job_id", "run_id", "run_attempt", "head_sha")
    ] == [
        "owner/repo",
        123,
        456,
        1,
        "a" * 40,
    ]
    assert payload["evidence"]["source"] == "job_logs"
    assert payload["evidence"]["record"] == 2
    assert FINALIZATION_403 in payload["evidence"]["excerpt"]
    assert calls == [
        ["api", "repos/owner/repo/actions/jobs/123"],
        ["api", "repos/owner/repo/actions/jobs/123/logs"],
    ]


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
            _result(0, json.dumps({**_job(), "check_run_url": "unused"})),
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
    assert calls[1] == ["api", "repos/owner/repo/actions/jobs/123/logs"]


def test_main_never_substitutes_later_attempt_logs(monkeypatch, capsys) -> None:
    """The requested failed job must not expose a later attempt's successful log."""

    def fake_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args == ["api", "repos/owner/repo/actions/jobs/123"]:
            return _result(0, json.dumps(_job()))
        if args == ["api", "repos/owner/repo/actions/jobs/123/logs"]:
            return _result(0, "original attempt: artifact finalization failed\n")
        if args[:2] == ["run", "view"]:
            return _result(0, "later attempt: upload succeeded\n")
        raise AssertionError(f"Unexpected remote operation: {args}")

    monkeypatch.setattr(diagnose_actions_job, "_gh", fake_gh)
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 0
    assert capsys.readouterr().out == "original attempt: artifact finalization failed\n"


def test_main_falls_back_to_check_run_annotations_when_logs_are_absent(monkeypatch, capsys) -> None:
    """An unavailable job log should expose GitHub's retained error annotation."""
    calls: list[list[str]] = []
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        **_job(),
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
                        **_job(),
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
                        **_job(),
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
                        **_job(),
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
                        **_job(),
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
                        **_job(),
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
                        **_job(),
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


def test_gh_timeout_returns_a_structured_failure(monkeypatch) -> None:
    """A hung gh request becomes a bounded diagnostic failure."""
    calls: list[tuple[list[str], dict[str, object]]] = []

    def raise_timeout(
        command: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del args
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(diagnose_actions_job.subprocess, "run", raise_timeout)

    result = diagnose_actions_job._gh(["api", "repos/owner/repo/actions/jobs/123"])

    assert result.returncode == 124
    assert result.stdout == ""
    assert result.stderr == "gh command timed out after 30 seconds"
    assert calls[0][1]["timeout"] == 30


@pytest.mark.parametrize("source", ["logs", "annotations"])
@pytest.mark.parametrize(
    "record",
    [
        "HTTP 403: Forbidden",
        "Failed to UploadArtifact: (403) Forbidden",
        FINALIZATION_403.replace("403", "503"),
        "FinalizeArtifact succeeded\nHTTP 403: Forbidden",
        "Failed to FinalizeArtifact\nHTTP 403: Forbidden",
        "##[error]Failed to FinalizeArtifact: other failure\n##[error]HTTP 403: Forbidden",
        "AssertionError: expected 3, got 4",
        "Artifact finalized successfully",
        FINALIZATION_403.replace("Forbidden: Error", "Forbidden:\nError"),
    ],
)
def test_json_does_not_join_unrelated_failure_records(monkeypatch, capsys, source, record):
    """Neither independent records nor other artifact phases/statuses prove this signature."""
    responses = [_result(0, json.dumps(_job()))]
    if source == "logs":
        responses.append(_result(0, record))
    else:
        responses.extend(
            [
                _result(1, stderr="logs unavailable"),
                _result(
                    0,
                    _include_page(json.dumps([{"message": line} for line in record.splitlines()])),
                ),
            ]
        )
    results = iter(responses)
    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostic_status"] == "unmatched"
    assert payload["classification"] is None
    assert payload["job_conclusion"] == "failure"


@pytest.mark.parametrize(
    "job_id,run_id,head_sha",
    [
        (102137512695, 34248777227, "11783ff1866ea98971220e9878690f003f05dfbe"),
        (102143335065, 34250375088, "11783ff1866ea98971220e9878690f003f05dfbe"),
        (102133036580, 34247287993, "f4e7a06617d2ce8b27ad2532bf851105709d10bc"),
    ],
)
def test_observed_incident_fixtures_preserve_provenance(
    monkeypatch, capsys, job_id, run_id, head_sha
):
    """The same sanitized signature classifies each observed failed attempt, not a later retry."""
    job = {
        **_job(),
        "id": job_id,
        "run_id": run_id,
        "head_sha": head_sha,
        "url": f"https://api.github.com/repos/owner/repo/actions/jobs/{job_id}",
    }
    results = iter([_result(0, json.dumps(job)), _result(0, f"##[error]{FINALIZATION_403}\n")])
    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))
    assert diagnose_actions_job.main([str(job_id), "--repo", "owner/repo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert (payload["job_id"], payload["run_id"], payload["run_attempt"], payload["head_sha"]) == (
        job_id,
        run_id,
        1,
        head_sha,
    )
    assert payload["classification"] == "artifact_finalization_403"


def test_json_recovers_signature_on_later_annotation_page(monkeypatch, capsys):
    """Complete annotation fallback retains its check-run endpoint and independent record index."""
    next_url = "https://api.github.com/repos/owner/repo/check-runs/789/annotations?page=2"
    calls = []
    results = iter(
        [
            _result(0, json.dumps(_job())),
            _result(1, stderr="HTTP 404"),
            _result(0, _include_page(json.dumps([{"message": "test failed"}]), next_url=next_url)),
            _result(0, _include_page(json.dumps([{"message": FINALIZATION_403}]))),
        ]
    )

    def fake_gh(args):
        calls.append(args)
        return next(results)

    monkeypatch.setattr(diagnose_actions_job, "_gh", fake_gh)
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["classification"] == "artifact_finalization_403"
    assert payload["evidence"] == {
        "source": "check_run_annotations",
        "endpoint": "repos/owner/repo/check-runs/789/annotations?per_page=100",
        "record": 2,
        "excerpt": FINALIZATION_403,
        "truncated": False,
    }
    assert calls == [
        ["api", "repos/owner/repo/actions/jobs/123"],
        ["api", "repos/owner/repo/actions/jobs/123/logs"],
        ["api", "--include", "repos/owner/repo/check-runs/789/annotations?per_page=100"],
        ["api", "--include", next_url],
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 124),
        ("id", True),
        ("run_id", None),
        ("run_id", "456"),
        ("run_attempt", None),
        ("run_attempt", 0),
        ("run_attempt", True),
        ("head_sha", None),
        ("head_sha", "bad"),
        ("status", None),
        ("conclusion", []),
        ("url", "https://api.github.com/repos/other/repo/actions/jobs/123"),
    ],
)
@pytest.mark.parametrize("json_mode", [False, True])
def test_missing_or_mismatched_metadata_stops_before_logs(
    monkeypatch, capsys, field, value, json_mode
):
    """Unproven identity never reaches a log endpoint, in either output mode."""
    calls = []

    def fake_gh(args):
        calls.append(args)
        return _result(0, json.dumps({**_job(), field: value}))

    monkeypatch.setattr(diagnose_actions_job, "_gh", fake_gh)
    argv = ["123", "--repo", "owner/repo"] + (["--json"] if json_mode else [])
    assert diagnose_actions_job.main(argv) == 1
    captured = capsys.readouterr()
    assert len(calls) == 1
    if json_mode:
        payload = json.loads(captured.out)
        assert payload["diagnostic_status"] == "unavailable"
        assert payload["reason"] == "job_metadata_identity_invalid"
        assert payload["classification"] is None
    else:
        assert captured.out == ""


@pytest.mark.parametrize(
    "response", [_result(1, stderr="HTTP 404"), _result(0, "{bad"), _result(0, "[]")]
)
def test_json_metadata_unavailable_is_explicit(monkeypatch, capsys, response):
    """Missing and malformed metadata produce one envelope and the existing failure exit."""
    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: response)
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostic_status"] == "unavailable"
    assert payload["reason"] == "job_metadata_unavailable"
    assert payload["run_attempt"] is None


@pytest.mark.parametrize(
    "page",
    [
        _result(1, stderr="HTTP 403"),
        _result(0, _include_page("[]")),
        _result(0, _include_page("{bad")),
        _result(0, _include_page('["bad"]')),
        _result(0, _include_page('[{"message": null}]')),
        _result(0, _include_page('[{"message": " "}]')),
        _result(0, _include_page("[{}]")),
        _result(0, '[{"message": "missing headers"}]'),
        _result(
            0,
            _include_page(
                json.dumps([{"message": FINALIZATION_403}]),
                next_url="https://api.github.com/repos/owner/repo/check-runs/789/annotations?page=2",
            ),
        ),
    ],
)
def test_json_unusable_or_incomplete_annotations_never_classify(monkeypatch, capsys, page):
    """A signature on an incomplete page is not successful diagnostic retrieval."""
    monkeypatch.setattr(diagnose_actions_job, "MAX_ANNOTATION_PAGES", 1)
    results = iter([_result(0, json.dumps(_job())), _result(0, " \n"), page])
    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostic_status"] == "unavailable"
    assert payload["classification"] is None


def test_annotation_pagination_cannot_change_check_run_identity(monkeypatch, capsys):
    """A foreign pagination target cannot provide evidence for the requested job."""
    calls = []
    results = iter(
        [
            _result(0, json.dumps(_job())),
            _result(1, stderr="no logs"),
            _result(
                0,
                _include_page(
                    '[{"message":"unmatched"}]',
                    next_url="https://api.github.com/repos/other/repo/check-runs/789/annotations?page=2",
                ),
            ),
        ]
    )

    def fake_gh(args):
        calls.append(args)
        return next(results)

    monkeypatch.setattr(diagnose_actions_job, "_gh", fake_gh)
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 1
    assert len(calls) == 3
    assert json.loads(capsys.readouterr().out)["diagnostic_status"] == "unavailable"


def test_json_excerpt_is_bounded_and_keeps_the_matched_evidence(monkeypatch, capsys):
    """A long record cannot flood JSON output or hide the actual matching signature."""
    log = "x" * 5000 + FINALIZATION_403 + "y" * 5000
    results = iter([_result(0, json.dumps(_job())), _result(0, log)])
    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evidence"]["truncated"] is True
    assert len(payload["evidence"]["excerpt"]) <= diagnose_actions_job.MAX_EXCERPT_CHARS
    assert FINALIZATION_403 in payload["evidence"]["excerpt"]


@pytest.mark.parametrize(
    "link",
    [
        'not-a-url; rel="next"',
        '<https://[broken/annotations>; rel="next"',
    ],
)
def test_json_malformed_pagination_cannot_look_complete(monkeypatch, capsys, link):
    """Malformed continuation metadata must not produce a partial successful diagnosis."""
    body = json.dumps([{"message": FINALIZATION_403}])
    response = f"HTTP/2.0 200 OK\r\nLink: {link}\r\n\r\n{body}"
    results = iter([_result(0, json.dumps(_job())), _result(1), _result(0, response)])
    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostic_status"] == "unavailable"
    assert payload["classification"] is None


@pytest.mark.parametrize(
    "check_run_url", [None, "https://api.github.com/repos/other/repo/check-runs/789"]
)
def test_json_annotation_fallback_requires_same_repository(monkeypatch, capsys, check_run_url):
    """Untrusted check-run links are not fetched when exact job logs are absent."""
    results = iter(
        [
            _result(0, json.dumps({**_job(), "check_run_url": check_run_url})),
            _result(1),
        ]
    )
    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == "check_run_identity_invalid"
    assert payload["job_conclusion"] == "failure"


@pytest.mark.parametrize(
    "field", ["id", "run_id", "run_attempt", "head_sha", "status", "conclusion", "url"]
)
def test_json_missing_required_metadata_field_is_unavailable(monkeypatch, capsys, field):
    """Missing identity/status fields are not filled from a run-level latest-attempt lookup."""
    job = _job()
    del job[field]
    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: _result(0, json.dumps(job)))
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["diagnostic_status"] == "unavailable"


@pytest.mark.parametrize("job_id,repo", [("0", "owner/repo"), ("123", "owner/repo/other")])
def test_json_invalid_request_never_calls_github(monkeypatch, capsys, job_id, repo):
    """Malformed identity inputs do not become API paths."""

    def reject_call(_args):
        raise AssertionError("Invalid input must not call GitHub")

    monkeypatch.setattr(diagnose_actions_job, "_gh", reject_call)
    assert diagnose_actions_job.main([job_id, "--repo", repo, "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "invalid_request"


def test_json_preserves_in_progress_job_without_conclusion(monkeypatch, capsys):
    """A legitimately null conclusion is retained without inventing success or failure."""
    job = {**_job(), "status": "in_progress", "conclusion": None}
    results = iter([_result(0, json.dumps(job)), _result(0, "test execution in progress")])
    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_status"] == "in_progress"
    assert payload["job_conclusion"] is None
    assert payload["diagnostic_status"] == "unmatched"


def test_repository_identity_is_case_insensitive_like_github(monkeypatch, capsys):
    """Canonical API capitalization must not break a valid repository override."""
    job = {
        **_job(),
        "url": "https://api.github.com/repos/Owner/Repo/actions/jobs/123",
        "check_run_url": "https://api.github.com/repos/Owner/Repo/check-runs/789",
    }
    results = iter(
        [
            _result(0, json.dumps(job)),
            _result(1),
            _result(0, _include_page(json.dumps([{"message": FINALIZATION_403}]))),
        ]
    )
    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))
    assert diagnose_actions_job.main(["123", "--repo", "owner/repo", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["classification"] == "artifact_finalization_403"
