"""Tests for the compute-window staging bundle builder (issue #8823)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/validation/build_compute_staging_bundle.py"
_SPEC = importlib.util.spec_from_file_location("_compute_staging_bundle", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _pin(root: Path, relative: str) -> dict:
    return {"path": relative, "sha256": _MODULE.sha256_file(root / relative)}


def _repo(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "worker@example.com")
    _git(root, "config", "user.name", "worker")
    files = {
        ".gitignore": "output/\n",
        "configs/campaign.yaml": "campaign: fixture\n",
        "configs/seeds.yaml": "seeds: [1, 2, 3]\n",
        "model/checkpoint.bin": "checkpoint-bytes\n",
        "uv.lock": "lock-bytes\n",
    }
    for relative, text in files.items():
        _write(root, relative, text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")
    return root


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    return _repo(tmp_path / "repo")


def _request(repo: Path) -> dict:
    return {
        "schema_version": _MODULE.REQUEST_SCHEMA,
        "issue": {"number": 8823, "url": "https://github.com/ll7/robot_sf_ll7/issues/8823"},
        "owner": "ll7",
        "source": {
            "commit": _git(repo, "rev-parse", "HEAD"),
            "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
            "dirty_policy": "reject",
        },
        "environment": "cpu-linux-x86_64",
        "configs": [_pin(repo, "configs/campaign.yaml")],
        "seed_set": {
            "identity": "fixture-seeds",
            "seed_count": 3,
            **_pin(repo, "configs/seeds.yaml"),
        },
        "checkpoints": [
            {
                "identity": "fixture-ckpt",
                "resolution": "local_file",
                **_pin(repo, "model/checkpoint.bin"),
            }
        ],
        "dependencies": {"lockfile": _pin(repo, "uv.lock")},
        "command_tokens": ["uv", "run", "python", "scripts/tools/run_camera_ready_benchmark.py"],
        "expected_row_count": 12,
        "resource_class": "cpu-standard",
        "output_contract": {"local_root": "output/bundle", "required_paths": ["manifest.json"]},
        "capability_class": "compute_window_expiring",
    }


def _receipt(config_sha: str, checkpoint_sha: str) -> dict:
    return {
        "schema_version": _MODULE.CHECKPOINT_STAGING_RECEIPT_SCHEMA,
        "status": "ok",
        "mode": "enforced_staged",
        "stage": True,
        "submit_safe": True,
        "campaign_config_sha256": config_sha,
        "arms": [{"planner_key": "ppo", "checkpoint_sha256": checkpoint_sha}],
    }


def _use_staging_receipt(repo: Path, request: dict, config_sha: str) -> None:
    checkpoint_sha = request["checkpoints"][0]["sha256"]
    request["checkpoints"][0] = {
        "identity": "fixture-ckpt",
        "resolution": "staging_receipt",
        "staging_receipt": "output/preflight/checkpoint_staging.json",
        "sha256": checkpoint_sha,
    }
    _write(
        repo,
        "output/preflight/checkpoint_staging.json",
        json.dumps(_receipt(config_sha, checkpoint_sha)),
    )


def _mutate(repo: Path, tmp_path: Path, case: str, request: dict) -> None:
    if case == "dirty":
        _write(repo, "configs/campaign.yaml", "campaign: changed\n")
    elif case == "changed_config":
        request["configs"][0]["sha256"] = "0" * 64
    elif case == "absent_checkpoint":
        request["checkpoints"][0]["path"] = "model/absent.bin"
    elif case == "symlink_escape":
        outside = _write(tmp_path, "outside.bin", "outside-bytes\n")
        (repo / "model/checkpoint.bin").unlink()
        (repo / "model/checkpoint.bin").symlink_to(outside)
        request["checkpoints"][0]["sha256"] = _MODULE.sha256_file(outside)
    elif case == "duplicate_member":
        request["configs"].append(dict(request["configs"][0]))
    elif case == "private_path":
        request["configs"][0]["path"] = "/etc/passwd"
    elif case == "untracked":
        _write(repo, "configs/untracked.yaml", "campaign: untracked\n")
        request["configs"][0] = _pin(repo, "configs/untracked.yaml")
    elif case == "stale_source":
        request["source"].update(commit="main", tree="2" * 40)
    elif case == "unbound_receipt":
        _use_staging_receipt(repo, request, "0" * 64)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("dirty", {"dirty_source"}),
        ("changed_config", {"checksum_mismatch"}),
        ("absent_checkpoint", {"missing_input"}),
        ("symlink_escape", {"symlink_input"}),
        ("duplicate_member", {"duplicate_member"}),
        ("private_path", {"path_escape"}),
        ("untracked", {"untracked_input"}),
        ("stale_source", {"mutable_ref", "stale_source"}),
        ("unbound_receipt", {"unauthorized_checkpoint"}),
    ],
)
def test_blocked_inputs_fail_closed_with_reason_codes(repo, tmp_path, case, expected) -> None:
    request = _request(repo)
    _mutate(repo, tmp_path, case, request)

    assert expected <= set(_MODULE.build_report(request, repo, 8823)["reason_codes"])


def test_clean_request_is_ready_and_byte_stable(repo: Path, tmp_path: Path) -> None:
    first = _MODULE.build_report(_request(repo), repo, 8823)
    assert first["status"] == "ready"
    assert first["file_count"] == len(first["members"]) == 4
    assert [m["path"] for m in first["members"]] == sorted(m["path"] for m in first["members"])
    assert first == _MODULE.build_report(_request(repo), repo, 8823)

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    _, written = _MODULE.write_bundle(first, out_a, repo)
    _MODULE.write_bundle(first, out_b, repo)
    assert len(written) == 4
    for name in written:
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    assert (out_a / _MODULE.SUMS_FILENAME).read_text(encoding="utf-8").count("\n") == 4


def test_staging_receipt_checkpoint_is_admitted(repo: Path) -> None:
    request = _request(repo)
    _use_staging_receipt(repo, request, request["configs"][0]["sha256"])

    report = _MODULE.build_report(request, repo, 8823)
    assert report["status"] == "ready"
    assert report["checkpoints"][0]["bytes_present"] is False
    assert "output/preflight/checkpoint_staging.json" in [m["path"] for m in report["members"]]


def test_cli_modes_and_exit_codes(repo: Path, tmp_path: Path, capsys) -> None:
    request_path = _write(tmp_path, "request.json", json.dumps(_request(repo)))
    base = ["--request", str(request_path), "--issue", "8823", "--repo-root", str(repo)]
    assert _MODULE.main([*base, "--check", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == _MODULE.BUNDLE_SCHEMA

    assert _MODULE.main([*base, "--issue", "1", "--check"]) == 2
    assert "issue_mismatch" in capsys.readouterr().out

    output_root = tmp_path / "bundle"
    assert _MODULE.main([*base, "--output-root", str(output_root)]) == 0
    (output_root / _MODULE.SUMS_FILENAME).write_text("tampered\n", encoding="utf-8")
    assert _MODULE.main([*base, "--output-root", str(output_root)]) == 2
    assert "immutable_output_conflict" in capsys.readouterr().out

    malformed = _write(tmp_path, "malformed.json", "{not json")
    args = ["--request", str(malformed), "--issue", "8823", "--repo-root", str(repo), "--check"]
    assert _MODULE.main(args) == 3
