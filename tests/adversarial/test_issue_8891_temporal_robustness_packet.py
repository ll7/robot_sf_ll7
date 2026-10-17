"""Focused outcome-free contract tests for issue #8891."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import robot_sf.adversarial.matched_budget_packet as packet_module
from robot_sf.adversarial.matched_budget_packet import (
    SIMULATOR_INVOCATIONS_PER_SLOT,
    TEMPORAL_SIDECAR_SCHEMA_VERSION,
    PacketError,
    build_canary_packet,
    build_expected_identities,
    canonical_sha256,
    validate_call_ledger,
    validate_packet,
    validate_result_rows,
    validate_temporal_sidecar,
)

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "configs/adversarial/issue_8891_temporal_robustness_packet.yaml"


@pytest.fixture()
def packet() -> dict:
    return yaml.safe_load(PACKET.read_text(encoding="utf-8"))


def test_packet_is_source_bound_and_diagnostic_only(packet: dict) -> None:
    result = validate_packet(packet, repo_root=ROOT)
    assert (
        result["status"] == "ok"
        and result["simulator_call_budget"] == 14112
        and not result["campaign_execution_allowed"]
    )
    assert packet["source"]["base_commit"] == "69580e4837ac96e9f658da92eb19e8b1a3a76950"


def test_packet_separates_immutable_and_working_tree_source_hashes(packet: dict) -> None:
    result = validate_packet(packet, repo_root=ROOT)
    assert result["status"] == "ok"
    for input_id in ("robustness", "samplers", "runner", "benchmark_runner", "benchmark_constants"):
        source_input = packet["source"]["inputs"][input_id]
        assert source_input["sha256"] != source_input["working_tree_sha256"]


def test_historical_packet_does_not_require_working_tree_refresh_hashes(packet: dict) -> None:
    """Historical provenance survives source evolution without pretending evidence was rerun."""
    for source_input in packet["source"]["inputs"].values():
        source_input.pop("working_tree_sha256", None)
    result = validate_packet(packet, repo_root=ROOT)
    assert result["status"] == "ok"
    assert result["campaign_execution_allowed"] is False
    assert any(item["input_id"] == "robustness" for item in result["source_staleness"])


def _historical_checkout(packet: dict, root: Path) -> Path:
    """Build a tiny Git fixture borrowing objects, without cloning the repository."""
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    common = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (root / ".git/objects/info/alternates").write_text(
        str(Path(common) / "objects") + "\n", encoding="utf-8"
    )
    commit = packet["source"]["base_commit"]
    subprocess.run(
        ["git", "-C", str(root), "update-ref", "refs/remotes/origin/main", commit], check=True
    )
    for item in packet["source"]["inputs"].values():
        target = root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            subprocess.run(
                ["git", "-C", str(root), "show", f"{commit}:{item['path']}"],
                capture_output=True,
                check=True,
            ).stdout
        )
        item.pop("working_tree_sha256", None)
    return root


@pytest.mark.parametrize("input_id", ["runner", "search_space", "scenario_template"])
@pytest.mark.parametrize("change", ["modified", "deleted"])
def test_historical_validation_uses_committed_inputs(
    packet: dict, tmp_path: Path, input_id: str, change: str
) -> None:
    """Current edits/deletions cannot invalidate or redefine a historical planning audit."""
    root = _historical_checkout(packet, tmp_path)
    assert validate_packet(packet, repo_root=root).get("source_staleness", []) == []
    item = packet["source"]["inputs"][input_id]
    path = root / item["path"]
    if change == "deleted":
        path.unlink()
    else:
        path.write_text("current: unrelated implementation\n", encoding="utf-8")
        # Repinning current bytes must not redefine the historical semantics.
        item["working_tree_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = validate_packet(packet, repo_root=root)
    assert result["status"] == "ok"
    assert result["claim_eligible"] is False
    assert result["campaign_execution_allowed"] is False
    assert result.get("source_staleness") == [
        {
            "input_id": input_id,
            "path": item["path"],
            "source_commit": packet["source"]["base_commit"],
            "message": f"{'unreadable' if change == 'deleted' else 'stale'} "
            f"since {packet['source']['base_commit']}",
        }
    ]


@pytest.mark.parametrize(
    "prefix", ["", "/", "../", "./", "scripts/../", "escape/../", "scripts//", "scripts/./"]
)
def test_pinned_path_rejects_unsafe_segments(packet: dict, tmp_path: Path, prefix: str) -> None:
    """Reject traversal before normalization, including the review's symlink/.. escape."""
    root = _historical_checkout(packet, tmp_path / "repo")
    outside = tmp_path / "outside/nested"
    outside.mkdir(parents=True)
    (root / "escape").symlink_to(outside, target_is_directory=True)
    item = packet["source"]["inputs"]["runner"]
    original = item["path"]
    item["path"] = prefix + original if prefix else str(root / original)
    with pytest.raises(PacketError, match=r"source.inputs.runner.path.*(repo-relative|path)"):
        validate_packet(packet, repo_root=root)


@pytest.mark.parametrize("value", ["", ".", "..", "scripts/", "scripts\\runner", "bad\0path"])
def test_pinned_path_rejects_malformed_spelling(packet: dict, tmp_path: Path, value: str) -> None:
    """Pins have one canonical, nonempty POSIX-relative spelling."""
    root = _historical_checkout(packet, tmp_path)
    packet["source"]["inputs"]["runner"]["path"] = value
    with pytest.raises(PacketError, match=r"source.inputs.runner.path.*(repo-relative|path)"):
        validate_packet(packet, repo_root=root)


@pytest.mark.parametrize("length", [128 * 1024, 1024 * 1024], ids=["128KiB", "1MiB"])
def test_oversized_pinned_paths_refused_with_packet_error(
    packet: dict, tmp_path: Path, length: int
) -> None:
    """Retain the reviewer's public-interface subprocess argument-size regressions."""
    root = _historical_checkout(packet, tmp_path)
    assert validate_packet(packet, repo_root=root)["status"] == "ok"
    packet["source"]["inputs"]["runner"]["path"] = "a" * length
    with pytest.raises(PacketError, match=r"source.inputs.runner.path.*path") as exc:
        validate_packet(packet, repo_root=root)
    assert len(str(exc.value)) < 200


@pytest.mark.parametrize(
    "value",
    ["a" * 256, "é" * 128, "/".join(["a" * 255] * 16) + "/a", "/".join(["é" * 127] * 17)],
    ids=["segment-256-bytes", "utf8-segment-256-bytes", "path-4097-bytes", "utf8-path-over-limit"],
)
def test_oversized_pinned_paths_checked_before_lookup(
    packet: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """All pins obey encoded byte limits before any immutable or checkout lookup."""
    root = _historical_checkout(packet, tmp_path)
    packet["source"]["inputs"]["scenario_template"]["path"] = value

    def forbidden(*args, **kwargs):
        pytest.fail("lookup preceded oversized pinned path rejection")

    with monkeypatch.context() as patch:
        patch.setattr(packet_module, "_git_output", forbidden)
        patch.setattr(Path, "lstat", forbidden)
        with pytest.raises(PacketError, match=r"source.inputs.scenario_template.path.*path") as exc:
            validate_packet(packet, repo_root=root)
    assert len(str(exc.value)) < 200


@pytest.mark.parametrize("failure", ["lstat", "digest"])
def test_current_oserror_is_informational_unreadable(
    packet: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Unavailable drift cannot crash or grant execution after historical validation."""
    root = _historical_checkout(packet, tmp_path)
    relative_path = packet["source"]["inputs"]["runner"]["path"]
    if failure == "lstat":
        original = Path.lstat

        def unreadable(path, *args, **kwargs):
            if path == root / relative_path:
                raise OSError("injected current-tree failure")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "lstat", unreadable)
    else:
        original = packet_module._current_source_digest

        def unreadable(current_root, path):
            if path == relative_path:
                raise OSError("injected remaining drift failure")
            return original(current_root, path)

        monkeypatch.setattr(packet_module, "_current_source_digest", unreadable)
    result = validate_packet(packet, repo_root=root)
    assert result["status"] == "ok"
    assert result["claim_eligible"] is False
    assert result["campaign_execution_allowed"] is False
    assert result["source_staleness"] == [
        {
            "input_id": "runner",
            "path": relative_path,
            "source_commit": packet["source"]["base_commit"],
            "message": f"unreadable since {packet['source']['base_commit']}",
        }
    ]


@pytest.mark.parametrize(
    "value",
    ["a" * 255, "é" * 127 + "a", "/".join(["a" * 255] * 15 + ["a" * 254, "a"])],
    ids=["segment-255-bytes", "utf8-segment-255-bytes", "path-4096-bytes"],
)
def test_pinned_path_byte_limits_are_inclusive(value: str) -> None:
    """Exact byte limits remain valid lexical paths, independent of tree membership."""
    assert packet_module._pinned_path(value, "source.inputs.runner.path") == value


def test_all_pinned_paths_checked_before_any_git_lookup(
    packet: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed late pin is rejected before even resolving the commit."""
    root = _historical_checkout(packet, tmp_path)
    packet["source"]["inputs"]["scenario_template"]["path"] = "./bad-path"

    def forbidden(*args, **kwargs):
        pytest.fail("Git lookup preceded pinned path validation")

    monkeypatch.setattr(packet_module, "_git_output", forbidden)
    with pytest.raises(PacketError, match="source.inputs.scenario_template.path"):
        validate_packet(packet, repo_root=root)


@pytest.mark.parametrize("failure", ["digest", "semantics", "execution"])
def test_immutable_failure_never_inspects_working_tree(
    packet: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Every historical gate must precede all current-tree operations."""
    root = _historical_checkout(packet, tmp_path)
    if failure == "digest":
        packet["source"]["inputs"]["confirmation"]["sha256"] = "0" * 64
        message = "immutable source hash confirmation"
    elif failure == "semantics":
        packet["scenario"]["parameters"][0]["bounds"]["min"] += 1
        message = "scenario parameters"
    else:
        packet["private_ops"]["execution_authorized"] = True
        message = "private execution boundary"

    def guard(original):
        def checked(path, *args, **kwargs):
            if Path(os.fsdecode(path)).is_relative_to(root):
                pytest.fail("historical validation touched the working tree")
            return original(path, *args, **kwargs)

        return checked

    with monkeypatch.context() as patch:
        for name in ("resolve", "stat", "lstat", "open", "read_bytes"):
            patch.setattr(Path, name, guard(getattr(Path, name)))
        patch.setattr(os.path, "abspath", guard(os.path.abspath))
        with pytest.raises(PacketError, match=message):
            validate_packet(packet, repo_root=root)


@pytest.mark.parametrize("kind", ["directory", "external-link", "internal-link", "loop"])
def test_current_objects_are_only_informational(packet: dict, tmp_path: Path, kind: str) -> None:
    """Current objects cannot redirect or invalidate regular committed inputs."""
    root = _historical_checkout(packet, tmp_path / "repo")
    item = packet["source"]["inputs"]["runner"]
    path = root / item["path"]
    original = path.read_bytes()
    path.unlink()
    if kind == "directory":
        path.mkdir()
    elif kind == "loop":
        path.symlink_to(path.name)
    else:
        target = tmp_path / "external" if kind == "external-link" else root / "internal"
        target.write_bytes(original)
        path.symlink_to(target)
    result = validate_packet(packet, repo_root=root)
    assert result["status"] == "ok"
    assert result["campaign_execution_allowed"] is False
    assert result["claim_eligible"] is False
    assert [entry["input_id"] for entry in result["source_staleness"]] == (
        [] if kind == "internal-link" else ["runner"]
    )


def test_current_fifo_does_not_hang_historical_validation(packet: dict, tmp_path: Path) -> None:
    """Run the review's FIFO-with-no-writer probe with bounded, explicit child cleanup."""
    root = _historical_checkout(packet, tmp_path)
    path = root / packet["source"]["inputs"]["runner"]["path"]
    path.unlink()
    os.mkfifo(path)
    code = (
        "import json, sys, yaml; from pathlib import Path; "
        "from robot_sf.adversarial.matched_budget_packet import validate_packet; "
        "print(json.dumps(validate_packet(yaml.safe_load(sys.stdin.read()), repo_root=Path(sys.argv[1]))))"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(root)],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        try:
            stdout, stderr = child.communicate(yaml.safe_dump(packet), timeout=60)
        except subprocess.TimeoutExpired:
            pytest.fail("historical validation hung opening a current FIFO without a writer")
        assert child.returncode == 0, stderr
        result = json.loads(stdout)
        assert result["status"] == "ok"
        assert result["campaign_execution_allowed"] is False
        assert result["claim_eligible"] is False
        assert [entry["input_id"] for entry in result["source_staleness"]] == ["runner"]
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate()
        path.unlink()


def _pin_source_entry(packet: dict, root: Path, kind: str) -> None:
    """Create real Git entries with matching pins, borrowing the historical source tree."""

    def git(*args: str, data: bytes | None = None) -> bytes:
        return subprocess.run(
            ["git", "-C", str(root), *args], input=data, capture_output=True, check=True
        ).stdout

    commit = packet["source"]["base_commit"]
    relative = "robot_sf/adversarial" if kind == "directory" else "source-entry"
    if kind != "directory":
        git("read-tree", commit)
        if kind == "gitlink":
            mode, oid = "160000", commit
        else:
            mode = {
                "symlink": "120000",
                "symlink-to-source": "120000",
                "regular": "100644",
                "executable": "100755",
            }[kind]
            content = b"absent-target" if mode == "120000" else b"source file bytes\n"
            if kind == "symlink-to-source":
                content = packet["source"]["inputs"]["runner"]["path"].encode()
                (root / relative).symlink_to(content.decode())
                # The old resolver follows this dangling link to a regular committed target.
                (root / content.decode()).unlink()
            oid = git("hash-object", "-w", "--stdin", data=content).decode().strip()
        git("update-index", "--add", "--cacheinfo", f"{mode},{oid},{relative}")
        tree = git("write-tree").decode().strip()
        commit = (
            git(
                "-c",
                "user.name=Packet Test",
                "-c",
                "user.email=packet@example.invalid",
                "commit-tree",
                tree,
                "-p",
                commit,
                data=b"source entry fixture\n",
            )
            .decode()
            .strip()
        )
        git("update-ref", "refs/remotes/origin/main", commit)
        packet["source"]["base_commit"] = commit
    pin_path = content.decode() if kind == "symlink-to-source" else relative
    shown = git("show", f"{commit}:{pin_path}")
    packet["source"]["inputs"]["runner"].update(
        path=relative, sha256=hashlib.sha256(shown).hexdigest()
    )


@pytest.mark.parametrize("kind", ["directory", "symlink", "gitlink", "symlink-to-source"])
def test_pinned_source_rejects_non_file_entries(packet: dict, tmp_path: Path, kind: str) -> None:
    """Matching pins cannot turn a tree, symlink or gitlink into a source file."""
    root = _historical_checkout(packet, tmp_path)
    assert validate_packet(packet, repo_root=root)["status"] == "ok"
    _pin_source_entry(packet, root, kind)
    with pytest.raises(PacketError, match=r"source.inputs.runner.path not found|regular.*blob"):
        validate_packet(packet, repo_root=root)


@pytest.mark.parametrize("kind", ["regular", "executable"])
def test_pinned_source_accepts_regular_blob_modes(packet: dict, tmp_path: Path, kind: str) -> None:
    """Both regular Git file modes remain valid even after current files are deleted."""
    root = _historical_checkout(packet, tmp_path)
    _pin_source_entry(packet, root, kind)
    result = validate_packet(packet, repo_root=root)
    assert result["status"] == "ok"
    assert result["campaign_execution_allowed"] is False
    assert result["claim_eligible"] is False
    assert [item["input_id"] for item in result["source_staleness"]] == ["runner"]


@pytest.mark.parametrize("input_id", ["runner", "search_space", "scenario_template"])
def test_immutable_source_hash_stays_blocking(packet: dict, input_id: str) -> None:
    """Commit-bound integrity remains a hard gate even when current drift is advisory."""
    packet["source"]["inputs"][input_id]["sha256"] = "0" * 64
    _bad(lambda: validate_packet(packet, repo_root=ROOT), f"immutable source hash {input_id}")


def test_working_tree_refresh_hash_format_is_fail_closed(packet: dict) -> None:
    packet["source"]["inputs"]["robustness"]["working_tree_sha256"] = "not-a-sha256"
    _bad(
        lambda: validate_packet(packet, repo_root=ROOT),
        "working-tree source hash robustness must be SHA-256",
    )


def test_identity_builder_is_deterministic_and_complete(packet: dict) -> None:
    first = build_expected_identities(packet, repo_root=ROOT)
    assert first == build_expected_identities(packet, repo_root=ROOT)
    assert (first["run_count"], first["candidate_slot_count"]) == (54, 2016)
    candidate_ids = [
        slot["candidate_id"] for run in first["runs"] for slot in run["candidate_slots"]
    ]
    assert len(candidate_ids) == 2016
    seeds = [
        seed
        for run in first["runs"]
        for slot in run["candidate_slots"]
        for seed in [slot["replay_seed"], *slot["confirmation_seeds"]]
    ]
    assert len(seeds) == len(set(seeds)) and not set(seeds) & set(
        packet["seed_policy"]["search_seeds"]
    )


def test_canary_is_disjoint_planned_data_and_uses_every_gate(packet: dict) -> None:
    canary = build_canary_packet(packet, repo_root=ROOT)
    assert canary == build_canary_packet(packet, repo_root=ROOT)
    assert (
        canary["claim_eligible"],
        canary["candidate_count"],
        canary["ledger"]["simulator_invocations"],
        {item["search_family"] for item in canary["candidates"]},
    ) == (False, 6, 42, {"random", "optuna", "cmaes"})
    seeds = [
        seed
        for item in canary["candidates"]
        for seed in [item["search_seed"], item["replay_seed"], *item["confirmation_seeds"]]
    ]
    assert len(seeds) == len(set(seeds))


def _sidecar(packet: dict, candidate_id: str, *, artifact_only: bool = False) -> dict:
    sidecar = copy.deepcopy(
        next(
            item["temporal_sidecar"]
            for item in build_canary_packet(packet, repo_root=ROOT)["candidates"]
            if "temporal_sidecar" in item
        )
    )
    sidecar.update(status="observed", candidate_id=candidate_id, admission_status="not_admitted")
    for item in sidecar["properties"]:
        item.update(
            signed_margin=-0.1,
            activation_time_s=0.2,
            execution_mode="native",
            certification_state="passed",
            replay_state="passed",
            independent_seed_state="passed",
            provenance={"source": "fixture", "packet_digest": canonical_sha256(packet)},
        )
    sidecar["monitor"].update(sample_count=3, artifact_only=artifact_only)
    return sidecar


def _bad(call, match: str) -> None:
    with pytest.raises(PacketError, match=match):
        call()


def test_temporal_sidecar_uses_a_distinct_versioned_schema(packet: dict) -> None:
    sidecar = _sidecar(packet, "fixture")
    assert sidecar["schema_version"] == TEMPORAL_SIDECAR_SCHEMA_VERSION
    runtime_shape = copy.deepcopy(sidecar)
    runtime_shape["schema_version"] = "robustness-report.v1"
    _bad(
        lambda: validate_temporal_sidecar(runtime_shape, packet, candidate_id="fixture"),
        "temporal sidecar schema",
    )


def test_confirmed_failure_rejects_all_positive_signed_margins(packet: dict) -> None:
    sidecar = _sidecar(packet, "fixture")
    sidecar["admission_status"] = "confirmed_failure"
    sidecar["failure_basis"] = "independent_confirmation"
    for property_item in sidecar["properties"]:
        property_item["signed_margin"] = 0.1
    _bad(
        lambda: validate_temporal_sidecar(sidecar, packet, candidate_id="fixture"),
        "negative signed margin",
    )


def test_confirmed_failure_requires_activation_time(packet: dict) -> None:
    sidecar = _sidecar(packet, "fixture")
    sidecar["admission_status"] = "confirmed_failure"
    sidecar["failure_basis"] = "independent_confirmation"
    sidecar["properties"][0]["activation_time_s"] = None
    _bad(
        lambda: validate_temporal_sidecar(sidecar, packet, candidate_id="fixture"),
        "activation time",
    )


def test_observed_sidecar_requires_activation_time_for_violated_property(packet: dict) -> None:
    sidecar = _sidecar(packet, "fixture")
    sidecar["properties"][0]["activation_time_s"] = None
    _bad(
        lambda: validate_temporal_sidecar(sidecar, packet, candidate_id="fixture"),
        "activation time",
    )


def test_planned_sidecar_rejects_observed_values(packet: dict) -> None:
    canary = build_canary_packet(packet, repo_root=ROOT)
    sidecar = copy.deepcopy(
        next(
            item["temporal_sidecar"] for item in canary["candidates"] if "temporal_sidecar" in item
        )
    )
    sidecar["properties"][0]["signed_margin"] = -0.1
    _bad(
        lambda: validate_temporal_sidecar(sidecar, packet, candidate_id=sidecar["candidate_id"]),
        "planned sidecar cannot contain observations",
    )


def test_sidecar_rejects_property_drift_monitor_artifact_and_weak_admission(packet: dict) -> None:
    sidecar = _sidecar(packet, "fixture")
    broken = copy.deepcopy(sidecar)
    broken["property_ids"][-1] = "different_property"
    _bad(lambda: validate_temporal_sidecar(broken, packet, candidate_id="fixture"), "property IDs")
    _bad(
        lambda: validate_temporal_sidecar(
            _sidecar(packet, "fixture", artifact_only=True), packet, candidate_id="fixture"
        ),
        "monitor-only",
    )
    broken = copy.deepcopy(sidecar)
    broken["properties"][0]["provenance"]["packet_digest"] = "a" * 64
    _bad(
        lambda: validate_temporal_sidecar(broken, packet, candidate_id="fixture"),
        "packet provenance",
    )
    broken = copy.deepcopy(sidecar)
    broken["properties"][0]["execution_mode"] = "fallback"
    broken["admission_status"] = "confirmed_failure"
    _bad(
        lambda: validate_temporal_sidecar(broken, packet, candidate_id="fixture"),
        "fallback/degraded",
    )
    sidecar["admission_status"], sidecar["failure_basis"] = "confirmed_failure", "objective_value"
    _bad(
        lambda: validate_temporal_sidecar(sidecar, packet, candidate_id="fixture"),
        "cannot establish failure",
    )


def test_ledger_rejects_hidden_retry_duplicate_and_non_native_admission(packet: dict) -> None:
    rows = build_canary_packet(packet, repo_root=ROOT)["rows"]
    run_budget_limits = {str(row["run_id"]): (1, SIMULATOR_INVOCATIONS_PER_SLOT) for row in rows}
    retry = copy.deepcopy(rows)
    retry[0]["retry_of"] = "prior"
    _bad(
        lambda: validate_call_ledger(packet, retry, run_budget_limits=run_budget_limits),
        "hidden retries",
    )
    _bad(
        lambda: validate_call_ledger(
            packet, [*rows, copy.deepcopy(rows[0])], run_budget_limits=run_budget_limits
        ),
        "duplicate search",
    )
    weak = copy.deepcopy(rows)
    weak[0].update(
        execution_mode="fallback", status="observed", admission_status="confirmed_failure"
    )
    _bad(
        lambda: validate_call_ledger(packet, weak, run_budget_limits=run_budget_limits),
        "must be excluded",
    )
    over_budget = copy.deepcopy(packet)
    over_budget["budget"]["simulator_call_budget"] = 41
    _bad(
        lambda: validate_call_ledger(over_budget, rows, run_budget_limits=run_budget_limits),
        "simulator call budget",
    )


def test_ledger_rejects_per_run_budget_overruns(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    run = identities["runs"][0]
    run_budget_limits = {
        item["run_id"]: (
            len(item["candidate_slots"]),
            len(item["candidate_slots"]) * SIMULATOR_INVOCATIONS_PER_SLOT,
        )
        for item in identities["runs"]
    }

    def row(candidate_id: str, *, phase: str, call_class: str, seed: int, index: int) -> dict:
        return {
            "schema_version": "temporal-robustness-call-ledger.v1",
            "candidate_id": candidate_id,
            "run_id": run["run_id"],
            "attempt_index": index if phase == "search" else 0,
            "phase": phase,
            "call_class": call_class,
            "simulator_invocations": 1,
            "simulator_call_id": f"budget-probe-{candidate_id}",
            "seed": seed,
            "seed_role": "search" if phase == "search" else "replay",
            "execution_mode": "native",
            "retry_of": None,
            "post_outcome_change": False,
        }

    search_rows = [
        row(
            f"search-probe-{index}",
            phase="search",
            call_class="search_evaluation",
            seed=run["search_seed"],
            index=index,
        )
        for index in range(len(run["candidate_slots"]) + 1)
    ]
    _bad(
        lambda: validate_call_ledger(packet, search_rows, run_budget_limits=run_budget_limits),
        "per-run search-slot budget",
    )

    within_search_budget = search_rows[:-1]
    replay_rows = [
        row(
            f"replay-probe-{index}",
            phase="replay",
            call_class="deterministic_replay",
            seed=9_100_000 + index,
            index=index,
        )
        for index in range(113)
    ]
    _bad(
        lambda: validate_call_ledger(
            packet,
            [*within_search_budget, *replay_rows],
            run_budget_limits=run_budget_limits,
        ),
        "per-run simulator call budget",
    )

    inflated_limits = dict(run_budget_limits)
    inflated_limits[run["run_id"]] = (1000, 1000)
    _bad(
        lambda: validate_call_ledger(
            packet,
            replay_rows,
            run_budget_limits=inflated_limits,
            repo_root=ROOT,
        ),
        "derived budget limits",
    )

    replay_only_limits = dict(run_budget_limits)
    replay_only_limits[run["run_id"]] = (0, 1)
    _bad(
        lambda: validate_call_ledger(
            packet,
            replay_rows[:2],
            run_budget_limits=replay_only_limits,
        ),
        "per-run simulator call budget",
    )


def _result_row(packet: dict, identities: dict) -> dict:
    run = next(item for item in identities["runs"] if item["objective_id"] == "temporal_robustness")
    slot = run["candidate_slots"][0]
    return {
        "schema_version": "temporal-robustness-call-ledger.v1",
        "candidate_id": slot["candidate_id"],
        "packet_digest": identities["packet_digest"],
        "identity_sha256": identities["identity_sha256"],
        "objective_id": run["objective_id"],
        "run_id": run["run_id"],
        "attempt_index": 0,
        "phase": "search",
        "call_class": "search_evaluation",
        "simulator_invocations": 1,
        "simulator_call_id": "fixture-call",
        "seed": run["search_seed"],
        "seed_role": "search",
        "execution_mode": "native",
        "retry_of": None,
        "post_outcome_change": False,
        "temporal_sidecar": _sidecar(packet, slot["candidate_id"]),
    }


def _result_lineage_rows(
    packet: dict,
    identities: dict,
    *,
    confirmed: bool = False,
    confirmation_states: list[str] | None = None,
) -> list[dict]:
    search = _result_row(packet, identities)
    run = next(item for item in identities["runs"] if item["run_id"] == search["run_id"])
    slot = next(
        item for item in run["candidate_slots"] if item["candidate_id"] == search["candidate_id"]
    )
    sidecar = search["temporal_sidecar"]
    sidecar["admission_status"] = "confirmed_failure" if confirmed else "not_admitted"
    if confirmed:
        sidecar["failure_basis"] = "independent_confirmation"
    rows = [search]
    certification = copy.deepcopy(search)
    certification.update(
        phase="certification",
        call_class="certification",
        seed=None,
        seed_role="none",
        simulator_invocations=0,
        simulator_call_id=None,
        certification_state="passed",
    )
    rows.append(certification)
    replay = copy.deepcopy(search)
    replay.update(
        phase="replay",
        call_class="deterministic_replay",
        seed=slot["replay_seed"],
        seed_role="replay",
        simulator_invocations=1,
        simulator_call_id="fixture-replay-call",
        replay_state="passed",
    )
    rows.append(replay)
    states = confirmation_states or ["passed"] * 5
    for offset, (seed, state) in enumerate(zip(slot["confirmation_seeds"], states, strict=True)):
        confirmation = copy.deepcopy(search)
        confirmation.update(
            phase="confirmation",
            call_class="independent_confirmation",
            seed=seed,
            seed_role="confirmation",
            simulator_invocations=1,
            simulator_call_id=f"fixture-confirmation-call-{offset}",
            independent_seed_state=state,
        )
        rows.append(confirmation)
    return rows


def test_result_lineage_requires_all_gate_records(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    search_only = _result_row(packet, identities)
    _bad(
        lambda: validate_result_rows(packet, [search_only], identities),
        "certification lineage",
    )
    rows = _result_lineage_rows(packet, identities)
    result = validate_result_rows(packet, rows, identities)
    assert (result["row_count"], result["simulator_invocations"]) == (8, 7)


def test_result_rows_reject_outcome_fields(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    rows = _result_lineage_rows(packet, identities)
    rows[0]["objective_value"] = 1.0
    _bad(
        lambda: validate_result_rows(packet, rows, identities),
        "outcome fields",
    )


def test_result_rows_reject_admission_status_drift_from_sidecar(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    rows = _result_lineage_rows(packet, identities)
    rows[0]["admission_status"] = "confirmed_failure"
    _bad(
        lambda: validate_result_rows(packet, rows, identities),
        "result/sidecar admission status",
    )


def test_result_rows_reject_supplied_identity_drift(packet: dict) -> None:
    identities = copy.deepcopy(build_expected_identities(packet, repo_root=ROOT))
    run = next(item for item in identities["runs"] if item["objective_id"] == "temporal_robustness")
    run["candidate_slots"][0]["candidate_id"] = "attacker_candidate"
    identities["identity_sha256"] = canonical_sha256(
        {key: value for key, value in identities.items() if key != "identity_sha256"}
    )
    rows = _result_lineage_rows(packet, identities)
    _bad(
        lambda: validate_result_rows(packet, rows, identities),
        "deterministic identities",
    )


def test_confirmed_result_lineage_enforces_three_of_five(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    rows = _result_lineage_rows(
        packet,
        identities,
        confirmed=True,
        confirmation_states=["passed", "passed", "failed", "failed", "failed"],
    )
    _bad(
        lambda: validate_result_rows(packet, rows, identities),
        "3-of-5 threshold",
    )


@pytest.mark.parametrize(
    "phase_state",
    [("certification", "certification_state"), ("replay", "replay_state")],
)
def test_confirmed_result_requires_passing_certification_and_replay(
    packet: dict, phase_state: tuple[str, str]
) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    rows = _result_lineage_rows(packet, identities, confirmed=True)
    phase, state_key = phase_state
    next(row for row in rows if row["phase"] == phase)[state_key] = "failed"
    _bad(
        lambda: validate_result_rows(packet, rows, identities),
        f"{phase} gate must pass",
    )


def test_invalid_search_proposal_cannot_claim_confirmed_failure(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    row = _result_row(packet, identities)
    row.pop("temporal_sidecar")
    row.update(
        call_class="search_invalid_proposal",
        simulator_invocations=0,
        simulator_call_id=None,
        admission_status="confirmed_failure",
    )
    _bad(
        lambda: validate_result_rows(packet, [row], identities),
        "invalid search proposals must be excluded",
    )


def test_invalid_search_proposal_cannot_carry_temporal_sidecar(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    row = _result_row(packet, identities)
    row.update(
        call_class="search_invalid_proposal",
        simulator_invocations=0,
        simulator_call_id=None,
        admission_status="invalid",
    )
    _bad(
        lambda: validate_result_rows(packet, [row], identities),
        "cannot carry temporal sidecar lineage",
    )


def test_invalid_search_proposal_rejects_additional_lineage(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    rows = _result_lineage_rows(packet, identities)
    invalid_search = rows[0]
    invalid_search.pop("temporal_sidecar")
    invalid_search.update(
        call_class="search_invalid_proposal",
        simulator_invocations=0,
        simulator_call_id=None,
        admission_status="invalid",
    )
    _bad(
        lambda: validate_result_rows(packet, rows, identities),
        "invalid search lineage",
    )


def test_result_lineage_rejects_post_outcome_identity_change(packet: dict) -> None:
    identities = build_expected_identities(packet, repo_root=ROOT)
    row = _result_row(packet, identities)
    row["post_outcome_change"] = True
    _bad(lambda: validate_result_rows(packet, [row], identities), "post-outcome")
    row["post_outcome_change"] = False
    row.pop("temporal_sidecar")
    _bad(lambda: validate_result_rows(packet, [row], identities), "temporal_sidecar")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("budget", "budgets"), [16, 32, 63], "budget grid"),
        (("seed_policy", "search_seeds"), [1101, 1101, 3303], "search seeds"),
        (("monitor_contract", "property_ids"), ["clearance"], "property IDs"),
        (("objectives",), [{"id": "new_objective"}], "roster"),
    ],
)
def test_packet_mutations_fail_closed(
    packet: dict, path: tuple[str, ...], value: object, message: str
) -> None:
    broken = copy.deepcopy(packet)
    target = broken
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _bad(lambda: validate_packet(broken, repo_root=ROOT), message)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda packet: packet.update(status="untrusted_status"), "packet metadata.status"),
        (lambda packet: packet.update(unreviewed_claim="failure"), "unsupported fields"),
        (lambda packet: packet["source"].update(base_commit="0" * 40), "cannot be resolved"),
        (
            lambda packet: packet["budget"].update(simulator_call_budget=1),
            "budget grid",
        ),
        (
            lambda packet: packet["budget"].update(unreviewed_field="unexpected"),
            "budget contains unsupported fields",
        ),
        (
            lambda packet: packet["scenario"]["parameters"][0].update(
                unreviewed_field="unexpected"
            ),
            "scenario.parameters",
        ),
        (
            lambda packet: packet["analysis_contract"].update(
                denominator="scheduled_search_attempt_slots"
            ),
            "analysis",
        ),
    ],
)
def test_packet_metadata_budget_and_source_are_fail_closed(
    packet: dict, mutation, message: str
) -> None:
    broken = copy.deepcopy(packet)
    mutation(broken)
    _bad(lambda: validate_packet(broken, repo_root=ROOT), message)


def test_cli_requires_one_explicit_operation_and_uses_check() -> None:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/validation/check_issue_8891_temporal_robustness_packet.py"),
                *args,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    checked = run("--check", "--format", "json")
    assert checked.returncode == 0
    assert json.loads(checked.stdout)["operation"] == "check"
    missing = run("--format", "json")
    assert missing.returncode == 2
    assert "one of the arguments --check --identities --canary is required" in missing.stderr
    conflicting = run("--check", "--identities", "--format", "json")
    assert conflicting.returncode == 2
    assert "not allowed with argument" in conflicting.stderr
