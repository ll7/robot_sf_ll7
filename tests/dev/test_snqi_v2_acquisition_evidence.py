# evidence-writer-exempt: Signed producer fixtures require plain JSONL/SHA256SUMS; retained fixture bytes use shared write_review_sidecar, while deliberate tamper witnesses must keep the old signatures.
"""Protect portable acquired-anchor evidence without resetting or stepping anything."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/context/evidence/2026-10-04_freeze008_calibration"
SPEC = importlib.util.spec_from_file_location(
    "determinism_receipt", ROOT / "scripts/dev/build_snqi_v2_determinism_receipt.py"
)
assert SPEC and SPEC.loader
RECEIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIPT)


def sample_row():
    """Return a minimal recorded row, with a fixed independent scalar example."""
    return {
        "metrics": {"robot_force_impulse_total": 2.0, "jerk_mean": 3.0, "curvature_mean": 4.0},
        "metric_values": {"clearance": float("nan")},
        "steps": 12,
        "status": "success",
        "wall_time": 1.0,
    }


def test_acquisition_proof_has_guard_counters_and_inert_source_audit():
    """The delivered proof must expose safety arbitration, not just mixed execution mode."""
    proof = json.loads((EVIDENCE / "acquisition-proof.json").read_text())
    comparison = proof["rehearsal_comparison"]
    assert "changed_runtime_config_paths" not in comparison
    assert comparison["changed_source_paths_behaviourally_inert_for_this_grid"] is True
    assert comparison["changed_source_paths_behaviourally_inert_reason"]
    assert comparison["changed_source_paths"]
    counters = proof["guard_arbitration_counts"]["guarded_ppo"]
    assert counters["fallback_safe"] > 0
    assert counters["ppo_clear"] > 0
    assert all(type(value) is int and value >= 0 for value in counters.values())


def test_repeat_proof_exposes_independent_snapshot_byte_verification():
    """Cold readback alone must not conceal whether the second copy was checked."""
    proof = json.loads((EVIDENCE / "acquisition-proof.json").read_text())
    custody = proof["repeat_preservation"]
    assert (
        custody["independent_snapshot_files_byte_verified"] == custody["cold_files_byte_verified"]
    )
    assert custody["independent_snapshot_files_byte_verified"] > 0


def test_delivered_determinism_receipt_binds_all_paired_rows():
    """The public claim must bind the actual complete comparison, not a summary-only receipt."""
    proof = json.loads((EVIDENCE / "acquisition-proof.json").read_text())
    binding = proof["determinism_receipt"]
    raw = (ROOT / binding["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
    receipt = json.loads(raw)
    comparison = receipt["original_vs_repeat"]
    pairs = comparison["row_hashes"]
    assert len(pairs) == comparison["rows"] == 1344
    assert len({(row["arm"], row["scenario"], row["seed"]) for row in pairs}) == 1344
    assert {row["seed"] for row in pairs} == {1001, 1002}
    different = sum(row["left_sha256"] != row["right_sha256"] for row in pairs)
    assert different == comparison["different_rows"] == len(comparison["differences"])
    same_environment = (
        receipt["execution_contexts"]["original"] == receipt["execution_contexts"]["repeat"]
    )
    assert receipt["same_recorded_environment"] == same_environment
    if receipt["classification"] == "a":
        assert same_environment and different == 0
    elif receipt["classification"] == "b":
        assert same_environment and different > 0
    else:
        assert not same_environment


def test_metric_row_hash_includes_metrics_steps_status_and_excludes_wall_time():
    """A trajectory change must affect the receipt; bookkeeping time must not."""
    row = sample_row()
    before = RECEIPT.metric_row_sha256(row)
    row["wall_time"] = 99.0
    assert RECEIPT.metric_row_sha256(row) == before
    for field, value in (("steps", 13), ("status", "collision")):
        changed = {**row, field: value}
        assert RECEIPT.metric_row_sha256(changed) != before
    row["metrics"]["jerk_mean"] = 3.0000000000000004
    assert RECEIPT.metric_row_sha256(row) != before


def test_signed_zero_and_nonfinite_sentinels_have_stable_hashes():
    """No approximate equality or JSON NaN extension can conceal a metric-byte change."""
    row = sample_row()
    row["metric_values"] = {"zero": -0.0, "nonfinite": float("inf")}
    expected_payload = {
        "metrics": {
            "robot_force_impulse_total": {"float64_hex": "0x1.0000000000000p+1"},
            "jerk_mean": {"float64_hex": "0x1.8000000000000p+1"},
            "curvature_mean": {"float64_hex": "0x1.0000000000000p+2"},
        },
        "metric_values": {
            "zero": {"float64_hex": "-0x0.0p+0"},
            "nonfinite": {"float64_hex": "inf"},
        },
        "steps": 12,
        "status": "success",
    }
    expected = hashlib.sha256(
        json.dumps(expected_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert RECEIPT.metric_row_sha256(row) == expected
    row["metric_values"]["zero"] = 0.0
    assert RECEIPT.metric_row_sha256(row) != expected


def test_comparison_reports_changed_columns_steps_and_status():
    """Every divergent row remains actionable, including a missing/null column change."""
    key = ("ppo", "classic_doorway_high", 1001)
    left, right = sample_row(), sample_row()
    right["steps"], right["status"] = 13, "collision"
    right["metric_values"]["extra"] = None
    right["metrics"]["jerk_mean"] = 5.0
    result = RECEIPT.compare_rows({key: left}, {key: right})
    assert (
        result["different_rows"] == result["step_differences"] == result["status_differences"] == 1
    )
    difference = result["differences"][0]
    assert difference["left"] == {"steps": 12, "status": "success"}
    assert difference["right"] == {"steps": 13, "status": "collision"}
    assert difference["changed_metric_columns"] == ["metrics.jerk_mean", "metric_values.extra"]


def test_comparison_refuses_missing_row():
    """An incomplete repeat cannot be described as reproducible."""
    with pytest.raises(ValueError, match="comparison grids differ"):
        RECEIPT.compare_rows({("ppo", "doorway", 1001): sample_row()}, {})


@pytest.mark.parametrize(
    ("different_rows", "same_environment", "expected"),
    [
        (0, True, "a"),
        (1, True, "b"),
        (0, False, "unresolved_environment"),
        (1, False, "unresolved_environment"),
    ],
)
def test_repeat_classification_requires_matching_environment(
    different_rows, same_environment, expected
):
    """Neither a reproducibility pass nor same-environment nondeterminism can mask an env change."""
    assert RECEIPT.classify_repeat(different_rows, same_environment) == expected


def test_interpolated_p95_need_not_be_a_raw_sample():
    """The interpolation itself explains an absent raw p95 value without implying lost rows."""
    left, right = sample_row(), sample_row()
    left["metrics"]["jerk_mean"], right["metrics"]["jerk_mean"] = 1.0, 3.0
    data = {("ppo", "doorway", 1001): left, ("ppo", "doorway", 1002): right}
    detail = RECEIPT.jerk_percentile_details(data, 2.0)
    assert detail["zero_based_index"] == 0.95
    assert detail["p95"] == 2.9
    assert detail["rows_above_original_p95"] == 1


@pytest.fixture
def acquisition_builder(monkeypatch):
    """Load the actual analysis builder; no environment or episode is constructed."""
    import sys

    monkeypatch.syspath_prepend(str(ROOT / "scripts/dev"))
    from build_snqi_v2_acquisition_evidence import compare_rehearsal

    return sys.modules[compare_rehearsal.__module__]


@pytest.fixture
def f2_custody(tmp_path, acquisition_builder, monkeypatch):
    """Write signed raw grids with independently chosen metrics and three source bindings."""
    import argparse

    from robot_sf.evidence.writers import write_review_sidecar

    builder = acquisition_builder
    historical = "3e73b04b43aa99b9fbe4a6ab34b89a5a9f1933b6"
    f2 = "66f402ba176b13e45210d0da0b2cf20fcdc0cc02"
    context = {
        "hostname": "fixture-node",
        "cpu_model": "fixture CPU",
        "platform": "Linux/glibc",
        "python_version": "3.13.14",
        "numpy_version": "2.4.6",
        "numba_version": "0.67.0",
        "torch_version": "2.13.0+cu130",
        "stable_baselines3_version": "2.9.0",
        "thread_env": dict.fromkeys(
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"), "1"
        ),
    }
    arms = ["ppo", *[f"arm{i}" for i in range(13)]]

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
        write_review_sidecar(path)

    def bundle(label, source, job):
        root = tmp_path / label
        campaign = root / "benchmarks" / label
        for arm in arms:
            file = campaign / "runs" / (arm + "__differential_drive") / "episodes.jsonl"
            file.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for scenario in range(48):
                for seed in (1001, 1002):
                    row = sample_row()
                    row.update(
                        scenario_id=f"scenario{scenario}",
                        seed=seed,
                        algo=arm,
                        algorithm_metadata={"execution_context": context},
                    )
                    rows.append(json.dumps(row, sort_keys=True))
            file.write_text("\n".join(rows) + "\n")
            write_review_sidecar(file)
        write(campaign / "run_meta.json", {"execution_context": context})
        producer = {
            "source_commit": source,
            "config_sha256": "fe55f5efb6fd885ae86fc978dffc01afd5928fba75128442a6dcd88ae9e94ff3",
            "private_ops_commit": "runtime",
            "launcher_sha256": "launcher",
        }
        write(root / "producer_provenance.json", producer)
        write(
            root / "producer_exit.json",
            {
                "output_status": "complete",
                "campaign_exit_code": 0,
                "sync_exit_code": 0,
                "finalization_errors": [],
            },
        )
        write(root / "startup.json", {"identities": {"public_commit": source, "job_id": job}})
        write(
            root / "f2-environment.json",
            {
                "source_commit": source,
                "slurm_job_id": job,
                "phase": "allocated",
                "execution_context": context,
                "installed_packages": [{"name": "torch", "version": "2.13.0"}],
            },
        )
        return root, campaign, producer

    old_root, old, _ = bundle("historical", historical, "21331")
    a_root, a, producer = bundle("A", f2, "21337")
    b_root, _, _ = bundle("B", f2, "21339")
    _, rehearsal, _ = bundle("rehearsal", "d56092ed", "20299")

    def sign(root):
        members = sorted(
            path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
        )
        (root / "SHA256SUMS").write_text(
            "".join(f"{builder.digest(path)}  {path.relative_to(root)}\n" for path in members)
        )

    for root in (old_root, a_root, b_root):
        sign(root)
    _, old_files = RECEIPT.load_rows(old)
    anchors = tmp_path / "anchors.v2.0.acquired.json"
    write(
        anchors,
        {
            "calibration": {
                "source_commit": historical,
                "run_id": "historical",
                "episode_files_sha256": old_files,
            }
        },
    )
    legacy = json.loads((EVIDENCE / "acquisition-proof.json").read_text())
    legacy.update(
        anchors_sha256=builder.digest(anchors),
        run_id="historical",
        producer_manifest_sha256=builder.digest(old_root / "SHA256SUMS"),
        producer_provenance_sha256=builder.digest(old_root / "producer_provenance.json"),
    )
    proof = tmp_path / "acquisition-proof.json"
    write(proof, legacy)
    monkeypatch.setattr(builder, "HISTORICAL_PROOF_SHA256", builder.digest(proof), raising=False)
    monkeypatch.setattr(builder, "HISTORICAL_ANCHOR_SHA256", builder.digest(anchors), raising=False)
    monkeypatch.setattr(
        builder, "verify_protected_inputs", lambda *_: {"uv.lock": "locked"}, raising=False
    )
    args = argparse.Namespace(
        historical_proof=proof,
        historical_producer_root=old_root,
        producer_root=a_root,
        repeat_producer_root=b_root,
        rehearsal_root=rehearsal,
        rehearsal=ROOT
        / "docs/context/evidence/2026-10-03_issue10112_mintorder/calibration-d56092ed-grid-proof.json",
    )
    scalars = [
        (arm, f"scenario{scenario}", seed, 2.0, 2.0, 0.0, 3.0, 4.0)
        for arm in arms
        for scenario in range(48)
        for seed in (1001, 1002)
    ]
    return args, producer, f2, scalars, a, sign


def test_f2_bound_custody_keeps_historical_audit_scope(acquisition_builder, f2_custody):
    """A fully bound neutral F2 grid is accepted without claiming a new rr10126 audit."""
    args, producer, head, scalars, _, _ = f2_custody
    old, comparison = acquisition_builder.compare_rehearsal(args, producer, head, scalars)
    legacy = json.loads((EVIDENCE / "acquisition-proof.json").read_text())
    assert old == json.loads(args.rehearsal.read_text())["anchors"]["anchors"]
    assert comparison == legacy["rehearsal_comparison"]
    neutrality = acquisition_builder.verify_f2_neutrality(args, producer, head)
    assert neutrality["source_commit"] == head
    assert neutrality["scheduler_job_ids"] == {
        "historical": "21331",
        "original": "21337",
        "repeat": "21339",
    }
    assert neutrality["historical_vs_original"]["identical_rows"] == 1344
    assert neutrality["same_source_repeat_classification"] == "a"


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("unbound", "hash-bound historical"),
        ("new_source", "reviewed freeze"),
        ("proof", "historical acquisition proof digest"),
        ("row", "historical-to-F2 metric"),
        ("repeat_row", "same-source F2 repeat"),
        ("context", "same-source F2 repeat"),
        ("inventory", "installed package inventories"),
        ("learned_context", "execution context mismatch"),
    ],
)
def test_f2_refuses_unbound_or_changed_custody(acquisition_builder, f2_custody, mutation, message):
    """Actual altered custody must refuse on its own gate, not on an unrelated missing fixture."""
    args, producer, head, scalars, _, sign = f2_custody
    if mutation == "unbound":
        args.historical_proof = None
    elif mutation == "new_source":
        head = "a" * 40
    elif mutation == "proof":
        args.historical_proof.write_text(args.historical_proof.read_text() + " ")
    elif mutation in {"row", "repeat_row", "learned_context"}:
        root = args.producer_root if mutation != "repeat_row" else args.repeat_producer_root
        file = (
            root
            / "benchmarks"
            / ("A" if mutation != "repeat_row" else "B")
            / "runs/ppo__differential_drive/episodes.jsonl"
        )
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        if mutation == "learned_context":
            rows[0]["algorithm_metadata"]["execution_context"]["torch_version"] = "changed"
        else:
            rows[0]["metrics"]["jerk_mean"] = 3.0000000000000004
        file.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
        sign(root)
    else:
        file = args.repeat_producer_root / (
            "f2-environment.json" if mutation == "inventory" else "benchmarks/B/run_meta.json"
        )
        data = json.loads(file.read_text())
        if mutation == "inventory":
            data["installed_packages"][0]["version"] = "changed"
        else:
            data["execution_context"]["torch_version"] = "changed"
        file.write_text(json.dumps(data) + "\n")
        sign(args.repeat_producer_root)
    with pytest.raises(ValueError, match=message):
        acquisition_builder.compare_rehearsal(args, producer, head, scalars)


def test_historical_renderer_stays_byte_identical(acquisition_builder, tmp_path):
    """Publishing F2 must not silently rewrite the historical interpretation or replay command."""
    import argparse

    from robot_sf.evidence.writers import write_review_sidecar

    (tmp_path / "determinism-receipt.json").write_bytes(
        (EVIDENCE / "determinism-receipt.json").read_bytes()
    )
    write_review_sidecar(tmp_path / "determinism-receipt.json")
    proof = json.loads((EVIDENCE / "acquisition-proof.json").read_text())
    acquisition_builder.write_review_outputs(
        argparse.Namespace(output_dir=tmp_path, anchors=EVIDENCE / "anchors.v2.0.acquired.json"),
        proof,
    )
    assert (tmp_path / "README.md").read_bytes() == (EVIDENCE / "README.md").read_bytes()
    assert (tmp_path / "metadata.json").read_bytes() == (EVIDENCE / "metadata.json").read_bytes()


@pytest.mark.parametrize("path", ["configs/changed.yaml", "uv.lock", "robot_sf/planner/changed.py"])
def test_f2_protected_input_refuses_changed_blob(acquisition_builder, monkeypatch, path):
    """A source/config drift cannot be hidden by matching metric-row comparisons."""
    monkeypatch.setattr(
        acquisition_builder.subprocess, "check_output", lambda *_a, **_k: path + "\n"
    )
    with pytest.raises(ValueError, match="protected input bytes differ"):
        acquisition_builder.verify_protected_inputs(ROOT, acquisition_builder.F2_SOURCE)


def test_f2_protected_inputs_rehash_actual_config_and_lock(
    acquisition_builder, monkeypatch, tmp_path
):
    """Matching Git objects still require the actual staged config and lock bytes."""
    from robot_sf.evidence.writers import write_review_sidecar

    members = [
        "configs/benchmarks/snqi_v2/calibration.dev1001_1002_scheduled_acquisition.yaml",
        "uv.lock",
    ]
    for name in members:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        write_review_sidecar(path)
    monkeypatch.setattr(
        acquisition_builder.subprocess,
        "check_output",
        lambda command, **_: "\n".join(members) if "ls-tree" in command else "",
    )
    hashes = acquisition_builder.verify_protected_inputs(tmp_path, acquisition_builder.F2_SOURCE)
    assert hashes["uv.lock"] == "def82098b23281e7c49f1f05e052e412c2de54ae2da53f1708bc0ddc2a30d023"
    (tmp_path / "uv.lock").write_bytes((tmp_path / "uv.lock").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="config or dependency lock digest mismatch"):
        acquisition_builder.verify_protected_inputs(tmp_path, acquisition_builder.F2_SOURCE)


def test_delivered_f2_proof_has_actual_sources_jobs_and_policy_versions():
    """The published binding must retain its literal F2 source, accepted hashes and A/B identities."""
    evidence = ROOT / "docs/context/evidence/2026-10-04_freeze008_f2_calibration"
    proof = json.loads((evidence / "acquisition-proof.json").read_text())
    assert proof["source_commit"] == "66f402ba176b13e45210d0da0b2cf20fcdc0cc02"
    assert proof["scheduler_job_id"] == "21337"
    assert (
        proof["anchors_sha256"]
        == "8d86636bcb33a27bab6ba97318516145112aaebe4a4a9713665ec2e39fbc7349"
    )
    binding = proof["determinism_receipt"]
    assert binding["sha256"] == "cd29d6c9a3213e4dbfabc1b8b8c23305066a6a39c5eb088f54d8f8d539827464"
    raw = (ROOT / binding["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
    receipt = json.loads(raw)
    assert receipt["scheduler_job_ids"] == {
        "original": "21337",
        "repeat": "21339",
        "rehearsal": "20299",
    }
    assert receipt["execution_contexts"]["original"]["torch_version"] == "2.13.0+cu130"
    assert receipt["execution_contexts"]["original"]["stable_baselines3_version"] == "2.9.0"
    assert "torch_version" not in receipt["execution_contexts"]["rehearsal"]
    historical = json.loads((EVIDENCE / "acquisition-proof.json").read_text())
    assert proof["rehearsal_comparison"] == historical["rehearsal_comparison"]
    neutrality = proof["historical_to_F2_neutrality"]
    assert neutrality["rehearsal_audit_scope"] == "d56092ed-to-3e73b04b only"
    data = (ROOT / neutrality["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == neutrality["sha256"]
    assert json.loads(data)["historical_vs_original"]["identical_rows"] == 1344


def test_delivered_f2_context_loads_through_release_admission():
    """The real source/anchor/receipt bindings must satisfy the existing release context loader."""
    from robot_sf.benchmark.snqi.execution_context import load_calibration_context

    evidence = ROOT / "docs/context/evidence/2026-10-04_freeze008_f2_calibration"
    actual = load_calibration_context(
        evidence / "anchors.v2.0.acquired.json",
        {
            "determinism_receipt_path": evidence / "determinism-receipt.json",
            "determinism_receipt_sha256": "cd29d6c9a3213e4dbfabc1b8b8c23305066a6a39c5eb088f54d8f8d539827464",
        },
    )
    assert actual["torch_version"] == "2.13.0+cu130"
    assert actual["stable_baselines3_version"] == "2.9.0"


@pytest.mark.parametrize(
    "protected_path",
    ["robot_sf/benchmark/constants.py", "robot_sf/benchmark/metric_definitions.py"],
)
@pytest.mark.parametrize("committed", [False, True], ids=["dirty", "committed"])
def test_protected_metric_inputs_refuse_changed_bytes(
    tmp_path, monkeypatch, acquisition_builder, protected_path, committed
):
    """Metric threshold/definition edits must trip custody checks in Git and on disk."""
    import subprocess

    builder = acquisition_builder

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    for relative in (
        "configs/benchmarks/snqi_v2/calibration.dev1001_1002_scheduled_acquisition.yaml",
        "uv.lock",
        "robot_sf/benchmark/constants.py",
        "robot_sf/benchmark/metric_definitions.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    git("add", "configs", "uv.lock", "robot_sf")

    def commit():
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture metric inputs",
        )

    commit()
    historical = git("rev-parse", "HEAD")
    monkeypatch.setattr(builder, "HISTORICAL_SOURCE", historical)
    builder.verify_protected_inputs(tmp_path, historical)
    target = tmp_path / protected_path
    target.write_bytes(target.read_bytes() + b"\n# changed metric input bytes\n")
    if committed:
        git("add", protected_path)
        commit()
    with pytest.raises(ValueError, match=f"F2 protected input bytes differ:.*{protected_path}"):
        builder.verify_protected_inputs(tmp_path, git("rev-parse", "HEAD"))
