"""Fail-closed candidate receipts for narrowly reviewed default-neutral edits.

This is a PR-only admission route. It never substitutes for release evidence.
Raw evidence stays durable; the compact payload binds independently reviewed
behavioral projections to exact execution sources and hashed raw sweep archives.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import math
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from scripts.ci import behaviour_receipt as gate
from scripts.ci import receipt_outcomes

VERSION = "default-path-equivalence.v1"
SEEDS = (1001, 1002, 1003)
RUNTIME_SOURCE_LEAVES = (
    ("git_hash",),
    ("provenance", "git_hash"),
    ("result_provenance", "repo_commit"),
    ("event_ledger", "software_commit"),
    ("algorithm_metadata", "paired_effect_native_trace", "source_commit"),
    ("algorithm_metadata", "obstacle_force_law", "sites", "fast_pysf", "source_commit"),
    ("algorithm_metadata", "obstacle_force_law", "sites", "socnav_social_force", "source_commit"),
    ("algorithm_metadata", "paired_effect_metric_producer", "source", "source_commit"),
)

# These are transport/provenance leaves, never entire timing/provenance objects.
# Unknown leaves remain compared; reward, actions, metrics and all trace bytes stay.
EXCLUDED_LEAVES = (
    ("git_hash",),
    ("scheduler_job_id",),
    ("scheduler_node",),
    ("provenance", "git_hash"),
    ("provenance", "planner_commit"),
    ("provenance", "artifact_uri"),
    ("provenance", "artifact_sha256"),
    ("result_provenance", "repo_commit"),
    ("event_ledger", "software_commit"),
    ("algorithm_metadata", "paired_effect_native_trace", "source_commit"),
    ("algorithm_metadata", "obstacle_force_law", "sites", "fast_pysf", "source_commit"),
    ("algorithm_metadata", "obstacle_force_law", "sites", "socnav_social_force", "source_commit"),
    ("algorithm_metadata", "paired_effect_metric_producer", "source", "source_commit"),
    ("timestamps", "start"),
    ("timestamps", "end"),
    ("wall_time_sec",),
    ("timing", "steps_per_second"),
)
# Explicit protected surfaces; everything else executable is unknown/refused,
# unless it matches a reviewed AST transition below. No declaration overrides it.
PROTECTED_PREFIXES = (
    "robot_sf/planner/",
    "robot_sf/baselines/",
    "robot_sf/sim/",
    "robot_sf/robot/",
    "robot_sf/nav/",
    "robot_sf/sensor/",
    "robot_sf/ped_npc/",
    "robot_sf/ped_ego/",
    "robot_sf/common/",
    "robot_sf/prediction/",
    "robot_sf/models/",
    "model/",
    "robot_sf/feature_extractors/",
    "fast-pysf/",
    "maps/",
    "configs/algos/",
    "configs/baselines/",
    "configs/planners/",
    "configs/robots/",
    "configs/scenarios/",
    "configs/benchmarks/",
    "robot_sf/benchmark/map_runner/",
    "robot_sf/benchmark/map_runner_policies/",
    "robot_sf/benchmark/safety/",
)
PROTECTED_FILES = gate.BEHAVIOUR_FILES | {
    "robot_sf/feature_extractor.py",
    "robot_sf/benchmark/algorithm_metadata.py",
    "robot_sf/benchmark/paired_effect_metric_contract.py",
    "robot_sf/benchmark/social_compliance.py",
    "robot_sf/benchmark/public_requirement_events.py",
    "robot_sf/benchmark/interaction_exposure.py",
}
# AST hashes include signatures, decorators, docstrings and every body expression.
# Only the documented saved-seed predicate / reward jerk proxy transitions qualify.
# All other module AST bytes must be unchanged. Extending this inventory changes
# admission authority and requires independent review and author promotion.
NEUTRAL_TRANSITIONS = {
    "robot_sf/benchmark/camera_ready/_run_state.py": (
        "_episode_identity",
        "d32535df145c63afe03319b795251a2ae66427ebaa271b8330b08b829d130a86",
        "70e6a192248ae4192871f455cedaccb162ed3b71f86ad6167ffb149d18769c27",
    ),
    "robot_sf/gym_env/reward.py": (
        "_reward_with_terms",
        "b91b0202748542825201ccd551186b3df7091b1bd34c3555664fbbc762ec5268",
        "d366a86eb6f33abdcef9dfe191ca66ea12259e162fe5e7f7ba4cec95cbb39d4d",
    ),
    "robot_sf/gym_env/snqi_proxy.py": (
        "compute_snqi_step_proxies",
        "0d8c7d179a09b17c33f8a96629c7aaeaa7d1ac1e7afcb280509fd704f3076cb0",
        "2504f5cddb3b0b8d2c95561f63592a16fe11fa81b7f92a6c136569f00cbe4318",
    ),
}

# Exact reviewed module transitions for optional training/reward helpers; these
# do not create a general path exemption. Any executable AST change refuses.
NEUTRAL_MODULE_TRANSITIONS = {
    "robot_sf/gym_env/reward.py": [
        (
            "ab08e7171ebfbc95110907c0cea31efb21d08ad7b03eeaed072b17d0cf9f5a71",
            "b61614fee05642448cd0925b5c457c9bfc26dc35ef6bddbbb77bbc4a71edf634",
            "reward-only TTC metadata consumption",
        ),
        (
            "ab08e7171ebfbc95110907c0cea31efb21d08ad7b03eeaed072b17d0cf9f5a71",
            "9cdf786160fbf5d9cb80aefad5e6bcfbe1048b73cd1273537b954789d1c52c65",
            "opt-in reward curriculum weight merge and logging",
        ),
    ],
    "robot_sf/gym_env/snqi_proxy.py": [
        (
            "152bbccf379b44623778b33d7882f2bf38ad94fbd2ce5de7dc112857b872d27b",
            "2d7f1de63661c1bb8ba9ff819e58ce7301ebd6ceef30864090d59452db6503bc",
            "read-only reward TTC scalar computation",
        )
    ],
    "robot_sf/training/released_warm_start.py": [
        (
            "absent",
            "41afb430d6d634c707661646c169896ffc64aa8f74f1040059aaaf9a3ac4991f",
            "standalone opt-in training module; no default imports/registry edits",
        )
    ],
    "scripts/training/prepare_released_ppo_warm_start.py": [
        (
            "absent",
            "655906788d21def7fb37017ad9ea41c64d6229f8a93372d559b1bb8c21e4fbf7",
            "standalone opt-in training module; no default imports/registry edits",
        )
    ],
    "robot_sf/training/scenario_family_weights.py": [
        (
            "absent",
            "3687008e002456ea29fd2078d85789c0eeb18fcec742ad0cca3b628e0d1654fa",
            "standalone opt-in training module; no default imports/registry edits",
        )
    ],
    "scripts/training/train_ppo_with_family_weights.py": [
        (
            "absent",
            "8fcc86d8d7b3e955e36b1a7ce4d9a4348ed11a4a3342068e76b74270bbc07a0a",
            "standalone opt-in training module; no default imports/registry edits",
        )
    ],
}

# Current-main offline analyzer drift: only saved-row audit consumers, no default rollout imports.
NEUTRAL_MODULE_TRANSITIONS["robot_sf/analysis_workbench/release_row_anomalies.py"] = [
    (
        "7c0201bb02313b38be9982d2b85c4b1099313eab1266e06b975706188789803b",
        "c48da00180a844736526440f6edd081a5a9bab05b7a30d510e8397bb253b8d4b",
        "offline saved-row analyzer identity validation drift",
    )
]

# Saved-row publication preflight only: no metric producers or episode mutation.
NEUTRAL_MODULE_TRANSITIONS["robot_sf/benchmark/artifact_publication.py"] = [
    (
        "b04637f955075dee1330546441fef2745f5a8d29b3da283b060d24e26d5ff4a9",
        "cd732059f2bc535a293a2879b23b300a03ed6d5fb07f241b66dd32d196d3980c",
        "saved-row SNQI-v2 publication validation",
    )
]

# Explicit predictive collection/training only; never imported by default matrix.
NEUTRAL_MODULE_TRANSITIONS.update(
    {
        "robot_sf/training/predictive_supervision.py": [
            (
                "abc0845d1ecfd1077593faaa0f39797449c103f7d1ea1fd9e79dce4275d19a6b",
                "c516d09a1fa4991ad7992e030ccde94f13c5d1ef22b5713b5d1d4d0570ba9812",
                "explicit predictive collection/training provenance only",
            )
        ],
        "scripts/training/build_predictive_mixed_dataset.py": [
            (
                "f3aecbb313b485666073a8b71ac917d2c7ff3f2fe7d0784478964a565d6c2ad2",
                "67530f620392ff0539dd066d36f07910577bd1f8db6c6779312fe2cee3a732fe",
                "explicit predictive collection/training provenance only",
            )
        ],
        "scripts/training/collect_predictive_hardcase_data.py": [
            (
                "151916195f3dae52813d23141ed5eff283adbaa9f371f0c794e115abc19194f1",
                "dfdf3108d2dc118f89df8fc847868d764f72180ff8766fee33ec4566f9a9a922",
                "explicit predictive collection/training provenance only",
            )
        ],
        "scripts/training/collect_predictive_planner_data.py": [
            (
                "347aa7da1331609da8a9b94212043adaea98abe913f043397704c5871cb3b0d0",
                "0563aa9c216d08b95a7cd648b1000beba975f7ea0284d1ebabc28bc81b01d9b3",
                "explicit predictive collection/training provenance only",
            )
        ],
        "scripts/training/run_predictive_training_pipeline.py": [
            (
                "69635694c9692be71251149f3b75ba6d7df3768d5219b85c5307ef5d0d634928",
                "ab6ed331902643dbbaa77389cd772ab1f4367f3179727fe3a92c1ad22b8855b7",
                "explicit predictive collection/training provenance only",
            )
        ],
    }
)

# Exact training-only data transitions. Read/hash bytes; never execute the seed config.
NEUTRAL_DATA_TRANSITIONS = {
    "configs/training/predictive/predictive_br07_all_maps_randomized.yaml": (
        "1635fe10709525d2a0555a42174fcb748a1285ca22f4d2532d15b7e7c4b0faa3",
        "a55c7be0be139ec6dc07cc08236a1ec64f250024c89ebbd2386526f1bdabbbb6",
    ),
    "configs/training/predictive/predictive_br07_all_maps_randomized_full.yaml": (
        "182facb7440c36e1e5f4dfafb7ad7ba9a249bfaa1a966d1a2063439d722f00e5",
        "d1e23e1f29477bbec6379d23340c9c517c719e3996233e718b014de9a1959594",
    ),
    "configs/training/predictive/predictive_br07_all_maps_randomized_xl.yaml": (
        "aa76332732631882fc7b9887e94da6289993dca6fd22ee05a6d8c70aa948d7d4",
        "90dba8a6662153d16644391be24bb0dcf119b618f5ab41728e3de59b17e2b21f",
    ),
    "configs/training/predictive/predictive_br07_all_maps_randomized_xl_ego.yaml": (
        "5ea8b2c4f9f6bee90cebca62f29eecd430ab28bf71ee8ab35fd150f94bd419f2",
        "1b3341acf1ccbfe794e53417e09b44750e1068bea95edb75c17ce513ce3c2e7e",
    ),
    "configs/training/predictive/predictive_crossing_conflict_weighted_issue_3254.yaml": (
        "775646625038e9d9f7f0ada0570176f5766f150eef90da16de3f64f7759aaf47",
        "3f1fcb9df0cd45970fc4bbb5b988d23da82b2f43ed3a93f3cc1c839286a570f6",
    ),
    "configs/training/predictive/predictive_development_seeds_v1.yaml": (
        "absent",
        "8601bb8a5eff6432f7e645bd6d422c2a5d91add741dffd5e88104c4ec6d1b920",
    ),
    "configs/training/predictive/predictive_hardcase_training_seeds_v1.yaml": (
        "absent",
        "bad423f3443d2b9df2606a3f21bb51cc676ca6ee5bf0e72494035a0b6acb9976",
    ),
    "configs/training/predictive/predictive_obstacle_features_v1_issue_1167.yaml": (
        "77abc125f1c2ef40e0e559a932c356c9c690915a220b12f9839b9c91d9eeb79b",
        "66678ef245febdb4e432237404dd3b710e7aa096f82d2daaf7df32a8fe6e8b0d",
    ),
}

# Gate authority is base-owned. These finite tools are never imported by default
# rollout code; only unchanged-on-the-PR-branch, base-only drift may be ignored.
# Actual PR edits/deletions of these files remain unknown and require full route.
BASE_ONLY_TOOLING_FILES = {
    "scripts/ci/behaviour_receipt.py",
    "scripts/ci/behaviour_receipt.schema.json",
    "scripts/ci/make_behaviour_receipt.py",
    "scripts/ci/default_path_equivalence.py",
    "scripts/ci/behaviour_receipt_nodes.json",
    "scripts/ci/receipt_provenance.py",
    "scripts/ci/receipt_outcomes.py",
    ".gitattributes",
    "scripts/dev/merge_queue_gate.py",
    "scripts/validation/profile_receipt_loading.py",
    ".github/workflows/merge-queue-gate.yml",
    ".github/workflows/scripts-catalog.yml",
    "docs/dev/ci_job_value_audit_2026_10_12/observations.csv",
    "docs/dev/ci_job_value_audit_2026_10_12/summary.json",
}


# Reviewed preparation/diagnostic blobs on current main, never a prefix waiver.
# Candidate changes refuse; base content and regular-file type must match exactly.
REVIEWED_BASE_ONLY_BLOBS = {
    "SLURM/ftplan/released_ppo_pilot.slurm": "e0e9dbb2127a1e6b491bedb4f2668eacbeee39924de39cf76aede783d678ad8c",
    "configs/training/ppo/ftplan/pilot_protocol_20261012.yaml": "50e891bfe72189dcbc667f8301f4b6357a0846575bd4c5c08c1d32e6cd976548",
    "configs/training/ppo/ftplan/released_warm_weighted_3m_20261012.template": "39b28f6451a64ce99067809e81fd1c875e36fc2735f729fb709fdabe4810d4ee",
    "docs/context/catalog.yaml": "fd7bf84b860af031003632a368bf480b5e4a16e129f3dca87e6b182d8602a4cb",
    "docs/context/evidence/ppo_wall_probe_dev_20261012/probe.json": "d12bb0c7b4c23cd429f840c1d23a50cba597f1f79bad0a9dbbcb3321c7bea6b3",
    "scripts/analysis/ppo_wall_probe_dev.py": "2b41527c564ab48fa6a1694e5875f8c28be67f7246e6c381a7e7b343feaca457",
    "scripts/validation/evidence_registry_baseline.json": "a31b2bffa4585496600b3bd2dcf71e2319049e3934ac7698e2d682458ebd7fd7",
}


def canonical_bytes(value: Any) -> bytes:
    """No numerical tolerance, rounding, trace truncation or reward exclusion."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    """Bind deterministic canonical bytes without floating-point tolerance."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=gate.ROOT, text=True).strip()


def _ast_bytes(node: Any) -> bytes:
    # Python versions add empty optional AST fields (e.g. type_params); omit
    # only empty/None fields so the inventory is stable across supported Python.
    def view(value: Any) -> Any:
        if isinstance(value, ast.AST):
            return {
                "node": type(value).__name__,
                "fields": {
                    key: view(item)
                    for key, item in ast.iter_fields(value)
                    if item is not None and item != []
                },
            }
        if isinstance(value, list):
            return [view(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"literal_type": type(value).__name__, "literal_repr": repr(value)}

    return canonical_bytes(view(node))


def _ast_view(source: str, name: str) -> tuple[str, bytes]:
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    gate._require(len(nodes) == 1, "scope uncertain: missing or duplicate reviewed function")
    node = nodes[0]
    signature = hashlib.sha256(_ast_bytes(node)).hexdigest()
    tree.body.remove(node)
    return signature, _ast_bytes(tree)


def _verify_base_git_attributes(base_source: str, head_source: str) -> None:
    """Only receipt-owned LFS rules may drift on a trusted promoted base."""
    authority_rules = {
        "receipts/behaviour/*.json filter=lfs diff=lfs merge=lfs -text",
        "receipts/behaviour/*-sources/**/*.tar.gz filter=lfs diff=lfs merge=lfs -text",
    }
    remaining = []
    for source in (base_source, head_source):
        result = subprocess.run(
            ["git", "show", f"{source}:.gitattributes"],
            cwd=gate.ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        remaining.append(
            [line for line in result.stdout.splitlines() if line not in authority_rules]
        )
    gate._require(
        remaining[0] == remaining[1], "scope uncertain: unreviewed base Git attribute drift"
    )


def _verify_base_policy_drift(path: str, base_source: str, head_source: str) -> None:
    if path == ".gitattributes":
        _verify_base_git_attributes(base_source, head_source)


def _verify_reviewed_base_blob(path: str, base_source: str) -> None:
    entry = _git("ls-tree", base_source, "--", path).split("\t")
    gate._require(
        len(entry) == 2
        and entry[1] == path
        and entry[0].split()[:2] in (["100644", "blob"], ["100755", "blob"]),
        f"scope refused: reviewed base-only path is not a regular blob {path}",
    )
    raw = subprocess.check_output(
        ["git", "cat-file", "blob", f"{base_source}:{path}"], cwd=gate.ROOT
    )
    gate._require(
        hashlib.sha256(raw).hexdigest() == REVIEWED_BASE_ONLY_BLOBS[path],
        f"scope refused: reviewed base-only blob changed {path}",
    )


def _reviewed_base_drift(path: str, base_source: str, introduced: set[str]) -> bool:
    """Check the finite reviewed map after protected paths have been refused.

    Returns:
        Whether an unchanged candidate path has the approved regular base blob.
    """
    if path not in REVIEWED_BASE_ONLY_BLOBS:
        return False
    gate._require(
        path not in introduced,
        f"scope refused: candidate changed reviewed base-only path {path}",
    )
    _verify_reviewed_base_blob(path, base_source)
    return True


def classify_scope(changed_files: list[str], base_source: str, head_source: str) -> list[str]:
    """Verify current Git bytes; unknown and default-path surfaces require full receipt."""
    base_source = _git("rev-parse", f"{base_source}^{{commit}}")
    head_source = _git("rev-parse", f"{head_source}^{{commit}}")
    # The PR merge-base diff is insufficient for a stale branch: execution runs
    # its exact source, including default changes absent from current main.
    actual = _git("diff", "--name-only", "--no-renames", base_source, head_source).splitlines()
    paths = sorted(set(changed_files) | set(actual))
    merge_base = _git("merge-base", base_source, head_source)
    introduced = set(
        _git("diff", "--name-only", "--no-renames", merge_base, head_source).splitlines()
    )
    # A candidate change ending at the same bytes as main is absent from the
    # two-tree diff. It must still refuse the finite reviewed-blob exception.
    paths = sorted(set(paths) | (introduced & REVIEWED_BASE_ONLY_BLOBS.keys()))
    for path in paths:
        if (
            path.startswith("receipts/behaviour/")
            or path.endswith(".md")
            or path.startswith("tests/")
        ):
            continue
        gate._require(
            path not in PROTECTED_FILES and not path.startswith(PROTECTED_PREFIXES),
            f"scope refused: protected default path {path}",
        )
    admitted = []
    for path in paths:
        if _reviewed_base_drift(path, base_source, introduced) or (
            path in BASE_ONLY_TOOLING_FILES and path not in introduced
        ):
            _verify_base_policy_drift(path, base_source, head_source)
            continue
        if (
            path.startswith("receipts/behaviour/")
            or path.endswith(".md")
            or path.startswith("tests/")
        ):
            continue
        gate._require(
            path not in PROTECTED_FILES and not path.startswith(PROTECTED_PREFIXES),
            f"scope refused: protected default path {path}",
        )
        if path in NEUTRAL_DATA_TRANSITIONS:
            observed = []
            for source in (base_source, head_source):
                result = subprocess.run(
                    ["git", "show", f"{source}:{path}"],
                    cwd=gate.ROOT,
                    capture_output=True,
                    check=False,
                )
                observed.append(
                    hashlib.sha256(result.stdout).hexdigest()
                    if result.returncode == 0
                    else "absent"
                )
            gate._require(
                tuple(observed) == NEUTRAL_DATA_TRANSITIONS[path],
                f"scope uncertain: unreviewed training data transition {path}",
            )
            admitted.append(f"{path}:reviewed-data-transition:{observed[1]}")
            continue
        if path in NEUTRAL_MODULE_TRANSITIONS:
            before_result = subprocess.run(
                ["git", "show", f"{base_source}:{path}"],
                cwd=gate.ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            before_source = before_result.stdout if before_result.returncode == 0 else None
            after_source = _git("show", f"{head_source}:{path}")
            before_hash = (
                "absent"
                if before_source is None
                else hashlib.sha256(_ast_bytes(ast.parse(before_source))).hexdigest()
            )
            after_hash = hashlib.sha256(_ast_bytes(ast.parse(after_source))).hexdigest()
            options = NEUTRAL_MODULE_TRANSITIONS[path]
            if any((before_hash, after_hash) == option[:2] for option in options):
                admitted.append(f"{path}:reviewed-module-transition:{after_hash}")
                continue
        gate._require(
            path in NEUTRAL_TRANSITIONS, f"scope uncertain: unreviewed executable path {path}"
        )
        name, old_hash, new_hash = NEUTRAL_TRANSITIONS[path]
        before = _ast_view(_git("show", f"{base_source}:{path}"), name)
        after = _ast_view(_git("show", f"{head_source}:{path}"), name)
        gate._require(
            before[1] == after[1] and (before[0], after[0]) == (old_hash, new_hash),
            f"scope uncertain: unreviewed AST transition {path}:{name}",
        )
        admitted.append(f"{path}:{name}")
    gate._require(bool(admitted), "scope uncertain: no reviewed neutral transition")
    return admitted


def _source_json_loads(raw: str | bytes, *, legacy_raw: bool = False) -> Any:
    """One duplicate/overflow guard; explicit legacy constants are raw-only."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            gate._hardware_require(
                key not in result, "equivalence source JSON duplicate object key refused"
            )
            result[key] = value
        return result

    def finite_float(literal: str) -> float:
        value = float(literal)
        gate._hardware_require(
            math.isfinite(value), "equivalence source JSON non-finite number refused"
        )
        return value

    def refuse_constant(literal: str) -> Any:
        if legacy_raw:
            return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[literal]
        raise gate.HardwareIdentityError(
            "hardware comparison invalid: equivalence source JSON non-finite number refused"
        )

    return json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_float=finite_float,
        parse_constant=refuse_constant,
    )


def strict_json_loads(raw: str | bytes) -> Any:
    """Claims and metadata require unique fields and finite standard JSON numbers."""
    return _source_json_loads(raw)


def raw_episode_loads(raw: str | bytes) -> Any:
    """Only raw episodes may retain explicit producer NaN/Infinity tokens."""
    row = _source_json_loads(raw, legacy_raw=True)
    validate_transport_numbers(row)
    return row


def raw_token_bytes(row: dict) -> bytes:
    """Keep legacy numerical tokens distinct; never use for receipt/claim hashing."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()


def same_raw_rows(left: dict, right: dict) -> bool:
    """Independent NaN parses must compare tokens, not float self-equality."""
    return set(left) == set(right) and all(
        raw_token_bytes(left[key]) == raw_token_bytes(right[key]) for key in left
    )


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(child) for child in value)
    return False


def validate_transport_numbers(row: dict) -> None:
    """Non-finite values cannot disappear in an excluded transport leaf."""
    for path in EXCLUDED_LEAVES:
        current = row
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            gate._hardware_require(
                not _contains_nonfinite(current),
                "equivalence source JSON non-finite number refused in transport leaf",
            )


def validate_source_json(root: Path) -> None:
    """Only this route rejects duplicate fields in every sealed semantic input."""
    from scripts.ci import receipt_provenance

    inputs = [*receipt_provenance.source_files(root), root / receipt_provenance.BINDING_FILE]
    inputs.extend(
        path
        for path in (root / "execution_request.json", root / "execution_scheduler.json")
        if path.is_file()
    )
    for path in inputs:
        if path.name.endswith(".jsonl"):
            decoder = (
                raw_episode_loads
                if path.name == "episodes.jsonl" and path.relative_to(root).parts[0] == "campaigns"
                else strict_json_loads
            )
            for line in path.read_text().splitlines():
                if line.strip():
                    decoder(line)
        else:
            strict_json_loads(path.read_bytes())


def validate_runtime_source(row: dict, source: str) -> None:
    """Only the finite producer-defined rollout aliases can be source transport."""
    for path in RUNTIME_SOURCE_LEAVES:
        current = row
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            gate._hardware_require(
                current == source, "provenance digest original row source differs"
            )


def episode_bytes(row: dict[str, Any]) -> bytes:
    """Compare all episode fields except the exact documented transport leaves."""
    receipt_outcomes.raw_outcomes(row)
    validate_transport_numbers(row)
    projected = copy.deepcopy(row)
    trace = projected.get("algorithm_metadata", {}).get("simulation_step_trace")
    gate._require(
        isinstance(trace, dict) and isinstance(trace.get("reset"), dict),
        "equivalence refused: missing reset/controller trace",
    )
    steps = trace.get("steps")
    gate._require(
        isinstance(steps, list) and bool(steps), "equivalence refused: missing step trace"
    )
    for step in steps:
        gate._require(
            isinstance(step, dict)
            and all(key in step for key in ("robot", "pedestrians", "planner", "collision", "rl")),
            "equivalence refused: incomplete action/pose/event/reward trace",
        )
        planner = step["planner"]
        gate._require(
            isinstance(planner, dict)
            and all(key in planner for key in ("selected_action", "applied_environment_action")),
            "equivalence refused: missing requested/applied action trace",
        )
    for path in EXCLUDED_LEAVES:
        parent = projected
        for part in path[:-1]:
            if not isinstance(parent, dict):
                break
            parent = parent.get(part)
        if isinstance(parent, dict):
            parent.pop(path[-1], None)
    return raw_token_bytes(projected)


def comparison_rows(
    head_rows: dict,
    baseline_rows: dict,
    scope: dict,
    *,
    head_execution_statuses: dict | None = None,
    baseline_execution_statuses: dict | None = None,
    allocation_identities: dict | None = None,
) -> list[dict]:
    """Produce complete paired digests for diagnostics, never admission by itself."""
    expected = set(itertools.product(scope["arms"], scope["maps"], SEEDS))
    gate._require(
        set(head_rows) == expected == set(baseline_rows),
        "equivalence refused: incomplete 14-arm/51-slot/three-seed matrix",
    )
    for statuses in (head_execution_statuses, baseline_execution_statuses):
        gate._require(
            statuses is None or set(statuses) == expected,
            "equivalence refused: missing paired summary execution audit",
        )
    from robot_sf.benchmark.algorithm_metadata import (
        canonical_algorithm_name,
        enrich_algorithm_metadata,
    )
    from scripts.ci.make_behaviour_receipt import _execution_evidence, require_valid_execution

    rows = []
    for key in sorted(expected):
        execution_evidence = {}
        for side, row in (("head", head_rows[key]), ("baseline", baseline_rows[key])):
            receipt_outcomes.raw_outcomes(row)
            if allocation_identities is not None:
                allocation = allocation_identities[side]
                gate._hardware_require(
                    row.get("scheduler_job_id") in gate.scheduler_job_ids(allocation["job_id"]),
                    f"{side} source row job differs from verified allocation",
                )
                gate.validate_source_row_context(row, row["scheduler_job_id"], allocation["node"])
            statuses = head_execution_statuses if side == "head" else baseline_execution_statuses
            # This separate audit view may bind the authoritative summary status.
            # The row passed to episode_bytes below remains completely untouched.
            audit_view = row if statuses is None else {**row, "execution_status": statuses[key]}
            evidence = _execution_evidence(audit_view)
            execution_evidence[side] = evidence
            expected_algorithm = canonical_algorithm_name(
                scope.get("arm_algorithms", {}).get(key[0], key[0])
            )
            profile = enrich_algorithm_metadata(algo=expected_algorithm)["planner_kinematics"]
            required_mode = "native" if profile["supports_native_commands"] else "adapter"
            gate._require(
                evidence["algorithm"] == expected_algorithm
                and evidence["execution_mode"] == required_mode
                and (profile["supports_native_commands"] or profile["supports_adapter_commands"]),
                f"equivalence refused: {side} wrong arm/command execution {key}",
            )
            gate._require(
                evidence["controller_executed"]
                and not evidence["fallback"]
                and not evidence["degraded"],
                f"equivalence refused: {side} fallback/degraded/unexecuted {key}",
            )
            require_valid_execution(evidence, expected_algorithm, profile)
        source_allocations = {}
        if allocation_identities is not None:
            source_allocations = {
                "scheduler_job_id": head_rows[key]["scheduler_job_id"],
                "scheduler_node": head_rows[key]["scheduler_node"],
                "baseline_scheduler_job_id": baseline_rows[key]["scheduler_job_id"],
                "baseline_scheduler_node": baseline_rows[key]["scheduler_node"],
            }
        base_bytes, head_bytes = episode_bytes(baseline_rows[key]), episode_bytes(head_rows[key])
        rows.append(
            {
                **source_allocations,
                "arm": key[0],
                "map": key[1],
                "seed": key[2],
                "baseline_sha256": hashlib.sha256(base_bytes).hexdigest(),
                "head_sha256": hashlib.sha256(head_bytes).hexdigest(),
                "head_execution": execution_evidence["head"],
                "baseline_execution": execution_evidence["baseline"],
            }
        )
    return rows


def compare_rows(head_rows: dict, baseline_rows: dict, scope: dict) -> list[dict]:
    """One changed behavioral byte refuses, including every learned arm."""
    rows = comparison_rows(head_rows, baseline_rows, scope)
    for row in rows:
        gate._require(
            row["baseline_sha256"] == row["head_sha256"],
            f"equivalence refused: changed behavioral byte {gate._key(row)}",
        )
    return rows


def validate_archive_paths(rows: list[dict], identities: dict[str, dict[str, str]]) -> None:
    """Bind compact raw artifact paths to each independently verified row allocation."""
    for row in rows:
        for side, prefix in (("head", ""), ("baseline", "baseline_")):
            jobs = gate.scheduler_job_ids(identities[side]["job_id"])
            path = row[f"{side}_path"]
            job = row[prefix + "scheduler_job_id"]
            gate._require(
                isinstance(path, str)
                and (
                    path.startswith(f"{job}/campaigns/")
                    or (len(jobs) == 1 and path.startswith("campaigns/"))
                ),
                f"equivalence archive path differs from {side} row job",
            )


def validate_receipt(
    receipt: dict,
    scope: dict,
    head: str,
    base_source: str,
    changed_files: list[str],
    *,
    is_release: bool = False,
) -> None:
    """Admission consumes committed evidence with independent exact-head review."""
    gate._require(not is_release, "equivalence refused: releases require full behaviour receipt")
    gate._require(receipt.get("schema_version") == VERSION, "invalid equivalence schema")
    gate._require(
        receipt.get("admission_context") == "pull_request",
        "equivalence refused: receipt is not PR-only",
    )
    gate._require(
        receipt.get("declaration") == "default_benchmark_behaviour_unchanged",
        "missing default-path equivalence declaration",
    )
    identities = gate.validate_hardware_pair(receipt.get("hardware"))
    gate.validate_scheduler_rows(receipt.get("rows", []), identities)
    validate_archive_paths(receipt.get("rows", []), identities)
    gate._require(receipt["head_sha"] == head, "equivalence receipt head is stale")
    gate._require(
        receipt["baseline"]["source_sha"] == base_source,
        "equivalence baseline differs from current main",
    )
    gate._require(receipt["scope_sha256"] == digest(scope), "equivalence scope inventory is stale")
    gate._require(
        len(scope["arms"]) == 14 and len(scope["maps"]) == 51,
        "equivalence requires the full 14-arm/51-slot scope",
    )
    gate._require(receipt["seeds"] == list(SEEDS), "equivalence requires dev seeds 1001-1003")
    gate._require(
        receipt["projection_sha256"] == digest(EXCLUDED_LEAVES),
        "equivalence transport whitelist differs",
    )
    execution = receipt["scheduler"]["source_sha"]
    gate._require(receipt["head_source_sha"] == execution, "equivalence execution source differs")
    gate.require_receipt_only_ancestor(execution, head, "equivalence job")
    gate._require(
        receipt["scope_transitions"] == classify_scope(changed_files, base_source, execution),
        "equivalence scope transition inventory differs",
    )
    for side, source in (("head", execution), ("baseline", base_source)):
        gate._require(identities[side]["source_sha"] == source, f"{side} hardware source differs")
        artifact = receipt["artifact"] if side == "head" else receipt["baseline"]["artifact"]
        gate._require(
            gate._uri(artifact["uri"]) and gate._digest(artifact["sha256"]),
            f"missing durable {side} raw artifact identity",
        )
    tooling = receipt["tooling_sha"]
    gate._require(gate._digest(tooling, 40), "missing acquisition tooling source")
    gate._require(
        receipt["tooling_path"] == "scripts/validation/run_empty_world_sweep.py",
        "unreviewed acquisition tooling path",
    )
    tooling_bytes = subprocess.check_output(
        ["git", "show", f"{tooling}:{receipt['tooling_path']}"], cwd=gate.ROOT
    )
    gate._require(
        hashlib.sha256(tooling_bytes).hexdigest() == receipt["tooling_file_sha256"],
        "acquisition tooling differs from pinned Git blob",
    )
    gate._require(gate._digest(receipt["harness_sha256"]), "missing acquisition wrapper digest")
    gate._require(
        receipt["harness_path"] == "acquisition_harness.py",
        "unreviewed acquisition wrapper artifact path",
    )
    jobs = [identities[side]["job_id"] for side in ("head", "baseline")]
    gate._require(
        receipt["scheduler"]["job_id"]
        == (jobs[0] if jobs[0] == jobs[1] else "_".join(jobs)).replace(",", "_"),
        "equivalence scheduler identity differs from hardware jobs",
    )
    expected = set(itertools.product(scope["arms"], scope["maps"], SEEDS))
    rows = receipt["rows"]
    keys = [gate._key(row) for row in rows]
    gate._require(
        len(keys) == len(set(keys)) and set(keys) == expected,
        "equivalence refused: incomplete/duplicate paired episode coverage",
    )
    from robot_sf.benchmark.algorithm_metadata import (
        canonical_algorithm_name,
        enrich_algorithm_metadata,
    )

    for row in rows:
        algorithm = canonical_algorithm_name(
            scope.get("arm_algorithms", {}).get(row["arm"], row["arm"])
        )
        profile = enrich_algorithm_metadata(algo=algorithm)["planner_kinematics"]
        mode = "native" if profile["supports_native_commands"] else "adapter"
        for side in ("head", "baseline"):
            evidence = row[f"{side}_execution"]
            gate._require(
                evidence["algorithm"] == algorithm
                and evidence["execution_mode"] == mode
                and (profile["supports_native_commands"] or profile["supports_adapter_commands"]),
                f"equivalence refused: {side} wrong arm/command execution {gate._key(row)}",
            )
            gate._require(
                evidence["controller_executed"] is True
                and evidence["fallback"] is False
                and evidence["degraded"] is False,
                f"equivalence refused: {side} fallback/degraded/unexecuted {gate._key(row)}",
            )
            from scripts.ci.make_behaviour_receipt import require_valid_execution

            require_valid_execution(evidence, algorithm, profile)
        gate._require(
            gate._digest(row["head_sha256"]) and row["head_sha256"] == row["baseline_sha256"],
            f"equivalence refused: changed behavioral byte {gate._key(row)}",
        )
    gate._require(
        receipt["totals"] == {"episodes": len(rows), "changed_episodes": 0},
        "equivalence totals contradict comparisons",
    )
    review = receipt["refute_review"]
    gate._require(
        review["head_sha"] == head and review["verdict"] == "accepted" and gate._uri(review["uri"]),
        "equivalence independent exact-head verdict is missing",
    )
    gate._require(
        review["comparison_sha256"] == digest(rows),
        "equivalence review does not bind the comparison evidence",
    )
