"""
Git hook to prevent absolute home-dir paths in ``configs/**`` and committed
evidence packets under ``docs/context/evidence/**``.

Autonomously-generated configs and evidence packets occasionally hardcode
absolute user-home paths (``/home/<user>/...``, including author-specific
worktree paths), which are non-portable for other contributors and automated
runners and violate the repository's reproducibility hard-rule. This hook fails
when a tracked config or evidence file contains such a path, UNLESS the line is
explicitly annotated as intentional (e.g. private-ops SLURM routing) with an
``allow-abs-path`` marker.

Configs coverage was added in issue #3605. Evidence-packet coverage was added in
issue #4324 after the same defect recurred in a committed provenance file
(``config.path`` / ``gate_spec.path`` baking in a local worktree path; cf. the
#4302/#4303 SNQI fix). To keep the guard cheap it scans only the files it is
handed (staged files at commit time), plus the whole tracked tree under ``--all``.
"""

import argparse
import hashlib
import logging
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# Absolute home-dir prefixes that should not appear in portable configs/evidence.
ABS_PATH_PATTERN = re.compile(r"(/home/|/Users/|/root/)")

# A line carrying this marker is an intentional, documented absolute path.
ALLOW_MARKER = "allow-abs-path"

CONFIG_ROOT = Path("configs")
EVIDENCE_ROOT = Path("docs/context/evidence")

# Directory roots whose committed files must stay free of absolute home-dir paths.
SCANNED_ROOTS = (CONFIG_ROOT, EVIDENCE_ROOT)

# Evidence packets committed *before* the guard was extended to
# ``docs/context/evidence/**`` (issue #4324). These historical benchmark
# command/provenance records already bake in absolute worktree paths in durable,
# checksummed artifacts. They are grandfathered so the guard can fail closed for
# NEW packets without forcing a retroactive rewrite (and SHA256SUMS churn) of
# durable historical evidence. This is tracked pre-existing debt: do NOT add new
# entries here — fix the leak at generation time instead.
LEGACY_EVIDENCE_ALLOWLIST = frozenset(
    {
        "issue_1023_candidate_augmented_local_full_2026-05-06",
        "issue_1023_scenario_horizons_local_full_2026-05-06",
        "issue_1023_scenario_horizons_preflight_2026-05-06",
        "issue_1454_stage_a_fixed_h100_2026-05-22",
        "issue_1470_oracle_imitation_traces_12911_2026-06-17",
        "issue_1475_orca_residual_bc_smoke_12913_2026-06-17",
        "issue_2258_topology_primary_route_audit_2026-06-05",
        "issue_2282_topology_selection_instrumentation_2026-06-05",
        "issue_3266_ppo_snqi_smoke_2026-06-23",
    }
)

# Verbatim evidence sources may retain absolute paths only when the exact producer bytes are
# separately bound and a portable normalized copy is retained. These exceptions are file- and
# digest-specific: any change to a pinned file restores the normal absolute-path scan.
PINNED_VERBATIM_EVIDENCE_SHA256 = {
    (
        "docs/context/evidence/issue_3810_h600_interpretation_2026-07/"
        "source_reports/13268/campaign_summary.json"
    ): "f29e6c5ee12679408b1d65add0149e4cfe07390f0c8828208114f39dd900c257",
    (
        "docs/context/evidence/issue_3810_h600_interpretation_2026-07/"
        "source_reports/13273/campaign_summary.json"
    ): "f456580bad70167e42d6e24c9570547042fd02ce39c35687ed152928f6a0698e",
    (
        "docs/context/evidence/issue_6474_social_compliance_nominal_campaign_manifest.json"
    ): "10c45f44ec5679144671c6247644a7e88b1444fdf9b25a7373b343bbf732e1bc",
    (
        "docs/context/evidence/issue_6102_robot_speed_tier_recovery/README.md"
    ): "14a131eceb2f767e70609d573d32a942ba15e703378d2bb921cb7da82e768179",
    (
        "docs/context/evidence/issue_6102_robot_speed_tier_recovery/recovery_manifest.json"
    ): "2e14b777170450825f7671418ea8ed7130576adbd6bd473bf0d63062d9ee49ae",
    # Issue #9645 retained exact producer records recovered from commit
    # 7588b785a607680400adcc6398d8c983c160e4bd. Portable analysis uses the separately
    # hash-bound payload/path_normalized_episode_records copies; only these exact source files
    # contain the original local route-override paths.
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0000/episode_records.jsonl": "34f84b7c414f963a1af78ee0c8db0a69c142119c4d1ebb8fb0be915c0170d847",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0001/episode_records.jsonl": "0f9ad563110a2e7e941606ce2a4b7507a4c34ef24887ba3561de1a095612a37b",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0002/episode_records.jsonl": "d7d8d1b0b43c97ac6d2e9e5c51f793d524a0f06ad0d2adecc2f1312d1052ad21",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0003/episode_records.jsonl": "6611858cd492dd56a5cd146d6da6159991fca3e519c19bed00de4db8ebab3688",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0004/episode_records.jsonl": "679d218d404a35eee5372e33e3ed193dbee397c42a5af00a21b88efc755a5494",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0005/episode_records.jsonl": "129f88933ba632bebb134ddf4edf85babc8ef75d4a0a2f3f72c6139516d6f78a",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0006/episode_records.jsonl": "19608886d21a45b10ad776ad33f7ca0a5dea7143b2fde0cc77efd56c63552ac9",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0007/episode_records.jsonl": "36509fe895aed6b77c5072cad3df4a0735d1f5fb1f92baca2d13df18df0959b8",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0008/episode_records.jsonl": "3d05b55ed21e3c2237360e0b4df4d21aae3cca4910de043fe967c154c4d21812",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0009/episode_records.jsonl": "bfc441ff29d1ea5e4d7ac5790d49e9acbee4d3f3f2d82c45e5a56783028a8711",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0010/episode_records.jsonl": "d58af050a3b43e187ab6437af2733164f62bb3efce572f56bc987b2ccd2b94ed",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0011/episode_records.jsonl": "c48d75e4a850de8d73499ef8c8d13d96d87d982c45bcff0a194f3b4947ab00b8",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0012/episode_records.jsonl": "bbf8cee9675afbfb539ed108d689381316b628ed10da4b1fcd985bc51a6dc632",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0013/episode_records.jsonl": "4fbdc7f47876c2c7c59e093bd04429f553cf678cdb1fd31a650141b1f4e6685c",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0014/episode_records.jsonl": "7e597221cb9a56a2a3c9fab6a57e4837ed154c28a0a848b4176690b0fe9bb762",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_optuna/candidate_0015/episode_records.jsonl": "65d8be5acb6889b55463e457dd25e5ca422022c7ef50063726993002d138e56d",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0000/episode_records.jsonl": "5a5608b8819dafc3766fd4a66ea5a4b77edffdaec247c052b2f22a1824ae8dc3",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0001/episode_records.jsonl": "fd757ec1066bd2a2a8b80c8f3bc4c4ae09349e8ccd46c5dfd8894e4557f30d3b",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0002/episode_records.jsonl": "8b7fd6b6f365610dcb0ce9cccf61dd3e01faff1df941fd2868296f01c44bd130",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0003/episode_records.jsonl": "d0f163d18de7603818fcb8d0cfb1871185c2c85db2adc2f01a12ec3351f04e72",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0004/episode_records.jsonl": "a6ef9029d4b7208f7053e8e197a8ff618a4360cb2be654e87f1dc5480e5675e3",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0005/episode_records.jsonl": "53584c4b586f2aa031524bd84d814930445f8ea9233bd0b3ce6e486f2282cb23",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0006/episode_records.jsonl": "9796c9f89f512f57d4c101d4b600a1c0befc10ed7ceefa1113e1a3f8d81cbaf2",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0007/episode_records.jsonl": "e8922b2cf62dbc462fb16a36a2485666a5209915a5bb3e5dbcd269d4ac80e789",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0008/episode_records.jsonl": "037e992b241b745c4482ba6b8ca4f757f897d978017fb33879c0ca9b8d1ad9cb",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0009/episode_records.jsonl": "28b2efe2e283ffebce3886be133b35be343614092d6b5629740c32e1720884dd",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0010/episode_records.jsonl": "a1c422c2a553017598731bd8669873e5ec72d7150eff457304f4129c10f424bf",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0011/episode_records.jsonl": "224eda922a6b3329ee085631a8cc8c649321492f6dd81f3103aa18df56419ebc",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0012/episode_records.jsonl": "5af855ee80ce54f402610b2b8e786a6fc0e9206a85cd4d6bf27f0a89eb8af834",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0013/episode_records.jsonl": "1fae12e439e1e66ebdb3327b93d0cc05674956144c2cc15a2895166169f90c30",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0014/episode_records.jsonl": "357ee6eb67c576c65306c7cd3ffd999bd41394d4fb2828c6cead8c02571c2f62",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_1101_random/candidate_0015/episode_records.jsonl": "c5e7ecfffdfd3a4eed8a70e4c5f8a0bd107d79bc07fe1cbcdc96f7dd3f62280c",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0000/episode_records.jsonl": "e6f46057e25fb908ca006eef30fbbe72a88c40850c9f20a035ac7f171b9c45f3",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0001/episode_records.jsonl": "79db78d67bb8a44e4fb82338e15d965feb9aecee13042e3f71330966963f9882",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0002/episode_records.jsonl": "28d2c178ac86c2286aa943d5b316dcff1de1fe9b48bdeebb3f8a851efcb22c8c",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0003/episode_records.jsonl": "6325b82cab5d9dfee988fb272992c70dfd2522d49a13357e5b56bfe6ba4f57e4",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0004/episode_records.jsonl": "a6043c6aaa6d01cd4a1600a6c832d21f17802e6ab7425ba0967553636dce223c",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0005/episode_records.jsonl": "3a95f10cd82718f2a750564f9f4aaac4ae0c92cef958bf69c5057640c6cb22b7",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0006/episode_records.jsonl": "8a88d4b73976b970fc66b5ec5efbf4eb1830fcdbb13f8c5c941e06e17cee8a0e",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0007/episode_records.jsonl": "859baf6772cb9261b631120a86a9a1cb7fd98d6cec35e148b113de14f61ac153",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0008/episode_records.jsonl": "ef9ceaa604a8f5aeb29218849ea3e5d061a716703b621815a782606d723aef2d",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0009/episode_records.jsonl": "dbad9f6f87687bf0f6c0c808aa6734b0376b80e43a198ab317a30d5de622912b",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0010/episode_records.jsonl": "292e9a10bb4d306ddd08e92891fcfc9be4efd3423f18ae613999c866e3e4e45f",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0011/episode_records.jsonl": "e8bb366f21f6f74338d6d6a07db891c9f288d117d5a9dab4071bd06670f2797e",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0012/episode_records.jsonl": "1dff92c2518384f0183c202dff4939a5b8c270fea05c0527fb4ef901c6c24b85",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0013/episode_records.jsonl": "4e3f0fc1e28fb5d63fc17a95bf5e06c9eb1e25edf6288fbfb8b97ea8a8a70550",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0014/episode_records.jsonl": "b3716b987c7ea4327a3ea4847b26275dd17b57e66ce7965528d8e01e611ef3fb",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_optuna/candidate_0015/episode_records.jsonl": "24b163e7e0f001d261041615801c8a7e4f2497c8e765cd7abd166c6c28fa36a8",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0000/episode_records.jsonl": "df0731a655d9de85a636c759df5cd490ddf8e457ebbda5837dd535949e4a8374",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0001/episode_records.jsonl": "e9e81b95d073c3afc7a8afb6001036f0ce47fe1de90b6a4a6a7b058cd0aff4d6",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0002/episode_records.jsonl": "3ccd6464d292593f5fb064256cdcb5b7c842305fd14d5ea8e66d29b23d17ea51",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0003/episode_records.jsonl": "a7812c2b475b1604ea3cda796c977285e101ae52114d418b3f4eb08d67c4a7c9",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0004/episode_records.jsonl": "0663cd167c642c0dc2efffdb24c5bb38a979a85b12b1a45ebb84ea1c642d2b82",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0005/episode_records.jsonl": "98204c97094d8cab790757a5ab9935a89d53a825ffb2e5a5962956a2a9ad483a",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0006/episode_records.jsonl": "2c7e47bf69dc0dc2d23eb575cb5ed1c0187ed76d3dc77b4eaaf052f27e6858b5",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0007/episode_records.jsonl": "d7de2c1b579db51ef83cf221cda6e082cab3aeddfacf472b306b902513cbbba6",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0008/episode_records.jsonl": "6f65a616b0b242501de15ee866bd77f08bb31e3ed946dcf9dff84c2d39a5fb05",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0009/episode_records.jsonl": "83b1a4a93d380731df944ddc3d4131c673962f9d2237d270b630804b89335142",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0010/episode_records.jsonl": "19a64982c299e9bb82ac754247f3162b89f64329e6e4ea28e0151b4c668bf2b3",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0011/episode_records.jsonl": "190df56bad20987c62f464c3f9e40226e61253dcdcb0b4982d7cf474f0ff47cb",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0012/episode_records.jsonl": "303be888223403f50f8cf500382577a9cecbf83bab02286041ccc4ff42f2587e",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0013/episode_records.jsonl": "3a0345e1696363d8c924beca7b290146bb3c398f69c993b60976b63af47a397f",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0014/episode_records.jsonl": "d1ffa6e9b7137e8d6615153544a58dc165460e542df03f237f93e2ca43c93b52",
    "docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_episode_records/seed_2202_random/candidate_0015/episode_records.jsonl": "1884640ac39e95fcda3558c02f50ef8068088b6f84b49d5f66bf290185ee0758",
}


def _under_root(path: Path, root: Path) -> bool:
    """Return whether ``path`` lives under ``root``.

    Membership is keyed on ``root``'s components appearing as a contiguous
    subsequence of ``path``'s parts, so both repo-relative paths (as pre-commit
    passes them, e.g. ``configs/training/x.yaml``) and absolute paths (as tests
    or direct calls may pass, e.g. ``/home/u/repo/configs/training/x.yaml``) are
    recognised.
    """
    parts = path.parts
    root_parts = root.parts
    span = len(root_parts)
    return any(parts[i : i + span] == root_parts for i in range(len(parts) - span + 1))


def _is_grandfathered_evidence(path: Path) -> bool:
    """Return whether ``path`` belongs to a grandfathered legacy evidence packet.

    The packet directory is the component immediately after the
    ``docs/context/evidence`` root; if that name is in
    :data:`LEGACY_EVIDENCE_ALLOWLIST` the file is skipped (pre-existing debt).
    """
    parts = path.parts
    root_parts = EVIDENCE_ROOT.parts
    span = len(root_parts)
    for i in range(len(parts) - span + 1):
        if parts[i : i + span] == root_parts and i + span < len(parts):
            return parts[i + span] in LEGACY_EVIDENCE_ALLOWLIST
    return False


def _repo_relative_path(path: Path) -> str | None:
    """Resolve a path relative to its repository root, or the test cwd.

    Git supplies repo-relative paths to ``--all``; tests and direct callers may supply
    absolute paths. Looking for the nearest ``.git`` marker keeps both forms tied to the
    actual repository path instead of accepting a matching suffix from a nested alias.
    Temporary test roots without Git metadata use their current working directory.
    """
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None

    anchor = resolved if resolved.is_dir() else resolved.parent
    for parent in (anchor, *anchor.parents):
        if (parent / ".git").exists():
            try:
                return resolved.relative_to(parent).as_posix()
            except ValueError:
                return None

    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError, RuntimeError):
        return None


def _is_pinned_verbatim_evidence(path: Path) -> bool:
    """Return whether ``path`` is an exact pinned recovery artifact.

    The path must resolve to the exact repository-relative key. The digest match makes the
    exception fail closed if a recovered artifact is edited or replaced.
    """

    normalized = _repo_relative_path(path)
    if normalized is None:
        return False
    for repo_path, expected_sha256 in PINNED_VERBATIM_EVIDENCE_SHA256.items():
        if normalized != repo_path:
            continue
        try:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 16), b""):
                    hasher.update(chunk)
            observed_sha256 = hasher.hexdigest()
        except OSError:
            return False
        return observed_sha256 == expected_sha256
    return False


def _iter_scanned_files(files: list[str]) -> list[Path]:
    """
    Return the subset of ``files`` under a scanned root, minus grandfathered ones.

    A file is in scope when it lives under any of :data:`SCANNED_ROOTS`
    (``configs/`` or ``docs/context/evidence/``) and is not part of a
    grandfathered legacy evidence packet.
    """
    selected: list[Path] = []
    for f in files:
        path = Path(f)
        if not path.is_file():
            continue
        if not any(_under_root(path, root) for root in SCANNED_ROOTS):
            continue
        if _is_grandfathered_evidence(path):
            continue
        if _is_pinned_verbatim_evidence(path):
            continue
        selected.append(path)
    return selected


def _iter_tracked_scanned_files() -> list[str]:
    """Return Git-tracked files under the scanned roots for ``--all`` checks.

    Uses ``git ls-files -z`` so paths containing spaces or other special
    characters survive verbatim (default output C-quotes such paths, which
    would no longer match a real file and could be silently skipped).
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *(str(root) for root in SCANNED_ROOTS)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [name for name in result.stdout.split("\0") if name and Path(name).is_file()]


def find_abs_path_violations(files: list[str]) -> dict:
    """
    Scan config/evidence files for unannotated absolute home-dir paths.

    Args:
        files: Candidate file paths (only those under a scanned root and not
            grandfathered are checked).

    Returns:
        Dict with ``status`` ("pass"/"fail"), ``violations`` (list of
        ``{file, line, text}``), and a human-readable ``message``.
    """
    scanned_files = _iter_scanned_files(files)

    if not scanned_files:
        return {
            "status": "pass",
            "violations": [],
            "message": "No config or evidence files in scope - nothing to check.",
        }

    violations: list[dict] = []
    for path in scanned_files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Non-text or unreadable file (e.g. binary asset) - skip.
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not ABS_PATH_PATTERN.search(line):
                continue
            if ALLOW_MARKER in line:
                # Explicitly annotated as an intentional absolute path.
                continue
            violations.append({"file": str(path), "line": lineno, "text": line.strip()})

    if violations:
        return {
            "status": "fail",
            "violations": violations,
            "message": (
                f"Found {len(violations)} unannotated absolute home-dir path(s) "
                f"in configs/ or docs/context/evidence/. Use a repo-relative path, "
                f"or annotate the line with '# {ALLOW_MARKER}: <reason>' if the "
                f"absolute path is intentional (e.g. private-ops routing)."
            ),
        }

    return {
        "status": "pass",
        "violations": [],
        "message": f"Checked {len(scanned_files)} file(s); no leaks found.",
    }


def main() -> None:
    """CLI entry point for the git hook."""
    parser = argparse.ArgumentParser(
        description=("Prevent absolute home-dir paths in configs/** and docs/context/evidence/**")
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to check (only those under a scanned root are scanned).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan every tracked file under the scanned roots instead of the given list.",
    )
    args = parser.parse_args()

    if args.all:
        files = _iter_tracked_scanned_files()
    else:
        files = args.files

    result = find_abs_path_violations(files)

    for v in result["violations"]:
        logging.error("Absolute path: %s:%s\n  %s", v["file"], v["line"], v["text"])
    if result["violations"]:
        logging.error(result["message"])

    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
