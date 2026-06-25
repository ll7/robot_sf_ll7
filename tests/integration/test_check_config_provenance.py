"""
Tests for the training-config provenance lint hook (issue #3606).

Covers the misattribution case (the #3570 pattern: header names a different
issue than the identity fields), the consistent case, the source-issue case
(header references both a leader issue and the canonical one), the
allow-annotation escape hatch, and the ambiguous/absent-identity skips.
"""

from pathlib import Path

from hooks.check_config_provenance import find_provenance_mismatches

_IDENTITY_3068 = """\
policy_id: ppo_expert_issue_3068_scout
tracking:
  wandb:
    group: issue-3068-ppo-curriculum
    tags:
      - issue-3068
      - ppo
"""


def _write(base: Path, rel: str, content: str) -> str:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestCheckConfigProvenance:
    """Behavioural tests for ``find_provenance_mismatches``."""

    def test_flags_misattributed_header(self, tmp_path):
        """Header naming issue-1024 while identity targets 3068 is flagged."""
        f = _write(
            tmp_path,
            "configs/training/ppo/ablations/scout.yaml",
            "# Issue-1024 h500 schedule retrain.\n" + _IDENTITY_3068,
        )
        result = find_provenance_mismatches([f])
        assert result["status"] == "fail"
        assert result["violations"][0]["identity_issue"] == 3068
        assert result["violations"][0]["header_issues"] == [1024]

    def test_passes_consistent_header(self, tmp_path):
        """Header naming the same issue as identity passes."""
        f = _write(
            tmp_path,
            "configs/training/ppo/ablations/scout.yaml",
            "# Issue-3068 PPO curriculum scout.\n" + _IDENTITY_3068,
        )
        result = find_provenance_mismatches([f])
        assert result["status"] == "pass"

    def test_passes_when_header_cites_source_and_canonical(self, tmp_path):
        """Header citing a leader issue AND the canonical one passes."""
        f = _write(
            tmp_path,
            "configs/training/ppo/ablations/scout.yaml",
            "# Issue-3068 scout; clones the issue-791 leader recipe.\n" + _IDENTITY_3068,
        )
        result = find_provenance_mismatches([f])
        assert result["status"] == "pass"

    def test_allow_marker_exempts_mismatch(self, tmp_path):
        """An annotated header is exempt even on mismatch."""
        f = _write(
            tmp_path,
            "configs/training/ppo/ablations/scout.yaml",
            "# Issue-1024 retrain.  # allow-provenance-mismatch: historical record\n"
            + _IDENTITY_3068,
        )
        result = find_provenance_mismatches([f])
        assert result["status"] == "pass"

    def test_skips_when_header_has_no_issue(self, tmp_path):
        """A header with no issue reference cannot be wrong - skipped."""
        f = _write(
            tmp_path,
            "configs/training/ppo/ablations/scout.yaml",
            "# Full-surface scout recipe.\n" + _IDENTITY_3068,
        )
        result = find_provenance_mismatches([f])
        assert result["status"] == "pass"

    def test_skips_when_identity_ambiguous(self, tmp_path):
        """Identity fields referencing multiple issues are not actioned."""
        content = (
            "# Issue-1024 retrain.\n"
            "policy_id: ppo_issue_3068_issue_999_mix\n"
            "tracking:\n  wandb:\n    group: issue-3068\n"
        )
        f = _write(tmp_path, "configs/training/ppo/ablations/mix.yaml", content)
        result = find_provenance_mismatches([f])
        assert result["status"] == "pass"

    def test_ignores_non_training_configs(self, tmp_path):
        """Configs outside configs/training/ are not scanned."""
        f = _write(
            tmp_path,
            "configs/benchmarks/scout.yaml",
            "# Issue-1024 retrain.\n" + _IDENTITY_3068,
        )
        result = find_provenance_mismatches([f])
        assert result["status"] == "pass"
        assert "nothing to check" in result["message"].lower()

    def test_repo_training_tree_is_consistent(self):
        """The real configs/training/ tree must already pass."""
        root = Path("configs/training")
        files = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix in {".yaml", ".yml"}]
        result = find_provenance_mismatches(files)
        assert result["status"] == "pass", result["message"]
