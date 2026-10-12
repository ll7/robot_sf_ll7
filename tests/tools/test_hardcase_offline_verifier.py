"""The retained verifier must ignore evidence-review companions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.new_feature_no_base(reason="base has no hardcase packet verifier")
def test_offline_verifier_ignores_review_sidecar_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Filesystem enumeration may return the review before the actual receipt."""
    path = ROOT / "docs/context/evidence/hardcase_positive_control/followup/verify.py"
    spec = importlib.util.spec_from_file_location("hardcase_offline_verifier", path)
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    original_glob = Path.glob

    def review_first(folder: Path, pattern: str):
        return iter(
            sorted(
                original_glob(folder, pattern), key=lambda p: not p.name.endswith(".review.json")
            )
        )

    monkeypatch.setattr(Path, "glob", review_first)
    verifier.main()
    assert "PASS: packet hashes" in capsys.readouterr().out
