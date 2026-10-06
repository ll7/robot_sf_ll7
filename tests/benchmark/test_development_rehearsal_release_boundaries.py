"""Fast non-stepping coverage of the permanent diagnostic release boundary.

Test value: a promoted rehearsal or lost bundle marker would break these real
validators; existing fast tests admit only release identities. The shared
witnesses consume public D-083 identity bytes and the real exporter, with an
environment sentinel. Running them together reuses one source fixture and
keeps the release refusals in PR CI without duplicating their assertions.
"""

from tests.benchmark import test_release_development_rehearsal as witnesses
from tests.benchmark.test_release_campaign_authority import no_execution as _no_execution
from tests.benchmark.test_sealed_source_pins import sealed_repository as _sealed_repository

no_execution = _no_execution
sealed_repository = _sealed_repository


def test_development_release_boundaries_and_publication_bytes(
    sealed_repository, tmp_path, monkeypatch
):
    """The real shared authority and bundle validators never promote a rehearsal."""
    code, identity = witnesses.generate(sealed_repository)
    assert code == 0

    def reuse_public_identity(repo, seeds="1001,1002,1003"):
        assert repo == sealed_repository
        assert seeds == "1001,1002,1003"
        return 0, identity

    # Reuse the public CLI's real bytes; every production loader/validator stays real.
    monkeypatch.setattr(witnesses, "generate", reuse_public_identity)
    for admission in (
        "sealed",
        "full",
        "mint",
        "doi",
        "tag",
        "comparator",
        "runtime-smoke",
        "metadata",
    ):
        witnesses.test_rehearsal_cannot_acquire_release_status(sealed_repository, admission)
    witnesses.test_shared_acceptance_counts_development_cells_without_promoting(sealed_repository)
    publication = tmp_path / "publication"
    publication.mkdir()
    witnesses.test_rehearsal_publication_contract_refuses_release(publication)
    export = tmp_path / "export"
    export.mkdir()
    witnesses.test_shared_exporter_preserves_non_release_marker(export)
