"""Fast real-input witness for diagnostic smoke admission.

Test value: release-mode promotion or missing source/checkpoint binding would
break this witness. Existing release-only tests cannot admit this identity.
The public resolver consumes real D-083 bytes; recorded validators and the
environment sentinel preserve the non-stepping boundary. The full negative
matrix remains in the slow release-development test file.
"""

from tests.benchmark.test_release_campaign_authority import no_execution as _no_execution
from tests.benchmark.test_release_development_rehearsal import (
    test_development_smoke_keeps_shared_admission_and_digest_checks as _check_smoke_admission,
)
from tests.benchmark.test_sealed_source_pins import sealed_repository as _sealed_repository

no_execution = _no_execution
sealed_repository = _sealed_repository


def test_real_development_smoke_admission(sealed_repository, monkeypatch):
    """The shared receipt verifier admits only a source-bound diagnostic result."""
    _check_smoke_admission(sealed_repository, monkeypatch, None)
