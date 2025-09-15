import pytest


@pytest.mark.skip(
    reason="Planned: add CI interval assertions for bootstrap outputs (optimize and recompute)."
)
def test_bootstrap_ci_intervals_placeholder():
    """Placeholder for bootstrap CI interval tests.

    Will run unified CLI for optimize/recompute with --bootstrap-samples and assert:
    - results.bootstrap.recommended_score exists
    - mean_mean is finite; std_mean >= 0
    - ci is a 2-tuple (lower <= upper)
    - recommended_score (point estimate) lies within the interval
    """
    assert True
