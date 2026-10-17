"""Contact results must depend on physical walkers, not their state-row labels."""

import numpy as np
import pytest

from tests.sim.test_pedestrian_contact_bounded import simulator

FIVE_STATE = np.array(
    [
        [0.0, 0.0, 0.4158914427198128, 0.9131155174916559, 10.0, 10.0, 0.5],
        [0.59, 0.0, -1.0032177897512744, -1.2775430984603262, 10.59, 10.0, 0.5],
        [1.18, 0.0, -0.3679931850105399, -1.1562032607511912, 11.18, 10.0, 0.5],
        [
            0.295,
            0.5109549882328187,
            0.7594572622184643,
            -1.2147475512371047,
            10.295,
            10.510954988232818,
            0.5,
        ],
        [
            0.885,
            0.5109549882328187,
            1.2693496953578995,
            0.764251783716553,
            10.885,
            10.510954988232818,
            0.5,
        ],
    ],
    dtype=float,
)
FIVE_PERMUTATION = np.array([4, 1, 0, 2, 3])
EIGHT_STATE = np.array(
    [
        [0.0, 0.0, 0.16301631279862083, 1.0083704079617235, 10.0, 10.0, 0.5],
        [0.59, 0.0, 0.6093749091068443, 0.5798439111233202, 10.59, 10.0, 0.5],
        [1.18, 0.0, -0.24267260315840722, 0.3995194994499136, 11.18, 10.0, 0.5],
        [
            0.295,
            0.5109549882328187,
            0.7568806410626394,
            0.9797320119390316,
            10.295,
            10.510954988232818,
            0.5,
        ],
        [
            0.885,
            0.5109549882328187,
            -0.5927894416825108,
            -0.9251068495320279,
            10.885,
            10.510954988232818,
            0.5,
        ],
        [
            1.4749999999999999,
            0.5109549882328187,
            1.1519203971353325,
            -0.7739601652298632,
            11.475,
            10.510954988232818,
            0.5,
        ],
        [
            0.0,
            1.0219099764656374,
            0.27384348011860005,
            0.26210447673496495,
            10.0,
            11.021909976465638,
            0.5,
        ],
        [
            0.59,
            1.0219099764656374,
            0.03690020600634791,
            1.0893007986067105,
            10.59,
            11.021909976465638,
            0.5,
        ],
    ],
    dtype=float,
)
EIGHT_PERMUTATION = np.array([5, 6, 2, 7, 0, 3, 1, 4])

SIXTEEN_STATE = np.array(
    [
        [0.0, 0.0, 0.198220283261219, 0.04431806359026158, 10.0, 10.0, 0.5],
        [0.59, 0.0, -0.9713632615724044, -0.17492763440779657, 10.59, 10.0, 0.5],
        [1.18, 0.0, -0.30254706567043055, -0.1934280507953139, 11.18, 10.0, 0.5],
        [
            0.295,
            0.5109549882328187,
            -0.9651926273809561,
            -0.629937653254759,
            10.295,
            10.510954988232818,
            0.5,
        ],
        [
            0.885,
            0.5109549882328187,
            0.7531415490945974,
            0.19665884069800055,
            10.885,
            10.510954988232818,
            0.5,
        ],
        [
            1.4749999999999999,
            0.5109549882328187,
            0.27172739386387557,
            -1.289124115818158,
            11.475,
            10.510954988232818,
            0.5,
        ],
        [
            0.0,
            1.0219099764656374,
            0.7721365690602842,
            0.11285982139406037,
            10.0,
            11.021909976465638,
            0.5,
        ],
        [
            0.59,
            1.0219099764656374,
            0.21452199245862724,
            -0.049519968882788756,
            10.59,
            11.021909976465638,
            0.5,
        ],
        [
            1.18,
            1.0219099764656374,
            1.046809243069249,
            -0.005009373906486925,
            11.18,
            11.021909976465638,
            0.5,
        ],
        [
            0.295,
            1.5328649646984562,
            0.2940288693549893,
            -0.0015295230855203457,
            10.295,
            11.532864964698456,
            0.5,
        ],
        [
            0.885,
            1.5328649646984562,
            0.9850859937444028,
            0.44636213435373473,
            10.885,
            11.532864964698456,
            0.5,
        ],
        [
            1.4749999999999999,
            1.5328649646984562,
            -0.24710158814309024,
            -0.7443125898545819,
            11.475,
            11.532864964698456,
            0.5,
        ],
        [
            0.0,
            2.043819952931275,
            0.3934637471053617,
            -0.49080397897816075,
            10.0,
            12.043819952931274,
            0.5,
        ],
        [
            0.59,
            2.043819952931275,
            -0.6142890674641489,
            0.6122241963417179,
            10.59,
            12.043819952931274,
            0.5,
        ],
        [
            1.18,
            2.043819952931275,
            0.1298525370656367,
            -0.5555259217993529,
            11.18,
            12.043819952931274,
            0.5,
        ],
        [
            0.295,
            2.5547749411640934,
            0.9896409413868252,
            -0.8314539134630936,
            10.295,
            12.554774941164094,
            0.5,
        ],
    ],
    dtype=float,
)
SIXTEEN_PERMUTATION = np.array([1, 7, 4, 10, 12, 13, 3, 2, 6, 8, 14, 0, 9, 11, 5, 15])


@pytest.mark.parametrize("passes", [3, 10])
def test_same_five_walkers_are_invariant_to_pair_processing_order(passes):
    """Replay the independent review's exact five-body zero-force refutation."""
    outputs = []
    for order in [np.arange(5), FIVE_PERMUTATION]:
        sim = simulator(FIVE_STATE[order], passes=passes)
        sim.peds.max_speeds = np.full(5, 3.0)
        sim.peds.step(np.zeros((5, 2)))
        assert sim.contact_projection_fallback_count == 0
        outputs.append(sim.peds.vel().copy()[np.argsort(order)])
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-10)
    # With no walls, holds or binding caps, pair impulses conserve the sum.
    # A label-independent component halt is not an acceptable "solution".
    np.testing.assert_allclose(outputs[0].sum(axis=0), FIVE_STATE[:, 2:4].sum(axis=0), atol=1e-12)


@pytest.mark.parametrize("passes", [3, 10])
def test_emergency_selection_is_invariant_to_walker_order(passes):
    """The review's eight-body witness also changed fallback admission by labeling."""
    outputs, receipts = [], []
    for order in [np.arange(8), EIGHT_PERMUTATION]:
        sim = simulator(EIGHT_STATE[order], passes=passes)
        sim.peds.max_speeds = np.full(8, 3.0)
        sim.peds.step(np.zeros((8, 2)))
        outputs.append(sim.peds.vel().copy()[np.argsort(order)])
        receipts.append(
            (sim.contact_projection_fallback_count, sim.contact_projection_unresolved_count)
        )
    assert receipts[0] == receipts[1]
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-10)


@pytest.mark.parametrize("population", range(2, 41))
@pytest.mark.parametrize("walls", [False, True])
@pytest.mark.parametrize("passes", [3, 10])
def test_projection_commutes_with_walker_permutations(population, walls, passes):
    """Cover every population boundary, walls, caps, holds and cache-bearing steps."""
    rng = np.random.default_rng(1001)
    index = np.arange(population)
    row, col = index // 6, index % 6
    xy = np.column_stack((0.59 * col + 0.295 * (row % 2), 0.32 + 0.59 * np.sqrt(3) / 2 * row))
    velocity = rng.normal(size=(population, 2))
    velocity *= 2.5 / np.linalg.norm(velocity, axis=1)[:, None]
    state = np.column_stack((xy, velocity, xy + 10, np.full(population, 0.5)))
    caps = rng.uniform(1.0, 3.0, population)
    obstacles = [(-2, 10, 0, 0)] if walls else None
    orders = [index, index[::-1], rng.permutation(population), rng.permutation(population)]
    reference = None
    for order in orders:
        sim = simulator(state[order], passes=passes, walls=obstacles)
        sim.peds.max_speeds = caps[order]
        # Map the same physical held actor rather than holding a different row.
        sim.peds.contact_held_indices = {int(np.flatnonzero(order == 0)[0])}
        samples = []
        for _ in range(3):
            sim.peds.step(np.zeros((population, 2)))
            samples.append(sim.peds.vel().copy()[np.argsort(order)])
        observed = (
            np.asarray(samples),
            sim.contact_projection_fallback_count,
            sim.contact_projection_unresolved_count,
        )
        if reference is None:
            reference = observed
        else:
            assert observed[1:] == reference[1:]
            np.testing.assert_allclose(observed[0], reference[0], rtol=0, atol=1e-10)


@pytest.mark.parametrize("passes", [3, 10])
def test_active_emergency_resolution_commutes_with_permutation(passes):
    """Exercise an actual emergency on both labels, not only fallback-free cases."""
    outputs, receipts = [], []
    for order in [np.arange(16), SIXTEEN_PERMUTATION]:
        sim = simulator(SIXTEEN_STATE[order], passes=passes)
        sim.peds.max_speeds = np.full(16, 3.0)
        sim.peds.step(np.zeros((16, 2)))
        outputs.append(sim.peds.vel().copy()[np.argsort(order)])
        receipts.append(
            (sim.contact_projection_fallback_count, sim.contact_projection_unresolved_count)
        )
    assert receipts[0] == receipts[1]
    assert receipts[0] == (1, 0), "this witness must reach the emergency solver"
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-10)
