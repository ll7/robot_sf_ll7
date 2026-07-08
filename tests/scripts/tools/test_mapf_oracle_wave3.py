"""Characterization wave 3 tests for scripts/tools/mapf_oracle_diagnostic.py.

Focuses on CBS/SIPP paths beyond the existing suite, specifically the
time_edges_blocked propagation cases.  NEW tests only, zero production
changes.
"""

from __future__ import annotations

from scripts.tools.mapf_oracle_diagnostic import (
    _build_time_blocked,
    _build_time_edges_blocked,
    _has_collision,
    cbs_search,
    sipp_search,
)

# ---------------------------------------------------------------------------
# _build_time_edges_blocked unit tests
# ---------------------------------------------------------------------------


class TestBuildTimeEdgesBlocked:
    """Direct unit tests for _build_time_edges_blocked."""

    def test_single_obstacle_single_edge(self) -> None:
        """One obstacle with one transition produces one edge."""
        obstacles = [{"id": 0, "trajectory": [[1, 0], [0, 0]]}]
        result = _build_time_edges_blocked(obstacles)
        assert result == {0: {((1, 0), (0, 0))}}

    def test_single_obstacle_multiple_edges(self) -> None:
        """Obstacle with multiple transitions produces edges at each time step."""
        obstacles = [{"id": 0, "trajectory": [[0, 0], [0, 1], [0, 2]]}]
        result = _build_time_edges_blocked(obstacles)
        assert 0 in result
        assert 1 in result
        assert ((0, 0), (0, 1)) in result[0]
        assert ((0, 1), (0, 2)) in result[1]

    def test_stationary_obstacle_produces_no_edges(self) -> None:
        """Obstacle that stays in place produces no edges."""
        obstacles = [{"id": 0, "trajectory": [[5, 5], [5, 5], [5, 5]]}]
        result = _build_time_edges_blocked(obstacles)
        assert result == {}

    def test_multiple_obstacles_same_transition(self) -> None:
        """Multiple obstacles on the same edge merge into one set."""
        obstacles = [
            {"id": 0, "trajectory": [[1, 0], [0, 0]]},
            {"id": 1, "trajectory": [[1, 0], [0, 0]]},
        ]
        result = _build_time_edges_blocked(obstacles)
        assert result == {0: {((1, 0), (0, 0))}}

    def test_multiple_obstacles_different_transitions(self) -> None:
        """Multiple obstacles with different transitions produce separate edges."""
        obstacles = [
            {"id": 0, "trajectory": [[1, 0], [0, 0]]},
            {"id": 1, "trajectory": [[0, 1], [1, 1]]},
        ]
        result = _build_time_edges_blocked(obstacles)
        assert ((1, 0), (0, 0)) in result[0]
        assert ((0, 1), (1, 1)) in result[0]

    def test_empty_obstacles(self) -> None:
        """No obstacles produces empty dict."""
        result = _build_time_edges_blocked([])
        assert result == {}

    def test_single_position_trajectory(self) -> None:
        """Obstacle with single position produces no edges."""
        obstacles = [{"id": 0, "trajectory": [[3, 3]]}]
        result = _build_time_edges_blocked(obstacles)
        assert result == {}

    def test_diagonal_movement(self) -> None:
        """Obstacle moving diagonally produces diagonal edge."""
        obstacles = [{"id": 0, "trajectory": [[0, 0], [1, 1]]}]
        result = _build_time_edges_blocked(obstacles)
        assert result == {0: {((0, 0), (1, 1))}}

    def test_reverse_path_edges(self) -> None:
        """Obstacle traversing path in reverse produces correct edges."""
        obstacles = [{"id": 0, "trajectory": [[0, 0], [0, 1], [0, 0]]}]
        result = _build_time_edges_blocked(obstacles)
        assert ((0, 0), (0, 1)) in result[0]
        assert ((0, 1), (0, 0)) in result[1]


# ---------------------------------------------------------------------------
# _has_collision with time_edges_blocked
# ---------------------------------------------------------------------------


class TestHasCollision:
    """Direct unit tests for _has_collision with time_edges_blocked."""

    def test_vertex_collision_detected(self) -> None:
        """Vertex collision when destination is blocked at arrival time."""
        time_blocked = {1: {(2, 2)}}
        time_edges: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {}
        assert _has_collision(2, 2, 1, 1, 0, 1, time_blocked, time_edges) is True

    def test_no_collision_when_clear(self) -> None:
        """No collision when destination is clear."""
        time_blocked: dict[int, set[tuple[int, int]]] = {}
        time_edges: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {}
        assert _has_collision(2, 2, 1, 1, 0, 1, time_blocked, time_edges) is False

    def test_edge_collision_detected(self) -> None:
        """Edge (swap) collision detected when obstacle traverses reverse edge."""
        time_blocked: dict[int, set[tuple[int, int]]] = {}
        time_edges = {0: {((2, 2), (1, 1))}}
        assert _has_collision(2, 2, 1, 1, 0, 1, time_blocked, time_edges) is True

    def test_no_edge_collision_when_different_edge(self) -> None:
        """No edge collision when obstacle traverses a different edge."""
        time_blocked: dict[int, set[tuple[int, int]]] = {}
        time_edges = {0: {((3, 3), (4, 4))}}
        assert _has_collision(2, 2, 1, 1, 0, 1, time_blocked, time_edges) is False

    def test_edge_collision_none_time_edges(self) -> None:
        """No edge collision check when time_edges_blocked is None."""
        time_blocked: dict[int, set[tuple[int, int]]] = {}
        assert _has_collision(2, 2, 1, 1, 0, 1, time_blocked, None) is False

    def test_vertex_collision_takes_precedence(self) -> None:
        """Vertex collision is checked before edge collision."""
        time_blocked = {1: {(2, 2)}}
        time_edges = {0: {((2, 2), (1, 1))}}
        assert _has_collision(2, 2, 1, 1, 0, 1, time_blocked, time_edges) is True

    def test_edge_collision_at_different_time(self) -> None:
        """Edge collision at different time step not detected."""
        time_blocked: dict[int, set[tuple[int, int]]] = {}
        time_edges = {1: {((2, 2), (1, 1))}}
        assert _has_collision(2, 2, 1, 1, 0, 1, time_blocked, time_edges) is False


# ---------------------------------------------------------------------------
# SIPP time_edges_blocked propagation
# ---------------------------------------------------------------------------


class TestSippTimeEdgesBlockedPropagation:
    """Tests for SIPP search with time_edges_blocked propagation."""

    def test_sipp_with_edge_collision_avoids_swap(self) -> None:
        """SIPP avoids swap collision when time_edges_blocked is provided."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        obstacles = [{"id": 0, "trajectory": [[1, 0], [0, 0]]}]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        path = sipp_search(
            grid, (0, 0), (1, 0), time_blocked, max_time=50, time_edges_blocked=time_edges
        )
        assert path is not None
        # The direct swap move must be avoided
        if len(path) >= 2:
            assert not (path[0] == (0, 0, 0) and path[1] == (1, 0, 1)), (
                "SIPP allowed a swap collision with time_edges_blocked"
            )

    def test_sipp_with_two_obstacles_no_false_swap(self) -> None:
        """Two independent obstacles must not cause a false swap block."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        obstacles = [
            {"id": 0, "trajectory": [[0, 1], [1, 1]]},
            {"id": 1, "trajectory": [[1, 0], [0, 0]]},
        ]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        path = sipp_search(
            grid, (0, 0), (0, 2), time_blocked, max_time=50, time_edges_blocked=time_edges
        )
        assert path is not None
        # Optimal direct path should be allowed
        assert path == [(0, 0, 0), (0, 1, 1), (0, 2, 2)]

    def test_sipp_with_stationary_obstacle_no_edge_collisions(self) -> None:
        """Stationary obstacles produce no edges, only vertex collisions."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        obstacles = [{"id": 0, "trajectory": [[1, 1], [1, 1], [1, 1]]}]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        path = sipp_search(
            grid, (0, 0), (2, 2), time_blocked, max_time=50, time_edges_blocked=time_edges
        )
        assert path is not None
        # Path must avoid (1,1) at all time steps
        for r, c, t in path:
            if t < 3:
                assert (r, c) != (1, 1)

    def test_sipp_edge_collision_with_wait(self) -> None:
        """SIPP can wait to avoid a swap collision."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # Obstacle moves (0,1)->(1,1) at t=0->t=1
        # Robot wants (0,0)->(0,1) - no swap here
        # Then obstacle moves (1,1)->(0,1) at t=1->t=2
        # Robot wants (0,1)->(0,2) - no swap here either
        obstacles = [{"id": 0, "trajectory": [[0, 1], [1, 1], [0, 1]]}]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        path = sipp_search(
            grid, (0, 0), (0, 2), time_blocked, max_time=50, time_edges_blocked=time_edges
        )
        assert path is not None
        # Must avoid (0,1) at t=0 and t=2
        for r, c, t in path:
            if t == 0:
                assert (r, c) != (0, 1)
            if t == 2:
                assert (r, c) != (0, 1)

    def test_sipp_without_time_edges_uses_vertex_only(self) -> None:
        """Without time_edges_blocked, SIPP uses vertex collision only."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        obstacles = [{"id": 0, "trajectory": [[0, 1], [1, 1]]}]
        time_blocked = _build_time_blocked(obstacles)

        # Without time_edges, swap detection is not possible
        path = sipp_search(grid, (0, 0), (0, 2), time_blocked, max_time=50)
        assert path is not None
        # (0,1) is only blocked at t=0, so at t=1 it's free
        assert path[0] == (0, 0, 0)
        # The path should be able to use (0,1) at t=1

    def test_sipp_multiple_edge_collisions_same_time(self) -> None:
        """Multiple obstacles can produce multiple edges at same time step."""
        grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        obstacles = [
            {"id": 0, "trajectory": [[1, 0], [0, 0]]},
            {"id": 1, "trajectory": [[0, 1], [1, 1]]},
        ]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        # Robot at (0,0) wants to go to (2,2)
        path = sipp_search(
            grid, (0, 0), (2, 2), time_blocked, max_time=50, time_edges_blocked=time_edges
        )
        assert path is not None
        # Must avoid swap with obstacle 0: (0,0)->(1,0) is a swap
        if len(path) >= 2:
            assert not (path[0] == (0, 0, 0) and path[1] == (1, 0, 1)), (
                "Swap with obstacle 0 not detected"
            )


# ---------------------------------------------------------------------------
# CBS time_edges_blocked propagation
# ---------------------------------------------------------------------------


class TestCbsTimeEdgesBlockedPropagation:
    """Tests for CBS search propagating time_edges_blocked to SIPP."""

    def test_cbs_with_single_obstacle_swap(self) -> None:
        """CBS avoids swap collision with a single moving obstacle."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        agents = [{"id": 0, "start": [0, 0], "goal": [2, 0]}]
        obstacles = [{"id": 0, "trajectory": [[1, 0], [0, 0]]}]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        result = cbs_search(grid, agents, time_blocked, max_time=50, time_edges_blocked=time_edges)
        assert result is not None
        path = result["solution"][0]
        # The direct swap move must be avoided
        if len(path) >= 2:
            assert not (path[0] == (0, 0, 0) and path[1] == (1, 0, 1)), (
                "CBS allowed a swap collision with a dynamic obstacle"
            )

    def test_cbs_two_agents_with_obstacle_swap(self) -> None:
        """CBS with two agents avoids dynamic obstacle swap collision."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        agents = [
            {"id": 0, "start": [0, 0], "goal": [2, 0]},
            {"id": 1, "start": [2, 0], "goal": [0, 0]},
        ]
        obstacles = [{"id": 0, "trajectory": [[1, 0], [1, 1]]}]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        result = cbs_search(grid, agents, time_blocked, max_time=50, time_edges_blocked=time_edges)
        assert result is not None
        assert result["solution"][0][-1][:2] == (2, 0)
        assert result["solution"][1][-1][:2] == (0, 0)

    def test_cbs_two_independent_obstacles_no_false_swap(self) -> None:
        """Two independent obstacles must not cause a false swap block."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        agents = [{"id": 0, "start": [0, 0], "goal": [0, 2]}]
        obstacles = [
            {"id": 0, "trajectory": [[0, 1], [1, 1]]},
            {"id": 1, "trajectory": [[1, 0], [0, 0]]},
        ]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        result = cbs_search(grid, agents, time_blocked, max_time=50, time_edges_blocked=time_edges)
        assert result is not None
        path = result["solution"][0]
        # Optimal direct path should be allowed
        assert path == [(0, 0, 0), (0, 1, 1), (0, 2, 2)]

    def test_cbs_with_stationary_obstacle(self) -> None:
        """CBS with stationary obstacle uses vertex collision only."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        agents = [{"id": 0, "start": [0, 0], "goal": [2, 2]}]
        obstacles = [{"id": 0, "trajectory": [[1, 1], [1, 1], [1, 1]]}]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        result = cbs_search(grid, agents, time_blocked, max_time=50, time_edges_blocked=time_edges)
        assert result is not None
        path = result["solution"][0]
        # Path must avoid (1,1) at all time steps
        for r, c, t in path:
            if t < 3:
                assert (r, c) != (1, 1)

    def test_cbs_without_time_edges_uses_vertex_only(self) -> None:
        """Without time_edges_blocked, CBS uses vertex collision only."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        agents = [{"id": 0, "start": [0, 0], "goal": [2, 0]}]
        obstacles = [{"id": 0, "trajectory": [[1, 0], [0, 0]]}]
        time_blocked = _build_time_blocked(obstacles)

        # Without time_edges, swap detection is not possible
        result = cbs_search(grid, agents, time_blocked, max_time=50)
        assert result is not None
        # The path may or may not avoid the swap - it's not detected without time_edges

    def test_cbs_multiple_obstacles_with_edges(self) -> None:
        """CBS with multiple obstacles and edge collisions."""
        grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        agents = [{"id": 0, "start": [0, 0], "goal": [3, 3]}]
        obstacles = [
            {"id": 0, "trajectory": [[1, 0], [0, 0]]},
            {"id": 1, "trajectory": [[0, 1], [1, 1]]},
        ]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        result = cbs_search(grid, agents, time_blocked, max_time=50, time_edges_blocked=time_edges)
        assert result is not None
        path = result["solution"][0]
        # Must avoid swap with obstacle 0
        if len(path) >= 2:
            assert not (path[0] == (0, 0, 0) and path[1] == (1, 0, 1)), (
                "Swap with obstacle 0 not detected"
            )

    def test_cbs_three_agents_with_obstacle_edges(self) -> None:
        """CBS with three agents and obstacle edge collisions."""
        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        agents = [
            {"id": 0, "start": [0, 0], "goal": [0, 4]},
            {"id": 1, "start": [2, 0], "goal": [2, 4]},
            {"id": 2, "start": [4, 0], "goal": [4, 4]},
        ]
        obstacles = [{"id": 0, "trajectory": [[2, 2], [2, 3]]}]
        time_blocked = _build_time_blocked(obstacles)
        time_edges = _build_time_edges_blocked(obstacles)

        result = cbs_search(grid, agents, time_blocked, max_time=50, time_edges_blocked=time_edges)
        assert result is not None
        # All agents should reach their goals
        assert result["solution"][0][-1][:2] == (0, 4)
        assert result["solution"][1][-1][:2] == (2, 4)
        assert result["solution"][2][-1][:2] == (4, 4)
