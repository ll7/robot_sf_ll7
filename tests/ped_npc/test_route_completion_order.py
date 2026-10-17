"""Route proximity must not respawn pedestrians before they traverse the route."""

import numpy as np
import pytest

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.ped_npc.ped_behavior import FollowRouteBehavior
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates


@pytest.mark.parametrize("end", [(0.0, 0.0), (0.5, 0.0)])
def test_loop_near_endpoint_keeps_group_until_final_segment(monkeypatch, end):
    """A loop/U-turn completes only after the intervening waypoints are visited."""
    data = np.zeros((1, 7))
    groups = PedestrianGroupings(PedestrianStates(lambda: data))
    gid = groups.new_group({0})
    zone = ((-1, -1), (1, -1), (1, 1))
    route = GlobalRoute(0, 0, [(0, 0), (5, 0), (5, 5), end], zone, zone)
    behavior = FollowRouteBehavior(groups, {gid: route}, [0])
    respawns = []
    monkeypatch.setattr(behavior, "respawn_group_at_start", respawns.append)
    behavior.step()
    assert respawns == []
    assert behavior.navigators[gid].waypoint_id == 1
    for point in [(5, 0), (5, 5)]:
        data[0, :2] = point
        behavior.step()
        assert respawns == []
    data[0, :2] = end
    behavior.step()
    assert respawns == [gid]
