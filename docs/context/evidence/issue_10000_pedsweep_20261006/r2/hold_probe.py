"""Diagnostic: contact projection and a held, zero-speed-cap pedestrian."""
import hashlib, json, pathlib
import numpy as np
from pysocialforce import Simulator
from pysocialforce.config import SimulatorConfig
from robot_sf.ped_npc.ped_behavior import SinglePedestrianBehavior
from robot_sf.ped_npc.ped_grouping import PedestrianStates, PedestrianGroupings
from robot_sf.nav.map_config import SinglePedestrianDefinition
np.random.seed(1001)
rows=[]
for selector in (False,True):
 cfg=SimulatorConfig()
 cfg.scene_config.agent_radius=.28
 cfg.scene_config.enable_group=False
 cfg.scene_config.max_speed_multiplier=1.3
 if selector: cfg.pedestrian_contact_rule='projection_v1'
 # Initially separated. Ped 0 models the zero-cap start-delay intervention.
 state=np.array([[0.,0.,1.,0.,10.,0.,.5],[.57,0.,-1.,0.,-10.,0.,.5]])
 sim=Simulator(state=state,config=cfg,obstacles=[])
 states=PedestrianStates(lambda: sim.peds.state)
 behavior=SinglePedestrianBehavior(states,PedestrianGroupings(states),[SinglePedestrianDefinition(id="held",start=(0.,0.),goal=(10.,0.),start_delay_s=10.)],0)
 behavior.bind_pysf_peds(sim.peds)
 behavior.step()
 before=sim.peds.state.copy()
 sim.step()
 rows.append({'contact_enabled':selector,'held_displacement_m':float(np.linalg.norm(sim.peds.pos()[0]-before[0,:2])), 'held_velocity_m_s':sim.peds.vel()[0].tolist(),'held_position_after':sim.peds.pos()[0].tolist(),'before':before.tolist(),'after':sim.peds.state.tolist(),'held_speed_cap_before':float(sim.peds.contact_step_speed_caps[0]) if selector else 0.,'dt_s':float(sim.peds.d_t),'start_delay_remaining_s':behavior._runtimes[0].start_delay_remaining_s})
print(json.dumps({'seed':1001,'purpose':'zero-speed-cap pedestrian hold; no robot controller instantiated','rows':rows,'contact_source_sha256':hashlib.sha256(pathlib.Path('fast-pysf/pysocialforce/contact.py').read_bytes()).hexdigest()},indent=2))
