"""Qualification probe: does an explicit desired_speed_mean erase the archetype mixture?"""
import copy, json, sys
import numpy as np
import robot_sf.benchmark.map_runner.map_runner_episode as E
from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios
M="configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
res={}
orig=E._prepare_episode_env
def wrap(env,**kw):
    obs=orig(env,**kw); res[kw["scenario"]["name"]]=sorted(round(float(x),4) for x in env.simulator.pysf_sim.peds.max_speeds); return obs
E._prepare_episode_env=wrap
base=[s for s in load_scenarios(M) if s["name"]=="classic_overtaking_medium"][0]
scs=[]
for tag,extra in (("mixed_no_override",{}),("mixed_with_desired_speed_mean_0.65",{"desired_speed_mean":0.65})):
    sc=copy.deepcopy(dict(base)); sc["name"]=tag; sc["seeds"]=[1001]
    sc["simulation_config"].update(population_size=12, archetype_composition={"slow":0.25,"standard":0.5,"fast":0.25}, archetype_speed_factors={"slow":0.5,"standard":1.0,"fast":1.5}, archetype_seed=1001, **extra)
    scs.append(sc)
run_map_batch(scs,"/tmp/xhet_override.jsonl","robot_sf/benchmark/schemas/episode.schema.v1.json",scenario_path=M,horizon=2,dt=0.1,algo="goal",workers=1,resume=False)
print(json.dumps(res,indent=1))
