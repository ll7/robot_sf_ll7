"""Diagnostic-only exact event replays. No parameter fitting or protected writes."""
from pathlib import Path
from collections import defaultdict
import argparse,copy,hashlib,json,subprocess,sys,traceback
import yaml
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config,run_campaign
from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.training.scenario_loader import build_robot_config_from_scenario
try:
 from robot_sf.scenario_certification.v1 import scenario_actor_source_census
except ImportError:
 # Version compatibility only: the exact 0.0.7 source predates that audit API.
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() in ('9c3452face9f361d764a48f1053eee074c809204', 'b8dccb9c9f9ddc1f1cc3bb980ee260f851a727b2', '46809b6bc18caef2a9cd82686e9670c224459c42', 'c9102a9c81bb60cebb760b2bac94af695bc087a4')
 def scenario_actor_source_census(config):
  assert config.sim_config.peds_per_area_m2==0
  assert getattr(config.sim_config,'population_size',None) in (None,0)
  for md in config.map_pool.map_defs.values():
   assert not md.single_pedestrians and not md.social_groups
  return {'verified_empty':True,'method':'legacy public-field assertions identical to prior baseline sweep; every reset/step actor list also checked'}

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--mode',required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 bundle=args.bundle.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);manifest=json.loads((bundle/'manifest.json').read_text());spec=next(x for x in manifest['layers']+manifest['crossovers'] if x['mode']==args.mode)
 sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert sha==spec['source_sha'];manifest['slots']=[s for s in manifest['slots'] if s['slot_id'] in spec['slot_ids']];assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
 sv,pv=spec['scenario_variant'],spec['planner_variant'];base=yaml.safe_load((bundle/'baseline/campaign.yaml').read_text());source=yaml.safe_load((bundle/sv/'campaign.yaml').read_text());planners=yaml.safe_load((bundle/pv/'campaign.yaml').read_text())['planners'];matrix=yaml.safe_load((bundle/sv/'scenarios.yaml').read_text())['scenarios'];rows=[];reports=[];identities=[]
 groups=defaultdict(list)
 for slot in manifest['slots']:
  arm,kin,sc,seed=slot['slot'];assert kin=='differential_drive' and seed in (1001,1002,1003);groups[arm].append(slot)
 for arm,slots in sorted(groups.items()):
  dest=out/arm;dest.mkdir(exist_ok=True);sc_seeds=defaultdict(list)
  for item in slots:sc_seeds[item['slot'][2]].append(item['slot'][3])
  selected=[]
  for raw in matrix:
   if raw['name'] not in sc_seeds:continue
   sc=copy.deepcopy(raw);sc['seeds']=sorted(sc_seeds[sc['name']]);sc['map_file']=sc['map_file'].replace('@BUNDLE@',str(bundle));assert Path(sc['map_file']).is_file();assert sc['simulation_config']['ped_density']==0 and not sc.get('single_pedestrians') and sc.get('_diagnostic_remove_pedestrian_actors') is True
   # Census without reset/step: explicit diagnostic maps, no actors, only allowed seeds.
   config=build_robot_config_from_scenario(sc,scenario_path=dest/'matrix.yaml')
   assert scenario_actor_source_census(config)["verified_empty"] is True
   identities.append({'mode':args.mode,'arm':arm,'scenario':sc['name'],'scenario_sha256':digest({k:v for k,v in sc.items() if k not in ('map_file','seeds')}),'map_sha256':hashlib.sha256(Path(sc['map_file']).read_bytes()).hexdigest(),'seeds':sc['seeds']})
   selected.append(sc)
  assert set(sc_seeds)=={s['name'] for s in selected}
  mat=dest/'matrix.yaml';mat.write_text(yaml.safe_dump({'scenarios':selected},sort_keys=False))
  payload=copy.deepcopy(base)
  # Only the assigned scenario/horizon or planner-profile axes differ from top.
  for k in ('horizon','scenario_horizons','scenario_horizons_sha256','horizon_policy'):
   payload.pop(k,None)
   if k in source:payload[k]=source[k]
  # Protocol marker belongs to the selected scenario/horizon contract.
  if sv=='baseline':
   payload.pop('protocol_version',None)
   if 'protocol_version' in source:payload['protocol_version']=source['protocol_version']
  # Original baseline fixed H600, top retains exact authored schedule unchanged.
  planner=copy.deepcopy(next(p for p in planners if manifest['replacements'].get(p['key'],p['key'])==arm))
  if planner.get('algo_config'):
   relative=planner['algo_config'];planner['algo_config']=str(bundle/pv/'refs'/relative)
   assert Path(planner['algo_config']).is_file()
  payload.update(scenario_matrix=str(mat),planners=[planner],seed_policy={'mode':'scenario-default'},workers=24,resume=False,stop_on_failure=False,paper_facing=False,export_publication_bundle=False,record_simulation_step_trace=True)
  # Reporting-only incompatible legacy anchors must never discard the executed rows.
  payload['snqi_weights']=None;payload['snqi_baseline']=None;payload['snqi_contract']['enabled']=False
  if True:  # Immutable legacy input schema for every adjacent runtime pair.
   # Legacy loader must retain its original unscheduled schema, metadata and geometry.
   for k in ('protocol_version','comparability_mapping'):
    payload.pop(k,None)
    if k in source:payload[k]=source[k]
  cp=dest/'campaign.yaml';cp.write_text(yaml.safe_dump(payload,sort_keys=False));cfg=load_campaign_config(cp)
  resolved=_load_campaign_scenarios(cfg);assert sorted((s['name'],n) for s in resolved for n in s['seeds'])==sorted((i['slot'][2],i['slot'][3]) for i in slots)
  try:
   result=run_campaign(cfg,output_root=dest/'campaigns',campaign_id='targeted',skip_publication_bundle=True,arm_isolation='in_process',invoked_command='pedsweep2 diagnostic exact-event replay')
   reports.append({'arm':arm,'result':str(result)[:1000]})
  except Exception as e:
   traceback.print_exc();reports.append({'arm':arm,'harness_error':str(e)})
  found=[]
  for file in (dest/'campaigns').rglob('episodes.jsonl'):
   for line in file.read_text().splitlines():
    if not line.strip():continue
    r=json.loads(line);assert r['seed'] in (1001,1002,1003);sc=r['scenario_id'];seed=r['seed'];key=next(i['slot_id'] for i in slots if i['slot'][2]==sc and i['slot'][3]==seed);md=r.get('algorithm_metadata',{});t=md.get('simulation_step_trace',{});steps=t.get('steps',[])
    assert t and len(steps)==r['steps'] and not t['reset'].get('pedestrians') and all(not s.get('pedestrians') for s in steps)
    # Common fields available across all six layers; do not hash added telemetry.
    trajectory=[{'robot':{'position':s.get('robot',{}).get('position'),'velocity':s.get('robot',{}).get('velocity'),'heading':s.get('robot',{}).get('heading')},'action':s.get('planner',{}).get('selected_action'),'rl':s.get('rl')} for s in steps]
    m=r['metrics'];item={'slot_id':key,'mode':args.mode,'source_sha':sha,'arm':arm,'actual_arm':planner['key'],'scenario':sc,'seed':seed,'success':bool(m.get('success')),'collision_count':m.get('total_collision_count',m.get('collisions')),'outcome':r.get('outcome'),'termination_reason':r.get('termination_reason'),'steps':r['steps'],'horizon':r.get('horizon'),'trajectory_sha256':digest(trajectory),'reset':t['reset'],'first_step':steps[0] if steps else None,'last_step':steps[-1] if steps else None,'trace_complete':True,'zero_actors':True,'planner_runtime':md.get('planner_runtime'),'guard_stats':md.get('guard_stats'),'shield_stats':md.get('shield_stats'),'planner_contract':md.get('planner_contract'),'action_semantics':md.get('action_semantics'),'action_adapter_version':md.get('action_adapter_version'),'config':md.get('config'),'raw_file':str(file.relative_to(out)),'episode_id':r['episode_id']}
    rows.append(item);found.append((sc,seed))
  assert len(found)==len(set(found)),('duplicate',args.mode,arm)
  reports[-1]['expected_rows']=len(slots);reports[-1]['written_rows']=len(found)
  (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'execution.json').write_text(json.dumps(reports,indent=2)+'\n');(out/'input-identity.json').write_text(json.dumps(identities,indent=2)+'\n')
  print(json.dumps({'mode':args.mode,'arm':arm,'expected':len(slots),'written':len(found)}),flush=True)
 assert len(rows)==len(spec['slot_ids']),('missing',args.mode,len(rows))
if __name__=='__main__':main()
