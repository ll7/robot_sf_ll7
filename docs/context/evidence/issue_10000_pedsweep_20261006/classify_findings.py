"""Classify every newly observed failure; no changes to simulator/planner settings."""
import collections, csv, json, math, pathlib, sys
folder = pathlib.Path(sys.argv[1]); raw = pathlib.Path(sys.argv[2])
data = json.loads((folder/'comparison.json').read_text())
exception = json.loads((folder/'checked-exceptions.json').read_text())
declared = {e['scenario']: e for e in exception['entries']}
audit = json.loads((folder/'runtime-audit.json').read_text())
runtime_issues = {variant: {tuple(r['slot']):r['issues'] for r in record['suites']['main']['runtime_issues']} for variant,record in audit['runs'].items()}
traces = {}; wanted = {(f['head']['arm'], f['head']['scenario'], f['head']['seed']) for f in data['new_findings']}
for p in sorted((raw/'top/results/campaigns/empty_world_main/runs').glob('*/episodes.jsonl')):
    arm = p.parent.name.removesuffix('__differential_drive')
    for line in p.open():
        row = json.loads(line); key = (arm,row['scenario_id'],row['seed'])
        if key not in wanted: continue
        trace = row['algorithm_metadata']['simulation_step_trace']; steps = trace['steps']; last = steps[-1]
        start = trace['reset']['robot']['position']; end = last['robot']['position']
        window = steps[-50:]; travel = sum(math.dist(a['robot']['position'],b['robot']['position']) for a,b in zip(window,window[1:]))
        commands = [s.get('command',s.get('action')) for s in window]
        traces[key] = {'source': str(p.relative_to(raw/'top/results')), 'raw_row_episode_id': row.get('episode_id'), 'reset_position': start, 'final_position': end, 'last_50_steps_travel_m': travel, 'reset_collision': trace['reset'].get('collision_at_reset'), 'zero_pedestrians_every_step': all(s['pedestrians']==[] for s in steps), 'trace_steps': len(steps), 'last_step': last, 'goal_position': row['algorithm_metadata'].get('paired_effect_native_trace',{}).get('goal_position'), 'collision_metric': row['metrics'].get('total_collision_count',row['metrics'].get('collisions'))}
for f in data['new_findings']:
    h=f['head']; key=(h['arm'],h['scenario'],h['seed'])
    if f['class']=='harness_issue': f['issue']=int(sys.argv[3]);continue
    f['trace_evidence']=traces[key]
    b=f['baseline']; baseline_key=(b['arm'],b['scenario'],b['seed'])
    f['runtime_issues']={'head':runtime_issues['top'].get(key,[]),'baseline':runtime_issues['baseline'].get(baseline_key,[])}
    if any(f['runtime_issues'].values()):
        f['class']='harness_issue';f['cause']='Unqualified intended-arm execution/runtime provenance; actual runtime marker(s) retained. Outcomes do not qualify as benchmark success evidence.';f['issue']=int(sys.argv[3]);continue
    if h['scenario'] in declared and h['outcome']=='timeout' and h['collisions']==0:
        f['class']='infeasible_by_design';f['cause']='Explicit checked 2.0m doorway declaration; contact-free timeout only.';f['issue']=10000
    elif h['collisions']>0:
        f['class']='real_defect';f['cause']='New physical contact in valid zero-actor execution. No collision exception. Planner/adaptation or changed release-input cause requires isolation; not attributed to the pedestrian stack.';f['issue']=int(sys.argv[3])
    else:
        assert h['outcome']=='timeout' and h['termination_reason']=='max_steps',h
        f['class']='real_defect';f['cause']='New release-contract progress failure: valid campaign episode reaches max_steps, without contact and without a declared infeasibility exception. Changed horizon/config/algorithm is a confound; underlying mechanism unresolved, not attributed to the pedestrian stack.';f['issue']=int(sys.argv[3])
data['unclassified_findings']=sum(f['class'] is None for f in data['new_findings'])
assert data['unclassified_findings']==0
data['classification_counts']=dict(collections.Counter(f['class'] for f in data['new_findings']))
(folder/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
with (folder/'new-failures.csv').open('w',newline='') as handle:
    writer=csv.writer(handle);writer.writerow(['arm','scenario','seed','events','baseline_outcome','head_outcome','baseline_steps','head_steps','baseline_horizon','head_horizon','baseline_collisions','head_collisions','classification','issue','cause'])
    for f in data['new_findings']:
        b,h=f['baseline'],f['head'];writer.writerow([h['arm'],h['scenario'],h['seed'],';'.join(f.get('events',[])),b['outcome'],h['outcome'],b['steps'],h['steps'],b['horizon'],h['horizon'],b['collisions'],h['collisions'],f['class'],f['issue'],f['cause']])
rows=[json.loads(s) for s in (raw/'top/results/episodes_main.jsonl').read_text().splitlines() if s.strip()]
checked=[]
for row in rows:
    if row['scenario'] not in declared: continue
    markers=runtime_issues['top'].get((row['arm'],row['scenario'],row['seed']),[])
    allowed=row['outcome']=='timeout' and row['collisions']==0 and row['trace_complete'] and row['execution_status']=='written' and not markers
    checked.append({'arm':row['arm'],'scenario':row['scenario'],'seed':row['seed'],'outcome':row['outcome'],'collisions':row['collisions'],'allowed_contact_free_timeout':allowed,'runtime_issues':markers,'classification':'infeasible_by_design' if allowed else ('real_defect' if row['execution_status']=='written' and not markers else 'harness_issue')})
assert len(checked)==exception['diagnostic_expected_probe_slots']
exception['observed_slots']=checked
exception['observed_outcome_counts']=dict(collections.Counter(r['outcome'] for r in checked))
exception['excepted_rows']=sum(r['allowed_contact_free_timeout'] for r in checked)
exception['non_excepted_rows']=len(checked)-exception['excepted_rows']
(folder/'checked-exceptions.json').write_text(json.dumps(exception,indent=2)+'\n')
print(json.dumps({'findings':len(data['new_findings']),'events':data['new_event_counts'],'classification_counts':data['classification_counts'],'exception_outcomes':exception['observed_outcome_counts'],'exception_allowed':exception['excepted_rows']}))
