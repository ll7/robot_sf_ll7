"""Fail-closed dev-only empty-world comparison; no metric or parameter tuning."""
from collections import Counter
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
replacements={
 'scenario_adaptive_hybrid_orca_v2_bottleneck_yield':'scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4',
 'scenario_adaptive_hybrid_orca_v2_collision_guard':'scenario_adaptive_hybrid_orca_v2_collision_guard_v4',
 'hybrid_rule_v3_fast_progress_static_escape':'hybrid_rule_v4_fast_progress_static_escape',
 'hybrid_rule_v3_fast_progress_static_escape_continuous':'hybrid_rule_v4_fast_progress_static_escape_continuous'}
def load(file):
 rows=[json.loads(s) for s in file.read_text().splitlines() if s.strip()]
 assert all(r['seed'] in (1001,1002,1003) for r in rows)
 return rows
def slot(r):return (replacements.get(r['arm'],r['arm']),r['kinematics'],r['scenario'],r['seed'])
def collision(r):
 value=r['collisions'];assert type(value) in (int,float) and value>=0,('invalid collision metric',r)
 return value>0
def results(variant):
 path=root/variant
 return path/'results' if (path/'results').is_dir() else path
brows=load(results('baseline')/'episodes_main.jsonl');hrows=load(results('top')/'episodes_main.jsonl')
b={slot(r):r for r in brows};h={slot(r):r for r in hrows}
assert len(b)==len(brows) and len(h)==len(hrows),'duplicate slots'
assert set(b)==set(h),{'missing_head':sorted(set(b)-set(h)),'missing_baseline':sorted(set(h)-set(b))}
findings=[];all_failures=[]
for key,r in sorted(h.items()):
 old=b[key]
 if r['execution_status']!='written' or old['execution_status']!='written':
  findings.append({'slot':list(key),'class':'harness_issue','cause':'invalid or missing campaign execution','baseline':old,'head':r});continue
 new=[]
 if old['success'] and not r['success']:new.append('success_to_failure')
 if collision(r) and not collision(old):new.append('new_collision')
 if not r['success'] or collision(r):all_failures.append(r)
 if new:findings.append({'slot':list(key),'events':new,'class':None,'cause':None,'baseline':old,'head':r})
widthroot=results('width-repaired') if (root/'width-repaired').exists() else results('top')
widthfile=widthroot/'episodes_width.jsonl'
width=load(widthfile) if widthfile.exists() else []
assert len({slot(r) for r in width})==len(width),'duplicate width slots'
result={'schema_version':'pedsweep-comparison.v1','evidence_tier':'diagnostic-only','paired_main_slots':len(b),'head_main_slots':len(hrows),'baseline_main_slots':len(brows),'width_unpaired_slots':len(width),'slot_replacements':replacements,'new_event_counts':dict(Counter(e for f in findings for e in f.get('events',[]))),'new_findings':findings,'head_failure_slots':len(all_failures),'width_failure_rows':[r for r in width if not r['success'] or (r['execution_status']=='written' and collision(r))],'unclassified_findings':sum(f['class'] is None for f in findings),'forbidden_inference':'Matching scenario IDs and arm slots do not establish byte-identical inputs or causal attribution to the pedestrian stack. Width slots have no 0.0.7 baseline.'}
(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('new_findings','width_failure_rows','slot_replacements')},indent=2))
