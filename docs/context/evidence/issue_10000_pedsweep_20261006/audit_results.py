"""Read-only, dev-only custody and runtime audit; never constructs an environment."""
import collections
import hashlib
import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import yaml

repo = pathlib.Path.cwd()
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip() == '52634e854076ad7a1da6be4edf17fa40baf93fd4', 'Audit parser must come from the pinned top runtime source.'
sys.path[:0] = [str(repo), str(repo / 'fast-pysf')]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj
audit = module('release_comparison', repo / 'scripts/analysis/compare_issue_9431_release.py')
sweep = module('empty_world_sweep', repo / 'scripts/validation/run_empty_world_sweep.py')
root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
heads = {'top': '52634e854076ad7a1da6be4edf17fa40baf93fd4', 'baseline': '07f7e8d43084de748915e1b1eb8b2a1603357c6e'}
result = {'schema_version': 'pedsweep-runtime-audit.v1', 'evidence_tier': 'diagnostic-only', 'method_source_sha': heads['top'], 'runs': {}}
def finite_numbers(value):
    if isinstance(value, (int,float)): return math.isfinite(value)
    if isinstance(value,dict): return all(finite_numbers(v) for v in value.values())
    if isinstance(value,list): return all(finite_numbers(v) for v in value)
    return True
for variant, sha in heads.items():
    if len(sys.argv) > 3 and variant not in sys.argv[3:]: continue
    run = root / variant
    primary_exit=int((run/'exit_code').read_text().strip())
    assert primary_exit == 0 or (variant=='top' and primary_exit==1)
    assert (run / 'production-diff.txt').read_text().strip() == ''
    record = {'source_sha': sha, 'production_diff_empty': True, 'primary_job_exit_code': primary_exit, 'primary_job_exit_scope': 'Top job can exit 1 because original width failed; each selected suite must independently reconcile complete. Width repair is a separate pinned-source job.', 'suites': {}, 'environment_sha256': hashlib.sha256((run/'environment.txt').read_bytes()).hexdigest()}
    for suite in ('main', 'width'):
        suite_run=root/'width-repaired' if variant=='top' and suite=='width' and (root/'width-repaired').exists() else run
        cfgpath = suite_run / 'results' / f'campaign_{suite}_empty_world.yaml'
        if not cfgpath.exists(): continue
        execution=json.loads((suite_run/'results'/f'execution_{suite}.json').read_text())
        assert execution['head_sha']==sha, (variant,suite,'wrong-source suite')
        declared_failures={ (a[0],a[2],a[3]) for a in execution.get('failed_slots',[]) }
        if declared_failures:
            assert variant=='top' and suite=='width' and len(declared_failures)==9
            assert {a[0] for a in declared_failures}=={'predictive_mppi'}
            failures=json.loads((suite_run/'results/campaigns/empty_world_width/runs/predictive_mppi__differential_drive/summary.json').read_text())['failures']
            assert len(failures)==9 and all('horizon_steps=12' in f['error'] and 'supported horizon of 8' in f['error'] for f in failures)
        else: assert execution['complete'] is True
        assert execution['seeds']==[1001,1002,1003]
        assert (suite_run/'production-diff.txt').read_text().strip()==''
        suite_job=int((suite_run/'job.log').open().readline().split()[0].split('=')[1])
        cfg = yaml.safe_load(cfgpath.read_text())
        expected = {p['key']: p['algo'] for p in cfg['planners']}
        scenarios = yaml.safe_load((suite_run/'results'/f'scenarios_{suite}_empty_world.yaml').read_text())['scenarios']
        slots = {(a, s['name'], seed) for a in expected for s in scenarios for seed in (1001,1002,1003)}
        seen = set(); modes = collections.Counter(); outcomes = collections.Counter(); issues = []; checkpoints = {}; horizons = collections.Counter(); reset_contacts = []; auxiliary_markers = collections.Counter(); reset_omissions = collections.Counter()
        compact = []; controls = None
        for p in sorted((suite_run/'results/campaigns'/f'empty_world_{suite}/runs').glob('*/episodes.jsonl')):
            arm = p.parent.name.removesuffix('__differential_drive')
            for line in p.open():
                row = json.loads(line); slot = (arm, row['scenario_id'], row['seed'])
                assert row['seed'] in (1001,1002,1003), slot
                assert slot in slots and slot not in seen, slot
                seen.add(slot)
                row['_source_file'] = str(p.relative_to(suite_run/'results'))
                original = audit._execution_audit(row, expected_algorithm=expected[arm])
                runtime_row = dict(row)
                runtime_metadata = dict(row.get('algorithm_metadata', {}))
                producer = runtime_metadata.get('paired_effect_metric_producer')
                # Typed auxiliary outcome measurements are not planner execution.
                # Retain the strict scanner's original markers, and rescan every
                # other metadata field so this first-marker collision cannot
                # hide a later actual fallback/unavailable runtime marker.
                if producer is not None:
                    assert producer.get('schema_version') == 'paired_effect_metric_producer.v1'
                    runtime_metadata.pop('paired_effect_metric_producer')
                trace_view = dict(runtime_metadata.get('simulation_step_trace', {}))
                reset_view = dict(trace_view.get('reset', {}))
                for field, reason in (('routes','route_objects_not_retained_per_episode'),('spawn','spawn_sampler_decision_not_retained')):
                    value = reset_view.get(field)
                    if isinstance(value,dict) and value.get('status')=='unavailable' and value.get('reason')==reason:
                        reset_omissions[f'simulation_step_trace.reset.{field}:{reason}'] += 1
                        reset_view.pop(field)
                trace_view['reset'] = reset_view
                if 'simulation_step_trace' in runtime_metadata: runtime_metadata['simulation_step_trace'] = trace_view
                runtime_row['algorithm_metadata'] = runtime_metadata
                found = audit._execution_audit(runtime_row, expected_algorithm=expected[arm])
                if controls is None:
                    controls = {}
                    for marker, value in (('fallback_triggered',True),('status','unavailable')):
                        bad = dict(runtime_row); badmeta = dict(runtime_metadata)
                        badruntime = dict(badmeta.get('planner_runtime',{})); badruntime[marker] = value
                        badmeta['planner_runtime'] = badruntime; bad['algorithm_metadata'] = badmeta
                        detected = audit._execution_audit(bad, expected_algorithm=expected[arm])
                        assert detected, ('runtime audit failed negative control',marker)
                        controls[marker] = detected
                for marker in original:
                    if marker.startswith('algorithm_metadata.paired_effect_metric_producer.'):
                        auxiliary_markers[marker] += 1
                try:
                    if audit._row_source_commit(row) != sha: found.append('wrong_source_sha')
                except ValueError as exc: found.append(str(exc))
                trace = sweep._trace_of(row)
                if not sweep._trace_complete(row): found.append('incomplete_trace')
                if trace is None: found.append('missing_trace')
                else:
                    if trace.get('reset', {}).get('pedestrians', []): found.append('pedestrians_at_reset')
                    if any(step.get('pedestrians') for step in trace.get('steps', [])): found.append('pedestrians_after_step')
                    if (trace.get('reset') or {}).get('collision_at_reset'): reset_contacts.append(list(slot))
                    for step in trace.get('steps', []):
                        planner=step.get('planner',{})
                        if not all(finite_numbers(v) for v in (step.get('robot',{}).get('velocity'),planner.get('selected_action'),planner.get('applied_environment_action'))):
                            found.append('nonfinite_velocity_or_action');break
                if found: issues.append({'slot': list(slot), 'issues': found})
                metadata = row.get('algorithm_metadata', {})
                modes[metadata.get('planner_kinematics', {}).get('execution_mode')] += 1
                outcomes[sweep.classify_outcome(row)] += 1
                horizons[row.get('horizon')] += 1
                cp = metadata.get('planner_runtime', {}).get('checkpoint_provenance')
                if cp:
                    public_cp = dict(cp)
                    if public_cp.get('checkpoint_path'):
                        path = public_cp.pop('checkpoint_path')
                        public_cp['checkpoint_path_basename'] = pathlib.PurePosixPath(path).name
                        public_cp['checkpoint_path_normalization'] = 'Private materialization prefix omitted; exact raw path remains in archive.'
                        if '/source/' in path: public_cp['checkpoint_repo_relative_path'] = path.split('/source/',1)[1]
                    checkpoints[json.dumps(public_cp, sort_keys=True)] = public_cp
                flat = sweep.flatten(row)
                compact.append(flat)
        assert seen == slots-declared_failures, {'missing': sorted(slots-seen-declared_failures), 'unexpected': sorted(seen-slots)}
        record['suites'][suite] = {'expected_rows': len(slots), 'written_rows': len(seen), 'unexplained_missing_duplicate_unexpected_slots': 0, 'execution_modes': dict(modes), 'outcomes': dict(outcomes), 'horizons': dict(horizons), 'runtime_issues': issues, 'strict_release_metadata_audit_passed': not auxiliary_markers and not reset_omissions and not issues, 'strict_scanner_auxiliary_measurement_markers': dict(auxiliary_markers), 'typed_reset_telemetry_omissions': dict(reset_omissions), 'negative_controls_detected': controls, 'negative_controls_are_metadata_only_not_episodes': True, 'runtime_view_exclusion': 'Typed paired_effect_metric_producer.v1 auxiliary measurements and exactly reason-tagged reset routes/spawn telemetry omissions only; original strict markers retained; all other metadata including commands and planner runtime rescanned. These omissions do not qualify for release/paper admission; this remains diagnostic-only.', 'reset_contact_slots': reset_contacts, 'checkpoint_provenance': list(checkpoints.values()), 'all_available_velocities_actions_finite': not any('nonfinite_velocity_or_action' in i['issues'] for i in issues), 'all_traces_complete': not any('incomplete_trace' in i['issues'] for i in issues), 'all_actor_lists_empty': not any(any('pedestrians_' in s for s in i['issues']) for i in issues)}
        record['suites'][suite].update(source_job_id=suite_job,suite_reconciliation_complete=execution['complete'],slot_grid_accounted_for=True,declared_failed_slots=[list(s) for s in sorted(declared_failures)],declared_failed_slot_classification='real_defect: protected width MPPI config requests 12 steps but predictor supports 8' if declared_failures else None,source_job_exit_code=int((suite_run/'exit_code').read_text().strip()),environment_sha256=hashlib.sha256((suite_run/'environment.txt').read_bytes()).hexdigest())
        (out / f'{variant}_{suite}_rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in compact))
    result['runs'][variant] = record
(out/'runtime-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({v: {s: {**{k: d[k] for k in ('written_rows','execution_modes','outcomes','horizons','strict_scanner_auxiliary_measurement_markers')}, 'runtime_issue_rows':len(d['runtime_issues']), 'first_runtime_issue':d['runtime_issues'][:1]} for s,d in r['suites'].items()} for v,r in result['runs'].items()}, indent=2))
