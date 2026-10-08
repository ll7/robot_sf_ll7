'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { allocate, route, watch, stranded } = require('../../scripts/ci/runner_fallback.cjs');

const names = ['fast-feedback', 'smoke-artifacts', 'wheel-smoke-install', 'examples-smoke', 'notebooks-smoke'];
const hosted = Object.fromEntries(names.map(name => [name, false]));
const runner = id => ({ id, status: 'online', busy: false, os: 'linux',
  labels: ['self-hosted', 'Linux', 'X64', 'robot-sf-ci-ephemeral'].map(name => ({ name })) });
const context = { repo: { owner: 'll7', repo: 'robot_sf_ll7' }, actor: 'll7',
  eventName: 'push', payload: {} };
const core = { info() {}, setOutput() {} };

test('allocate only healthy idle registered capacity; no runners or bad inventory is hosted', () => {
  assert.deepEqual(allocate({ total_count: 1, runners: [runner(1)] }),
    { ...hosted, 'smoke-artifacts': true });
  assert.deepEqual(allocate({ total_count: 1, runners: [{ ...runner(1), os: 'Linux' }] }),
    { ...hosted, 'smoke-artifacts': true });
  assert.deepEqual(allocate({ total_count: 10, runners: Array.from({ length: 10 }, (_, i) => runner(i + 1)) }),
    Object.fromEntries(names.map(name => [name, true])));
  for (const change of [{ status: 'offline' }, { busy: true }, { busy: undefined },
    { os: 'windows' }, { os: 42 }, { labels: [{ name: 'self-hosted' }] }]) {
    assert.deepEqual(allocate({ total_count: 1, runners: [{ ...runner(1), ...change }] }), hosted);
  }
  for (const data of [null, {}, { total_count: 0, runners: [] },
    { total_count: 101, runners: [runner(1)] }, { total_count: 1, runners: [runner(0)] },
    { total_count: 2, runners: [runner(1), runner(1)] }]) {
    assert.deepEqual(allocate(data), hosted);
  }
});

test('route uses bounded API and emits hosted on API error; disabled/untrusted/retry never probes', async () => {
  process.env.GITHUB_TRIGGERING_ACTOR = 'll7';
  process.env.GITHUB_RUN_ATTEMPT = '1';
  let calls = 0;
  const github = { rest: { actions: { async listSelfHostedRunnersForRepo(args) {
    calls++;
    assert.equal(args.request.timeout, 5000);
    assert.ok(args.request.signal instanceof AbortSignal);
    return { data: { total_count: 1, runners: [runner(1)] } };
  } } } };
  const outputs = {};
  const recordingCore = { info() {}, setOutput(name, value) { outputs[name] = value; } };
  assert.equal((await route({ github, context, core: recordingCore, enabled: 'true', provenance: 'true' }))['smoke-artifacts'], true);
  assert.equal(outputs.smoke_artifacts, 'true');
  assert.equal(outputs.fast_feedback, 'false');
  const denied = [
    { enabled: 'false' }, { enabled: '' }, { provenance: 'false' },
    { provenance: '' }, { provenance: undefined }, { context: { ...context, actor: 'other' } },
    { context: { ...context, eventName: 'workflow_dispatch' } },
    { context: { ...context, eventName: 'pull_request', payload: { pull_request: {
      head: { repo: { full_name: 'outsider/repo' } }, user: { login: 'll7' } } } } },
    { context: { ...context, eventName: 'pull_request', payload: { pull_request: {
      head: { repo: { full_name: 'll7/robot_sf_ll7' } }, user: { login: 'other' } } } } },
  ];
  for (const override of denied) assert.deepEqual(await route({ github, context, core, enabled: 'true', provenance: 'true', ...override }), hosted);
  process.env.GITHUB_RUN_ATTEMPT = '2';
  assert.deepEqual(await route({ github, context, core, enabled: 'true', provenance: 'true' }), hosted);
  process.env.GITHUB_RUN_ATTEMPT = '1';
  process.env.GITHUB_TRIGGERING_ACTOR = 'other';
  assert.deepEqual(await route({ github, context, core, enabled: 'true', provenance: 'true' }), hosted);
  process.env.GITHUB_TRIGGERING_ACTOR = 'll7';
  assert.equal(calls, 1);
  const pr = { base: { ref: 'main' }, head: { repo: { full_name: 'll7/robot_sf_ll7' } }, user: { login: 'll7' } };
  const prContext = { ...context, eventName: 'pull_request', payload: { pull_request: pr } };
  assert.equal((await route({ github, context: prContext, core, enabled: 'true', provenance: 'true' }))['smoke-artifacts'], true);
  assert.deepEqual(await route({ github, context: { ...prContext, payload: { pull_request: {
    ...pr, base: { ref: 'collaborator-branch' } } } }, core, enabled: 'true', provenance: 'true' }), hosted);
  assert.equal(calls, 2);
  github.rest.actions.listSelfHostedRunnersForRepo = async () => { throw new Error('sensitive API error'); };
  const logs = [];
  assert.deepEqual(await route({ github, context, core: { ...core, info(x) { logs.push(x); } }, enabled: 'true', provenance: 'true' }), hosted);
  assert.equal(logs.some(x => x.includes('sensitive')), false);
});

test('a hung inventory call falls back after five seconds', async () => {
  process.env.GITHUB_TRIGGERING_ACTOR = 'll7';
  process.env.GITHUB_RUN_ATTEMPT = '1';
  const github = { rest: { actions: { listSelfHostedRunnersForRepo: () => new Promise(() => {}) } } };
  assert.deepEqual(await route({ github, context, core, enabled: 'true', provenance: 'true' }), hosted);
});

const target = { id: 123, workflow_id: 9, run_attempt: 1, head_sha: 'a'.repeat(40),
  path: '.github/workflows/ci.yml', repository: { full_name: 'll7/robot_sf_ll7' },
  actor: { login: 'll7' }, triggering_actor: { login: 'll7' }, event: 'push',
  head_branch: 'main', status: 'in_progress' };
const queued = { id: 456, name: 'smoke-artifacts', labels: ['robot-sf-ci-ephemeral'], status: 'queued' };

function harness(overrides = {}) {
  let run = { ...target, ...overrides.run };
  let clock = 0;
  let head = run.head_sha;
  const writes = [];
  let jobCalls = 0;
  const github = { rest: { actions: {
    async getWorkflowRun() { return { data: { ...run } }; },
    async listJobsForWorkflowRunAttempt(args) {
      jobCalls++;
      assert.ok(args.request.signal instanceof AbortSignal);
      if (overrides.listError) throw new Error('jobs unavailable');
      if (overrides.incomplete) return { data: { total_count: 2, jobs: [queued] } };
      if (overrides.pages) return { data: { total_count: 101,
        jobs: args.page === 1 ? Array.from({ length: 100 }, (_, i) => ({ id: i + 1000, name: 'other', status: 'completed' })) : [queued] } };
      const jobs = overrides.jobs ? overrides.jobs(jobCalls, clock) : [queued];
      return { data: { total_count: jobs.length, jobs } };
    },
    async cancelWorkflowRun(args) {
      writes.push(['cancel', args.run_id, clock]);
      if (overrides.cancelError) throw new Error('cancel failed');
      if (!overrides.cancelStuck) run = { ...run, status: 'completed', conclusion: 'cancelled' };
      if (overrides.cancelConclusion) run.conclusion = overrides.cancelConclusion;
      if (overrides.movedOnCancel) run.run_attempt = 2;
      if (overrides.supersedeOnCancel) head = 'b'.repeat(40);
    },
    async reRunWorkflow(args) {
      writes.push(['rerun', args.run_id, clock]);
      if (!overrides.unconfirmed) run = { ...run, run_attempt: 2, status: 'queued', conclusion: null };
    },
  }, repos: { async getBranch() { return { data: { commit: { sha: overrides.stale ? 'b'.repeat(40) : head } } }; } },
  pulls: { async get() { return { data: overrides.pr }; } } } };
  return { writes, args: { github, core, context: { ...context, payload: { workflow_run: { ...target, ...overrides.run } } },
    now: () => clock, wait: async ms => { clock += overrides.clockJump && ms === 60000 ? overrides.clockJump : ms; } } };
}

test('queue timing tracks continuous observed jobs; ignores hosted, running and dependencies', () => {
  const since = new Map();
  assert.equal(stranded([queued], 0, since), false);
  assert.equal(stranded([queued], 599999, since), false);
  assert.equal(stranded([queued], 600000, since), true);
  assert.equal(stranded([{ ...queued, status: 'in_progress' }], 600001, since), false);
  assert.equal(stranded([queued], 600002, since), false);
  assert.equal(stranded([{ ...queued, labels: ['ubuntu-latest'] }], 1200002, since), false);
  assert.equal(stranded([{ ...queued, name: 'other-job' }], 1800002, since), false);
});

test('watchdog cancels after ten queued minutes, reruns all once and reads back attempt two', async () => {
  const h = harness();
  const logs = [];
  h.args.core = { ...core, info(x) { logs.push(x); } };
  await watch(h.args);
  assert.deepEqual(h.writes, [['cancel', 123, 600000], ['rerun', 123, 602000]]);
  assert.ok(logs.includes('ci-runner-watchdog run=123 verified: hosted rerun accepted'));
});

test('watchdog never writes for stale, retry, unrelated, completed or recovered runs', async () => {
  for (const options of [{ stale: true }, { run: { run_attempt: 2 } },
    { run: { path: '.github/workflows/other.yml' } }, { run: { actor: { login: 'other' } } },
    { run: { status: 'completed' } }]) {
    const h = harness(options);
    await watch(h.args);
    assert.deepEqual(h.writes, []);
  }
  const h = harness({ jobs: call => call === 12 ? [{ ...queued, status: 'in_progress' }] : [queued] });
  await watch(h.args);
  assert.deepEqual(h.writes, []);
});

test('cancellation error/unconfirmed cancellation or superseded head cannot rerun', async () => {
  for (const options of [{ cancelError: true }, { cancelStuck: true }]) {
    const h = harness(options);
    await assert.rejects(watch(h.args));
    assert.deepEqual(h.writes, [['cancel', 123, 600000]]);
  }
  const h = harness({ supersedeOnCancel: true });
  await watch(h.args);
  assert.deepEqual(h.writes, [['cancel', 123, 600000]]);
});

test('PR recovery requires current open author PR in the same repository', async () => {
  const pr = { state: 'open', draft: false, base: { ref: 'main' }, user: { login: 'll7' },
    head: { sha: target.head_sha, repo: { full_name: 'll7/robot_sf_ll7' } } };
  const run = { event: 'pull_request', pull_requests: [{ number: 321 }] };
  for (const change of [{ state: 'closed' }, { draft: true }, { base: { ref: 'other' } }, { user: { login: 'other' } },
    { head: { ...pr.head, repo: { full_name: 'outsider/repo' } } }, { head: { ...pr.head, sha: 'b'.repeat(40) } }]) {
    const h = harness({ run, pr: { ...pr, ...change } });
    await watch(h.args);
    assert.deepEqual(h.writes, []);
  }
  const h = harness({ run, pr });
  await watch(h.args);
  assert.equal(h.writes.length, 2);
});

test('hosted/running/skipped workload admission finishes observation without waiting', async () => {
  const jobs = names.flatMap(name => name === 'fast-feedback' ? Array.from({ length: 6 }, (_, i) =>
    ({ id: i + 1, name: `fast-feedback (${i + 1})`, status: 'in_progress' })) :
    [{ id: name, name, status: 'completed', conclusion: 'skipped' }]);
  jobs[0] = { ...jobs[0], status: 'queued', labels: ['ubuntu-latest'] };
  const h = harness({ jobs: () => jobs });
  h.args.wait = async () => { assert.fail('completed admission must not sleep'); };
  await watch(h.args);
  assert.deepEqual(h.writes, []);
});

test('complete pagination finds stranded jobs; API failure/incomplete inventory never writes', async () => {
  const h = harness({ pages: true });
  await watch(h.args);
  assert.equal(h.writes.length, 2);
  for (const options of [{ listError: true }, { incomplete: true }]) {
    const denied = harness(options);
    await assert.rejects(watch(denied.args));
    assert.deepEqual(denied.writes, []);
  }
});

test('unconfirmed rerun and wall-clock deadlines fail visibly without another write', async () => {
  const h = harness({ unconfirmed: true });
  await assert.rejects(watch(h.args), /rerun not confirmed/);
  assert.equal(h.writes.length, 2);
  const late = harness({ clockJump: 104 * 60 * 1000 });
  await assert.rejects(watch(late.args), /recovery deadline/);
  assert.deepEqual(late.writes, []);
  const empty = harness({ jobs: () => [] });
  await assert.rejects(watch(empty.args), /observation deadline/);
  assert.deepEqual(empty.writes, []);
});

test('non-cancelled completion, moved attempt and ambiguous PR list cannot rerun', async () => {
  for (const options of [{ cancelConclusion: 'success' }, { movedOnCancel: true }]) {
    const h = harness(options);
    await watch(h.args);
    assert.equal(h.writes.length, 1);
  }
  for (const pull_requests of [[], [{ number: 1 }, { number: 2 }]]) {
    const h = harness({ run: { event: 'pull_request', pull_requests } });
    await watch(h.args);
    assert.deepEqual(h.writes, []);
  }
});
