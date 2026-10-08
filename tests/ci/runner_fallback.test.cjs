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
  assert.deepEqual(allocate({ total_count: 10, runners: Array.from({ length: 10 }, (_, i) => runner(i + 1)) }),
    Object.fromEntries(names.map(name => [name, true])));
  for (const change of [{ status: 'offline' }, { busy: true }, { busy: undefined },
    { os: 'windows' }, { labels: [{ name: 'self-hosted' }] }]) {
    assert.deepEqual(allocate({ total_count: 1, runners: [{ ...runner(1), ...change }] }), hosted);
  }
  for (const data of [null, {}, { total_count: 0, runners: [] },
    { total_count: 101, runners: [runner(1)] }, { total_count: 2, runners: [runner(1), runner(1)] }]) {
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
    return { data: { total_count: 1, runners: [runner(1)] } };
  } } } };
  const outputs = {};
  const recordingCore = { info() {}, setOutput(name, value) { outputs[name] = value; } };
  assert.equal((await route({ github, context, core: recordingCore, enabled: 'true' }))['smoke-artifacts'], true);
  assert.equal(outputs.smoke_artifacts, 'true');
  assert.equal(outputs.fast_feedback, 'false');
  const denied = [
    { enabled: 'false' }, { enabled: '' }, { context: { ...context, actor: 'other' } },
    { context: { ...context, eventName: 'workflow_dispatch' } },
    { context: { ...context, eventName: 'pull_request', payload: { pull_request: {
      head: { repo: { full_name: 'outsider/repo' } }, user: { login: 'll7' } } } } },
    { context: { ...context, eventName: 'pull_request', payload: { pull_request: {
      head: { repo: { full_name: 'll7/robot_sf_ll7' } }, user: { login: 'other' } } } } },
  ];
  for (const override of denied) assert.deepEqual(await route({ github, context, core, enabled: 'true', ...override }), hosted);
  process.env.GITHUB_RUN_ATTEMPT = '2';
  assert.deepEqual(await route({ github, context, core, enabled: 'true' }), hosted);
  process.env.GITHUB_RUN_ATTEMPT = '1';
  process.env.GITHUB_TRIGGERING_ACTOR = 'other';
  assert.deepEqual(await route({ github, context, core, enabled: 'true' }), hosted);
  process.env.GITHUB_TRIGGERING_ACTOR = 'll7';
  assert.equal(calls, 1);
  github.rest.actions.listSelfHostedRunnersForRepo = async () => { throw new Error('sensitive API error'); };
  const logs = [];
  assert.deepEqual(await route({ github, context, core: { ...core, info(x) { logs.push(x); } }, enabled: 'true' }), hosted);
  assert.equal(logs.some(x => x.includes('sensitive')), false);
});

test('a hung inventory call falls back after five seconds', async () => {
  process.env.GITHUB_TRIGGERING_ACTOR = 'll7';
  process.env.GITHUB_RUN_ATTEMPT = '1';
  const github = { rest: { actions: { listSelfHostedRunnersForRepo: () => new Promise(() => {}) } } };
  assert.deepEqual(await route({ github, context, core, enabled: 'true' }), hosted);
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
    async listJobsForWorkflowRunAttempt() {
      jobCalls++;
      const jobs = overrides.jobs ? overrides.jobs(jobCalls, clock) : [queued];
      return { data: { total_count: jobs.length, jobs } };
    },
    async cancelWorkflowRun(args) {
      writes.push(['cancel', args.run_id, clock]);
      if (overrides.cancelError) throw new Error('cancel failed');
      if (!overrides.cancelStuck) run = { ...run, status: 'completed', conclusion: 'cancelled' };
      if (overrides.supersedeOnCancel) head = 'b'.repeat(40);
    },
    async reRunWorkflow(args) {
      writes.push(['rerun', args.run_id, clock]);
      run = { ...run, run_attempt: 2, status: 'queued', conclusion: null };
    },
  }, repos: { async getBranch() { return { data: { commit: { sha: overrides.stale ? 'b'.repeat(40) : head } } }; } },
  pulls: { async get() { return { data: overrides.pr }; } } } };
  return { writes, args: { github, core, context: { ...context, payload: { workflow_run: { ...target, ...overrides.run } } },
    now: () => clock, wait: async ms => { clock += ms; } } };
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
  const pr = { state: 'open', draft: false, user: { login: 'll7' },
    head: { sha: target.head_sha, repo: { full_name: 'll7/robot_sf_ll7' } } };
  const run = { event: 'pull_request', pull_requests: [{ number: 321 }] };
  for (const change of [{ state: 'closed' }, { draft: true }, { user: { login: 'other' } },
    { head: { ...pr.head, repo: { full_name: 'outsider/repo' } } }, { head: { ...pr.head, sha: 'b'.repeat(40) } }]) {
    const h = harness({ run, pr: { ...pr, ...change } });
    await watch(h.args);
    assert.deepEqual(h.writes, []);
  }
  const h = harness({ run, pr });
  await watch(h.args);
  assert.equal(h.writes.length, 2);
});
