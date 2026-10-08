'use strict';

// Availability/recovery only. The workflow and runner hook still own admission.
const LABEL = 'robot-sf-ci-ephemeral';
const COSTS = { 'fast-feedback': 6, 'smoke-artifacts': 1, 'wheel-smoke-install': 1,
  'examples-smoke': 1, 'notebooks-smoke': 1 };
const QUEUE_LIMIT_MS = 10 * 60 * 1000;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const bounded = args => ({ ...args, request: { timeout: 5000, signal: AbortSignal.timeout(5000) } });

function eligible(context, enabled) {
  const pr = context.payload.pull_request;
  return enabled === 'true' && context.repo.owner === 'll7' &&
    context.repo.repo === 'robot_sf_ll7' && context.actor === 'll7' &&
    process.env.GITHUB_TRIGGERING_ACTOR === 'll7' &&
    process.env.GITHUB_RUN_ATTEMPT === '1' &&
    (context.eventName === 'push' || (context.eventName === 'pull_request' &&
      pr?.base?.ref === 'main' && pr?.head?.repo?.full_name === 'll7/robot_sf_ll7' && pr?.user?.login === 'll7'));
}

function allocate(data) {
  const result = Object.fromEntries(Object.keys(COSTS).map(name => [name, false]));
  // Truncated, malformed, or ambiguous inventory is not proof of availability.
  if (!Number.isInteger(data?.total_count) || !Array.isArray(data.runners) ||
      data.total_count !== data.runners.length || data.total_count < 0) return result;
  const ids = new Set();
  let idle = 0;
  for (const runner of data.runners) {
    if (!Number.isSafeInteger(runner?.id) || runner.id <= 0 || ids.has(runner.id) ||
        !Array.isArray(runner.labels)) return result;
    ids.add(runner.id);
    const labels = runner.labels.map(label => typeof label?.name === 'string' ? label.name.toLowerCase() : null);
    if (runner.status === 'online' && runner.busy === false && typeof runner.os === 'string' && runner.os.toLowerCase() === 'linux' &&
        ['self-hosted', 'linux', 'x64', LABEL].every(label => labels.includes(label))) idle++;
  }
  for (const [name, cost] of Object.entries(COSTS)) {
    if (idle >= cost) { result[name] = true; idle -= cost; }
  }
  return result;
}

async function route({ github, context, core, enabled }) {
  let result = allocate(null);
  let reason = 'hosted: admission disabled or retry';
  if (eligible(context, enabled)) {
    let timer;
    try {
      const response = await Promise.race([
        github.rest.actions.listSelfHostedRunnersForRepo(bounded({ ...context.repo, per_page: 100 })),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('timeout')), 5000); }),
      ]);
      result = allocate(response.data);
      reason = Object.values(result).some(Boolean) ? 'idle capacity allocated' : 'hosted: no proven idle capacity';
    } catch {
      // Never log response bodies, inventory names, credentials or error text.
      reason = 'hosted: inventory API unavailable';
    } finally { clearTimeout(timer); }
  }
  for (const [name, useSelfHosted] of Object.entries(result)) core.setOutput(name.replaceAll('-', '_'), String(useSelfHosted));
  core.info(`ci-runner-route ${reason}`);
  return result;
}

function sameAttempt(run, target) {
  return run.id === target.id && run.run_attempt === 1 && run.head_sha === target.head_sha &&
    run.workflow_id === target.workflow_id && run.path === '.github/workflows/ci.yml' &&
    run.repository?.full_name === 'll7/robot_sf_ll7' && run.actor?.login === 'll7' &&
    run.triggering_actor?.login === 'll7' && ['push', 'pull_request'].includes(run.event);
}

function stranded(jobs, now, queuedSince) {
  const queued = jobs.filter(job =>
    (Object.hasOwn(COSTS, job.name) || /^fast-feedback \([1-6]\)$/.test(job.name)) &&
    job.labels?.includes(LABEL) && job.status === 'queued');
  const ids = new Set(queued.map(job => job.id));
  for (const id of queuedSince.keys()) if (!ids.has(id)) queuedSince.delete(id);
  for (const job of queued) if (!queuedSince.has(job.id)) queuedSince.set(job.id, now);
  // Jobs API has no documented queue timestamp. Measure continuous observed
  // queue time, not run creation or dependency-wait time.
  return queued.some(job => now - queuedSince.get(job.id) >= QUEUE_LIMIT_MS);
}

function admissionFinished(jobs) {
  return Object.entries(COSTS).every(([name, count]) => {
    const matching = jobs.filter(job => job.name === name ||
      (name === 'fast-feedback' && /^fast-feedback \([1-6]\)$/.test(job.name)));
    return matching.length >= count && matching.every(job =>
      ['in_progress', 'completed'].includes(job.status) || job.labels?.includes('ubuntu-latest'));
  });
}

async function currentHead(github, repo, run) {
  if (run.event === 'push') {
    if (run.head_branch !== 'main') return false;
    const { data } = await github.rest.repos.getBranch(bounded({ ...repo, branch: 'main' }));
    return data.commit.sha === run.head_sha;
  }
  if (run.pull_requests?.length !== 1) return false;
  const { data } = await github.rest.pulls.get(bounded({ ...repo, pull_number: run.pull_requests[0].number }));
  return data.state === 'open' && !data.draft && data.base?.ref === 'main' && data.user.login === 'll7' &&
    data.head.repo?.full_name === 'll7/robot_sf_ll7' && data.head.sha === run.head_sha;
}

async function listJobs(github, repo, target) {
  const jobs = [];
  for (let page = 1; page <= 10; page++) {
    const { data } = await github.rest.actions.listJobsForWorkflowRunAttempt(bounded({ ...repo,
      run_id: target.id, attempt_number: 1, per_page: 100, page }));
    if (!Array.isArray(data.jobs) || !Number.isInteger(data.total_count)) throw new Error('incomplete jobs');
    jobs.push(...data.jobs);
    if (jobs.length === data.total_count) return jobs;
    if (jobs.length > data.total_count || data.jobs.length < 100) throw new Error('incomplete jobs');
  }
  throw new Error('job pagination limit');
}

async function watch({ github, context, core, wait = sleep, now = Date.now }) {
  const repo = context.repo;
  const target = context.payload.workflow_run;
  const log = verdict => core.info(`ci-runner-watchdog run=${target?.id} ${verdict}`);
  if (repo.owner !== 'll7' || repo.repo !== 'robot_sf_ll7' || !target || !sameAttempt(target, target)) {
    log('decline: outside first-attempt CI scope'); return;
  }
  const args = { ...repo, run_id: target.id };
  const getRun = async () => (await github.rest.actions.getWorkflowRun(bounded(args))).data;
  const queuedSince = new Map();
  const deadline = now() + 110 * 60 * 1000;
  // A separate hosted workflow survives cancellation of the target CI run.
  // Includes the dispatch gate (55m) and subsequent job admission; bounded at 110m.
  while (now() < deadline) {
    const run = await getRun();
    if (!sameAttempt(run, target) || run.status === 'completed') { log('decline: finished or moved'); return; }
    if (!await currentHead(github, repo, run)) { log('decline: superseded or untrusted head'); return; }
    const jobs = await listJobs(github, repo, target);
    if (admissionFinished(jobs)) { log('decline: all workload admission finished or hosted'); return; }
    if (stranded(jobs, now(), queuedSince)) {
      // Re-read immediately before the write. Cancel the whole run: GitHub has
      // no job-level cancel or supported way to change a queued job's label.
      const fresh = await getRun();
      if (!sameAttempt(fresh, target) || fresh.status === 'completed' ||
          !await currentHead(github, repo, fresh) ||
          !stranded(await listJobs(github, repo, target), now(), queuedSince)) {
        log('decline: queue recovered or run moved'); return;
      }
      if (now() + 7 * 60 * 1000 >= deadline) {
        log('decline: insufficient time to confirm recovery');
        throw new Error('watchdog recovery deadline; inspect and rerun manually hosted');
      }
      log('act: cancel first attempt after 10m self-hosted queue');
      await github.rest.actions.cancelWorkflowRun(bounded(args));
      const cancelDeadline = now() + 6 * 60 * 1000;
      while (now() < cancelDeadline) {
        await wait(2000);
        const cancelled = await getRun();
        if (!sameAttempt(cancelled, target)) { log('decline: attempt changed during cancellation'); return; }
        if (cancelled.status !== 'completed') continue;
        if (cancelled.conclusion !== 'cancelled' || !await currentHead(github, repo, cancelled)) {
          log('decline: not cancelled or superseded'); return;
        }
        log('act: rerun all jobs; attempt >1 forces hosted');
        await github.rest.actions.reRunWorkflow(bounded(args));
        // Successful POST alone does not establish effective state.
        const verifyDeadline = now() + 30000;
        while (now() < verifyDeadline) {
          await wait(2000);
          const resumed = await getRun();
          if (resumed.head_sha === target.head_sha && resumed.run_attempt > 1) {
            log('verified: hosted rerun accepted'); return;
          }
        }
        throw new Error('hosted rerun not confirmed; inspect before retrying');
      }
      throw new Error('cancellation not confirmed; no rerun issued');
    }
    log('observe: no bounded self-hosted queue yet');
    await wait(60000);
  }
  throw new Error('watchdog observation deadline exceeded');
}

module.exports = { route, allocate, eligible, watch, sameAttempt, stranded };
