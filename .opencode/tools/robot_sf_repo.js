import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

function makeSchemaNode() {
  return {
    describe() {
      return this;
    },
    default() {
      return this;
    },
    int() {
      return this;
    },
    min() {
      return this;
    },
    max() {
      return this;
    },
  };
}

function makeFallbackTool() {
  const schema = {
    string: () => makeSchemaNode(),
    number: () => makeSchemaNode(),
    enum: () => makeSchemaNode(),
  };
  const fallback = (definition) => definition;
  fallback.schema = schema;
  return fallback;
}

let tool;

try {
  ({ tool } = await import("@opencode-ai/plugin"));
} catch {
  tool = makeFallbackTool();
}

const execFileAsync = promisify(execFile);
const COMMAND_TIMEOUT_MS = 30 * 60 * 1000;

async function runCommand(command, args, options = {}) {
  const { cwd, env = {} } = options;
  try {
    const { stdout, stderr } = await execFileAsync(command, args, {
      cwd,
      env: { ...process.env, ...env },
      maxBuffer: 1024 * 1024 * 10,
      timeout: COMMAND_TIMEOUT_MS,
    });
    return {
      command: [command, ...args].join(" "),
      cwd,
      exitCode: 0,
      stdout: stdout.trim(),
      stderr: stderr.trim(),
    };
  } catch (error) {
    const timedOut =
      error.killed === true || error.code === "ETIMEDOUT" || error.signal === "SIGTERM";
    const stderr = timedOut
      ? `Command timed out after ${COMMAND_TIMEOUT_MS} ms: ${[command, ...args].join(" ")}`
      : (error.stderr ?? error.message ?? "").trim();
    const normalizedExitCode =
      typeof error.code === "number"
        ? error.code
        : Number.isFinite(Number.parseInt(error.code, 10))
          ? Number.parseInt(error.code, 10)
          : 1;
    return {
      command: [command, ...args].join(" "),
      cwd,
      exitCode: normalizedExitCode,
      stdout: (error.stdout ?? "").trim(),
      stderr,
    };
  }
}

function existingDirs(root, dirs) {
  return dirs.map((dir) => path.join(root, dir)).filter((candidate) => fs.existsSync(candidate));
}

export const find_tests = tool({
  description: "Find likely Robot SF test files and test references for a repo path.",
  args: {
    repo_path: tool.schema.string().describe("Repository-relative file path to inspect."),
    limit: tool.schema.number().int().min(1).max(50).default(20),
  },
  async execute(args, context) {
    const root = context.worktree;
    const lookup = path.basename(args.repo_path);
    const stem = lookup.replace(path.extname(lookup), "").replace(/^test_/, "");
    const dirs = existingDirs(root, ["tests", "test_pygame", "fast-pysf/tests"]);
    if (dirs.length === 0) {
      return JSON.stringify(
        { repo_path: args.repo_path, stem, matches: [], references: [] },
        null,
        2,
      );
    }

    const fileList = await runCommand("rg", ["--files", ...dirs], { cwd: root });
    const fileMatches = fileList.stdout
      .split("\n")
      .filter(Boolean)
      .filter((entry) => entry.includes(stem) || entry.includes(lookup))
      .slice(0, args.limit);
    const contentMatches = await runCommand(
      "rg",
      ["-n", "--glob", "test_*.py", "--", stem, ...dirs],
      { cwd: root },
    );

    return JSON.stringify(
      {
        repo_path: args.repo_path,
        stem,
        matches: fileMatches,
        references: contentMatches.stdout.split("\n").filter(Boolean).slice(0, args.limit),
      },
      null,
      2,
    );
  },
});

export const search_configs = tool({
  description: "Search benchmark, training, and validation config surfaces in Robot SF.",
  args: {
    query: tool.schema.string().describe("Text or regex to search for."),
    limit: tool.schema.number().int().min(1).max(100).default(30),
  },
  async execute(args, context) {
    const root = context.worktree;
    const searchRoots = existingDirs(root, ["configs", "docs/context", "scripts/validation"]);
    if (searchRoots.length === 0) {
      return JSON.stringify({ query: args.query, exitCode: 0, matches: [] }, null, 2);
    }
    const result = await runCommand(
      "rg",
      [
        "-n",
        "--glob",
        "*.yaml",
        "--glob",
        "*.yml",
        "--glob",
        "*.md",
        "--glob",
        "*.py",
        "--",
        args.query,
        ...searchRoots,
      ],
      { cwd: root },
    );
    return JSON.stringify(
      {
        query: args.query,
        exitCode: result.exitCode,
        matches: result.stdout.split("\n").filter(Boolean).slice(0, args.limit),
      },
      null,
      2,
    );
  },
});

export const run_validation = tool({
  description: "Run one of the canonical Robot SF validation entry points.",
  args: {
    target: tool.schema
      .enum(["ruff_fix_format", "tests_parallel", "pr_ready_check"])
      .describe("Validation entry point to execute."),
  },
  async execute(args, context) {
    const root = context.worktree;
    const commandMap = {
      ruff_fix_format: {
        command: path.join(root, "scripts/dev/ruff_fix_format.sh"),
        args: [],
      },
      tests_parallel: {
        command: path.join(root, "scripts/dev/run_tests_parallel.sh"),
        args: [],
      },
      pr_ready_check: {
        command: path.join(root, "scripts/dev/pr_ready_check.sh"),
        args: [],
        env: { BASE_REF: "origin/main" },
      },
    };
    const selected = commandMap[args.target];
    const result = await runCommand(selected.command, selected.args, {
      cwd: root,
      env: selected.env,
    });
    return JSON.stringify(result, null, 2);
  },
});
