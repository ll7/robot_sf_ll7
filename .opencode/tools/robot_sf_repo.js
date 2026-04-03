import { tool } from "@opencode-ai/plugin";
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

async function runCommand(command, args, cwd) {
  try {
    const { stdout, stderr } = await execFileAsync(command, args, {
      cwd,
      maxBuffer: 1024 * 1024 * 10,
    });
    return {
      command: [command, ...args].join(" "),
      cwd,
      exitCode: 0,
      stdout: stdout.trim(),
      stderr: stderr.trim(),
    };
  } catch (error) {
    return {
      command: [command, ...args].join(" "),
      cwd,
      exitCode: error.code ?? 1,
      stdout: (error.stdout ?? "").trim(),
      stderr: (error.stderr ?? error.message ?? "").trim(),
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
      return JSON.stringify({ repo_path: args.repo_path, matches: [], references: [] }, null, 2);
    }

    const fileList = await runCommand("rg", ["--files", ...dirs], root);
    const fileMatches = fileList.stdout
      .split("\n")
      .filter(Boolean)
      .filter((entry) => entry.includes(stem) || entry.includes(lookup))
      .slice(0, args.limit);
    const contentMatches = await runCommand(
      "rg",
      ["-n", "--glob", "test_*.py", stem, ...dirs],
      root,
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
        args.query,
        ...searchRoots,
      ],
      root,
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
        command: "/bin/bash",
        args: ["scripts/dev/ruff_fix_format.sh"],
      },
      tests_parallel: {
        command: "/bin/bash",
        args: ["scripts/dev/run_tests_parallel.sh"],
      },
      pr_ready_check: {
        command: "/bin/bash",
        args: ["-lc", "BASE_REF=origin/main scripts/dev/pr_ready_check.sh"],
      },
    };
    const selected = commandMap[args.target];
    const result = await runCommand(selected.command, selected.args, root);
    return JSON.stringify(result, null, 2);
  },
});
