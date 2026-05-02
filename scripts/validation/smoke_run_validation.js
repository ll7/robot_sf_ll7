#!/usr/bin/env node

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { run_validation } from "../../.opencode/tools/robot_sf_repo.js";

function writeExecutable(filePath, body) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, body, { encoding: "utf-8", mode: 0o755 });
}

function makeStubWorktree() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "robot-sf-run-validation-"));
  writeExecutable(
    path.join(root, "scripts/dev/ruff_fix_format.sh"),
    "#!/usr/bin/env bash\nset -euo pipefail\necho 'stub:ruff_fix_format'\n",
  );
  writeExecutable(
    path.join(root, "scripts/dev/run_tests_parallel.sh"),
    "#!/usr/bin/env bash\nset -euo pipefail\necho 'stub:tests_parallel'\n",
  );
  writeExecutable(
    path.join(root, "scripts/dev/pr_ready_check.sh"),
    "#!/usr/bin/env bash\nset -euo pipefail\necho \"stub:pr_ready_check:${BASE_REF:-unset}\"\n",
  );
  return root;
}

async function main() {
  const root = makeStubWorktree();
  try {
    const expected = {
      ruff_fix_format: {
        command: path.join(root, "scripts/dev/ruff_fix_format.sh"),
        stdout: "stub:ruff_fix_format",
      },
      tests_parallel: {
        command: path.join(root, "scripts/dev/run_tests_parallel.sh"),
        stdout: "stub:tests_parallel",
      },
      pr_ready_check: {
        command: path.join(root, "scripts/dev/pr_ready_check.sh"),
        stdout: "stub:pr_ready_check:origin/main",
      },
    };

    for (const [target, expectation] of Object.entries(expected)) {
      const raw = await run_validation.execute({ target }, { worktree: root });
      const result = JSON.parse(raw);
      assert.equal(result.command, expectation.command, `unexpected command for ${target}`);
      assert.equal(result.cwd, root, `unexpected cwd for ${target}`);
      assert.equal(result.exitCode, 0, `unexpected exit code for ${target}`);
      assert.equal(result.stdout, expectation.stdout, `unexpected stdout for ${target}`);
      assert.equal(result.stderr, "", `unexpected stderr for ${target}`);
    }
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error.stack ?? String(error));
  process.exit(1);
});
