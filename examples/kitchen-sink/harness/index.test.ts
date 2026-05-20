import { expect, test } from "bun:test";

test("kitchen-sink harness passes", async () => {
  const proc = Bun.spawn(["bun", "run", "index.ts"], {
    cwd: import.meta.dir,
    env: process.env,
    stdout: "pipe",
    stderr: "pipe",
  });

  const [exitCode, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);

  if (exitCode !== 0) {
    throw new Error(`Harness failed with exit ${exitCode}\nSTDOUT:\n${stdout}\nSTDERR:\n${stderr}`);
  }

  expect(stdout).toContain("ALL PASSED");
}, 120_000);
