import { appendFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

import type * as GeneratedClient from "../frontend/src/generated/api";

type Client = typeof GeneratedClient;
type Task = GeneratedClient.Task;
type WeatherUpdate = GeneratedClient.WeatherUpdate;

const ROOT = new URL("../../..", import.meta.url).pathname.replace(/\/$/, "");
const BACKEND_DIR = `${ROOT}/examples/kitchen-sink/backend`;
const PYTHON_GENERATED_DIR = `${ROOT}/examples/kitchen-sink/frontend/src/generated`;
const PYTHON_API_PATH = `${PYTHON_GENERATED_DIR}/api.ts`;
const RUST_GENERATED_DIR = `${import.meta.dir}/generated/rust`;
const RUST_API_PATH = `${RUST_GENERATED_DIR}/api.ts`;
const PYTHON_EXECUTABLE = `${ROOT}/bindings/python/.venv/bin/python`;

const mode = parseMode();
const PYTHON_BASE_URL = process.env.ZYNK_PYTHON_BASE_URL ?? (mode === "python" ? process.env.ZYNK_BASE_URL : undefined) ?? "http://127.0.0.1:8100";
const RUST_BASE_URL = process.env.ZYNK_RUST_BASE_URL ?? (mode === "rust" || mode === "cross" ? process.env.ZYNK_BASE_URL : undefined) ?? "http://127.0.0.1:8101";
const runId = new Date().toISOString().replace(/[:.]/g, "-");
const LOG_DIR = `${import.meta.dir}/logs/${runId}${mode === "python" ? "" : `-${mode}`}`;

interface ServerProcess {
  name: string;
  proc: ReturnType<typeof Bun.spawn>;
  stdout: Promise<void>;
  stderr: Promise<void>;
}

interface EvidenceEntry {
  suite?: string;
  kind: string;
  name: string;
  request?: unknown;
  response?: unknown;
  raw?: string;
}

interface SuiteContext {
  label: string;
  baseUrl: string;
  client: Client;
}

const evidence: EvidenceEntry[] = [];
const testResults: { name: string; status: "PASS" | "FAIL"; detail?: string }[] = [];

await Bun.$`mkdir -p ${LOG_DIR}`;

function parseMode(): "python" | "rust" | "cross" | "all" {
  const arg = Bun.argv.slice(2).find((value) => value.startsWith("--mode="));
  const value = (arg?.split("=", 2)[1] ?? process.env.ZYNK_HARNESS_MODE ?? "python") as "python" | "rust" | "cross" | "all";
  if (!["python", "rust", "cross", "all"].includes(value)) {
    throw new Error(`Unknown harness mode ${value}; expected python, rust, cross, or all`);
  }
  return value;
}

function portOf(baseUrl: string): string {
  const url = new URL(baseUrl);
  return url.port || (url.protocol === "https:" ? "443" : "80");
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function assertEqual<T>(actual: T, expected: T, message: string): void {
  if (actual !== expected) {
    throw new Error(`${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

async function appendLog(file: string, text: string): Promise<void> {
  await appendFile(`${LOG_DIR}/${file}`, text);
}

async function record(entry: EvidenceEntry): Promise<void> {
  evidence.push(entry);
  await Bun.write(`${LOG_DIR}/evidence.json`, `${JSON.stringify(evidence, null, 2)}\n`);
}

async function step(name: string, fn: () => Promise<void>): Promise<void> {
  console.log(`[RUN] ${name}`);
  try {
    await fn();
    testResults.push({ name, status: "PASS" });
    await appendLog("harness.log", `[PASS] ${name}\n`);
    console.log(`[PASS] ${name}`);
  } catch (error) {
    const detail = error instanceof Error ? error.stack ?? error.message : String(error);
    testResults.push({ name, status: "FAIL", detail });
    await appendLog("harness.log", `[FAIL] ${name}\n${detail}\n`);
    console.error(`[FAIL] ${name}`);
    throw error;
  }
}

async function pump(stream: ReadableStream<Uint8Array> | null, file: string): Promise<void> {
  if (!stream) return;
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    await appendLog(file, decoder.decode(value, { stream: true }));
  }
}

async function runCommand(name: string, command: string[], options: { cwd: string; env?: Record<string, string | undefined> }): Promise<void> {
  await appendLog("harness.log", `[COMMAND] ${name}: ${command.join(" ")} (cwd=${options.cwd})\n`);
  const proc = Bun.spawn(command, {
    cwd: options.cwd,
    env: { ...process.env, ...options.env },
    stdout: "pipe",
    stderr: "pipe",
  });
  const [exitCode, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  await appendLog(`${name}.stdout.log`, stdout);
  await appendLog(`${name}.stderr.log`, stderr);
  if (exitCode !== 0) {
    throw new Error(`${name} failed with exit ${exitCode}; see ${LOG_DIR}/${name}.stderr.log`);
  }
}

async function generatePythonClient(): Promise<void> {
  await runCommand("gen-python-client", [
    "cargo",
    "run",
    "-q",
    "-p",
    "zynk-cli",
    "--",
    "gen",
    "typescript",
    "--target",
    "python",
    "--out",
    PYTHON_GENERATED_DIR,
    "--app",
    "main:app",
    "--python",
    PYTHON_EXECUTABLE,
  ], {
    cwd: BACKEND_DIR,
    env: {
      PYTHONPATH: `${ROOT}/bindings/python${process.env.PYTHONPATH ? `:${process.env.PYTHONPATH}` : ""}`,
    },
  });
}

async function generateRustClient(): Promise<void> {
  await runCommand("gen-rust-client", [
    "cargo",
    "run",
    "-q",
    "-p",
    "zynk-cli",
    "--",
    "gen",
    "typescript",
    "--target",
    "rust",
    "--out",
    RUST_GENERATED_DIR,
    "--rust-cmd",
    "cargo",
    "--rust-arg=run",
    "--rust-arg=-q",
    "--rust-arg=-p",
    "--rust-arg=rust-axum-kitchen-sink",
    "--rust-arg=--",
  ], { cwd: ROOT });
}

async function loadClient(apiPath: string): Promise<Client> {
  const href = pathToFileURL(apiPath).href;
  return (await import(`${href}?run=${runId}-${Math.random()}`)) as Client;
}

async function startPythonServer(baseUrl: string): Promise<ServerProcess> {
  const proc = Bun.spawn(["uv", "run", "python", "-c", "import main; main.app.port = int(__import__('os').environ['PORT']); main.app.run(dev=False)"], {
    cwd: BACKEND_DIR,
    env: {
      ...process.env,
      PORT: portOf(baseUrl),
      ZYNK_DEV: "0",
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: `${ROOT}/bindings/python${process.env.PYTHONPATH ? `:${process.env.PYTHONPATH}` : ""}`,
    },
    stdout: "pipe",
    stderr: "pipe",
  });

  const stdout = pump(proc.stdout, "python-server.stdout.log");
  const stderr = pump(proc.stderr, "python-server.stderr.log");
  return { name: "python", proc, stdout, stderr };
}

async function startRustServer(baseUrl: string): Promise<ServerProcess> {
  const proc = Bun.spawn(["cargo", "run", "--release", "--bin", "rust-axum-kitchen-sink", "--", "--port", portOf(baseUrl)], {
    cwd: `${ROOT}/examples/rust-axum-kitchen-sink`,
    env: process.env,
    stdout: "pipe",
    stderr: "pipe",
  });

  const stdout = pump(proc.stdout, "rust-server.stdout.log");
  const stderr = pump(proc.stderr, "rust-server.stderr.log");
  return { name: "rust", proc, stdout, stderr };
}

async function stopServer(server: ServerProcess): Promise<void> {
  server.proc.kill("SIGTERM");
  const exited = await Promise.race([
    server.proc.exited,
    new Promise<"timeout">((resolve) => setTimeout(() => resolve("timeout"), 5_000)),
  ]);
  if (exited === "timeout") {
    server.proc.kill("SIGKILL");
    await server.proc.exited;
  }
  await Promise.allSettled([server.stdout, server.stderr]);
}

async function waitForHealth(baseUrl: string, suite: string): Promise<void> {
  const started = Date.now();
  let lastError = "server did not respond";
  while (Date.now() - started < 30_000) {
    try {
      const response = await fetch(`${baseUrl}/`);
      if (response.ok) {
        const body = await response.json();
        await record({ suite, kind: "health", name: "GET /", response: body });
        return;
      }
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await Bun.sleep(250);
  }
  throw new Error(`Backend healthcheck failed for ${baseUrl}: ${lastError}`);
}

async function rawJson(ctx: SuiteContext, path: string, body: unknown): Promise<unknown> {
  const response = await fetch(`${ctx.baseUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  await record({ suite: ctx.label, kind: "raw-http", name: path, request: body, raw: text });
  assert(response.ok, `${path} failed with HTTP ${response.status}: ${text}`);
  return JSON.parse(text);
}

async function testCommands(ctx: SuiteContext): Promise<void> {
  const user = await ctx.client.getUser({ userId: 1 });
  await record({ suite: ctx.label, kind: "command", name: "getUser", request: { userId: 1 }, response: user });
  assertEqual(user.name, "Alice", "getUser returns fixture user");
  assertEqual(user.email, "alice@example.com", "getUser maps email");
  assertEqual(user.isActive, true, "getUser maps isActive");

  const cities = await ctx.client.listCities();
  await record({ suite: ctx.label, kind: "command", name: "listCities", response: cities });
  assert(cities.includes("Tokyo"), "listCities includes Tokyo fixture");

  const forecast = await ctx.client.getForecast({ city: "Tokyo", days: 2 });
  await record({ suite: ctx.label, kind: "command", name: "getForecast", request: { city: "Tokyo", days: 2 }, response: forecast });
  assertEqual(forecast.length, 2, "getForecast honors days arg");
  assert(forecast.every((day) => typeof day.precipitationChance === "number"), "forecast maps precipitationChance");

  const stats = await ctx.client.getTaskStats();
  await record({ suite: ctx.label, kind: "command", name: "getTaskStats", response: stats });
  assert(stats.total >= 4, "getTaskStats sees seeded tasks");
  assert(stats.byPriority.high >= 2, "getTaskStats maps byPriority record");

  const updated = await ctx.client.updateUser({ userId: 2, name: null, email: null, isActive: false });
  await record({ suite: ctx.label, kind: "command", name: "updateUser", request: { userId: 2, name: null, email: null, isActive: false }, response: updated });
  assertEqual(updated.name, "Bob", "nullable name does not overwrite when null");
  assertEqual(updated.email, "bob@example.com", "nullable email does not overwrite when null");
  assertEqual(updated.isActive, false, "nullable boolean updates when false");
}

async function testTypeFidelity(ctx: SuiteContext): Promise<void> {
  const created = await ctx.client.createTask({
    title: `Harness urgent task ${ctx.label}`,
    description: null,
    priority: "urgent",
    dueDate: null,
    labelIds: [1, 2],
  });
  await record({ suite: ctx.label, kind: "type-fidelity", name: "createTask", response: created });

  const priority: "urgent" = created.priority as "urgent";
  const status: "todo" = created.status as "todo";
  assertEqual(priority, "urgent", "enum-with-values priority survives runtime");
  assertEqual(status, "todo", "enum-with-values status survives runtime");
  assertEqual(created.description, null, "optional+nullable description round-trips null");
  assertEqual(created.dueDate, null, "optional+nullable dueDate round-trips null");
  assert(created.labels?.some((label) => label.name === "Bug"), "nested label model maps camelCase fields");

  const filtered = await ctx.client.listTasks({ status: "todo", priority: "urgent", labelId: null });
  await record({ suite: ctx.label, kind: "type-fidelity", name: "listTasks", request: { status: "todo", priority: "urgent", labelId: null }, response: filtered });
  assert(filtered.some((task: Task) => task.id === created.id), "nullable filter param accepted as null");
}

async function testRawWireAndStatic(ctx: SuiteContext): Promise<void> {
  const raw = await rawJson(ctx, "/command/get_user", { user_id: 1 });
  assertEqual((raw as { result: { is_active: boolean } }).result.is_active, true, "raw command response preserves snake_case wire");

  const url = ctx.client.downloadSampleUrl({ filename: "sample.txt" });
  const response = await fetch(url);
  const text = await response.text();
  await record({ suite: ctx.label, kind: "static", name: "downloadSampleUrl", request: { url }, response: { status: response.status, text } });
  assertEqual(response.status, 200, "static handler returns 200");
  assertEqual(text, "Zynk kitchen sink static sample\n", "static file fixture content matches");
}

async function testChannel(ctx: SuiteContext): Promise<void> {
  const update = await new Promise<WeatherUpdate>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("streamWeather timed out")), 8_000);
    const channel = ctx.client.streamWeather({ city: "Tokyo", intervalSeconds: 1 });
    channel.subscribe((message) => {
      clearTimeout(timeout);
      channel.close();
      resolve(message);
    });
    channel.onError((error) => {
      clearTimeout(timeout);
      channel.close();
      reject(new Error(`channel error: ${JSON.stringify(error)}`));
    });
  });
  await record({ suite: ctx.label, kind: "channel", name: "streamWeather", request: { city: "Tokyo", intervalSeconds: 1 }, response: update });
  assertEqual(update.city, "Tokyo", "channel update maps city");
  assert(typeof update.temperature === "number", "channel update has numeric temperature");
  assert(typeof update.conditions === "string", "channel update has conditions");
}

async function testUpload(ctx: SuiteContext): Promise<void> {
  const bytes = new TextEncoder().encode("hello from bun harness\n");
  const file = new File([bytes], "harness.txt", { type: "text/plain" });
  const body = new FormData();
  body.append("files", file);
  body.append("_args", JSON.stringify({}));
  const rawResponse = await fetch(`${ctx.baseUrl}/upload/upload_file`, { method: "POST", body });
  const rawText = await rawResponse.text();
  await record({ suite: ctx.label, kind: "upload-raw", name: "POST /upload/upload_file", request: { filename: file.name, size: file.size, type: file.type }, raw: rawText });
  assert(rawResponse.ok, `raw upload failed with HTTP ${rawResponse.status}: ${rawText}`);

  class HarnessXMLHttpRequest {
    upload: { onprogress: ((event: { lengthComputable: boolean; loaded: number; total: number }) => void) | null } = { onprogress: null };
    status = 0;
    responseText = "";
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    onabort: (() => void) | null = null;
    private method = "POST";
    private url = "";
    private aborted = false;

    open(method: string, url: string): void {
      this.method = method;
      this.url = url;
    }

    send(body: BodyInit | null): void {
      void fetch(this.url, { method: this.method, body })
        .then(async (response) => {
          this.status = response.status;
          this.responseText = await response.text();
          this.upload.onprogress?.({ lengthComputable: true, loaded: file.size, total: file.size });
          if (!this.aborted) this.onload?.();
        })
        .catch(() => {
          if (!this.aborted) this.onerror?.();
        });
    }

    abort(): void {
      this.aborted = true;
      this.onabort?.();
    }
  }

  (globalThis as unknown as { XMLHttpRequest: typeof HarnessXMLHttpRequest }).XMLHttpRequest = HarnessXMLHttpRequest;
  const result = await ctx.client.uploadFile({ file }).promise;
  await record({ suite: ctx.label, kind: "upload", name: "uploadFile", response: result });
  assertEqual(result.filename, "harness.txt", "uploadFile returns filename");
  assertEqual(result.size, bytes.length, "uploadFile returns byte size");
  assert(result.contentType.startsWith("text/plain"), "uploadFile returns a text/plain content type");
  assertEqual(result.checksum.length, 16, "uploadFile returns truncated checksum");
}

async function testWebSocket(ctx: SuiteContext): Promise<void> {
  const socket = ctx.client.createChatSocket();
  const now = new Date().toISOString();
  const user = `harness-${ctx.label}-${runId}`;
  const chatText = "hello over websocket";

  const result = await new Promise<{ joined: string; text: string; typing: boolean }>((resolve, reject) => {
    const state = { joined: "", text: "", typing: false };
    const timeout = setTimeout(() => {
      socket.disconnect();
      reject(new Error(`websocket timed out with state ${JSON.stringify(state)}`));
    }, 10_000);

    function maybeResolve(): void {
      if (state.joined && state.text && state.typing) {
        clearTimeout(timeout);
        socket.disconnect();
        resolve(state);
      }
    }

    socket.onUserJoined((message) => {
      if (message.user === user) state.joined = message.user;
      maybeResolve();
    });
    socket.onChatMessage((message) => {
      if (message.user === user && message.text === chatText) state.text = message.text;
      maybeResolve();
    });
    socket.on("typing", (message) => {
      const isTyping = message.isTyping ?? (message as unknown as { is_typing?: boolean }).is_typing;
      if (message.user === user && isTyping) state.typing = true;
      maybeResolve();
    });
    socket.onError((error) => {
      clearTimeout(timeout);
      reject(new Error(`websocket error: ${String(error)}`));
    });
    socket.onConnect(() => {
      socket.sendJoin({ user, timestamp: now });
      socket.sendChatMessage({ user, text: chatText, timestamp: now });
      socket.sendTyping({ user, isTyping: true });
    });
    socket.connect();
  });

  await record({ suite: ctx.label, kind: "websocket", name: "chat", request: { user, chatText }, response: result });
  assertEqual(result.joined, user, "websocket join event echoes user");
  assertEqual(result.text, chatText, "websocket chat_message echoes text");
  assertEqual(result.typing, true, "websocket typing event maps isTyping");
}

async function runSuite(ctx: SuiteContext): Promise<void> {
  await appendLog("harness.log", `suite=${ctx.label} baseUrl=${ctx.baseUrl}\n`);
  ctx.client.initBridge(ctx.baseUrl);

  await step(`${ctx.label}: command endpoints`, () => testCommands(ctx));
  await step(`${ctx.label}: type fidelity fixtures`, () => testTypeFidelity(ctx));
  await step(`${ctx.label}: static endpoint and raw wire`, () => testRawWireAndStatic(ctx));
  await step(`${ctx.label}: channel SSE endpoint`, () => testChannel(ctx));
  await step(`${ctx.label}: upload multipart endpoint`, () => testUpload(ctx));
  await step(`${ctx.label}: websocket endpoint`, () => testWebSocket(ctx));
}

async function main(): Promise<void> {
  await appendLog("harness.log", `run=${runId} mode=${mode} pythonBaseUrl=${PYTHON_BASE_URL} rustBaseUrl=${RUST_BASE_URL}\n`);

  if (mode === "rust" || mode === "all") await generateRustClient();
  if (mode === "cross" || mode === "all") await generatePythonClient();

  const pythonClient = mode === "python" || mode === "cross" || mode === "all" ? await loadClient(PYTHON_API_PATH) : undefined;
  const rustClient = mode === "rust" || mode === "all" ? await loadClient(RUST_API_PATH) : undefined;

  const servers: ServerProcess[] = [];
  try {
    if (mode === "python" || mode === "all") servers.push(await startPythonServer(PYTHON_BASE_URL));
    if (mode === "rust" || mode === "cross" || mode === "all") servers.push(await startRustServer(RUST_BASE_URL));

    await Promise.all([
      ...(mode === "python" || mode === "all" ? [waitForHealth(PYTHON_BASE_URL, "python")] : []),
      ...(mode === "rust" || mode === "cross" || mode === "all" ? [waitForHealth(RUST_BASE_URL, "rust")] : []),
    ]);

    if (mode === "python") await runSuite({ label: "python", baseUrl: PYTHON_BASE_URL, client: pythonClient! });
    if (mode === "rust") await runSuite({ label: "rust", baseUrl: RUST_BASE_URL, client: rustClient! });
    if (mode === "cross") await runSuite({ label: "cross-python-client-on-rust", baseUrl: RUST_BASE_URL, client: pythonClient! });
    if (mode === "all") {
      await runSuite({ label: "python", baseUrl: PYTHON_BASE_URL, client: pythonClient! });
      await runSuite({ label: "rust", baseUrl: RUST_BASE_URL, client: rustClient! });
      await runSuite({ label: "cross-python-client-on-rust", baseUrl: RUST_BASE_URL, client: pythonClient! });
    }

    await Bun.write(`${LOG_DIR}/summary.json`, `${JSON.stringify({ mode, pythonBaseUrl: PYTHON_BASE_URL, rustBaseUrl: RUST_BASE_URL, testResults, evidenceCount: evidence.length }, null, 2)}\n`);
    console.log(`ALL PASSED (${testResults.length}/${testResults.length}); logs: ${LOG_DIR}`);
  } finally {
    await Promise.allSettled(servers.reverse().map((server) => stopServer(server)));
  }
}

await main();
