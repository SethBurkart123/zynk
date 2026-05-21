import { appendFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

import type * as GeneratedClient from "../frontend/src/generated/api";

type Client = typeof GeneratedClient;
type Task = GeneratedClient.Task;
type TaskWireCheck = GeneratedClient.TaskWireCheck;
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

interface SseFrame {
  event: string;
  dataRaw: string;
  data: unknown;
}

interface SuiteContext {
  label: string;
  baseUrl: string;
  client: Client;
}

const evidence: EvidenceEntry[] = [];
const testResults: { name: string; status: "PASS" | "FAIL"; detail?: string }[] = [];

await Bun.$`mkdir -p ${LOG_DIR}`;

type HarnessMode = "python" | "rust" | "cross" | "all" | "parity" | "errors" | "debug-flag" | "case-normalization" | "wire-fidelity";

function parseMode(): HarnessMode {
  const arg = Bun.argv.slice(2).find((value) => value.startsWith("--mode="));
  const value = (arg?.split("=", 2)[1] ?? process.env.ZYNK_HARNESS_MODE ?? "python") as HarnessMode;
  if (!["python", "rust", "cross", "all", "parity", "errors", "debug-flag", "case-normalization", "wire-fidelity"].includes(value)) {
    throw new Error(`Unknown harness mode ${value}; expected python, rust, cross, all, parity, errors, debug-flag, case-normalization, or wire-fidelity`);
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

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function assertDeepEqual(actual: unknown, expected: unknown, message: string): void {
  const actualJson = stableJson(actual);
  const expectedJson = stableJson(expected);
  if (actualJson !== expectedJson) {
    throw new Error(`${message}: expected ${expectedJson}, got ${actualJson}`);
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

async function startPythonServer(baseUrl: string, debug = false): Promise<ServerProcess> {
  const proc = Bun.spawn(["uv", "run", "python", "-c", "import main; main.app.port = int(__import__('os').environ['PORT']); main.app.debug = __import__('os').environ.get('ZYNK_DEBUG') == '1'; main.app.run(dev=False)"], {
    cwd: BACKEND_DIR,
    env: {
      ...process.env,
      PORT: portOf(baseUrl),
      ZYNK_DEV: "0",
      ZYNK_DEBUG: debug ? "1" : "0",
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

async function startRustServer(baseUrl: string, debug = false): Promise<ServerProcess> {
  const args = ["cargo", "run", "--release", "--bin", "rust-axum-kitchen-sink", "--", "--port", portOf(baseUrl)];
  if (debug) args.push("--debug");
  const proc = Bun.spawn(args, {
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

async function stopServers(servers: ServerProcess[]): Promise<void> {
  await Promise.allSettled([...servers].reverse().map((server) => stopServer(server)));
  servers.length = 0;
}

async function restartServersForFreshParity(servers: ServerProcess[]): Promise<void> {
  // --mode=all runs stateful WebSocket suites before byte parity. Restart both
  // servers here so parity observes the same fresh process state as --mode=parity
  // instead of comparing cumulative counters such as connected_users.
  await appendLog("harness.log", "[RESTART] restarting Python and Rust servers before cross-server byte parity\n");
  await stopServers(servers);
  servers.push(await startPythonServer(PYTHON_BASE_URL));
  servers.push(await startRustServer(RUST_BASE_URL));
  await Promise.all([
    waitForHealth(PYTHON_BASE_URL, "python-parity-fresh"),
    waitForHealth(RUST_BASE_URL, "rust-parity-fresh"),
  ]);
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

async function httpJson(baseUrl: string, path: string, body: unknown, method = "POST"): Promise<{ status: number; json: unknown; raw: string }> {
  const init: RequestInit = {
    method,
    headers: { "content-type": "application/json" },
  };
  if (method !== "GET" && method !== "HEAD") init.body = JSON.stringify(body);
  const response = await fetch(`${baseUrl}${path}`, init);
  const raw = await response.text();
  return { status: response.status, json: raw ? JSON.parse(raw) : null, raw };
}

async function rawJson(ctx: SuiteContext, path: string, body: unknown): Promise<unknown> {
  const { status, json, raw } = await httpJson(ctx.baseUrl, path, body);
  await record({ suite: ctx.label, kind: "raw-http", name: path, request: body, raw });
  assert(status >= 200 && status < 300, `${path} failed with HTTP ${status}: ${raw}`);
  return json;
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
      resolve(message as WeatherUpdate);
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

function parseSseFrames(raw: string): SseFrame[] {
  return raw
    .split("\n\n")
    .filter((chunk) => chunk.trim().length > 0)
    .map((chunk) => {
      const lines = chunk.split("\n");
      const eventLine = lines.find((line) => line.startsWith("event: "));
      const dataLine = lines.find((line) => line.startsWith("data: "));
      assert(eventLine, `SSE frame missing event line: ${chunk}`);
      assert(dataLine, `SSE frame missing data line: ${chunk}`);
      const dataRaw = dataLine.slice("data: ".length);
      return { event: eventLine.slice("event: ".length), dataRaw, data: JSON.parse(dataRaw) };
    });
}

function normalizeSseFrame(frame: SseFrame): unknown {
  if (frame.event === "message" && typeof frame.data === "object" && frame.data && "timestamp" in frame.data) {
    return { event: frame.event, data: { ...frame.data, timestamp: "<normalized>" } };
  }
  if (frame.event === "close" && typeof frame.data === "object" && frame.data && "channelId" in frame.data) {
    return { event: frame.event, data: { channelId: "<normalized>" } };
  }
  return { event: frame.event, data: frame.data };
}

async function captureSse(baseUrl: string, name: string, request: unknown): Promise<{ status: number; contentType: string; raw: string; frames: SseFrame[] }> {
  const response = await fetch(`${baseUrl}/channel/${name}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  });
  const raw = await response.text();
  return {
    status: response.status,
    contentType: response.headers.get("content-type") ?? "",
    raw,
    frames: parseSseFrames(raw),
  };
}

function stableUploadResult(payload: unknown): unknown {
  const result = (payload as { result?: Record<string, unknown> }).result;
  assert(result && typeof result === "object", `upload response missing object result: ${JSON.stringify(payload)}`);
  return {
    filename: result.filename,
    size: result.size,
    contentType: result.content_type,
    checksum: result.checksum,
  };
}

function stableUploadError(payload: unknown): unknown {
  const error = payload as { code?: string; details?: unknown };
  return { code: error.code, details: error.details };
}

function errorPayload(json: unknown): { code?: string; message?: string; details?: unknown } {
  const payload = json as { code?: string; message?: string; details?: unknown; error?: { code?: string; message?: string; details?: unknown } };
  return payload.error ?? payload;
}

function stableErrorEnvelope(json: unknown): unknown {
  const error = errorPayload(json);
  const envelope: Record<string, unknown> = { code: error.code };
  if (error.details !== undefined) envelope.details = error.details;
  return envelope;
}

function assertErrorShape(json: unknown, label: string): void {
  const error = errorPayload(json);
  assert(error && typeof error === "object", `${label} error response is an object`);
  assert(typeof error.code === "string", `${label} error code is present`);
  assert(typeof error.message === "string", `${label} error message is present`);
  const keys = Object.keys(error).sort();
  assert(keys.every((key) => ["code", "details", "message"].includes(key)), `${label} has only code/message/details keys: ${keys.join(",")}`);
}

async function assertParityHttpError(name: string, path: string, body: unknown, expectedStatus: number, expectedCode: string, method = "POST"): Promise<void> {
  const [python, rust] = await Promise.all([
    httpJson(PYTHON_BASE_URL, path, body, method),
    httpJson(RUST_BASE_URL, path, body, method),
  ]);
  await record({ kind: "error-parity", name: `${name}/python`, request: { path, body }, response: { status: python.status, json: python.json }, raw: python.raw });
  await record({ kind: "error-parity", name: `${name}/rust`, request: { path, body }, response: { status: rust.status, json: rust.json }, raw: rust.raw });
  assertEqual(python.status, expectedStatus, `${name}: Python HTTP status`);
  assertEqual(rust.status, expectedStatus, `${name}: Rust HTTP status`);
  assertErrorShape(python.json, `${name}: Python`);
  assertErrorShape(rust.json, `${name}: Rust`);
  assertEqual(errorPayload(python.json).code, expectedCode, `${name}: Python error code`);
  assertEqual(errorPayload(rust.json).code, expectedCode, `${name}: Rust error code`);
  assertDeepEqual(stableErrorEnvelope(python.json), stableErrorEnvelope(rust.json), `${name}: stable error envelope fields match`);
}

async function assertWebSocketClose(baseUrl: string, path: string, firstMessage?: unknown): Promise<{ code: number; reason: string }> {
  const wsUrl = `${baseUrl.replace(/^http/, "ws")}${path}`;
  return await new Promise((resolve, reject) => {
    const socket = new WebSocket(wsUrl);
    const timeout = setTimeout(() => {
      socket.close();
      reject(new Error(`websocket close timed out for ${wsUrl}`));
    }, 10_000);
    socket.onopen = () => {
      if (firstMessage !== undefined) socket.send(JSON.stringify(firstMessage));
    };
    socket.onerror = () => {
      // Some runtimes surface an immediate close after upgrade as an error before onclose.
    };
    socket.onclose = (event) => {
      clearTimeout(timeout);
      resolve({ code: event.code, reason: event.reason });
    };
  });
}

async function testDebugFlagParity(): Promise<void> {
  const defaultServers = [await startPythonServer(PYTHON_BASE_URL, false), await startRustServer(RUST_BASE_URL, false)];
  try {
    await Promise.all([waitForHealth(PYTHON_BASE_URL, "python-debug-default"), waitForHealth(RUST_BASE_URL, "rust-debug-default")]);
    const [pythonDefault, rustDefault] = await Promise.all([
      httpJson(PYTHON_BASE_URL, "/command/get_user", { user_id: -500 }),
      httpJson(RUST_BASE_URL, "/command/get_user", { user_id: -500 }),
    ]);
    await record({ kind: "debug-flag", name: "default/python", response: { status: pythonDefault.status, json: pythonDefault.json }, raw: pythonDefault.raw });
    await record({ kind: "debug-flag", name: "default/rust", response: { status: rustDefault.status, json: rustDefault.json }, raw: rustDefault.raw });
    assertEqual(errorPayload(pythonDefault.json).message, "An internal error occurred", "Python debug=False hides exception text");
    assertEqual(errorPayload(rustDefault.json).message, "An internal error occurred", "Rust debug=False hides exception text");
  } finally {
    await Promise.allSettled(defaultServers.reverse().map((server) => stopServer(server)));
  }

  const debugServers = [await startPythonServer(PYTHON_BASE_URL, true), await startRustServer(RUST_BASE_URL, true)];
  try {
    await Promise.all([waitForHealth(PYTHON_BASE_URL, "python-debug"), waitForHealth(RUST_BASE_URL, "rust-debug")]);
    const [pythonDebug, rustDebug] = await Promise.all([
      httpJson(PYTHON_BASE_URL, "/command/get_user", { user_id: -500 }),
      httpJson(RUST_BASE_URL, "/command/get_user", { user_id: -500 }),
    ]);
    await record({ kind: "debug-flag", name: "enabled/python", response: { status: pythonDebug.status, json: pythonDebug.json }, raw: pythonDebug.raw });
    await record({ kind: "debug-flag", name: "enabled/rust", response: { status: rustDebug.status, json: rustDebug.json }, raw: rustDebug.raw });
    assertEqual(errorPayload(pythonDebug.json).message, "super secret stack info", "Python debug=True exposes str(e)");
    assertEqual(errorPayload(rustDebug.json).message, "super secret stack info", "Rust debug=True exposes str(e)");
  } finally {
    await Promise.allSettled(debugServers.reverse().map((server) => stopServer(server)));
  }
}

async function testCaseNormalizationParity(): Promise<void> {
  const cases = [
    { name: "get_user snake_case", path: "/command/get_user", body: { user_id: 1 } },
    { name: "get_user camelCase", path: "/command/get_user", body: { userId: 1 } },
    { name: "create_task snake_case", path: "/command/create_task", body: { title: "snake case task", description: null, priority: "urgent", due_date: null, label_ids: [1, 2] } },
    { name: "create_task camelCase", path: "/command/create_task", body: { title: "camel case task", description: null, priority: "urgent", dueDate: null, labelIds: [1, 2] } },
  ];
  for (const item of cases) {
    const [python, rust] = await Promise.all([
      httpJson(PYTHON_BASE_URL, item.path, item.body),
      httpJson(RUST_BASE_URL, item.path, item.body),
    ]);
    await record({ kind: "case-normalization", name: `${item.name}/python`, request: item.body, response: { status: python.status, json: python.json }, raw: python.raw });
    await record({ kind: "case-normalization", name: `${item.name}/rust`, request: item.body, response: { status: rust.status, json: rust.json }, raw: rust.raw });
    assertEqual(python.status, 200, `${item.name}: Python accepts request keys`);
    assertEqual(rust.status, 200, `${item.name}: Rust accepts request keys`);
    assert((python.json as { result?: unknown }).result !== undefined, `${item.name}: Python returns result envelope`);
    assert((rust.json as { result?: unknown }).result !== undefined, `${item.name}: Rust returns result envelope`);
  }
}

function taskWireCheckFixture(): TaskWireCheck {
  return {
    kind: "task_wire_check",
    priority: "urgent",
    status: "in_progress",
    numericStatus: 2,
  };
}

function assertTaskWireCheck(payload: TaskWireCheck, message: string): void {
  const kind: "task_wire_check" = payload.kind;
  const priority: "low" | "medium" | "high" | "urgent" = payload.priority;
  const status: "todo" | "in_progress" | "done" | "cancelled" = payload.status;
  const numericStatus: 1 | 2 | 3 = payload.numericStatus;

  assertEqual(kind, "task_wire_check", `${message}: literal kind preserved`);
  assertEqual(priority, "urgent", `${message}: enum priority preserved`);
  assertEqual(status, "in_progress", `${message}: enum status preserved`);
  assertEqual(numericStatus, 2, `${message}: numeric literal union preserved`);
  assertEqual(typeof numericStatus, "number", `${message}: numeric enum/literal stays numeric on the wire`);
}

function unwrapResult<T>(payload: unknown): T {
  const result = (payload as { result?: T }).result;
  assert(result !== undefined, `response missing result envelope: ${JSON.stringify(payload)}`);
  return result;
}

async function testWireFidelityWithClient(ctx: SuiteContext): Promise<void> {
  ctx.client.initBridge(ctx.baseUrl);
  const fixture = taskWireCheckFixture();
  const literal: "task_wire_check" = fixture.kind;
  const priority: "low" | "medium" | "high" | "urgent" = fixture.priority;
  const status: "todo" | "in_progress" | "done" | "cancelled" = fixture.status;
  const numericStatus: 1 | 2 | 3 = fixture.numericStatus;
  assertEqual(literal, "task_wire_check", `${ctx.label}: compile-time literal fixture value`);
  assertEqual(priority, "urgent", `${ctx.label}: compile-time priority union fixture value`);
  assertEqual(status, "in_progress", `${ctx.label}: compile-time status union fixture value`);
  assertEqual(numericStatus, 2, `${ctx.label}: compile-time numeric literal union fixture value`);

  const echoed = await ctx.client.echoTaskWireCheck({ payload: fixture });
  await record({ suite: ctx.label, kind: "wire-fidelity-client", name: "echoTaskWireCheck", request: fixture, response: echoed });
  assertTaskWireCheck(echoed, `${ctx.label}: client echoTaskWireCheck`);

  const canonical = await ctx.client.getTaskWireCheck();
  await record({ suite: ctx.label, kind: "wire-fidelity-client", name: "getTaskWireCheck", response: canonical });
  assertTaskWireCheck(canonical, `${ctx.label}: client getTaskWireCheck`);
}

async function testRawWireFidelityParity(): Promise<void> {
  const wireFixture = {
    kind: "task_wire_check",
    priority: "urgent",
    status: "in_progress",
    numeric_status: 2,
  } satisfies {
    kind: "task_wire_check";
    priority: "urgent";
    status: "in_progress";
    numeric_status: 2;
  };
  const request = { payload: wireFixture };

  const [pythonEcho, rustEcho] = await Promise.all([
    httpJson(PYTHON_BASE_URL, "/command/echo_task_wire_check", request),
    httpJson(RUST_BASE_URL, "/command/echo_task_wire_check", request),
  ]);
  await record({ kind: "wire-fidelity-raw", name: "echo_task_wire_check/python", request, response: { status: pythonEcho.status, json: pythonEcho.json }, raw: pythonEcho.raw });
  await record({ kind: "wire-fidelity-raw", name: "echo_task_wire_check/rust", request, response: { status: rustEcho.status, json: rustEcho.json }, raw: rustEcho.raw });
  assertEqual(pythonEcho.status, 200, "Python echo_task_wire_check returns 200");
  assertEqual(rustEcho.status, 200, "Rust echo_task_wire_check returns 200");
  assertDeepEqual(unwrapResult(pythonEcho.json), wireFixture, "Python echo_task_wire_check raw result preserves literal/enum wire values");
  assertDeepEqual(unwrapResult(rustEcho.json), wireFixture, "Rust echo_task_wire_check raw result preserves literal/enum wire values");
  assertDeepEqual(unwrapResult(pythonEcho.json), unwrapResult(rustEcho.json), "Python and Rust echo_task_wire_check raw results match");
  assert(pythonEcho.raw.includes('"kind":"task_wire_check"'), "Python raw echo response contains exact literal value");
  assert(rustEcho.raw.includes('"kind":"task_wire_check"'), "Rust raw echo response contains exact literal value");
  assert(pythonEcho.raw.includes('"numeric_status":2'), "Python raw echo response keeps numeric status numeric");
  assert(rustEcho.raw.includes('"numeric_status":2'), "Rust raw echo response keeps numeric status numeric");

  const [pythonCanonical, rustCanonical] = await Promise.all([
    httpJson(PYTHON_BASE_URL, "/command/get_task_wire_check", {}),
    httpJson(RUST_BASE_URL, "/command/get_task_wire_check", {}),
  ]);
  await record({ kind: "wire-fidelity-raw", name: "get_task_wire_check/python", response: { status: pythonCanonical.status, json: pythonCanonical.json }, raw: pythonCanonical.raw });
  await record({ kind: "wire-fidelity-raw", name: "get_task_wire_check/rust", response: { status: rustCanonical.status, json: rustCanonical.json }, raw: rustCanonical.raw });
  assertEqual(pythonCanonical.status, 200, "Python get_task_wire_check returns 200");
  assertEqual(rustCanonical.status, 200, "Rust get_task_wire_check returns 200");
  assertDeepEqual(unwrapResult(pythonCanonical.json), wireFixture, "Python emitted canonical literal/enum payload matches expected wire fixture");
  assertDeepEqual(unwrapResult(rustCanonical.json), wireFixture, "Rust emitted canonical literal/enum payload matches expected wire fixture");
  assertDeepEqual(unwrapResult(pythonCanonical.json), unwrapResult(rustCanonical.json), "Python and Rust get_task_wire_check raw results match");
}

async function testSseErrorTermination(): Promise<void> {
  const request = { city: "__error__", interval_seconds: 1 };
  const [python, rust] = await Promise.all([
    captureSse(PYTHON_BASE_URL, "stream_weather", request),
    captureSse(RUST_BASE_URL, "stream_weather", request),
  ]);
  await record({ kind: "sse-error-termination", name: "stream_weather/python", request, raw: python.raw, response: python.frames });
  await record({ kind: "sse-error-termination", name: "stream_weather/rust", request, raw: rust.raw, response: rust.frames });
  for (const [label, stream] of [["Python", python], ["Rust", rust]] as const) {
    assertEqual(stream.status, 200, `${label} SSE error stream starts with 200`);
    assertDeepEqual(stream.frames.map((frame) => frame.event), ["message", "message", "error"], `${label} SSE terminates immediately after error frame`);
    const errorFrame = stream.frames[2];
    assertDeepEqual(errorFrame.data, { error: "boom" }, `${label} SSE error payload`);
  }
  assertDeepEqual(python.frames.map(normalizeSseFrame), rust.frames.map(normalizeSseFrame), "SSE error termination frame sequence matches across servers");
}

async function testErrorEnvelopeParity(): Promise<void> {
  await assertParityHttpError("VALIDATION_ERROR", "/command/get_user", {}, 400, "VALIDATION_ERROR");
  await assertParityHttpError("COMMAND_NOT_FOUND", "/command/does_not_exist", {}, 404, "COMMAND_NOT_FOUND");
  await assertParityHttpError("EXECUTION_ERROR", "/command/get_user", { user_id: 999999 }, 500, "EXECUTION_ERROR");
  await assertParityHttpError("INTERNAL_ERROR", "/command/get_user", { user_id: -500 }, 500, "INTERNAL_ERROR");
  await assertParityHttpError("HANDLER_NOT_FOUND", "/command/does_not_exist", {}, 404, "COMMAND_NOT_FOUND");
  await assertParityHttpError("UPLOAD_HANDLER_NOT_FOUND", "/upload/does_not_exist", {}, 404, "UPLOAD_HANDLER_NOT_FOUND");
  await assertParityHttpError("STATIC_HANDLER_NOT_FOUND", "/static/does_not_exist", {}, 404, "STATIC_HANDLER_NOT_FOUND", "GET");

  const badFile = new File([new Uint8Array(5 * 1024 * 1024 + 1)], "too-large.png", { type: "image/png" });
  const [pythonUploadError, rustUploadError] = await Promise.all([
    postUpload(PYTHON_BASE_URL, "upload_image", badFile, { generate_thumbnail: false }),
    postUpload(RUST_BASE_URL, "upload_image", badFile, { generate_thumbnail: false }),
  ]);
  await record({ kind: "error-parity", name: "UPLOAD_VALIDATION_ERROR/python", response: { status: pythonUploadError.status, json: pythonUploadError.json }, raw: pythonUploadError.raw });
  await record({ kind: "error-parity", name: "UPLOAD_VALIDATION_ERROR/rust", response: { status: rustUploadError.status, json: rustUploadError.json }, raw: rustUploadError.raw });
  assertEqual(pythonUploadError.status, 400, "Python UPLOAD_VALIDATION_ERROR status");
  assertEqual(rustUploadError.status, 400, "Rust UPLOAD_VALIDATION_ERROR status");
  assertDeepEqual(stableErrorEnvelope(pythonUploadError.json), stableErrorEnvelope(rustUploadError.json), "UPLOAD_VALIDATION_ERROR envelopes match");

  const [pythonChannel, rustChannel] = await Promise.all([
    captureSse(PYTHON_BASE_URL, "stream_weather", { city: "__error__", interval_seconds: 1 }),
    captureSse(RUST_BASE_URL, "stream_weather", { city: "__error__", interval_seconds: 1 }),
  ]);
  const pythonErrorFrame = pythonChannel.frames.find((frame) => frame.event === "error");
  const rustErrorFrame = rustChannel.frames.find((frame) => frame.event === "error");
  await record({ kind: "error-parity", name: "CHANNEL_ERROR/python", raw: pythonChannel.raw, response: pythonChannel.frames });
  await record({ kind: "error-parity", name: "CHANNEL_ERROR/rust", raw: rustChannel.raw, response: rustChannel.frames });
  assert(pythonErrorFrame, "Python CHANNEL_ERROR emits error SSE frame");
  assert(rustErrorFrame, "Rust CHANNEL_ERROR emits error SSE frame");
  assertDeepEqual(pythonErrorFrame.data, rustErrorFrame.data, "CHANNEL_ERROR SSE error payloads match");

  const [pythonWsMissing, rustWsMissing] = await Promise.all([
    assertWebSocketClose(PYTHON_BASE_URL, "/ws/does_not_exist"),
    assertWebSocketClose(RUST_BASE_URL, "/ws/does_not_exist"),
  ]);
  assertEqual(pythonWsMissing.code, 4004, "Python WS handler-not-found close code");
  assertEqual(rustWsMissing.code, 4004, "Rust WS handler-not-found close code");

  const panicFrame = { event: "join", data: { user: "__panic__", timestamp: "2024-01-01T00:00:00.000Z" } };
  const [pythonWsPanic, rustWsPanic] = await Promise.all([
    assertWebSocketClose(PYTHON_BASE_URL, "/ws/chat", panicFrame),
    assertWebSocketClose(RUST_BASE_URL, "/ws/chat", panicFrame),
  ]);
  assertEqual(pythonWsPanic.code, 1011, "Python WS handler-panic close code");
  assertEqual(rustWsPanic.code, 1011, "Rust WS handler-panic close code");
  await record({ kind: "error-parity", name: "WEBSOCKET_ERROR", response: { pythonMissing: pythonWsMissing, rustMissing: rustWsMissing, pythonPanic: pythonWsPanic, rustPanic: rustWsPanic } });
}

async function postUpload(baseUrl: string, path: string, file: File, args: unknown): Promise<{ status: number; raw: string; json: unknown }> {
  const body = new FormData();
  body.append("files", file);
  body.append("_args", JSON.stringify(args));
  const response = await fetch(`${baseUrl}/upload/${path}`, { method: "POST", body });
  const raw = await response.text();
  return { status: response.status, raw, json: JSON.parse(raw) };
}

async function captureWebSocket(baseUrl: string): Promise<{ sent: unknown[]; received: unknown[] }> {
  const wsUrl = `${baseUrl.replace(/^http/, "ws")}/ws/chat`;
  const user = `parity-${runId}`;
  const now = "2024-01-01T00:00:00.000Z";
  const sent = [
    { event: "join", data: { user, timestamp: now } },
    { event: "chat_message", data: { user, text: "hello parity", timestamp: now } },
    { event: "typing", data: { user, is_typing: true } },
  ];
  const received = await new Promise<unknown[]>((resolve, reject) => {
    const frames: unknown[] = [];
    const socket = new WebSocket(wsUrl);
    let sentFrames = false;
    const timeout = setTimeout(() => {
      socket.close();
      reject(new Error(`websocket parity timed out for ${wsUrl}: ${JSON.stringify(frames)}`));
    }, 10_000);

    socket.onopen = () => {
      if (sentFrames) return;
      sentFrames = true;
      for (const frame of sent) socket.send(JSON.stringify(frame));
    };
    socket.onerror = () => {
      clearTimeout(timeout);
      reject(new Error(`websocket parity failed for ${wsUrl}`));
    };
    socket.onmessage = (event) => {
      frames.push(JSON.parse(event.data));
      const events = new Set(frames.map((frame) => (frame as { event?: string }).event));
      if (events.has("user_joined") && events.has("status") && events.has("chat_message") && events.has("typing")) {
        clearTimeout(timeout);
        socket.close();
        resolve(frames);
      }
    };
  });
  return { sent, received };
}

function normalizeWsFrames(frames: unknown[]): unknown[] {
  return frames
    .map((frame) => {
      const message = frame as { event: string; data: Record<string, unknown> };
      const data = { ...message.data };
      if ("timestamp" in data) data.timestamp = "<normalized>";
      if ("uptime_seconds" in data) data.uptime_seconds = "<normalized>";
      return { event: message.event, data };
    })
    .sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)));
}

async function testCrossServerParity(): Promise<void> {
  const weatherRequest = { city: "__parity__", interval_seconds: 60 };
  const [pythonSse, rustSse] = await Promise.all([
    captureSse(PYTHON_BASE_URL, "stream_weather", weatherRequest),
    captureSse(RUST_BASE_URL, "stream_weather", weatherRequest),
  ]);
  await Bun.write(`${LOG_DIR}/parity-sse-python.txt`, pythonSse.raw);
  await Bun.write(`${LOG_DIR}/parity-sse-rust.txt`, rustSse.raw);
  await record({ kind: "parity-sse", name: "stream_weather/python", request: weatherRequest, raw: pythonSse.raw, response: { status: pythonSse.status, contentType: pythonSse.contentType } });
  await record({ kind: "parity-sse", name: "stream_weather/rust", request: weatherRequest, raw: rustSse.raw, response: { status: rustSse.status, contentType: rustSse.contentType } });
  assertEqual(pythonSse.status, 200, "Python stream_weather returns 200");
  assertEqual(rustSse.status, 200, "Rust stream_weather returns 200");
  assert(pythonSse.contentType.startsWith("text/event-stream"), `Python SSE content-type: ${pythonSse.contentType}`);
  assert(rustSse.contentType.startsWith("text/event-stream"), `Rust SSE content-type: ${rustSse.contentType}`);
  assertDeepEqual(pythonSse.frames.map(normalizeSseFrame), rustSse.frames.map(normalizeSseFrame), "stream_weather SSE frames match after channelId/timestamp normalization");
  assert(pythonSse.raw.includes('event: close\ndata: {"channelId": '), "Python close frame uses camelCase channelId");
  assert(rustSse.raw.includes('event: close\ndata: {"channelId": '), "Rust close frame uses camelCase channelId");

  const idleRequest = { city: "__idle_keepalive__", interval_seconds: 60 };
  const [pythonIdle, rustIdle] = await Promise.all([
    captureSse(PYTHON_BASE_URL, "stream_weather", idleRequest),
    captureSse(RUST_BASE_URL, "stream_weather", idleRequest),
  ]);
  await Bun.write(`${LOG_DIR}/parity-sse-idle-python.txt`, pythonIdle.raw);
  await Bun.write(`${LOG_DIR}/parity-sse-idle-rust.txt`, rustIdle.raw);
  await record({ kind: "parity-sse-idle", name: "stream_weather/python", request: idleRequest, raw: pythonIdle.raw });
  await record({ kind: "parity-sse-idle", name: "stream_weather/rust", request: idleRequest, raw: rustIdle.raw });
  for (const [label, idle] of [["Python", pythonIdle], ["Rust", rustIdle]] as const) {
    const keepaliveCount = idle.frames.filter((frame) => frame.event === "keepalive" && frame.dataRaw === "{}").length;
    assert(keepaliveCount >= 1, `${label} emitted at least one 35s idle keepalive`);
    const stringFrame = idle.frames.find((frame) => frame.event === "message");
    assert(stringFrame, `${label} emitted string message after idle keepalive`);
    assertEqual(stringFrame.dataRaw, '"hello"', `${label} JSON-encodes string SSE data`);
  }
  assertDeepEqual(pythonIdle.frames.map(normalizeSseFrame), rustIdle.frames.map(normalizeSseFrame), "idle keepalive/string SSE frames match after channelId normalization");

  const [pythonWs, rustWs] = await Promise.all([captureWebSocket(PYTHON_BASE_URL), captureWebSocket(RUST_BASE_URL)]);
  await Bun.write(`${LOG_DIR}/parity-ws-python.json`, `${JSON.stringify(pythonWs, null, 2)}\n`);
  await Bun.write(`${LOG_DIR}/parity-ws-rust.json`, `${JSON.stringify(rustWs, null, 2)}\n`);
  await record({ kind: "parity-websocket", name: "chat/python", request: pythonWs.sent, response: pythonWs.received });
  await record({ kind: "parity-websocket", name: "chat/rust", request: rustWs.sent, response: rustWs.received });
  assertDeepEqual(normalizeWsFrames(pythonWs.received), normalizeWsFrames(rustWs.received), "WebSocket received frame structures match");
  assert(pythonWs.received.every((frame) => typeof (frame as { event?: unknown }).event === "string" && !(frame as { event: string }).event.includes("-")), "Python WS event names are stable strings");
  assert(rustWs.received.every((frame) => typeof (frame as { event?: unknown }).event === "string" && !(frame as { event: string }).event.includes("-")), "Rust WS event names are stable strings");

  const uploadBytes = new TextEncoder().encode("hello parity upload\n");
  const uploadFile = new File([uploadBytes], "parity.txt", { type: "text/plain" });
  const [pythonUpload, rustUpload] = await Promise.all([
    postUpload(PYTHON_BASE_URL, "upload_file", uploadFile, {}),
    postUpload(RUST_BASE_URL, "upload_file", uploadFile, {}),
  ]);
  await record({ kind: "parity-upload", name: "upload_file/python", request: { filename: uploadFile.name, size: uploadFile.size, type: uploadFile.type }, raw: pythonUpload.raw });
  await record({ kind: "parity-upload", name: "upload_file/rust", request: { filename: uploadFile.name, size: uploadFile.size, type: uploadFile.type }, raw: rustUpload.raw });
  assertEqual(pythonUpload.status, 200, "Python upload_file returns 200");
  assertEqual(rustUpload.status, 200, "Rust upload_file returns 200");
  assertDeepEqual(stableUploadResult(pythonUpload.json), stableUploadResult(rustUpload.json), "upload_file response structures match after generated fields normalization");

  const badFile = new File([new Uint8Array(5 * 1024 * 1024 + 1)], "too-large.png", { type: "image/png" });
  const [pythonUploadError, rustUploadError] = await Promise.all([
    postUpload(PYTHON_BASE_URL, "upload_image", badFile, { generate_thumbnail: false }),
    postUpload(RUST_BASE_URL, "upload_image", badFile, { generate_thumbnail: false }),
  ]);
  await record({ kind: "parity-upload-error", name: "upload_image/python", raw: pythonUploadError.raw, response: pythonUploadError.json });
  await record({ kind: "parity-upload-error", name: "upload_image/rust", raw: rustUploadError.raw, response: rustUploadError.json });
  assertEqual(pythonUploadError.status, 400, "Python oversized upload_image returns 400");
  assertEqual(rustUploadError.status, 400, "Rust oversized upload_image returns 400");
  assertDeepEqual(stableUploadError(pythonUploadError.json), stableUploadError(rustUploadError.json), "oversized upload error code/details match");
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
  if (mode === "cross" || mode === "all" || mode === "parity" || mode === "errors" || mode === "case-normalization" || mode === "wire-fidelity") await generatePythonClient();

  const pythonClient = mode === "python" || mode === "cross" || mode === "all" || mode === "wire-fidelity" ? await loadClient(PYTHON_API_PATH) : undefined;
  const rustClient = mode === "rust" || mode === "all" ? await loadClient(RUST_API_PATH) : undefined;

  const servers: ServerProcess[] = [];
  try {
    if (mode === "debug-flag") {
      await step("cross-server debug flag hides and exposes internal errors", testDebugFlagParity);
    } else {
      const needsPython = mode === "python" || mode === "all" || mode === "parity" || mode === "errors" || mode === "case-normalization" || mode === "wire-fidelity";
      const needsRust = mode === "rust" || mode === "cross" || mode === "all" || mode === "parity" || mode === "errors" || mode === "case-normalization" || mode === "wire-fidelity";
      if (needsPython) servers.push(await startPythonServer(PYTHON_BASE_URL));
      if (needsRust) servers.push(await startRustServer(RUST_BASE_URL));

      await Promise.all([
        ...(needsPython ? [waitForHealth(PYTHON_BASE_URL, "python")] : []),
        ...(needsRust ? [waitForHealth(RUST_BASE_URL, "rust")] : []),
      ]);

      if (mode === "python") await runSuite({ label: "python", baseUrl: PYTHON_BASE_URL, client: pythonClient! });
      if (mode === "rust") await runSuite({ label: "rust", baseUrl: RUST_BASE_URL, client: rustClient! });
      if (mode === "cross") await runSuite({ label: "cross-python-client-on-rust", baseUrl: RUST_BASE_URL, client: pythonClient! });
      if (mode === "parity") await step("cross-server byte parity: channel/ws/upload", testCrossServerParity);
      if (mode === "errors") {
        await step("cross-server error envelope and WS close code parity", testErrorEnvelopeParity);
        await step("cross-server SSE error termination", testSseErrorTermination);
      }
      if (mode === "case-normalization") await step("cross-server snake/camel request key acceptance", testCaseNormalizationParity);
      if (mode === "wire-fidelity") {
        await step("wire fidelity: Python server with Python-generated client", () => testWireFidelityWithClient({ label: "wire-python", baseUrl: PYTHON_BASE_URL, client: pythonClient! }));
        await step("wire fidelity: Rust server with Python-generated client", () => testWireFidelityWithClient({ label: "wire-rust-python-client", baseUrl: RUST_BASE_URL, client: pythonClient! }));
        await step("wire fidelity: raw Python/Rust literal and enum parity", testRawWireFidelityParity);
      }
      if (mode === "all") {
        await runSuite({ label: "python", baseUrl: PYTHON_BASE_URL, client: pythonClient! });
        await runSuite({ label: "rust", baseUrl: RUST_BASE_URL, client: rustClient! });
        await runSuite({ label: "cross-python-client-on-rust", baseUrl: RUST_BASE_URL, client: pythonClient! });
        await restartServersForFreshParity(servers);
        await step("cross-server byte parity: channel/ws/upload", testCrossServerParity);
        await step("cross-server error envelope parity", testErrorEnvelopeParity);
        await step("cross-server SSE error termination", testSseErrorTermination);
        await step("cross-server snake/camel request key acceptance", testCaseNormalizationParity);
        await step("wire fidelity: Python server with Python-generated client", () => testWireFidelityWithClient({ label: "wire-python", baseUrl: PYTHON_BASE_URL, client: pythonClient! }));
        await step("wire fidelity: Rust server with Python-generated client", () => testWireFidelityWithClient({ label: "wire-rust-python-client", baseUrl: RUST_BASE_URL, client: pythonClient! }));
        await step("wire fidelity: raw Python/Rust literal and enum parity", testRawWireFidelityParity);
      }
    }

    await Bun.write(`${LOG_DIR}/summary.json`, `${JSON.stringify({ mode, pythonBaseUrl: PYTHON_BASE_URL, rustBaseUrl: RUST_BASE_URL, testResults, evidenceCount: evidence.length }, null, 2)}\n`);
    console.log(`ALL PASSED (${testResults.length}/${testResults.length}); logs: ${LOG_DIR}`);
  } finally {
    await stopServers(servers);
  }
}

await main();
