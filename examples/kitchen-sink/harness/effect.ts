import { appendFile } from "node:fs/promises";

import { Chunk, Effect, Stream } from "effect";

import {
  connectChat,
  createTask,
  downloadSampleUrl,
  getForecast,
  getTaskStats,
  getUser,
  initZynk,
  listCities,
  listTasks,
  streamWeather,
  updateUser,
  uploadFile,
  type Task,
  type WeatherUpdate,
} from "../frontend/src/generated-effect/api";

const BASE_URL = process.env.ZYNK_BASE_URL ?? "http://127.0.0.1:8100";
const PORT = new URL(BASE_URL).port || "8100";
const ROOT = new URL("../../..", import.meta.url).pathname.replace(/\/$/, "");
const BACKEND_DIR = `${ROOT}/examples/kitchen-sink/backend`;
const runId = new Date().toISOString().replace(/[:.]/g, "-");
const LOG_DIR = `${import.meta.dir}/logs/${runId}-effect`;
const ZynkLayer = initZynk({ baseUrl: BASE_URL });

interface ServerProcess {
  proc: ReturnType<typeof Bun.spawn>;
  stdout: Promise<void>;
  stderr: Promise<void>;
}

interface EvidenceEntry {
  kind: string;
  name: string;
  request?: unknown;
  response?: unknown;
  raw?: string;
}

const evidence: EvidenceEntry[] = [];
const testResults: { name: string; status: "PASS" | "FAIL"; detail?: string }[] = [];

await Bun.$`mkdir -p ${LOG_DIR}`;

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function assertEqual<T>(actual: T, expected: T, message: string): void {
  if (actual !== expected) {
    throw new Error(`${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function runEffect<A>(program: Effect.Effect<A, unknown, unknown>): Promise<A> {
  return Effect.runPromise(program.pipe(Effect.provide(ZynkLayer)) as Effect.Effect<A>);
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

async function startServer(): Promise<ServerProcess> {
  const proc = Bun.spawn(["uv", "run", "python", "-c", "import main; main.app.port = int(__import__('os').environ['PORT']); main.app.run(dev=False)"], {
    cwd: BACKEND_DIR,
    env: {
      ...process.env,
      PORT,
      ZYNK_DEV: "0",
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: `${ROOT}/bindings/python${process.env.PYTHONPATH ? `:${process.env.PYTHONPATH}` : ""}`,
    },
    stdout: "pipe",
    stderr: "pipe",
  });

  const stdout = pump(proc.stdout, "server.stdout.log");
  const stderr = pump(proc.stderr, "server.stderr.log");
  return { proc, stdout, stderr };
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

async function waitForHealth(): Promise<void> {
  const started = Date.now();
  let lastError = "server did not respond";
  while (Date.now() - started < 20_000) {
    try {
      const response = await fetch(`${BASE_URL}/`);
      if (response.ok) {
        const body = await response.json();
        await record({ kind: "health", name: "GET /", response: body });
        return;
      }
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await Bun.sleep(250);
  }
  throw new Error(`Backend healthcheck failed: ${lastError}`);
}

async function rawJson(path: string, body: unknown): Promise<unknown> {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  await record({ kind: "raw-http", name: path, request: body, raw: text });
  assert(response.ok, `${path} failed with HTTP ${response.status}: ${text}`);
  return JSON.parse(text);
}

async function testCommands(): Promise<void> {
  const user = await runEffect(getUser({ userId: 1 }));
  await record({ kind: "effect-command", name: "getUser", request: { userId: 1 }, response: user });
  assertEqual(user.name, "Alice", "Effect getUser decodes fixture user");
  assertEqual(user.email, "alice@example.com", "Effect getUser decodes email");
  assertEqual(user.isActive, true, "Effect getUser decodes isActive via Schema.fromKey");

  const cities = await runEffect(listCities());
  await record({ kind: "effect-command", name: "listCities", response: cities });
  assert(cities.includes("Tokyo"), "Effect listCities includes Tokyo fixture");

  const forecast = await runEffect(getForecast({ city: "Tokyo", days: 2 }));
  await record({ kind: "effect-command", name: "getForecast", request: { city: "Tokyo", days: 2 }, response: forecast });
  assertEqual(forecast.length, 2, "Effect getForecast honors days arg");
  assert(forecast.every((day) => typeof day.precipitationChance === "number"), "Effect Schema decodes precipitationChance");

  const stats = await runEffect(getTaskStats());
  await record({ kind: "effect-command", name: "getTaskStats", response: stats });
  assert(stats.total >= 4, "Effect getTaskStats sees seeded tasks");
  assert(stats.byPriority.high >= 2, "Effect Schema decodes byPriority record");

  const updated = await runEffect(updateUser({ userId: 2, name: null, email: null, isActive: false }));
  await record({ kind: "effect-command", name: "updateUser", request: { userId: 2, name: null, email: null, isActive: false }, response: updated });
  assertEqual(updated.name, "Bob", "Effect nullable name does not overwrite when null");
  assertEqual(updated.email, "bob@example.com", "Effect nullable email does not overwrite when null");
  assertEqual(updated.isActive, false, "Effect nullable boolean updates when false");
}

async function testTypeFidelity(): Promise<void> {
  const created = await runEffect(createTask({
    title: "Effect harness urgent task",
    description: null,
    priority: "urgent",
    dueDate: null,
    labelIds: [1, 2],
  }));
  await record({ kind: "effect-type-fidelity", name: "createTask", response: created });

  const priority: "urgent" = created.priority as "urgent";
  const status: "todo" = created.status as "todo";
  assertEqual(priority, "urgent", "Effect enum-with-values priority survives runtime");
  assertEqual(status, "todo", "Effect enum-with-values status survives runtime");
  assert(created.description === null || created.description === undefined, "Effect optional+nullable description decodes as absent-or-null");
  assert(created.dueDate === null || created.dueDate === undefined, "Effect optional+nullable dueDate decodes as absent-or-null");
  assert(created.labels?.some((label) => label.name === "Bug"), "Effect nested label model maps camelCase fields");

  const filtered = await runEffect(listTasks({ status: "todo", priority: "urgent", labelId: null }));
  await record({ kind: "effect-type-fidelity", name: "listTasks", request: { status: "todo", priority: "urgent", labelId: null }, response: filtered });
  assert(filtered.some((task: Task) => task.id === created.id), "Effect nullable filter param accepted as null");
}

async function testRawWireAndStatic(): Promise<void> {
  const raw = await rawJson("/command/get_user", { user_id: 1 });
  assertEqual((raw as { result: { is_active: boolean } }).result.is_active, true, "raw command response preserves snake_case wire");

  const url = await runEffect(downloadSampleUrl({ filename: "sample.txt" }));
  const response = await fetch(url);
  const text = await response.text();
  await record({ kind: "effect-static", name: "downloadSampleUrl", request: { url }, response: { status: response.status, text } });
  assertEqual(response.status, 200, "Effect static URL returns 200");
  assertEqual(text, "Zynk kitchen sink static sample\n", "Effect static URL fixture content matches");
}

async function testChannel(): Promise<void> {
  const update = await runEffect(
    streamWeather({ city: "Tokyo", intervalSeconds: 1 }).pipe(
      Stream.take(1),
      Stream.runCollect,
      Effect.map((chunk) => Chunk.unsafeHead(chunk)),
    ),
  );
  await record({ kind: "effect-channel", name: "streamWeather", request: { city: "Tokyo", intervalSeconds: 1 }, response: update });
  assertEqual((update as WeatherUpdate).city, "Tokyo", "Effect channel update decodes city");
  assert(typeof update.temperature === "number", "Effect channel update has numeric temperature");
  assert(typeof update.conditions === "string", "Effect channel update has conditions");
}

async function testUpload(): Promise<void> {
  const bytes = new TextEncoder().encode("hello from effect bun harness\n");
  const file = new File([bytes], "effect-harness.txt", { type: "text/plain" });

  class HarnessXMLHttpRequest {
    upload: { onprogress: ((event: { lengthComputable: boolean; loaded: number; total: number }) => void) | null } = { onprogress: null };
    status = 0;
    responseText = "";
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    onabort: (() => void) | null = null;
    setRequestHeader(_key: string, _value: string): void {}
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
  const result = await runEffect(uploadFile({ file }));
  await record({ kind: "effect-upload", name: "uploadFile", response: result });
  assertEqual(result.filename, "effect-harness.txt", "Effect uploadFile decodes filename");
  assertEqual(result.size, bytes.length, "Effect uploadFile decodes byte size");
  assert(result.contentType.startsWith("text/plain"), "Effect uploadFile decodes content type");
  assertEqual(result.checksum.length, 16, "Effect uploadFile decodes truncated checksum");
}

async function waitForSocketOpen(socket: WebSocket): Promise<void> {
  if (socket.readyState === WebSocket.OPEN) return;
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("websocket open timed out")), 5_000);
    socket.addEventListener("open", () => {
      clearTimeout(timeout);
      resolve();
    }, { once: true });
    socket.addEventListener("error", () => {
      clearTimeout(timeout);
      reject(new Error("websocket open failed"));
    }, { once: true });
  });
}

async function testWebSocket(): Promise<void> {
  const now = new Date().toISOString();
  const user = `effect-harness-${runId}`;
  const chatText = "hello over effect websocket";
  const socket = await runEffect(connectChat());

  try {
    await waitForSocketOpen(socket.socket);
    const resultPromise = runEffect(
      Effect.all({
        joined: socket.on("user_joined").pipe(Stream.take(1), Stream.runCollect, Effect.map(Chunk.unsafeHead)),
        message: socket.on("chat_message").pipe(Stream.take(1), Stream.runCollect, Effect.map(Chunk.unsafeHead)),
        typing: socket.on("typing").pipe(Stream.take(1), Stream.runCollect, Effect.map(Chunk.unsafeHead)),
      }).pipe(Effect.timeout("10 seconds")),
    );

    await runEffect(socket.send("join", { user, timestamp: now }));
    await runEffect(socket.send("chat_message", { user, text: chatText, timestamp: now }));
    await runEffect(socket.send("typing", { user, is_typing: true } as never));

    const result = await resultPromise;
    await record({ kind: "effect-websocket", name: "chat", request: { user, chatText }, response: result });
    assertEqual(result.joined.user, user, "Effect websocket join event echoes user");
    assertEqual(result.message.text, chatText, "Effect websocket chat_message event echoes text");
    const isTyping = result.typing.isTyping ?? (result.typing as unknown as { is_typing?: boolean }).is_typing;
    assertEqual(isTyping, true, "Effect websocket typing event preserves typing flag");
  } finally {
    await runEffect(socket.close);
  }
}

async function main(): Promise<void> {
  await appendLog("harness.log", `run=${runId} baseUrl=${BASE_URL}\n`);
  const server = await startServer();
  try {
    await waitForHealth();

    await step("effect command endpoints", testCommands);
    await step("effect type fidelity fixtures", testTypeFidelity);
    await step("effect static endpoint and raw wire", testRawWireAndStatic);
    await step("effect channel SSE endpoint", testChannel);
    await step("effect upload multipart endpoint", testUpload);
    await step("effect websocket endpoint", testWebSocket);

    await Bun.write(`${LOG_DIR}/summary.json`, `${JSON.stringify({ baseUrl: BASE_URL, testResults, evidenceCount: evidence.length }, null, 2)}\n`);
    console.log(`ALL PASSED (${testResults.length}/${testResults.length}); logs: ${LOG_DIR}`);
  } finally {
    await stopServer(server);
  }
}

await main();
