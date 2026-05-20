import { appendFile } from "node:fs/promises";

import {
  createChatSocket,
  createTask,
  downloadSampleUrl,
  getForecast,
  getTaskStats,
  getUser,
  initBridge,
  listCities,
  listTasks,
  streamWeather,
  updateUser,
  uploadFile,
  type Task,
  type WeatherUpdate,
} from "../frontend/src/generated/api";

const BASE_URL = process.env.ZYNK_BASE_URL ?? "http://127.0.0.1:8100";
const PORT = new URL(BASE_URL).port || "8100";
const ROOT = new URL("../../..", import.meta.url).pathname.replace(/\/$/, "");
const BACKEND_DIR = `${ROOT}/examples/kitchen-sink/backend`;
const runId = new Date().toISOString().replace(/[:.]/g, "-");
const LOG_DIR = `${import.meta.dir}/logs/${runId}`;

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
  const proc = Bun.spawn(["uv", "run", "python", "-c", "import main; main.app.port = int(__import__('os').environ['PORT']); main.app.generate_ts = None; main.app.run(dev=False)"], {
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
  const user = await getUser({ userId: 1 });
  await record({ kind: "command", name: "getUser", request: { userId: 1 }, response: user });
  assertEqual(user.name, "Alice", "getUser returns fixture user");
  assertEqual(user.email, "alice@example.com", "getUser maps email");
  assertEqual(user.isActive, true, "getUser maps isActive");

  const cities = await listCities();
  await record({ kind: "command", name: "listCities", response: cities });
  assert(cities.includes("Tokyo"), "listCities includes Tokyo fixture");

  const forecast = await getForecast({ city: "Tokyo", days: 2 });
  await record({ kind: "command", name: "getForecast", request: { city: "Tokyo", days: 2 }, response: forecast });
  assertEqual(forecast.length, 2, "getForecast honors days arg");
  assert(forecast.every((day) => typeof day.precipitationChance === "number"), "forecast maps precipitationChance");

  const stats = await getTaskStats();
  await record({ kind: "command", name: "getTaskStats", response: stats });
  assert(stats.total >= 4, "getTaskStats sees seeded tasks");
  assert(stats.byPriority.high >= 2, "getTaskStats maps byPriority record");

  const updated = await updateUser({ userId: 2, name: null, email: null, isActive: false });
  await record({ kind: "command", name: "updateUser", request: { userId: 2, name: null, email: null, isActive: false }, response: updated });
  assertEqual(updated.name, "Bob", "nullable name does not overwrite when null");
  assertEqual(updated.email, "bob@example.com", "nullable email does not overwrite when null");
  assertEqual(updated.isActive, false, "nullable boolean updates when false");
}

async function testTypeFidelity(): Promise<void> {
  const created = await createTask({
    title: "Harness urgent task",
    description: null,
    priority: "urgent",
    dueDate: null,
    labelIds: [1, 2],
  });
  await record({ kind: "type-fidelity", name: "createTask", response: created });

  const priority: "urgent" = created.priority as "urgent";
  const status: "todo" = created.status as "todo";
  assertEqual(priority, "urgent", "enum-with-values priority survives runtime");
  assertEqual(status, "todo", "enum-with-values status survives runtime");
  assertEqual(created.description, null, "optional+nullable description round-trips null");
  assertEqual(created.dueDate, null, "optional+nullable dueDate round-trips null");
  assert(created.labels?.some((label) => label.name === "Bug"), "nested label model maps camelCase fields");

  const filtered = await listTasks({ status: "todo", priority: "urgent", labelId: null });
  await record({ kind: "type-fidelity", name: "listTasks", request: { status: "todo", priority: "urgent", labelId: null }, response: filtered });
  assert(filtered.some((task: Task) => task.id === created.id), "nullable filter param accepted as null");
}

async function testRawWireAndStatic(): Promise<void> {
  const raw = await rawJson("/command/get_user", { user_id: 1 });
  assertEqual((raw as { result: { is_active: boolean } }).result.is_active, true, "raw command response preserves snake_case wire");

  const url = downloadSampleUrl({ filename: "sample.txt" });
  const response = await fetch(url);
  const text = await response.text();
  await record({ kind: "static", name: "downloadSampleUrl", request: { url }, response: { status: response.status, text } });
  assertEqual(response.status, 200, "static handler returns 200");
  assertEqual(text, "Zynk kitchen sink static sample\n", "static file fixture content matches");
}

async function testChannel(): Promise<void> {
  const update = await new Promise<WeatherUpdate>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("streamWeather timed out")), 8_000);
    const channel = streamWeather({ city: "Tokyo", intervalSeconds: 1 });
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
  await record({ kind: "channel", name: "streamWeather", request: { city: "Tokyo", intervalSeconds: 1 }, response: update });
  assertEqual(update.city, "Tokyo", "channel update maps city");
  assert(typeof update.temperature === "number", "channel update has numeric temperature");
  assert(typeof update.conditions === "string", "channel update has conditions");
}

async function testUpload(): Promise<void> {
  const bytes = new TextEncoder().encode("hello from bun harness\n");
  const file = new File([bytes], "harness.txt", { type: "text/plain" });
  const body = new FormData();
  body.append("files", file);
  body.append("_args", JSON.stringify({}));
  const rawResponse = await fetch(`${BASE_URL}/upload/upload_file`, { method: "POST", body });
  const rawText = await rawResponse.text();
  await record({ kind: "upload-raw", name: "POST /upload/upload_file", request: { filename: file.name, size: file.size, type: file.type }, raw: rawText });
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
  const result = await uploadFile({ file }).promise;
  await record({ kind: "upload", name: "uploadFile", response: result });
  assertEqual(result.filename, "harness.txt", "uploadFile returns filename");
  assertEqual(result.size, bytes.length, "uploadFile returns byte size");
  assert(result.contentType.startsWith("text/plain"), "uploadFile returns a text/plain content type");
  assertEqual(result.checksum.length, 16, "uploadFile returns truncated checksum");
}

async function testWebSocket(): Promise<void> {
  const socket = createChatSocket();
  const now = new Date().toISOString();
  const user = `harness-${runId}`;
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
      socket.send("join", { user, timestamp: now });
      socket.send("chat_message", { user, text: chatText, timestamp: now });
      socket.send("typing", { user, is_typing: true } as unknown as { user: string; isTyping: boolean });
    });
    socket.connect();
  });

  await record({ kind: "websocket", name: "chat", request: { user, chatText }, response: result });
  assertEqual(result.joined, user, "websocket join event echoes user");
  assertEqual(result.text, chatText, "websocket chat_message echoes text");
  assertEqual(result.typing, true, "websocket typing event maps isTyping");
}

async function main(): Promise<void> {
  await appendLog("harness.log", `run=${runId} baseUrl=${BASE_URL}\n`);
  const server = await startServer();
  try {
    await waitForHealth();
    initBridge(BASE_URL);

    await step("command endpoints", testCommands);
    await step("type fidelity fixtures", testTypeFidelity);
    await step("static endpoint and raw wire", testRawWireAndStatic);
    await step("channel SSE endpoint", testChannel);
    await step("upload multipart endpoint", testUpload);
    await step("websocket endpoint", testWebSocket);

    await Bun.write(`${LOG_DIR}/summary.json`, `${JSON.stringify({ baseUrl: BASE_URL, testResults, evidenceCount: evidence.length }, null, 2)}\n`);
    console.log(`ALL PASSED (${testResults.length}/${testResults.length}); logs: ${LOG_DIR}`);
  } finally {
    await stopServer(server);
  }
}

await main();
