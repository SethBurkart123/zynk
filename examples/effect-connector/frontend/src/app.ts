/**
 * Example consumer of the generated Effect client.
 *
 * The generated module is `./api`. It exposes:
 *  - `initZynk(config)`: a Layer that provides ZynkClient with global defaults.
 *  - Per-call options on every endpoint that override the global defaults.
 *  - `Effect.Effect<...>` for RPC/uploads, `Stream.Stream<...>` for channels.
 */

import { Effect, Stream } from "effect"

import {
  getUser,
  initZynk,
  listUsers,
  streamTicks,
  type CallOptions,
} from "./api"

// Global defaults: applied to every request unless overridden per-call.
const ZynkLayer = initZynk({
  baseUrl: "http://127.0.0.1:8000",
  timeout: "10 seconds",
  retry: {
    times: 3,
    backoff: {
      kind: "exponential",
      base: "200 millis",
      factor: 2,
      max: "5 seconds",
    },
    jitter: true,
    while: (error) =>
      error._tag === "ZynkNetworkError" ||
      error._tag === "ZynkTimeoutError" ||
      (error._tag === "ZynkHttpError" && error.status >= 500),
  },
  headers: { "X-Client": "effect-demo" },
})

// One-off override on a single call (no retries, longer timeout).
const noRetry: CallOptions = {
  timeout: "30 seconds",
  retry: { times: 0 },
}

const program = Effect.gen(function* () {
  const me = yield* getUser({ userId: 1 })
  const users = yield* listUsers({ activeOnly: true }, noRetry)

  yield* Effect.log(`me=${me.name}, total=${users.length}`)

  // Stream consumption: ticks flow as a fully typed Effect Stream.
  yield* streamTicks({ label: "demo" }).pipe(
    Stream.tap((tick) => Effect.log(`tick #${tick.n} (${tick.label})`)),
    Stream.runDrain,
  )
})

Effect.runPromise(program.pipe(Effect.provide(ZynkLayer))).catch((error) => {
  console.error("program failed", error)
})
