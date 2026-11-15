# Design

## Context

HeartMuLa.Net exposes `HeartMuLaModel`, `MusicGenerationPipeline` (`Preprocess` + `Generate`) and `AudioStage` (`ReleaseGeneration`, `Unload`, `DecodeToWav`); the CLI wires them in `Program.cs`. Measured on this machine: a song of 10 s takes about 9 s of generation on the 4070 (RTF 0.85) and the weights take several seconds to load, so reloading per song is the dominant cost for short songs. The model needs about 8 GB of VRAM with its caches (bf16) and the codec 6.3 GB (fp32); `ReleaseGeneration` was added so a loaded model can serve several songs. `Generate` calls an optional `progress(i, n)` after every frame and `DecodeToWav` calls one per codec window, so both can also be used to abort by throwing. The backbone has 8192 positions, which bounds prompt + song length. See proposal.md for the motivation.

## Goals / Non-Goals

**Goals:**
- A separate server project; `HeartMuLa.Net` (library, CLI, tests) does not change.
- The same audio as the CLI for the same inputs, seed and devices.
- Predictable use of the two GPUs, with the model resident and the codec transient.
- Everything testable without a GPU or the 7 GB checkpoint, plus a few real-checkpoint tests.

**Non-Goals:**
- Several generations in parallel, batching across requests, or multi-GPU scheduling (one worker, one model).
- Streaming audio while it is generated.
- Multi-user accounts, quotas or a database.
- A Dockerfile (the docker deployment is being dropped), model download, or reference audio.

## Decisions

**1. ASP.NET Core minimal API + Swashbuckle, in `src/HeartMuLa.Net.Server`.** Minimal APIs keep the surface to five endpoints. `Swashbuckle.AspNetCore` 6.x supports net8.0 and gives the Swagger UI; alternatives: `Microsoft.AspNetCore.OpenApi` (only the document, no UI on net8) and NSwag (heavier). The project targets net8.0 with `RollForward=Major` like the others.

**2. One `GenerationWorker` as a `BackgroundService` on a dedicated long-running thread, fed by a bounded `Channel<Job>`.** TorchSharp and the CUDA context are used from one thread, FIFO order is the channel's, and the bound gives the `429`. A job store (`ConcurrentDictionary<Guid, Job>`) holds state; the worker is the only writer of a job's state after it starts. Alternative: a queue library or Hangfire (a database for no benefit).

**3. Model loaded once at startup, in the background.** The host starts listening immediately and `/health` reports `loading` until the weights are on the device (`503`), so a supervisor can wait for `ready`. Jobs are accepted meanwhile and wait in the queue. Between songs the worker calls `AudioStage.ReleaseGeneration(model, frames, prompt tensors…)`; `Unload` is never used. Alternative: load lazily on the first job (first request pays 10+ s and a failed load is discovered late).

**4. The worker repeats the CLI's calls, in the CLI's order, and nothing else.** `Preprocess` → `Seeding.Apply(seed)` → `Generate` → copy frames to CPU → `ReleaseGeneration` → `DecodeToWav` with `AudioOptions(path, Pcm16, CodecDevice, CodecTf32: false)`. That order is what makes the WAV byte-identical to `generate --wav` (the seed is applied right before `Generate`; the codec noise comes from the same stream afterwards). The server asks `Seeding.NewSeed()` when no seed is given and always returns the seed used. A test pins the equality on the real checkpoint (`[RequiresCheckpoint]`) by comparing against a CLI-equivalent call.

**5. Cancellation by exception from the progress callbacks.** Each job has a `CancellationTokenSource`. The `progress` callbacks of `Generate` and `DecodeToWav` call `token.ThrowIfCancellationRequested()`, which unwinds the frame loop (its `NewDisposeScope` frees the step tensors) within one frame (~80 ms of audio, ~0.1 s of work on GPU) or one codec window. The worker then runs `ReleaseGeneration` in a `finally`. This needs no change in the library. Alternative: a `CancellationToken` parameter on `Generate` (cleaner, but touches the validated pipeline; can be done later without changing the HTTP contract).

**6. Result storage: files on disk; index in memory, rebuilt from sidecars.** `{Storage:Directory}/{id}.wav` and `{id}.json` (request, seed, state, timestamps, error). The WAV is written to `{id}.wav.tmp` and renamed when complete, so a crash never leaves a truncated song that looks finished. At startup the directory is scanned: finished jobs are restored, unfinished sidecars become `failed` ("interrupted by restart"). A timer deletes results older than `Storage:RetentionHours` (default 24). Alternatives: memory only (loses songs on restart, and a 4-minute WAV is ~46 MB), SQLite (a dependency for five fields).

**7. Duration limit.** `MaxDurationMs` defaults to 360 000 (4500 frames, under the 8192 positions of the backbone including the prompt; it is also the ceiling the orchestrator's service accepts, see decision 14) and is configurable; `DefaultDurationMs` is 30 000. The request value is passed as `maxAudioLengthMs`. The token loop can end earlier on the audio EOS token, so the real duration is reported in the job.

**8. Authentication: optional static API key, loopback by default.** `Security:ApiKey` empty means no auth and `Urls` defaults to `http://127.0.0.1:5080`; with a key, a small middleware checks `X-Api-Key` (constant-time comparison) on everything except `/health` and Swagger. the intended caller runs on the same host or network, so the full identity stack (JWT, OIDC) is not worth it now; a reverse proxy can add more.

**9. Deployment: a plain process managed by systemd (unit example in the README), no container.** Configuration through `appsettings.json` and environment variables (`HEARTMULA_CKPT`, `Model__Device`, `Codec__Device`, …), the same variable the tests and CLI use for the checkpoint. The service is built with the same `-p:TorchBackend=cuda-linux|cuda-windows|cpu` as the rest.

**10. How an orchestrator uses it: over HTTP.** That keeps TorchSharp, CUDA and the 7 GB checkpoint out of its process and lets the service restart without taking it down. The library route (README section "Using it from your code") stays valid for anyone who prefers to host the model in-process.

**11. Errors as RFC 7807 problem details** (`application/problem+json`) with `AddProblemDetails`; validation by hand on the request record (five fields) instead of a validation library.

**12. Webhook: a separate `WebhookDispatcher` with its own channel, HMAC-signed, retried.** The worker only hands a finished job to the dispatcher (an in-memory enqueue), so a slow receiver never touches the GPU path. Body: `id`, `state`, `seed`, `durationMs`, `error`, timestamps, relative `audioUrl` (the receiver already knows the service's base address, and the URL stays valid behind a proxy); header `X-HeartMuLa-Event: song.finished`. Optional `X-HeartMuLa-Signature: sha256=<HMAC-SHA256(body, Webhook:Secret)>` so the receiver can authenticate the call. Delivery: 1 attempt + 3 retries after 1 s, 5 s and 30 s, `Webhook:TimeoutSeconds` (default 10) per attempt, a shared `HttpClient` with redirects disabled, `2xx` = delivered. The delivery state (`pending | delivered | failed`, attempts, last error) lives in the job's sidecar, so it is shown by `GET /songs/{id}` and the pending ones are resumed at startup (bounded by the retention). `Webhook:AllowedHosts` (empty = any) limits where the service may call: without it, anyone who can reach the service can make it send requests to arbitrary URLs, which is why the default listener is loopback and the README recommends the API key or the host list outside it. Alternatives: delivering inside the worker (a dead receiver would hold the GPU for the whole retry time), no retries (a restart of the receiver at the wrong second loses the event), a durable outbox table (more machinery than a sidecar field), and polling only (the original plan; kept as the fallback because the webhook is optional).

**13. A second front door for the orchestrator (added after reading its client).** The orchestrator already talks to its HeartMuLa and DiffRhythm2 containers with one shared Go client (`internal/musicsvc`): `GET /health` (`ready, version, busy, queued, gpu, missing`), `POST /generate` (`lyrics, tags, max_ms, seed, temperature, topk, cfg_scale`) → `202 {id}`, `GET /jobs/{id}` (`state` queued|running|done|failed, `progress{done,total}`, `error`, `seconds`), `/jobs/{id}/audio` (409/410), `DELETE /jobs/{id}` → `{ok:true}`, and a bearer token. The sister DiffRhythm2 service implements exactly that so the client works by changing the address. This service does the same, as a thin layer: `CompatEndpoints` validates the request with the orchestrator's limits and defaults (which differ from the native API: guidance 3.0, seed 0, `max_ms` 5 000–360 000), builds the same `SongRequest` (which gains `Temperature` and `TopK`, also optional in `POST /songs`) and uses the same `SongService`, so there is one queue, one worker and one set of files; only the JSON in and out differs (`ToCompatView`: `succeeded→done`, `cancelled→failed "cancelled"`, `seconds`, progress). Consequences decided with it:
  - `GET /health` is shared and now always `200`, with the orchestrator's fields plus the native ones: its client treats a 4xx/5xx as an error, and `ready:false` carries the news. The `gpu` object comes from `nvidia-smi` for the model's CUDA device (cached a few seconds; `null` on CPU or when it cannot be read), since TorchSharp 0.107 has no `mem_get_info`.
  - `DELETE /jobs/{id}` means *forget*: cancel if needed, remove from the store, delete the files. A running job that is forgotten ends in the worker, which must then not write its sidecar or WAV nor notify the webhook (checked against the store before saving); that is why a forgotten job cannot come back after a restart.
  - Bearer is accepted next to `X-Api-Key` against the same configured key; the 401 body on the orchestrator's routes is its `{"error": ...}` shape, on the others problem-details.
  - Environment aliases (`HEARTMULA_TOKEN/JOBS/VERSION/PORT`) are read as configuration fallbacks so a container's environment can be reused; `HEARTMULA_PORT` listens on all interfaces like the container does (the loopback default only applies when nothing says otherwise). `HEARTMULA_DEVICE` is deliberately not an alias: the test suite uses it.
  - Alternatives: replacing the native API with the orchestrator's (one contract, but loses `/songs`, Swagger-first documentation and the nicer names, and the orchestrator contract has no webhook), or leaving the native API only (the orchestrator would need a new client). The contract is proven by running the orchestrator's own `Client` code against a live instance (task 7.4).

**14. Tests.** Unit and HTTP tests use `WebApplicationFactory` (and a fake webhook receiver served by a second test host, which can answer `500`, hang or redirect) with a **fake generator** behind an `ISongGenerator` interface (returns a tiny WAV after N progress ticks, can be told to fail or block), so queueing, cancellation, retention, restart recovery, auth and the OpenAPI document run in milliseconds without GPU or checkpoint. The real implementation (`HeartMuLaSongGenerator`) is covered by `[RequiresCheckpoint]` tests: equality with the CLI-equivalent path and two songs on one loaded model. Parity with Python: not applicable, nothing in the model changes.

## Risks / Trade-offs

- [The 4070 holds the model (≈8 GB) permanently, so it cannot be shared with other heavy jobs such as `ollama` or other GPU jobs] → documented; `/health` shows the devices, and a `Model:Device` / `Codec:Device` setting lets the service live on whichever GPU is free. Future option: an idle timeout that unloads the model.
- [Out-of-memory on a long song or when another process takes the GPU] → the job fails with a message, `ReleaseGeneration` runs, the next job proceeds (spec: generation-worker). A real OOM test is hard to provoke deterministically, so the worker is tested with the fake generator throwing, and the OOM path with `ReleaseGeneration` is checked by hand once.
- [Cancel is only honoured between frames or codec windows] → a codec window can take several seconds on CPU; acceptable, and bounded.
- [libtorch's allocator keeps freed blocks (no `empty_cache()` in TorchSharp 0.107)] → memory shown by `nvidia-smi` does not shrink between songs; it is reused. Documented.
- [In-memory queue is lost on restart] → queued jobs become `failed` ("interrupted by restart"); clients resubmit. Acceptable for a single-host tool.
- [A webhook can be lost if the receiver stays down longer than ~36 s of retries, or the result expires before it returns] → the state `failed` is visible in `GET /songs/{id}` and the audio stays downloadable by polling for 24 h; [Outgoing requests to attacker-chosen URLs when the service is reachable by untrusted clients] → loopback default, API key and `Webhook:AllowedHosts`, no redirects, and no response body is read or returned; [Swashbuckle is a new dependency] → only in the server project; the library stays dependency-free.
- [`Seeding.Apply` seeds process-wide generators] → safe because exactly one generation runs at a time (decision 2).

## Migration Plan

Additive. Nothing existing changes behaviour. Rollback is deleting the two new projects and their solution entries. The public snapshot script (`tools/update_public_branch.sh`) builds the solution, so it is run once after this change to confirm the new projects restore and build there (Swashbuckle comes from NuGet).

## Open Questions

- Whether the orchestrator later needs progress events (not only the final one). The final-event webhook covers the first version; a progress stream can be added as a new endpoint without changing the existing ones.

## Settled with the user

- Webhook: wanted, optional (decision 12).
- Retention: 24 h is enough (`Storage:RetentionHours` = 24).
