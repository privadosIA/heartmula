# Proposal

## Why

HeartMuLa.Net can only be used through the `generate` CLI (one process per song, 7 GB of weights reloaded every time) or by referencing the project from C#.
A media orchestrator, like any other tool, needs to ask for songs without owning TorchSharp, CUDA or the checkpoint. A small HTTP service with an
OpenAPI/Swagger description gives them a stable contract, keeps the model loaded between songs (`AudioStage.ReleaseGeneration` already allows it) and
uses both GPUs (model on one, codec on the other) without reloading. Python stays out of it, as in the rest of the project.

## What Changes

- New project `src/HeartMuLa.Net.Server` (ASP.NET Core minimal API, net8.0) that references `HeartMuLa.Net`; the CLI, the library and their tests do not change.
- Asynchronous job API, because a song takes seconds to minutes:
  - `POST /songs` (tags, lyrics, `durationMs`, `seed`, `cfgScale`, optional `callbackUrl`) → `202` with a job id.
  - `GET /songs/{id}`: state (`queued | running | succeeded | failed | cancelled`), progress, seed used, timings.
  - `GET /songs/{id}/audio`: the WAV when the job has succeeded.
  - `DELETE /songs/{id}`: cancels a queued or running job, or removes a finished one.
  - `GET /health`: model loaded, devices, queue length.
- **Orchestrator-compatible routes** (`POST /generate`, `GET /jobs/{id}`, `/jobs/{id}/audio`, `DELETE /jobs/{id}`, bearer token, the orchestrator's health shape and request limits) over the same queue and worker, so the orchestrator's existing music-service client works against this service by changing only the address. The service's own `/songs` API, Swagger and webhook stay.
- Optional webhook: `POST /songs` may carry a `callbackUrl`; when the job ends (succeeded, failed or cancelled) the service notifies it with a signed JSON `POST`, retrying on failure, so the caller does not need to poll.
- One generation worker that owns the GPUs: a FIFO queue, one job at a time, the model loaded once and reused through `ReleaseGeneration`; the codec device is configurable (`Model:Device`, `Codec:Device`).
- Swagger UI and `openapi.json` generated from the endpoints.
- Results are WAV files on disk with a retention time; the job index survives a restart. Optional API key; localhost binding by default.
- README section on running and calling the service. No Dockerfile in this change (the docker deployment is being dropped); a systemd unit example only.
- Parity criterion: **not applicable** (no change in model behaviour). The service must produce the same WAV as the CLI for the same checkpoint, inputs, seed and devices; the tests check that on the real checkpoint.

## Capabilities

### New Capabilities
- `song-service-api`: HTTP contract: endpoints, request validation, job states and errors, webhook notification, OpenAPI document, API key, health.
- `engine-compat-api`: the orchestrator's HTTP contract (routes, limits, answers, errors, bearer token, health fields, environment variables) served over the same jobs.
- `generation-worker`: queueing and execution: single worker, model reuse between jobs, cancellation, progress, device selection, result storage and retention, webhook delivery, failure isolation.

### Modified Capabilities
<!-- none: audio-generation and generation-reproducibility are used as they are -->

## Impact

- New: `src/HeartMuLa.Net.Server/`, `tests/HeartMuLa.Net.Server.Tests/`; both added to `HeartMuLa.Net.sln`.
- New dependencies in the server only: `Swashbuckle.AspNetCore` (Swagger UI / OpenAPI) and the `Microsoft.AspNetCore.App` framework reference. `HeartMuLa.Net` gains none.
- New environment aliases `HEARTMULA_TOKEN`, `HEARTMULA_JOBS`, `HEARTMULA_VERSION`, `HEARTMULA_PORT`; `/health` now answers `200` always (with `ready`), as the orchestrator's client expects.
- New settings `Webhook:Secret`, `Webhook:AllowedHosts`, `Webhook:TimeoutSeconds`; outgoing HTTP calls from the service (the only ones it makes).
- `Directory.Build.props` already selects the TorchSharp backend and the tokenizer runtime; the server is built with the same `-p:TorchBackend=...`.
- README gains a "Running as a service" section; `tools/update_public_branch.sh` must be checked so the public snapshot includes the new projects.
- Operational: the service holds ~8 GB of VRAM on the model device while running and ~6.3 GB on the codec device while decoding. It is not meant to share a GPU with other heavy jobs.
