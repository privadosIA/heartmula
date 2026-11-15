# Proposal

## Why

The HTTP service (change `http-service`) keeps its GPUs busy after the first song: TorchSharp 0.107 has no `empty_cache`, so libtorch's allocator keeps what the model and the codec used until the process exits. Measured on 2026-10-22 with one 10 s song, about 10 GB stay on the model's GPU and about 13.5 GB on the codec's GPU. The container service it replaces freed the GPU after every song, and the orchestrator shares those GPUs with other engines, so this is a regression for the main use of the service. The sister DiffRhythm2 service solved the same problem by making each song in a short-lived worker process: when the process ends the system returns everything, video memory included, and a song can be stopped instantly by killing it.

## What Changes

- The work of a song can run in a **worker process** started from the service's own executable (`--worker`): the request goes in as one JSON line on its standard input, progress comes out as JSON lines on its standard output, diagnostics and the final error go to standard error, the WAV is written to the path it is given, exit code 0 means done.
- New setting `Service:Isolation` (environment alias `HEARTMULA_ISOLATION`) with three values:
  - `codec` (default): the model stays loaded in the service (about 10 GB stay on its GPU) and only the HeartCodec decode runs in a worker, so the codec's GPU (the larger footprint, about 13.5 GB) is freed; it costs about a second per song over `inprocess`.
  - `process`: the whole song, model load included, runs in the worker; the service process loads no model and creates no GPU context, so both GPUs are free between songs, as the container behaved; each song pays the model load (about 18 s more, measured).
  - `inprocess`: today's behaviour (everything in the service process, nothing returned until exit).
- Cancelling or forgetting a running job, and stopping the service, kill the worker at once (all memory returns, no wait for a frame or a window). A worker that crashes or is killed makes the job `failed` with a clear error and the service goes on. A worker whose standard input closes stops by itself and removes its partial files.
- In `process` mode `GET /health` says `ready` when the checkpoint files exist and the configured devices exist (there is no loading phase); the job shows a new progress stage `loading` while the worker reads the model.
- README: the modes, the measured cost (load time per song, memory before and after) and a corrected "things to know".
- HTTP APIs, webhook, queue, cancel/forget semantics: unchanged. Python: not involved.
- Parity criterion: **not applicable** (no model code changes). The WAV of `process` and `inprocess` must equal `generate --wav` byte for byte for the same checkpoint, inputs, seed and devices; `codec` must be deterministic and equal to it when the model and codec devices differ (see design, decision 4). The tests check this on the real checkpoint.

## Capabilities

### New Capabilities
- `song-worker-process`: running a song, or its codec decode, in a short-lived worker process: the isolation modes and setting, the worker protocol, memory returned to the system, instant cancellation, failure reporting, and the health/progress changes of the on-demand mode.

### Modified Capabilities
<!-- none in openspec/specs: the requirement of `generation-worker` that the model is loaded once belongs to a change (http-service) that is not archived yet; this capability states that it applies to `inprocess` and `codec` isolation only -->

## Impact

- Code: `src/HeartMuLa.Net.Server` (new `Worker/` pieces: protocol, worker entry, process runner; `Program` handles `--worker`; options and health), tests in `tests/HeartMuLa.Net.Server.Tests`. `HeartMuLa.Net` (library, CLI) unchanged.
- Dependency on `http-service`: this change builds on its code and should be applied after it, and archived after it (its `generation-worker` spec must exist in `openspec/specs` before this one is archived).
- Operations: with `process` isolation each song pays the model load (to be measured, see tasks) but nothing stays on the GPUs; with `codec` the model GPU stays occupied; with `inprocess` both do.
