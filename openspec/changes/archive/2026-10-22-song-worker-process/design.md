# Design

## Context

`HeartMuLaSongGenerator` (change `http-service`) loads the model once and keeps it, runs `Preprocess` → `Seeding.Apply` → `Generate` → frames to CPU → `ReleaseGeneration` → `DecodeToWav`, and the worker thread calls it one job at a time. TorchSharp 0.107 exports no `empty_cache` (checked in the native library's symbols), so libtorch's caching allocator keeps every freed block until the process exits: measured after one 10 s song, about 10 GB on the model's RTX 4070 and 13.5 GB on the codec's RTX 5060 Ti. The sister DiffRhythm2 service (`~/Proyectos/diffrhythm2`, spec `song-worker-process`, `ProcessSongMaker.cs`, `Worker.cs`, `WorkerProtocol.cs`) already solves this with a worker process per song; this design follows its protocol and adapts it to a two-phase model (generate, then decode) and two devices. See proposal.md for the motivation.

## Goals / Non-Goals

**Goals:**
- The larger GPU footprint (the codec's) free between songs by default, and both GPUs free on request, as the container service the orchestrator used to talk to.
- Instant cancellation (kill the process) instead of "within one frame / one window".
- No change to either HTTP API, the queue, the webhook or the stored results.
- The same WAV as `generate --wav` in `process` and `inprocess`; deterministic in `codec`.
- Everything but the model itself testable without a GPU.

**Non-Goals:**
- Several songs at once, or a pool of warm workers.
- Making `inprocess` return memory (impossible without `empty_cache`; it stays as the fast mode).
- Moving the HTTP layer or the queue out of the service process.
- Changes to `HeartMuLa.Net` (library, CLI).

## Decisions

**1. The worker is the service's own executable.** `Program` looks at `args[0] == "--worker"` before it builds the web host and runs `WorkerEntry` instead; the parent starts `dotnet HeartMuLa.Net.Server.dll --worker <song|codec> <wavPath>` (or the apphost directly), the way the sister service does. The same native libraries and TorchSharp backend are therefore used by construction. Alternative: a separate worker project (a second thing to publish and keep in step).

**2. Everything the worker needs travels in the request line, not in its environment.** The parent already holds the effective settings (some may come from the command line or `appsettings`, which a child does not inherit). The request line carries the song (`tags, lyrics, durationMs, seed, cfgScale, temperature, topK`) and the resolved `checkpoint, version, modelDevice, codecDevice, codecTf32` (and `framesPath` in codec mode). One record in `WorkerProtocol`, used by both sides, so they cannot drift.

**3. Protocol: JSON lines, three exit codes.** stdin: the request line, then kept open (EOF means "the service is gone"); stdout: only progress lines `{"stage":"loading"}`, `{"stage":"generating","current":i,"total":n}`, `{"stage":"decoding","current":w,"total":m}` (one per frame / window, as the in-process callbacks); stderr: diagnostics, and as the last line the cause of a failure. Exit codes 0 done, 1 failed, 3 cancelled. The pipeline prints human text with `Console.WriteLine` ("Loading HeartMuLa weights…", the RTF line), so the worker captures the real stdout stream first and then redirects `Console.Out` to stderr; the parent also ignores any stdout line that is not a progress line (native libraries sometimes write there).

**4. Seeds and byte-identity.** `process` runs exactly the CLI's sequence in one process (`Preprocess`, `Seeding.Apply(seed)`, `Generate`, then the decode noise from the same seeded generators), so its WAV equals the CLI's. In `codec` the decode runs in another process, which calls `Seeding.Apply(seed)` itself before decoding. The decode noise is `randn` on the codec device's generator: when that is a different device from the model's, generation never touched it, so it is in the same just-seeded state as in the CLI and the WAV is identical; when both are the same device (or the CPU) the CLI's noise continues the stream after the sampling draws and the worker's does not, so the WAV differs from the CLI's but is still fully determined by the request. This is stated in the spec (identical with two devices, deterministic always) and checked by a test on the two GPUs; if the test shows the two-device case also differs, the spec scenario is reduced to determinism and the README says so (decision to revisit, not to hide). Alternative: draw the noise in the parent and hand it over (identical in all cases, but it needs a CUDA context on the codec device in the parent, which is what this change removes).

**5. Split the generator into steps instead of copying it.** `HeartMuLaSongGenerator` is refactored into reusable steps with no change of behaviour: `LoadModel`, `GenerateFrames(request, progress)` (preprocess, seed, generate, frames to CPU, release) and `DecodeFrames(frames, wavPath, progress)`. The in-process generator calls them in sequence; the `song` worker calls them in sequence; the `codec` worker calls only `DecodeFrames` after `Seeding.Apply`. The existing real-model tests (CLI equality) guard the refactor.

**6. Generators by mode.** `Service:Isolation` picks the registration in `ServiceHost`: `inprocess` → today's generator; `process` → `ProcessSongGenerator` (an `ISongGenerator` that starts a `song` worker per job and loads nothing); `codec` → the in-process generator with its decode step replaced by a `WorkerFrameDecoder` that writes the frames to `{id}.frames.tmp` (the core's `SafeTensorsWriter`; the name makes the existing startup sweep of `*.tmp` files remove a leftover), starts a `codec` worker and removes the file in a `finally`. Leftover files of a crashed run are deleted at startup by the existing sweep of `*.tmp` files.

**7. The parent kills, the child also watches.** On cancellation the parent calls `Process.Kill(entireProcessTree: true)`, waits for the exit, and deletes `{wav}.tmp`. Independently the worker has a thread blocked on stdin: when it reads EOF before finishing it cancels the song, deletes its partial files and exits with 3, so a service killed with `kill -9` does not leave an orphan holding 20 GB of video memory.

**8. `process` mode touches no CUDA in the parent.** The parent needs the number of CUDA devices only to validate settings and to default the devices; it asks `nvidia-smi` (the `IGpuProbe` it already has gains a device count) instead of `torch.cuda`, so libtorch is not even loaded in the service process. `ready` in health is "checkpoint files present and configured devices exist"; `model` shows `on-demand`; `gpu` is still the model device's `nvidia-smi` row. A test asserts the parent's own memory use on the cards is within a few hundred MiB of the baseline.

**9. Progress.** The native API's `progress.stage` gains `loading` (shown while the worker reads the model, `current`/`total` 0); the orchestrator-compatible view keeps `{done, total}` (0, 0 during loading). Nothing else in either contract changes.

**10. Default `codec`.** The goal is to return the GPUs to the orchestrator's other engines without making every song slow. Measured (task 6.1): the codec is read for every song in every mode, so `codec` costs about one second over `inprocess` and frees the card that held about 13.5 GB, while `process` frees the model's card too (about 10 GB) for about 18 s more per song. The user chose `codec` as the default (and said the rest did not matter to them); `process` stays one setting away for when the model's card must be free as well. Originally `process` was proposed as the default because the container behaved like it; the numbers changed the choice.

**11. Tests.** Without a GPU: `WorkerProtocol` round trips; `Worker` driven with in-memory stdin/stdout/stderr and a fake pipeline (success, failure, EOF cancel, partial-file removal, stray output); `ProcessSongGenerator` against a stub worker (a shell script that speaks the protocol; the tests skip on Windows) for progress, failure cause, killed worker, stdout noise, cancellation within a second, orphan stop; settings, health and stage tests on the HTTP factory. With the GPUs (`[RequiresCheckpoint]`, heavy, announced first): CLI equality in `process` and `inprocess`, `codec` on two devices, memory back to baseline after a song and after a kill, load time per song. Parity with Python: not applicable. Reference for the audio: the WAV that `generate --wav` writes in the same test run (generated by the CLI at test time, no stored fixture).

## Risks / Trade-offs

- [Each song reloads about 8 GB of weights in `process` mode] → measure it and write it down; `codec` and `inprocess` remain available; the sister service accepts the same trade-off for the same reason.
- [A worker killed in the middle of CUDA work could leave the driver in a bad state] → the kernel releases a process's GPU memory on exit; the test of killing a worker mid-generation checks that the memory is back and the next song succeeds.
- [Two processes on one GPU in `codec` mode (model in the service, codec in the worker)] → by design they use different phases; the README says the codec's device should have room for the codec while the model stays.
- [The stdout protocol is fragile if something writes to it] → `Console.Out` is redirected to stderr in the worker and the parent ignores non-progress lines.
- [The refactor of the generator could change the audio] → the real-model CLI-equality tests are run before and after (task 6.1 compares with the CLI in all modes).
- [Orphan workers if the service dies] → stdin EOF stops the worker (decision 7).

## Migration Plan

Additive, but it changes the default behaviour of the service built by `http-service` (new default `codec`). Apply after `http-service`; archive `http-service` first so that `generation-worker` is a main spec when this one is archived. Rollback: set `Service:Isolation=inprocess` (or `HEARTMULA_ISOLATION`), or revert the change.

## Open Questions

- None: the default was decided with the numbers of task 6.1 (see decision 10).
