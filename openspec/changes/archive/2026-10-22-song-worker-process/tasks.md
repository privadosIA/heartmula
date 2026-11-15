# Tasks

## 1. Reusable steps and the worker protocol

- [x] 1.1 Split `HeartMuLaSongGenerator` into `LoadModel`, `GenerateFrames` and `DecodeFrames` with no change of behaviour (the in-process generator calls them in order), and add `WorkerProtocol` (request record with the song and the resolved settings, progress lines, exit codes, `Argument = "--worker"`). Verify the existing 162 server tests still pass, and unit tests for the protocol: request and each progress line round-trip, a stray non-JSON line and a JSON line that is not progress are ignored.

## 2. The worker side

- [x] 2.1 Implement `WorkerEntry`/`Worker.Run(args, stdin, stdout, stderr, pipeline)`: reads the request line, reports `loading`/`generating`/`decoding` lines, writes the WAV to the given path, exits 0; on failure writes the cause as the last stderr line and exits 1; a thread on stdin cancels on EOF, removes partial files and exits 3; `Console.Out` is redirected to stderr after the real stdout is captured; `Program` dispatches to it when `args[0]` is `--worker`. Verify unit tests with in-memory streams and a fake pipeline: success output, failure last line, EOF before the end cancels and deletes `{wav}.tmp`, noise written with `Console.WriteLine` never reaches stdout.

## 3. The service side

- [x] 3.1 Implement `ProcessSongGenerator` (an `ISongGenerator` for `process` isolation) over a command factory: starts the worker, writes the request line and keeps stdin open, turns progress lines into job progress (stage `loading` added to `JobStage`; the orchestrator-compatible view shows 0 of 0 meanwhile), keeps the last 20 stderr lines, kills the process tree on cancellation and deletes partial files, maps exit codes to errors as in the spec (cause line, or "the song process ended unexpectedly (exit code N)"). Verify tests against a stub worker script: progress sequence, failure cause, a worker that kills itself with SIGKILL, noise on stdout, no WAV written, cancellation ends the process within one second, the next job runs after each failure (tests skip on Windows).

## 4. Settings, health and wiring

- [x] 4.1 Add `Service:Isolation` (`codec` default, `process`, `inprocess`) with the alias `HEARTMULA_ISOLATION` (own setting wins; an invalid value stops startup naming it), register the generator by mode in `ServiceHost`, make `process` mode validate settings and default devices through `nvidia-smi` (device count added to `IGpuProbe`) without loading libtorch, and give health the on-demand state (`ready` = checkpoint files present and devices exist, `model: "on-demand"`, `missing`). Verify tests: default and alias, invalid value, health in each mode (complete and incomplete checkpoint), the `loading` stage visible through `/songs/{id}` and `/jobs/{id}`, and that the parent in `process` mode never constructs the in-process generator.

## 5. Codec in a process

- [x] 5.1 Implement `codec` isolation: `WorkerFrameDecoder` writes the frames to `{id}.frames.tmp`, starts a `codec` worker (it calls `Seeding.Apply(seed)` and `DecodeFrames`), and removes the frames file whatever the outcome; a leftover `.frames.tmp` is deleted at startup. Verify tests with the stub worker: the file exists while the worker runs and is gone after success, failure, kill and cancellation; the startup sweep removes a leftover; progress shows `generating` then `decoding`.

## 6. Real-model verification (heavy)

- [x] 6.1 `[RequiresCheckpoint]` tests on the GPUs: (a) `process` and `inprocess` give the CLI's WAV byte for byte; (b) `codec` with model on `cuda:0` and codec on `cuda:1` gives the CLI's WAV (otherwise record the finding and reduce the spec as decision 4 says) and is deterministic; (c) after a song in `process` the memory in use on each card is back to the baseline (plus a few hundred MiB at most), after a song in `codec` the codec's card is back and the model's holds the model; (d) a running job killed through `DELETE /jobs/{id}` frees the memory within a second and the next song succeeds; (e) record the time of a song in each mode (model load included). **Heavy: both GPUs and about 8 GB of RAM for several minutes. Announce to the user and check `nvidia-smi` first.** Also run the orchestrator's own client (as in `http-service` task 7.4) against `process` mode. Paste the results and the numbers.

## 7. Documentation and close

- [x] 7.1 README: the three modes (table with what stays on each GPU and what each song costs, using the measured numbers), the new setting and alias, the `loading` stage, a corrected "things to know" (memory is returned in `process`; in `inprocess` and `codec` the model's card stays occupied), and a note that the default can be changed. Verify the documented environment variables work (`HEARTMULA_ISOLATION=inprocess` starts the old behaviour) and run `openspec validate song-worker-process`, the server tests and the library tests without the checkpoint; paste the results.

### Results of 6.1 (2026-10-22; model on the RTX 4070, codec on the RTX 5060 Ti; each test in its own `dotnet test` process, then the orchestrator's client)

- `process`: the WAV equals the CLI's byte for byte; after the song the cards are at their baseline (227 MiB and 13 MiB, unchanged); a 2 s song takes 31.4 s (second one 30.9 s), of which about 18 s is reading the model.
- `codec` (model on card 0, codec on card 1): the WAV equals the CLI's byte for byte and the same request twice gives the same WAV; after the song card 1 is at 16 MiB (baseline 13) and card 0 holds the model (10 333 MiB); songs take 15.2 s and 13.6 s.
- `inprocess`: the WAV equals the CLI's byte for byte; 15.4 s then 12.6 s; the cards stay at 10 333 MiB and 13 561 MiB.
- Killing a running job in `process` mode: the model card went from 10 333 MiB back to its baseline in 0.23 s, and the next song was the CLI's WAV.
- The four real-model tests of `http-service` (now in `inprocess`, where their meaning is) still pass.
- The orchestrator's own client against `process` mode: Health, Generate, Job → done (41.4 s for a 10 s song, model load included), Audio, Forget, a 400 naming `max_ms` and `ErrUnauthorized`; idle afterwards the cards were at their baseline.
- Finding for the open question of the design: the codec is read for every song in every mode (most of the time of a short song), so `codec` costs about one second over `inprocess` while freeing the card that held ≈13.5 GB, and `process` adds ≈18 s per song to also free the model's ≈10 GB. The default stays `process` as proposed; `codec` is the cheap middle ground.
- **7.1** (2026-10-22): `openspec validate song-worker-process` valid; server tests 217 passed and 8 skipped (the real-model ones, without the checkpoint), stable over three runs; library tests without the checkpoint 156 passed, 35 skipped, 0 failed. The README documents the modes with the measured numbers above; the environment alias and the setting are covered by unit and HTTP tests, and the real runs used the default (`process`), `codec` and `inprocess` through `Service:Isolation`.
- **Default changed to `codec`** (2026-10-22, the user's decision after the numbers of 6.1): `ServiceOptions.IsolationMode` falls back to `codec`; spec, proposal, design, README and the two tests that assumed `process` were updated.
