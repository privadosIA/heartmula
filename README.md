# HeartMuLa.Net

> Part of [mediaOrquestator](https://orquestator.xuxo.net/): a song made with this port, and examples of every other engine, are on its public page (es/en).

C# port (.NET 8 + TorchSharp) of [heartlib](https://github.com/HeartMuLa/heartlib), the HeartMuLa music generation model.

Status: backbone, decoder and generation loop (LLM + codebook decoder + sampling) validated against Python with automated tests (C1–C4, see `openspec/`); real tokenizer and Python-compatible text preparation (C5); HeartCodec vocoder, flow matching and windowed decode (C6–C8). `generate --wav` goes from text to a WAV file end to end.

## Using it from your code

Everything runs in .NET: no Python. To call it over HTTP instead of hosting the model in your process, see "Running as a service". In-process there is no sidecar service either. The only runtime inputs are the checkpoint directory and the native libraries that come with the NuGet packages.

**Requirements**

- A checkpoint directory with `HeartMuLa-oss-3B/`, `HeartCodec-oss/`, `tokenizer.json` and `gen_config.json` (the layout of the official download).
- .NET 8 or newer (`RollForward=Major`, so a machine with only .NET 10 runs it).
- The sibling `musicgen-core` repository (see "Shared core and CI").
- One native backend, chosen at build time with `-p:TorchBackend=cpu|cuda-linux|cuda-windows`. The consuming project must be built with the same value as this one, because
  `Directory.Build.props` adds the TorchSharp native package and the tokenizer's native library.
- Memory: about 7.3 GB of VRAM (bf16) for the model plus its KV cache, then 6.3 GB (fp32) for the codec once the model has been released; both do not fit together on a 12 GB card, which is why `AudioStage.Unload` exists. On CPU expect roughly 17x slower than real time.

**Referencing it.** `HeartMuLa.Net` is an executable project, but another project can reference it directly:

```xml
<ProjectReference Include="/path/to/heartmula/src/HeartMuLa.Net/HeartMuLa.Net.csproj" />
```

and be built with `-p:TorchBackend=cuda-linux` (or the one you need). There is no NuGet package yet. Alternatively, run the CLI (`generate --wav`) as a process.

**Minimal example** (compiled and run against the real checkpoint: 2.08 s of audio on CPU; 30 s of music takes about a minute on a 12 GB GPU):

```csharp
using HeartMuLa.Model;
using HeartMuLa.Pipeline;
using MusicGen.Core;
using MusicGen.Core.Audio;
using TorchSharp;
using static TorchSharp.torch;

var ckpt = Environment.GetEnvironmentVariable("HEARTMULA_CKPT")!;
var device = new Device(torch.cuda.is_available() ? "cuda:0" : "cpu");
var dtype = ScalarType.BFloat16;

// 1. Load the language model and the tokenizer.
var mulaDir = Path.Combine(ckpt, "HeartMuLa-oss-3B");
var model = HeartMuLaModel.Create(HeartMuLaConfig.FromFile(Path.Combine(mulaDir, "config.json")), dtype);
model.eval();
model.LoadWeights(mulaDir, dtype, device);
using var tokenizer = new HeartMuLaTokenizer(Path.Combine(ckpt, "tokenizer.json"));
var gen = GenConfig.FromFile(Path.Combine(ckpt, "gen_config.json"));
var pipe = new MusicGenerationPipeline(model, tokenizer, gen, device, dtype);

// 2. Tags and lyrics -> codec frames [8, T]. Seeding.Apply (not torch.manual_seed) seeds CPU and CUDA.
var inputs = pipe.Preprocess("piano,happy", "[verse]\nla la la", cfgScale: 1.5);
Seeding.Apply(1234);
var frames = pipe.Generate(inputs, maxAudioLengthMs: 30_000, cfgScale: 1.5);

// 3. Free the model, then decode the frames with HeartCodec into a 48 kHz stereo WAV.
var framesCpu = frames.cpu().clone();
AudioStage.Unload(model, frames, inputs.Tokens, inputs.TokensMask, inputs.MuqEmbed, inputs.Pos);
var options = new AudioOptions("song.wav", WavFormat.Pcm16, CodecDevice: null, CodecTf32: false);
long samples = AudioStage.DecodeToWav(framesCpu, ckpt, device, options);
```

**Things to know when integrating**

- `Generate` and `DecodeToWav` are synchronous and heavy: run them on a worker, not on a request thread. Both take optional `progress` callbacks (`Generate`: frame `i` of `n`; `DecodeToWav`: codec window `w` of `n`).
- `AudioStage.Unload` disposes the model (needed when the codec shares a 12 GB GPU with it). To generate several songs with one loaded model, call `AudioStage.ReleaseGeneration(model, frames, inputs.Tokens, …)` instead: it frees the caches and the song's tensors and keeps the weights. Put the codec on another GPU (`CodecDevice: "cuda:1"`) or make sure both fit.
- `Pipeline.Generate` prints one summary line to the console; there is no logger abstraction yet.
- A fixed seed on the same checkpoint, inputs and device gives a byte-identical WAV. Another device is not guaranteed to give the same song.
- `Preprocess` lowercases and wraps the tags in `<tag>…</tag>` as heartlib does; pass them as plain comma-separated text. Reference audio (`ref_audio`) is not supported by the open-source checkpoint.
- Keep `AudioOptions.CodecTf32` off unless you want heartlib's behaviour: with TF32 the GPU audio drifts from the CPU's (see the flow-matching section).
- The `HeartMuLa.Net` namespace is not a stable public API yet: types are public so the CLI and the tests can use them, but there is no versioning promise.

Python is only needed to regenerate the test fixtures (`tools/*.py`, once) or to compare by ear with the reference codec; nothing in the build or at runtime calls it.

## Running as a service

`src/HeartMuLa.Net.Server` wraps the library in an HTTP service with a Swagger UI, for callers that should not host TorchSharp, CUDA and the 7 GB checkpoint themselves (an orchestrator, scripts, other machines). It keeps the model loaded between songs and generates **one song at a time**; requests wait in a queue.

```
dotnet run --project src/HeartMuLa.Net.Server -c Release -p:TorchBackend=cuda-linux     # or cpu | cuda-windows; same flag as the rest of the solution
```

With `HEARTMULA_CKPT` pointing at the checkpoint directory, it listens on `http://127.0.0.1:5080`; open `http://127.0.0.1:5080/swagger` for the interactive documentation (the OpenAPI document is `/swagger/v1/swagger.json`). `GET /health` always answers `200`; `"ready":false` (`"model":"loading"`) says the weights are still being read, and jobs sent meanwhile simply wait.

### Calling it

A song takes seconds to minutes, so a request creates a job:

```
curl -s -X POST http://127.0.0.1:5080/songs -H 'Content-Type: application/json' \
     -d '{"tags":"piano,happy","lyrics":"[verse]\nla la la","durationMs":30000,"seed":1234}'
# 202 Accepted, Location: /songs/{id}, body: {"id":"…","state":"queued",…}

curl -s http://127.0.0.1:5080/songs/{id}                       # state: queued | running | succeeded | failed | cancelled, plus progress
curl -s -o song.wav http://127.0.0.1:5080/songs/{id}/audio     # 200 audio/wav once succeeded (409 before)
curl -s -X DELETE http://127.0.0.1:5080/songs/{id}             # cancels a queued/running job (202) or removes a finished one (204)
```

| Field of `POST /songs` | Meaning |
|---|---|
| `tags`, `lyrics` | Required. Tags are comma-separated (`"piano,happy"`); lyrics may use section markers such as `[verse]`. |
| `durationMs` | Optional. Maximum length (the model may stop earlier); default 30 000, maximum 360 000. |
| `seed` | Optional 64-bit integer. Same checkpoint, inputs, seed and devices give the same WAV as `generate --wav`. Chosen by the server if omitted and always returned. |
| `cfgScale` | Optional guidance scale, default 1.5. |
| `temperature`, `topK` | Optional sampling parameters, 0.1 to 2.0 (default 1.0) and 1 to 500 (default 50). |
| `callbackUrl` | Optional absolute `http(s)` URL, see below. |

Errors are `application/problem+json`: `400` names the invalid fields, `404` unknown or expired job, `409` audio requested before the job succeeded, `429` (with `Retry-After`) when `Jobs:MaxQueued` jobs are already waiting, `401` wrong or missing API key.

`progress` is `{"stage":"generating","current":57,"total":375}` while the tokens are produced (frames of 80 ms; `total` is the frame limit, the song may end earlier) and `{"stage":"decoding","current":1,"total":2}` while the codec runs (window `current` of `total`). In `process` isolation (see below) a job first shows `{"stage":"loading"}` while the song's worker process reads the model.

### Replacing the container: the orchestrator-compatible API

The orchestrator talks to its music engines (the HeartMuLa container, DiffRhythm2) with one shared client and one contract. This service speaks that contract too, over the same queue and worker, so pointing the orchestrator at it needs only a new address (and the same token). Its own API above stays available.

| Orchestrator's call | Answer |
|---|---|
| `GET /health` | `{ready, version, busy, queued, gpu: {name, free_mb, total_mb} \| null, missing[]}` plus this service's own fields. Always `200`. `gpu` is read with `nvidia-smi` for the model's device (`null` on CPU). |
| `POST /generate` `{lyrics, tags, max_ms?, seed?, temperature?, topk?, cfg_scale?}` | `202 {"id"}`. |
| `GET /jobs/{id}` | `{id, state: queued\|running\|done\|failed, progress: {done, total}, error, seconds}`. |
| `GET /jobs/{id}/audio` | `200 audio/wav`; `409 {"error": "the job is running"}` before it is done; `410` if the file is gone. |
| `DELETE /jobs/{id}` | `{"ok": true}`: cancels the job if needed and **forgets** it (files included). |

Limits and defaults are the container's, so the same request gives the same answer: `lyrics` ≤ 20 000 characters, `tags` as text or a list (lower-cased, trimmed), `max_ms` 5 000–360 000 (default 240 000), `temperature` 0.1–2.0 (1.0), `topk` 1–500 (50), `cfg_scale` 1.0–10.0 (**3.0**, unlike `/songs`' 1.5), `seed` ≥ 0 (**0**, not random). Errors are `{"error": "..."}` (`400` naming the field, `401 a bearer token is required`, `404 unknown job` / `not found`, `413` over 1 000 000 bytes, `429` when the queue is full). A job made through either API is visible through both.

The container's environment can be reused as it is: `HEARTMULA_CKPT` (checkpoint), `HEARTMULA_TOKEN` (= `Security:ApiKey`), `HEARTMULA_JOBS` (= `Storage:Directory`), `HEARTMULA_VERSION` (= `Service:Version`), `HEARTMULA_ISOLATION` (= `Service:Isolation`) and `HEARTMULA_PORT` (listen on all interfaces on that port, as the container does; the loopback default applies otherwise). The service's own settings win over their aliases, and a port that is not a number stops the service at startup naming the variable. Differences from the container worth knowing: results survive restarts and expire after `Storage:RetentionHours` instead of keeping the last 20, and the GPU memory is given back between songs as the container did only in `process` isolation (next section), where each song is made by a worker process that loads the model and exits; the default, `codec`, gives back the codec's card and keeps the model's.

### Webhook

Instead of polling, give a `callbackUrl`. When the job ends (`succeeded`, `failed` or `cancelled`, including a job cancelled while it waited) the service sends one `POST`:

```
POST {callbackUrl}
Content-Type: application/json
X-HeartMuLa-Event: song.finished
X-HeartMuLa-Signature: sha256=<hex HMAC-SHA256 of the body>      (only when Webhook:Secret is set)

{"id":"…","state":"succeeded","seed":1234,"durationMs":29840,"createdAt":"…","startedAt":"…","endedAt":"…","audioUrl":"/songs/…/audio"}
```

`audioUrl` is relative to the service; `error` is present for `failed`. Any `2xx` answer counts as delivered. Otherwise the call is retried after 1 s, 5 s and 30 s (`Webhook:RetryDelaysSeconds`), each attempt limited to `Webhook:TimeoutSeconds` (10); redirects are not followed. The outcome is in the job (`"webhook":{"state":"pending|delivered|failed","attempts":2,"lastError":"HTTP 500"}`); a failed webhook never changes the job, and the audio can still be downloaded. Deliveries still pending when the service restarts are resumed. Deleting a finished result sends nothing.

Verifying the signature in C# (compare in constant time):

```csharp
var expected = "sha256=" + Convert.ToHexString(HMACSHA256.HashData(Encoding.UTF8.GetBytes(secret), bodyBytes)).ToLowerInvariant();
bool ok = CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(expected), Encoding.ASCII.GetBytes(headerValue));
```

### Settings

In `appsettings.json` or environment variables (`Section__Key`, e.g. `Codec__Device=cuda:1`):

| Setting | Default | |
|---|---|---|
| `Urls` | `http://127.0.0.1:5080` | Loopback only. To serve other machines set e.g. `http://0.0.0.0:5080` **and** an API key. |
| `Model:Checkpoint` / `HEARTMULA_CKPT` | – | Required. Directory with `HeartMuLa-oss-3B/`, `HeartCodec-oss/`, `tokenizer.json`, `gen_config.json`. |
| `Model:Device` | `cuda:0` if CUDA exists, else `cpu` | Where HeartMuLa (bf16, ≈8 GB with its caches) stays loaded. |
| `Codec:Device` | the model's | Where HeartCodec (fp32, ≈6.3 GB) runs for each song. On a single 12 GB card both do not fit together: put the codec on another GPU, or share a card with enough memory. |
| `Codec:Tf32` | `false` | Keep off so GPU audio matches the CPU's (see the flow-matching section). |
| `Storage:Directory`, `Storage:RetentionHours` | `songs`, `24` | Finished songs are WAV + JSON files; they survive restarts and are deleted after the retention time. |
| `Jobs:DefaultDurationMs`, `Jobs:MaxDurationMs`, `Jobs:MaxQueued`, `Jobs:DefaultCfgScale` | 30 000, 360 000, 16, 1.5 | |
| `Security:ApiKey` | empty | When set, every endpoint but `/health` and `/swagger` needs `X-Api-Key: <key>` or `Authorization: Bearer <key>`. |
| `Service:Version` | `1` | Reported by `/health`. |
| `Service:Isolation` | `codec` | `process`, `codec` or `inprocess`: where the work of a song runs, and therefore what stays on the GPUs between songs (next section). |
| `Webhook:Secret`, `Webhook:AllowedHosts`, `Webhook:TimeoutSeconds`, `Webhook:RetryDelaysSeconds` | empty, any host, 10, `[1,5,30]` | `AllowedHosts` lists the hosts a `callbackUrl` may point to. |

A wrong setting (missing checkpoint, a `cuda:3` that does not exist, …) stops the service at startup with a message naming it.

### Giving the GPUs back: `Service:Isolation`

TorchSharp 0.107 has no `empty_cache`, so a process that has used a GPU keeps what it used until it exits. To hand the cards back between songs (the orchestrator shares them with other engines) the work of a song can run in a **worker process**: the service's own executable started with `--worker`, which makes the song, writes the WAV and exits, so the system takes everything back. Three modes:

| `Service:Isolation` | Where the song runs | Between songs the GPUs hold | Cost |
|---|---|---|---|
| `codec` (default) | generation in the service (model stays loaded), only the codec decode in a worker | the model, on its card (≈10 GB); the codec's card is free | about a second per song over `inprocess` |
| `process` | the whole song, model load included, in a worker process | nothing: both cards are back at their baseline | the model is read for every song (about 18 s more) |
| `inprocess` | everything in the service process | model **and** the codec's cache: ≈10 GB and ≈13.5 GB, until the service exits | none |

Measured on this project's machine (RTX 4070 12 GB for the model, RTX 5060 Ti 16 GB for the codec, a 2 s song, the codec is also read for every song in every mode):

| | song time | model card 0 after the song | codec card 1 after the song |
|---|---|---|---|
| baseline (nothing running) | | 227 MiB | 13 MiB |
| `process` | 31 s (first and second the same; about 18 s more than `codec`, which is reading the model plus starting the process) | **227 MiB** | **13 MiB** |
| `codec` | 15 s first, 14 s second | 10 333 MiB (the model) | **16 MiB** |
| `inprocess` | 15 s first, 13 s second | 10 333 MiB | 13 561 MiB |

The WAV is byte-identical to `generate --wav` for the same inputs, seed and devices in `process`, `codec` (with the model and the codec on different cards, as above) and `inprocess`. In `codec` with both on the same card, the decode noise continues the generator the sampling used in the CLI but starts fresh in the worker, so that WAV differs from the CLI's (it is still fully determined by the request).

Cancelling or forgetting a running job kills its worker process: in `process` mode the model card went from 10 333 MiB back to its baseline in 0.23 s. A worker that crashes or is killed makes the job `failed` ("the song process ended unexpectedly (exit code N)", or the cause it printed) and the next job runs. A worker whose service disappears stops by itself within seconds. `GET /health` in `process` mode says `ready` when the checkpoint files are there and the configured cards exist, with `"model": "on-demand"`; nothing is loaded in the service process, which does not touch the GPUs at all.

Which mode to use is a trade-off you can change with one setting: `codec` (the default) returns the card that held the codec (the larger footprint, ≈13.5 GB) for about a second per song and keeps the model resident; `process` returns everything but adds the model load to every song; `inprocess` is the fastest and keeps both cards occupied while the service runs.

### Things to know

- In `codec` (the default) and `inprocess` isolation the model stays on its card for as long as the service runs, and in `inprocess` the codec's card stays occupied too (see the table above). Do not share those cards with other heavy jobs while the service runs in those modes; stop it, or use `process`. In `process` mode `GET /health` is `ready` as soon as the files and cards are there (the model is loaded for each song).
- Cancelling a running job ends its worker process at once (`process` and `codec`), or stops it within one frame while it is generating or within one codec window while it decodes (`inprocess`).
- A job that fails (for example a CUDA out-of-memory) is reported as `failed` and the next one runs. Jobs that were queued or running when the service stopped come back as `failed` with the reason `interrupted by restart`.
- Anyone who can reach the service can make it call arbitrary URLs through `callbackUrl`. Keep it on loopback, or set the API key and `Webhook:AllowedHosts`.
- There is no `Dockerfile`; this project runs it as a plain process.

### Deploying with systemd

```
dotnet publish src/HeartMuLa.Net.Server -c Release -p:TorchBackend=cuda-linux -o /opt/heartmula-server
```

```ini
# /etc/systemd/system/heartmula.service
[Unit]
Description=HeartMuLa song service
After=network.target

[Service]
WorkingDirectory=/opt/heartmula-server
ExecStart=/usr/bin/dotnet /opt/heartmula-server/HeartMuLa.Net.Server.dll
Environment=HEARTMULA_CKPT=/srv/heartmula-models
Environment=Model__Device=cuda:0
Environment=Codec__Device=cuda:1
Environment=Storage__Directory=/var/lib/heartmula/songs
EnvironmentFile=-/etc/heartmula.env          # Security__ApiKey=…  and  Webhook__Secret=…  (keep it readable by root only)
User=heartmula
Restart=on-failure
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

The tests of the service (`tests/HeartMuLa.Net.Server.Tests`) run on a stand-in for the model, in a few seconds and without a GPU or the checkpoint: `dotnet test tests/HeartMuLa.Net.Server.Tests -p:TorchBackend=cpu`.

## Plan

| Step | What | Status |
|---|---|---|
| 1 | HeartCodec in isolation (tokens → wav) | done (C6 vocoder, C7 flow matching, C8 windowed decode) |
| 2 | Llama 3.2 3B backbone + 300M decoder, weights loaded, logits = Python | **this repo** |
| 3 | Full generation loop with CFG | validated (C4): sampling identical to Python given its logits and identical first frames; see "Generation" |
| 4 | Text tokenizer in .NET | done (C5): HuggingFace library + Python-style lowercasing and file reading; see "Text and tokenizer" |
| 5 | HeartCodec in C# (flow matching + vocoder) | done (C6–C8); see the HeartCodec sections below |

The order recommended in the conversation was 1 → 2 → 3, but 2 made the most sense to start with because it is almost a direct translation and it is where the weight loader and the Llama architecture are validated.

## Structure

```
HeartMuLa.Net.sln
Directory.Build.props         TorchBackend (cpu by default | cuda-windows | cuda-linux), TorchSharpVersion, MusicGenCoreDir
tests/HeartMuLa.Net.Tests/    xUnit; [RequiresCheckpoint] reads HEARTMULA_CKPT; tests/fixtures/ (golden run, sha256 in MANIFEST.md)
src/HeartMuLa.Net/
  HeartMuLa.Net.csproj        net8.0, TorchSharp, project reference to the core
  Program.cs                  CLI: inspect | check | generate
  Codec/Vocoder.cs            HeartCodec vocoder (scalar_model.decoder): latents → 48 kHz stereo audio, built from the core's conv layers
  Codec/CodecWeights.cs       StrictLoader adapter: prefix scalar_model.decoder., weight_norm folding
  Codec/FlowMatching.cs, Dit*.cs  HeartCodec flow matching: RVQ lookup, DiT estimator, Euler/CFG solver
  Codec/FlowMatchingWeights.cs    StrictLoader adapter: prefix flow_matching., allow-list of RVQ training state
  Codec/HeartCodecRope.cs     HeartCodec's rotary embedding: adjacent pairs, base 10000, cos/sin per call in the input dtype
  Codec/DetokenizePlan.cs, HeartCodecDecoder.cs  windowed decode of codes of any length (in-context windows; the core's WindowAssembler cross-fades)
  Pipeline/HeartMuLaTokenizer.cs  real tokenizer (HuggingFace) over the checkpoint's tokenizer.json
  Pipeline/AudioStage.cs      `generate --wav` options, model unloading and codec decode
  Pipeline/MusicGenerationPipeline.cs  preprocess (prompt) + frame loop
  Pipeline/ITextTokenizer.cs  interface + "precomputed ids" tokenizer for tests
  Model/Llama.cs              GQA attention + KV cache, FFN, TransformerDecoder (torchtune), on the core's RmsNorm and RotaryEmbedding
  Model/HeartMuLaModel.cs     HeartMuLa: embeddings, heads, BackboneLogits, GenerateFrame, top-k sampling; LoadWeights = StrictLoader adapter
src/HeartMuLa.Net.Server/     HTTP service: job queue + worker (model loaded once), webhook, Swagger; see "Running as a service"
tests/HeartMuLa.Net.Server.Tests/  its tests, on a fake model (no GPU, no checkpoint)
tools/dump_reference.py       golden run (Python, once)
tools/dump_text_reference.py  text fixtures (Python, once)
tools/decode_frames.py        decode .NET frames with the Python codec (reference for listening comparisons)
tools/update_public_branch.sh sanitized public snapshot, vendoring the core
```

## API reference

The public API is documented with XML comments (`///`, the C# counterpart of Javadoc). They show up as tooltips in the editor and feed an HTML
reference built with [DocFX](https://dotnet.github.io/docfx/), pinned as a local tool in `.config/dotnet-tools.json`:

```bash
dotnet tool restore
TorchBackend=cpu dotnet docfx docs/docfx.json --serve   # builds docs/_site and serves it on http://localhost:8080
```

`docs/api` and `docs/_site` are generated and ignored by git. The home page is `docs/index.md`; the rest comes from the source, so a new public member
needs a `<summary>` (the compiler warns with CS1591 when it is missing). The CI builds the site on every push and fails the step on broken comments.

## Shared core and CI

The model-agnostic code (safetensors, strict loader, rotary embedding, norms, convolutions, window assembly, WAV writer, seeding, test helpers) lives in
the sibling repository `musicgen-core`. `Directory.Build.props` finds it at `../musicgen-core` (or `third_party/musicgen-core` in the public snapshot);
override it with `-p:MusicGenCoreDir=/path/to/musicgen-core`. Both repositories must use the same `TorchSharpVersion` (a test checks it).

`.github/workflows/ci.yml` checks out both repositories side by side (`heartmula/` and `musicgen-core/`), builds and tests this solution and then the
core's own. Configure it once in the repository settings (Settings → Secrets and variables → Actions):

What lives in the core (namespaces `MusicGen.Core.*`):

| Namespace | Types |
|---|---|
| `Weights` | `SafeTensors`, `CheckpointIndex`, `SafeTensorsWriter`, `StrictLoader`, `WeightNorm` |
| `Layers` | `RmsNorm`, `FlowRmsNorm`, `RotaryEmbedding`, `FlowAttention`, `FlowMlp`, `Conv1dLayer`, `ConvTranspose1dLayer`, `PReLULayer` |
| `Audio` | `WavWriter`, `WindowAssembler` |
| `Text` | `PythonText` (+ generated tables) |
| `MusicGen.Core` | `Seeding` |
| `MusicGen.Core.Testing` | `Parity`, `FixtureDirectory`, `CheckpointEnvironment`, `Requires*FactAttribute`, `Bf16`, `Rng`, `WavFile` |

Three things that look alike but are not interchangeable (the compiler cannot tell you):

- **RMSNorm**: `RmsNorm` is torchtune's (parameter `scale`, computed in float32 and cast back); `FlowRmsNorm` is HeartCodec's (parameter `weight`, computed in the input dtype). HuggingFace's `LlamaRMSNorm` (`weight`, float32) is a third one that is not in the core yet.
- **RoPE**: `RotaryEmbedding` makes the choice explicit and has no default layout: `RopeLayout.AdjacentPairs` (torchtune, used here) or `HalfRotation` (HuggingFace `rotate_half`), the frequency source (`Plain` or `Llama3Scaled`) and the table (`Float32Cached` for HeartMuLa, `InputDtypePerCall` for HeartCodec's flow transformer).
- **Seeding**: TorchSharp's `torch.manual_seed` seeds only the CPU generator. Always seed through `Seeding.Apply`, which also seeds CUDA (without it two generations with the same seed differed in 126 of 128 tokens). See the core's `docs/torchsharp-findings.md`.

- **Variable `MUSICGEN_CORE_REPO`** (required): `owner/name` of the core repository. The first step fails with a clear message when it is missing.
- **Secret `MUSICGEN_CORE_TOKEN`** (only if the core repository is private): a token with read access to it. Without it the default token is used.

## How to validate (step 2)

1. The fixtures are already versioned in `tests/fixtures/` (see `tests/fixtures/MANIFEST.md`: versions, commands and sha256).
   They only need regenerating if the checkpoint or the script changes. The environment and the checkpoints live outside the repo
   (`~/.venvs/heartlib-golden`, heartlib in `~/Proyectos/heartlib`, weights in `~/Proyectos/heartmula-models`); the script has one subcommand per group:
   ```
   ~/.venvs/heartlib-golden/bin/python tools/dump_reference.py {prompt|rope|frames|tokenizer|codec}
   ```

2. Build. Choose the native backend with `-p:TorchBackend=cuda-windows|cuda-linux|cpu`:
   ```
   dotnet build HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu   # or cuda-windows | cuda-linux
   dotnet test  HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu
   ```

   `[RequiresCheckpoint]` tests are skipped unless `HEARTMULA_CKPT` points to an existing directory. It must be the directory that contains
   `HeartMuLa-oss-3B/` and `HeartCodec-oss/` (the same one passed to `--ckpt`):
   ```
   HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" dotnet test HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu
   ```
   (in fish: `set -x HEARTMULA_CKPT $HOME/Proyectos/heartmula-models`). With it the loading and parity tests against HeartMuLa-oss-3B run too.
   They all share a single loaded model (the `RealModel` collection), so the model-related part of the suite takes about 3 minutes on CPU and the peak memory
   of the load (the `PeakMemory` test) is measured once, with no other model alive.

3. Parity with Python (`ModelParityTests`, `RopeParityTests`; they need `HEARTMULA_CKPT`, except RoPE). They compare against the golden run in
   `tests/fixtures/` with tolerances `cos_sim ≥ 0.999` and `max_abs ≤ 0.5` on bf16 logits (RoPE: `1e-4` in F32 and 1 ulp in bf16):

   | Test | What it checks |
   |---|---|
   | `RopeParityTests` | torchtune RoPE: F32 at positions 0..15 and 4000..4015, and bf16. No checkpoint. |
   | `Backbone_FirstFrame_MatchesPython` | `last_h` and codebook 0 logits of the prompt (batch 2, CFG). |
   | `Backbone_TeacherForcing_MatchesPythonOnAll126Calls` | All 126 calls feeding Python's frames back in, with the KV cache growing; conditional and unconditional rows separately, before CFG. |
   | `Decoder_TeacherForced_MatchesPython` | Logits of codebooks 1–7 with Python's `last_h` and tokens (6 frames, `decoder_logits.safetensors`). |

   ```
   HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" dotnet test HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu --filter "FullyQualifiedName~Parity"
   ```
   To run them on the GPU (with the CUDA backend): `HEARTMULA_DEVICE=cuda dotnet test ... -p:TorchBackend=cuda-linux`. The peak-memory test only
   applies on CPU and is skipped with `HEARTMULA_DEVICE=cuda`. There is no fp32 parity of the whole model: the backbone fixtures are CPU bf16.

   `check` remains as a manual diagnostic; it prints `cos_sim` and `max_abs` of the first frame:
   ```
   dotnet run --project src/HeartMuLa.Net -c Release -p:TorchBackend=cpu -- check --ckpt ~/Proyectos/heartmula-models --reference tests/fixtures --dtype bf16
   ```

4. Generation (`SamplingParityTests`, `GenerationLoopTests`, `GenerationParityTests`). The loop is a stochastic process, so parity is
   measured in parts:

   | Test | What it checks | Needs checkpoint |
   |---|---|---|
   | `SamplingParityTests` | With Python's logits, seed 1234 and the same RNG consumption as heartlib, sampling (CFG + top-k + temperature) picks **the same tokens**: 126/126 for codebook 0 and 42/42 for codebooks 1–7 (frames 0, 1, 2, 63, 124, 125). | no |
   | `GenerationLoopTests` | Invariants with a small model: shape `[K, N+1]`, Int64, range, same seed → same output, equal CFG rows, stop at EOS (the EOS frame is not returned) and maximum length. | no |
   | `GenerationParityTests` | Full loop with HeartMuLa-oss-3B: prompt identical to Python's, first 2 frames identical to `frames_py` (3 in practice), report of the first divergence, diagnosis of that divergence and invariants with the real model. | yes (CPU, except the invariants) |

   ```
   HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" dotnet test HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu --filter "FullyQualifiedName~Generation|FullyQualifiedName~SamplingParity"
   ```
   The whole filter takes about 4 minutes on CPU; the 126-frame loop takes about 3 (RTF ≈ 17, against ~12 for Python on the same machine; optimising is not a goal yet).

   **Why exact equality of the 126 frames is not expected.** With the same seed libtorch's CPU RNG matches Python's (which is why the first 3
   frames are identical), but a minimal logit difference in bf16 can change the chosen token and, since each token is fed back into the next
   step, the error propagates: at frame 3, codebook 5, C# picks 3908 and Python 7232, and from there to the end only 42 of 1008 tokens agree. The diagnosis
   (`FirstDivergence_Frame3Codebook5_…`) shows that with Python's inputs Python's token is the most probable one (p=0.097) and C#'s is the fourth
   (p=0.059), and that the exponential noise tips the balance by only 0.064 in `log(p/q)`: one bf16 ulp in one logit. It is a technical tie, not a failure.
   So the criterion is: identical sampling when the inputs are identical, loop invariants and a reproducible prefix, and not equality of the 126 frames.
   The generation tests do not run in parallel because libtorch's RNG is global.

5. Generate frames and listen to them with the reference codec:
   ```
   dotnet run --project src/HeartMuLa.Net -c Release -p:TorchBackend=cpu -- generate --ckpt ~/Proyectos/heartmula-models --tags tests/fixtures/inputs/tags.txt --lyrics tests/fixtures/inputs/lyrics.txt --ms 30000 --out frames.safetensors
   python tools/decode_frames.py --model_path ./ckpt --frames frames.safetensors --out out.wav
   ```
   (`generate --wav song.wav` does the decoding in C#; see "Windowed decode and `generate --wav`".)

   **Seed and reproducibility.** `generate` accepts `--seed N` (64-bit integer). If omitted it picks a random one and **always prints it** on the first line
   (`seed=N`) and stores it, together with `cfg_scale`, `max_audio_length_ms`, `temperature`, `topk`, `dtype`, `device` and the origin of the prompt, in the metadata of
   `frames.safetensors` (read with `safe_open(path, framework="numpy").metadata()` from Python's `safetensors` library). To repeat a generation just
   run again with that `--seed` and the same inputs:
   ```
   dotnet run --project src/HeartMuLa.Net -c Release -p:TorchBackend=cpu -- generate --seed 1234 --ckpt ~/Proyectos/heartmula-models --tags tests/fixtures/inputs/tags.txt --lyrics tests/fixtures/inputs/lyrics.txt --ms 400 --out frames.safetensors
   ```
   - Same seed, same inputs, same machine and same device → the file comes out byte-identical. With another seed the frames change.
   - With `--seed 1234` and the reference prompt, on CPU, the first 3 frames match `tests/fixtures/frames_py.safetensors` (the golden run seeded 1234 by hand;
     heartlib's example offers no seed). From frame 3 they diverge because of bf16 noise (see "Generation").
   - Equality between CPU and CUDA, between different GPUs or drivers, or between machines is **not guaranteed**.
   - The seed is applied right before generating (after loading the model and preparing the prompt) and seeds the CPU **and** all GPUs: TorchSharp's `torch.manual_seed`
     only seeds the CPU, unlike PyTorch in Python (`Seeding.Apply`).

## Text and tokenizer

`generate` tokenizes with `<ckpt>/tokenizer.json` through the HuggingFace library (`Tokenizers.DotNet` wraps the same Rust library Python uses),
so the ids are exactly heartlib's: the tokenizer prepends BOS and the pipeline appends EOS. `--ids prompt_ids.json` uses the precomputed ids of the
golden run instead (`JsonIdsTokenizer`); with seed 1234 both paths generate identical frames.

**Native library per platform.** The `Tokenizers.DotNet` package does not ship the native library: `Directory.Build.props` adds `Tokenizers.DotNet.runtime.<rid>` according to the
operating system and architecture, and `-p:TokenizerRuntime=<rid>` forces it (to build for another machine).

| RID | Verified |
|---|---|
| linux-x64 | running (build, tests, `generate`) |
| linux-arm64, win-x64, win-arm64, osx-x64, osx-arm64 | only `dotnet restore` (exit code 0, no warnings) and that the package contains the native library (`libhf_tokenizers.so`, `hf_tokenizers.dll`, `libhf_tokenizers.dylib`); **not run there** |

**Text is prepared as in heartlib**, not as .NET would do by default:
- **Lowercasing** (`PythonText.Lower`) like Python's `str.lower()`: `ΑΣ` → `ας` (final sigma), `İ` → `i̇`. `ToLowerInvariant` gave `ασ` and left the `İ`, and it also uses ICU, so
  its result changes with each system's ICU version. Tables generated from Python 3.11 (Unicode 14.0) are used: same result on Linux, Windows and macOS; characters
  added after Unicode 14 do not change to lowercase, as in that Python version. `PythonText` now lives in the shared `musicgen-core` repository, whose `tools/gen_python_text_data.py` regenerates the tables from
  `tests/fixtures/lower_reference.json`, and a test there walks all 1 112 064 code points comparing with Python.
- **File reading** (`PythonText.ReadAllText`) like `open(path, encoding="utf-8").read()`: strict UTF-8, `\r\n` and `\r` become `\n` (with CRLF on Windows, `File.ReadAllText` gave different ids)
  and a BOM is kept as U+FEFF.
- **`<tag>`/`</tag>`** are checked with ordinal comparisons: `StartsWith(string)` is culture-dependent and, with ICU, ignores a zero-width space before `<tag>`.

New fixtures (`tests/fixtures/text_cases.json`, `lower_reference.json`): 28 text cases and 8 tags+lyrics pairs with Python's ids, and the exhaustive lowercase dump.
They were generated once by `tools/dump_text_reference.py` in a minimal venv with only `tokenizers==0.22.1` (recipe in `tests/fixtures/MANIFEST.md`). The tests that use the real
tokenizer carry `[RequiresCheckpoint]` because `tokenizer.json` (9 MB) is not in the repo:
```
HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" dotnet test HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu --filter "FullyQualifiedName~Tokenizer|FullyQualifiedName~PythonText|FullyQualifiedName~Preprocess"
```

## HeartCodec vocoder

`ScalarVocoder` (`Codec/`) is the second half of the codec: it turns the latents `[B, 128, T]` into audio `[B, T·1920]` (25 Hz × 1920 = 48 kHz; the batch of 2 is the two
stereo channels, decoded as mono). The latents are produced by the flow matching (next sections); here the vocoder is tested on its own, with the golden run's reference latents (`decode_in_seg0`).

- **Strict loading** (`CodecWeights.LoadVocoder`): only the 224 `scalar_model.decoder.*` keys (335 MB of the checkpoint's 6.2 GB; the `scalar_model` encoder and the
  `flow_matching` are not even read). It validates keys and shapes from the headers alone and fails with a single error that lists everything. It folds the 57 `weight_norm` pairs
  (`parametrizations.weight.original0/1`, `w = v·g/‖v‖`) into plain weights, in fp32.
- **Always fp32** (the audio degrades in bf16). Parameter names replicate the keys of the folded checkpoint, with no mapping table.
- **Parity** against Python's audio (`codec_wav.safetensors`, 483 840 samples, both channels, peak 0.609), measured:

  | Device | `cos_sim` | `max_abs` | signal/error | decoding the window (744 frames) |
  |---|---|---|---|---|
  | CPU (x86, 20 threads) | 1.0000000 | 1.0e-6 | 121 dB | 9.3 s (peak 3.7 GB of RAM) |
  | RTX 4070, default TF32 | 0.9999999 | 4.1e-4 | 66 dB | 0.3 s warm (≈ 8.8 GiB of VRAM) |
  | RTX 5060 Ti, default TF32 | 0.9999999 | 3.7e-4 | 66 dB | 0.4 s warm (≈ 9.3 GiB of VRAM) |
  | any GPU, TF32 off | 1.0000000 | 8.2e-7 | 122 dB | — |

  On GPU the error is ~400 times larger because of **TF32**: cuDNN convolutions use it by default on Ampere and later GPUs (and so does heartlib's Python). It is
  noise at −66 dB, but for the same result as on CPU it is enough to set `torch.backends.cudnn.allow_tf32 = false`. The CPU tests require `cos_sim ≥ 0.999999`,
  `max_abs ≤ 4e-6` and 100 dB; the CUDA ones `0.9999`, `1e-3` and 60 dB, and also that with TF32 off the GPU equals the CPU.
- **WAV writer** (`WavWriter`): 16-bit PCM (clips at ±1, rounds to nearest) or 32-bit float, interleaved channels; validated with `file`, Python's `wave` module and `ffprobe`.

```
HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" dotnet test HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu --filter "FullyQualifiedName~Vocoder|FullyQualifiedName~CodecWeights|FullyQualifiedName~WeightNorm|FullyQualifiedName~WavWriter"
HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" HEARTMULA_DEVICE=cuda:0 dotnet test HeartMuLa.Net.sln -p:TorchBackend=cuda-linux --filter "FullyQualifiedName~VocoderParity"   # on the GPU
```

**Audio for listening.** The tests do not write audio by default. With `HEARTMULA_AUDIO_DIR` they write, in 16-bit PCM and with the same format, the audio of our vocoder and Python's
reference to compare them by ear:
```
HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" HEARTMULA_AUDIO_DIR="$HOME/heartmula-audio" dotnet test HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu --filter "FullyQualifiedName~AudioForListening"
# → c6_vocoder_from_reference_latents.wav and c6_reference_python.wav
```

## HeartCodec flow matching

`FlowMatching` (`Codec/`) is the first half of the codec: it turns the 8 RVQ code streams `[1, 8, T]` (12.5 Hz) into the latents `[1, 2T, 256]` that the vocoder consumes,
and `HeartCodecDecoder.DecodeSegment` chains it with the stereo reordering and the vocoder, so **codes → audio works for one segment** (up to 372 frames, 29.76 s). `generate --wav` and the windowed decode for any length are described in the next section.

- **What it does**: RVQ lookup (the 8 per-layer code vectors are summed, then `project_out`), `cond_feature_emb`, nearest ×2 upsampling in time, masks with
  `zero_cond_embedding1`, and an Euler solver (10 steps, classifier-free guidance 1.25, batch of 2) around a 30-layer DiT estimator (24 blocks of width 1536 and 6 of width 3072,
  AdaLN-single, RoPE, SwiGLU). The in-context path (`incontextLength > 0`) is checked against real data in the windowed decode below.
- **Strict loading** (`FlowMatchingWeights.Load`): 339 of the 365 `flow_matching.*` tensors (5.96 GB in fp32, 4.8 s on this machine). Keys and shapes are validated from the headers
  alone, with a single error listing every discrepancy. The 26 RVQ training-state tensors (`embed_avg`, `cluster_size`, `initted`, `project_in`) are skipped through an explicit
  allow-list, so any other key the model does not have is still reported. Parameter names are the checkpoint keys, with no mapping table.
- **Parity** against the Python golden run (`codes_seg0` plus the recorded noise `randn_1_inference_codes` → `latents_seg0`, then the whole chain → `codec_wav`), measured:

  | Device | latents `max_abs` / SNR | 10 Euler steps | wav `max_abs` / SNR (483 840 samples) | peak memory |
  |---|---|---|---|---|
  | CPU (x86, 20 threads) | 4.9e-6 / 118.7 dB | 43 s | 1.0e-6 / 121.3 dB (59 s for the whole chain) | 9.9 GB RAM |
  | RTX 4070, defaults | 5.5e-3 / 68.6 dB | 3.5 s | 3.7e-2 / 36.1 dB | 10.8 GiB of 12 (tight) |
  | RTX 5060 Ti, defaults | 5.5e-3 / 68.5 dB | 4.2 s | 3.6e-2 / 38.1 dB | 15.3 GiB of 16 |
  | any GPU, cuDNN TF32 off | 6.5e-6 to 1.2e-5 / 114 to 117 dB | same | 8.2e-7 / 122 dB | same |

  The cause on GPU is **cuDNN's TF32 convolutions** (on by default on Ampere and later, as in heartlib's Python); cuBLAS TF32 is already off by default. The 5.5e-3 error in the
  latents is noise at −68 dB, but it grows through the vocoder to −36 dB. For the same result as on the CPU set `torch.backends.cudnn.allow_tf32 = false`. The tests require, on CPU,
  `cos_sim ≥ 0.999999`, `max_abs ≤ 2e-5` and 100 dB for the latents (`4e-6` for the wav); on CUDA with the defaults `0.99999`, `2e-2` and 60 dB (wav: `0.999`, `0.15`, 30 dB); and that
  with cuDNN TF32 off the GPU matches the CPU.
- **Memory**: 5.96 GB of weights plus the vocoder (0.3 GB) do not fit next to the 7.3 GB HeartMuLa model on a 12 GB GPU, so `generate --wav` unloads HeartMuLa before loading the codec.

```
HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" dotnet test HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu --filter "FullyQualifiedName~Flow|FullyQualifiedName~Dit|FullyQualifiedName~LatentReorder"
HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" HEARTMULA_DEVICE=cuda:0 dotnet test HeartMuLa.Net.sln -p:TorchBackend=cuda-linux --filter "FullyQualifiedName~FlowMatchingParity"   # on the GPU
```

**Audio for listening.** With `HEARTMULA_AUDIO_DIR` set, a test writes the first audio produced from tokens (`c7_codes_to_wav_csharp.wav`) next to the Python reference
(`c6_reference_python.wav`, same format; at most 1 PCM16 step apart):
```
HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" HEARTMULA_AUDIO_DIR="$HOME/heartmula-audio" dotnet test HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu --filter "FullyQualifiedName~AudioFromTokens"
```

## Windowed decode and `generate --wav`

`HeartCodecDecoder.Decode` decodes codes of **any length**, as heartlib's `detokenize`: windows of 372 code frames (29.76 s) with a hop of 320 (overlap 52 frames), codes shorter than a window
repeated up to one window and longer ones repeated up to the next `52 + k·320` frames (`DetokenizePlan`); each window after the first receives the last 104 latent frames of the previous one as
in-context latents (completed with fresh noise) and the audio of consecutive windows is joined with a linear cross-fade over 199 680 samples (`WindowAssembler`) and trimmed to
`int(frames / 12.5 · 48000)`. Memory stays at one window: a 10-minute song is 17 windows.

`generate` writes the audio when asked:

```
HeartMuLa.Net generate --ckpt ~/Proyectos/heartmula-models --tags tags.txt --lyrics lyrics.txt --ms 30000 --seed 1234 --wav song.wav
   [--wav-format pcm16|float32] [--codec-device cpu|cuda:N] [--codec-tf32]
```

- After the token loop HeartMuLa is **released** (parameters, KV caches, masks and prompt tensors) and HeartCodec is loaded in fp32, on the model's device unless `--codec-device` says otherwise.
  They do not fit together on a 12 GB GPU (7.3 GB bf16 + 6.3 GB fp32). TorchSharp 0.107 has no `cuda.empty_cache()`; freed blocks stay in libtorch's caching allocator, which reuses them.
  `frames.safetensors` is still written first, so a codec failure never loses the tokens.
- The codec runs with **cuDNN TF32 off** by default and restores the setting afterwards: with TF32 a GPU's audio is at about 38 dB against the CPU's, without it GPU and CPU agree (see the flow-matching section).
  `--codec-tf32` opts back in (heartlib's default behaviour).
- The WAV is PCM16 stereo 48 kHz unless `--wav-format float32`. Invalid values, or codec options without `--wav`, fail in milliseconds with exit code 2, before any model is loaded.
- Same checkpoint, inputs, seed and device give a **byte-identical** WAV; another seed gives another one. The noise of the codec comes from the seeded stream, after generation.

**A long song.** `--ms` is a maximum, not a length: the model stops by itself when it emits the end-of-sequence token, and with little text it does so early. With the 160-word lyrics of the
test fixtures `--ms 240000` ended at 134 s (1 672 frames); a song of the full length needs lyrics with enough verses and choruses (355 words reached 240 s). Measured on the machine of
this repository (seed 42, `--ms 240000`, HeartMuLa on the RTX 4070 and the codec on the RTX 5060 Ti with `--codec-device cuda:1`, TF32 off):

| | frames / audio | HeartMuLa | whole command | peak VRAM (model card / codec card) | peak host RAM |
|---|---|---|---|---|---|
| fixture lyrics (160 words) | 1 672 / 133.8 s | 118 s (RTF 0.88) | 176 s | 10.2 GiB / 13.2 GiB | 8.2 GiB |
| long lyrics (355 words) | 2 999 / 239.9 s | 203 s (RTF 0.85) | 276 s | 10.4 GiB / 13.2 GiB | 7.3 GiB |

The memory does not grow with the length: the codec card peaks at the same 13.2 GiB with 6 windows and with 10, and the model card changes by 0.2 GiB between 134 s and 240 s (VRAM as `nvidia-smi` reports it, including the CUDA context). The
model card's memory stays reserved after HeartMuLa is released (libtorch's caching allocator, as above), so do not count on it for other programs while the process runs.
With `--codec-tf32` the same frames take 273 s in all (no measurable gain) and the audio differs from the TF32-off one (29 dB signal-to-error between the two files, listened to only informally),
which is why it stays off.

Measured (heartlib golden run of 400 frames = 2 windows, x86 CPU fp32; `tools/dump_reference.py codec_long`):

| | latents window 0 | latents window 1 (in-context from our own window 0) | audio | time (2 windows) |
|---|---|---|---|---|
| CPU | 1.2e-6 / 124.9 dB | 5.8e-6 / 123.0 dB | 125.0 dB from the reference's own vocoder inputs; see below for the whole chain | 105 s |
| RTX 4070, TF32 off | 1.6e-5 / 113.2 dB | 4.5e-5 / 108.3 dB | 120.3 dB | 8.1 s |
| RTX 5060 Ti, TF32 off | 1.3e-5 / 115.6 dB | 3.3e-5 / 111.2 dB | 120.3 dB | 9.9 s |
| GPU, TF32 on | 5.4e-3 / 68.7 dB | 2.0e-3 / 70.4 dB | 37.8 dB (4070), 40.7 dB (5060 Ti) | 8 to 9 s |

(`max_abs` / signal-to-error ratio.) The window-1 numbers are the first check of the in-context path against real data. The in-context frames come back exactly, and window 1 fed with the reference's
window-0 tail gives 124.0 dB.

**The vocoder is discontinuous.** Its first operation is `round(9·x)/9`, so a latent error of 1e-6 (the float32 noise floor) occasionally moves one of the ~190 000 values per window across a rounding
boundary and changes the audio locally. On the CPU run above exactly one value flips versus Python (window 0, frame 358), which changes the audio by up to 4e-3 for 0.8 s (65.9 dB overall); everything else is
below 1e-5, and the reference's own inputs through our vocoder and assembler give 125 dB, cross-fade included. Python on another CPU shows the same effect, so the tests assert the latents and the
reference-input path exactly and the whole-chain audio by "few flips, global agreement, deviations only near a flipped frame".

Real run, `generate --wav` with the model and the codec on the **RTX 4070 alone** (12 GB), 30 s of music: 57 s in total including both loads, peak VRAM 10 965 MiB sampled with `nvidia-smi`.

```
HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" dotnet test HeartMuLa.Net.sln -c Release -p:TorchBackend=cpu --filter "FullyQualifiedName~DetokenizePlan|FullyQualifiedName~WindowAssembler|FullyQualifiedName~HeartCodecDecoder|FullyQualifiedName~AudioOptions|FullyQualifiedName~LongDecodeParity"
HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" HEARTMULA_DEVICE=cuda:0 dotnet test HeartMuLa.Net.sln -p:TorchBackend=cuda-linux --filter "FullyQualifiedName~LongDecodeParity"   # on the GPU
HEARTMULA_CKPT="$HOME/Proyectos/heartmula-models" HEARTMULA_AUDIO_DIR="$HOME/heartmula-audio" dotnet test HeartMuLa.Net.sln -p:TorchBackend=cpu --filter "FullyQualifiedName~LongAudioForListening"   # c8_reference_long_{csharp,python}.wav
```

## Weight loading

`HeartMuLaModel.Create(config, dtype)` builds the model already in the target dtype (otherwise the ~3.9 B parameters are allocated first in fp32, ~14.7 GiB)
and `LoadWeights` fills them from the checkpoint (on disk it is F32) with these guarantees:

- **Strict**: before reading any tensor data it compares the model's keys and shapes with the checkpoint's **headers**. If anything does not match
  it throws a single `InvalidDataException` that lists all the missing keys, the extra ones and the different shapes (model vs checkpoint), without having
  modified the model. There is no `strict: false` and no silently ignored keys.
- **Streaming**: it copies tensor by tensor under `no_grad`, converting each one to the requested dtype (F32→BF16 with round-to-nearest-even). Memory only
  holds the model and one tensor in transit; the peak does not exceed 1.5 × the model size in the target dtype (measured on CPU, bf16:
  7.34 GiB of weights, peak 9.78 GiB = 1.33 ×, 16 s with the checkpoint in the disk cache).
- **Consistent index**: `CheckpointIndex.Open` checks that each shard contains exactly the keys that `model.safetensors.index.json` assigns to it,
  and rejects dtypes it does not support, naming the tensor and the dtype.

## Where it will hurt (list of suspects if `check` fails)

- **RoPE**: torchtune rotates adjacent pairs `(x[2i], x[2i+1])`, not `rotate_half` as HuggingFace does. It also applies Llama 3's frequency scaling (`scale_factor=32`). It is `RotaryEmbedding` in the core, configured in `Attention` as `AdjacentPairs` + `Llama3Scaled` + `Float32Cached`.
- **RMSNorm**: computes in float32 and the parameter is called `scale`.
- **Mask**: bool with `true = attend`, shape `[b, s, max_seq]` indexing the causal one with `input_pos`; it is passed to SDPA with `unsqueeze(1)`.
- **KV cache**: here it is written by absolute position (`index_copy_` at `input_pos`); torchtune uses an internal cursor. Equivalent result as long as positions are increasing and have no gaps.
- **CFG**: batch 2, row 0 conditional, row 1 unconditional; the guided logits are `uncond + (cond - uncond) * cfg`.
- **TorchSharp API**: indexing with C# ranges (`t[.., ^1..]`), `topk` tuples, the signature of `scaled_dot_product_attention` and `load_state_dict`. If something does not compile it is almost surely one of these.

## Memory

3B in bf16 ≈ 6.5 GB + KV cache (8192 positions × 28 layers × 8 kv heads × 128 × 2 × bf16 × batch 2 ≈ 1.9 GB). It fits in a 12 GB card. The codec is loaded separately (see "Windowed decode and `generate --wav`").

## CLI messages

All the text the command line prints is English now (it used to be Spanish). The option names, the exit codes and the `seed=N` line did not change. The user-visible strings that changed language:

| Where | Before | Now |
|---|---|---|
| unknown command | `comandos: inspect \| check \| generate (mira Program.cs)` | `commands: inspect \| check \| generate (see Program.cs)` |
| missing `--ckpt` | `--ckpt requerido` | `--ckpt is required` |
| invalid `--seed` | `--seed debe ser un entero de 64 bits, recibido: 'x'` | `--seed must be a 64-bit integer, got: 'x'` |
| `generate` result | `frames guardados en <path> (8xN)` | `frames saved to <path> (8xN)` |
| generation summary | `N frames (X s de audio) en Y s -> RTF Z` | `N frames (X s of audio) in Y s -> RTF Z` |
| `check` hint | `Si cos_sim > 0.999 y max_abs es pequeño ...` | `If cos_sim > 0.999 and max_abs is small ...` |
| loading errors | `faltan en el checkpoint N claves del modelo: ...` | `N model keys are missing from the checkpoint: ...` |
| | `sobran en el checkpoint N claves que el modelo no tiene: ...` | `N checkpoint keys are not in the model: ...` |
| | `shapes distintas (N): ...` | `shape mismatches (N): ...` |
| | `shard S: faltan N claves del índice: ...` / `shard S: sobran N claves que el índice no lista: ...` | `shard S: missing N keys from the index: ...` / `shard S: N extra keys that the index does not list: ...` |
| | `No se encontró safetensors en <path>` | `No safetensors found at <path>` |

## Related repositories

Depends on `musicgen-core` (shared code) and follows `heartlib`, the original. Used by `veritmo`, which starts it as a service.

| Repository | What it is |
|---|---|
| [`veritmo`](https://github.com/privadosIA/veritmo) | The server: the HTTP API and its contract, the pipelines (video, images, music), the run history and the music services it starts. |
| [`dashboards-labs`](https://github.com/privadosIA/dashboards-labs) | The dashboards (Angular) that talk to the server through its OpenAPI contract. |
| [`heartmula`](https://github.com/privadosIA/heartmula) (this repository) | C# (.NET 8 + TorchSharp) port of [heartlib](https://github.com/HeartMuLa/heartlib): the HeartMuLa song generator, run by the server as a service. |
| [`DiffRhythm2`](https://github.com/privadosIA/DiffRhythm2) | C# port of [DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2): a diffusion model that writes a whole song. |
| [`lion-clap`](https://github.com/privadosIA/lion-clap) | C# port of [LAION-CLAP](https://github.com/LAION-AI/CLAP): audio and text embeddings, used to recognise songs. |
| [`musicgen-core`](https://github.com/privadosIA/musicgen-core) | The TorchSharp building blocks shared by the five ports. |
| [`triposr`](https://github.com/privadosIA/triposr) | C# port of [TripoSR](https://github.com/VAST-AI-Research/TripoSR): one picture in, one 3D model (`.glb`) out; makes the models of the *models3d* category (MIT original). |
| [`stable-fast-3d`](https://github.com/privadosIA/stable-fast-3d) | C# port of [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d): one picture in, one textured 3D model (`.glb`) out. **Under the Stability AI Community License, not Apache-2.0** (free for research and non-commercial use, and for commercial use under US$1M of yearly revenue). |
| [`forms-lib`](https://github.com/privadosIA/forms-lib) | `@core/dynamic-forms`, the Angular library that draws forms and tables from JSON. |

The links are to the public copies of these repositories. The third-party tools, models and libraries they build on, with their licenses, are listed in [THIRD_PARTY.md](https://github.com/privadosIA/veritmo/blob/main/THIRD_PARTY.md).

## License

Apache 2.0, same as heartlib (see [LICENSE](LICENSE)).

This licence covers the code of this repository. The model weights (HeartMuLa, HeartCodec) are downloaded from their publishers and are under their own licences: see their model cards.

## Maintenance

Maintained by one person in their free time: issues and pull requests are welcome, with no promised response time. See [CONTRIBUTING.md](CONTRIBUTING.md) and, for vulnerabilities, [SECURITY.md](SECURITY.md).
