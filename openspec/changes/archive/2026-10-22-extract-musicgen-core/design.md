# Design

## Context

HeartMuLa.Net is ~3 200 lines of source and ~5 000 of tests, validated against a golden run of heartlib (fixtures in `tests/fixtures`, regenerated with `tools/dump_reference.py`). See proposal.md for why a second port (DiffRhythm2) needs a shared base. Observed in the current code:

- The strict "model vs checkpoint" check exists three times: `CodecWeights` (vocoder, with weight-norm folding), `FlowMatchingWeights` (with a training-state allow-list) and `HeartMuLaModel.ValidateAgainst`/`LoadWeights`. They differ only in the key prefix, the allow-list and the folding, and in the dtype read.
- Two RoPE implementations with opposite layout of `x`: `ScaledRope` works on `[b, s, h, d]`, computes a Llama-3-scaled `cos/sin` table once in fp32 and rotates adjacent pairs; `FlowRope` works on `[B, H, T, D]`, computes `sin/cos` in the input dtype per call (base 10000, no scaling), also adjacent pairs, and can rotate only the first `ropeDim` channels. DiffRhythm2 needs the third variant: half-rotation (`rotate_half`) from `transformers`.
- `FlowRmsNorm` (`weight`, input dtype) and `RmsNorm` (`scale`, fp32 round trip) are intentionally different and must stay bit-identical.
- `Rng.Seed` in the tests already delegates to `Seeding.Apply`. The test helpers `Parity`, `Fixtures`, `Bf16`, `Rng`, `RequiresCheckpoint*` mention `HEARTMULA_*` environment variables and `HeartMuLa.*` types.
- `.github/workflows/ci.yml` builds and tests `HeartMuLa.Net.sln` on a single checkout; `tools/update_public_branch.sh` publishes `git archive main` and then builds the snapshot with `HeartMuLa.Net.sln`.
- `Directory.Build.props` selects the native TorchSharp package (`cpu | cuda-windows | cuda-linux`) and adds `Tokenizers.DotNet` to every project.

## Goals / Non-Goals

**Goals:**
- One tested copy of everything model-agnostic, consumable by HeartMuLa.Net now and by DiffRhythm2.Net next.
- Zero numerical change: same tests, same counts, same measured figures before and after.
- Rotary layout and norm flavor that cannot be shared by mistake.

**Non-Goals:**
- The numerical findings of the HeartMuLa review (fp32 softmax in sampling, guidance ≤ 1 in the codec, `no_grad` in public codec APIs, fp32 for the codec's noise and timestep, RNG draw order, construction memory, attention over 8192 positions). They are follow-up changes in HeartMuLa.Net, each with its own parity run, because bundling behavior changes into a refactor would make a regression impossible to attribute.
- Moving anything coupled to HeartCodec or HeartMuLa (listed in proposal.md).
- Publishing a NuGet package. The core is consumed by path.

## Decisions

1. **Separate repo, consumed by relative path.** `~/musicgen-core` holds `src/MusicGen.Core`, `tests/MusicGen.Core.Tests`, its own `Directory.Build.props` (only the TorchSharp backend switch, no Tokenizers) and its own solution. Consumers reference `$(MusicGenCoreDir)/src/MusicGen.Core/MusicGen.Core.csproj`, with `MusicGenCoreDir` defaulting to `$(MSBuildThisFileDirectory)../musicgen-core` and overridable (`-p:MusicGenCoreDir=...`). *Alternative:* a project inside the HeartMuLa repo that DiffRhythm2 references; rejected because the next port would depend on the layout of an unrelated repository. *Alternative:* NuGet package; rejected for now (a publishing pipeline for two consumers, and fixtures/tests move together with the code).
2. **`TorchBackend` is a global MSBuild property**, so core and consumer resolve the same native package when built from the consumer with `-p:TorchBackend=...`. Core and consumer must not reference different TorchSharp versions: the version lives in each `Directory.Build.props` and a test in the core's CI asserts both files carry the same `TorchSharpVersion` (the consumer's CI checks it against the core it checks out).
3. **Namespaces:** `MusicGen.Core.Weights` (SafeTensors, CheckpointIndex, SafeTensorsWriter, StrictLoader, WeightNorm), `MusicGen.Core.Layers` (norms, rope, convs, MLP, attention), `MusicGen.Core.Audio` (WavWriter, WindowAssembler), `MusicGen.Core.Text` (PythonText), `MusicGen.Core` (Seeding), `MusicGen.Core.Testing` (test helpers, shipped in a second project `MusicGen.Core.Testing` so that the library does not reference xunit). Checkpoint parameter names do not depend on C# namespaces or class names (they come from `RegisterComponents` field names), which is why a rename is safe; a test asserts it by listing `named_parameters()` of the moved layers.
4. **One `RotaryEmbedding` with two independent axes of variation**, because the existing implementations differ in more than the layout:
   - `RopeLayout { AdjacentPairs, HalfRotation }`, required constructor argument.
   - Frequency source: `Llama3Scaled(base, scaleFactor, lowFreqFactor=1, highFreqFactor=4, oldContextLen=8192)` or `Plain(base)`.
   - Table precision: a cos/sin table computed once in fp32 for `maxSeqLen` (torchtune) or computed per call in the input dtype (the flow transformer). It is part of the configuration, not a hidden branch, so a consumer states it. It stays configurable because torchtune and the flow code take different numerical paths and the zero-change requirement is checked bit for bit.
   - Tensor layout: `Apply(x, positions, seqDim)` with `seqDim = 1` for `[b, s, h, d]` and `seqDim = 2` for `[b, h, t, d]`, so no caller transposes (a transpose is a view, but the bf16 path must see the same memory order to stay bit-identical).
   - Optional `ropeDim < headDim`: only the leading channels are rotated.

   torchtune vs HuggingFace differences respected: torchtune/HeartMuLa/HeartCodec rotate adjacent pairs `(x[2i], x[2i+1])`; HuggingFace Llama (DiffRhythm2) rotates halves `(x[i], x[i + d/2])` after building `cos/sin` as `cat(freqs, freqs)`. The half-rotation scenario in the spec is checked against a fixture generated from `transformers.models.llama.modeling_llama` (see Fixtures below). RMSNorm: torchtune computes in fp32 with parameter `scale`; the flow norm computes in the input dtype with `weight`; HuggingFace `LlamaRMSNorm` (DiffRhythm2) computes in fp32 with parameter `weight`: a third flavor is **not** created here, DiffRhythm2.Net will add it when it needs it, and the three flavors are named in the doc comment so nobody swaps them.
5. **Norm family.** `RmsNorm` (torchtune flavor, `scale`) and `FlowRmsNorm` (`weight`, input dtype) move unchanged under `MusicGen.Core.Layers`. Merging them into one class with a flag would make the parameter name data-dependent, which breaks the "names replicate checkpoint keys" rule.
6. **`StrictLoader`.** One class configured with: key prefix to strip, an allow-list predicate for keys to skip, a `foldWeightNorm` flag, the target model's `named_parameters` (plus `named_buffers` when asked), and the read dtype. Algorithm (identical to today's): plan entries from header information only (folding `original0/original1` pairs into `<layer>.weight`); compute missing / unexpected / shape-mismatch sets; if any, throw one `InvalidDataException` whose text lists every discrepancy before any data was read; otherwise move the model to the device, then read tensors ordered by file and offset under `no_grad`, copying each into its parameter. The three callers become 3-line adapters that keep their report records (`VocoderLoadReport`, `FlowMatchingLoadReport`) so their tests do not change. Error message prefixes ("The HeartCodec checkpoint does not match the vocoder:") stay, passed in by the caller, because existing tests assert on them.
7. **`ConvTranspose1dLayer` non-causal mode** (needed by BigVGAN): `padding = (k − s) / 2`, output length `(T − 1)·s + k − 2p`. The existing `narrow().contiguous()` copy in the causal mode (review finding) is **not** touched.
8. **`WindowAssembler`** takes `(windowSamples, hopSamples, overlapSamples, totalSamples/ windows, channels)` and the HeartCodec `DetokenizePlan` provides them. The assembler keeps the float64 ramp mixing exactly as is.
9. **Attention variants.** `FlowAttention` gains an optional boolean mask `[B, 1, Tq, Tk]` (true = attend, as torchtune and torch SDPA) and an optional `kvInput` for cross-attention; absent, the call path is the same `F.scaled_dot_product_attention(q, k, v)` as today. The mask scenario in the spec is checked with a naive loop.
10. **Public branch of HeartMuLa.** `update_public_branch.sh` gains `--core <path>` (default `$MUSICGEN_CORE_DIR` or the sibling folder): it exports the core's current commit with `git archive` into `third_party/musicgen-core/` of the snapshot, sets `MusicGenCoreDir` through a `Directory.Build.props` line that prefers `third_party/musicgen-core` when it exists, runs the same forbidden-string and secret checks over it, and records both commits in the snapshot message. *Alternative:* block publishing until the core is public; rejected because it would freeze the public repo.
11. **CI.** `ci.yml` checks out two repositories side by side (`path: heartmula` and `path: musicgen-core`) and builds/tests `HeartMuLa.Net.sln`; the core repo gets its own `ci.yml` (build + test its suite). Where the core is hosted is an open question (below); the workflow takes the repository name from a repository variable so it can be set later without editing the file.
12. **Fixtures.** The HeartMuLa fixtures (`tests/fixtures`, manifest, `tools/dump_reference.py`) stay in HeartMuLa. The core only needs *small* generated fixtures for its own checks: (a) the half-rotation RoPE reference (a few positions and heads, fp32) generated with a `tools/dump_rope_hf.py` script in the core that imports `transformers.models.llama.modeling_llama` (Python only there, run once, sha256 in a manifest as in HeartMuLa); (b) none for the other pieces: they are verified against naive loops or by identity with the previous implementation. The identity check ("bit for bit equal to the old implementation") is performed once during the move by running the old and new code in the same test run, and the temporary old copies are deleted in the last task.
13. **History.** Moved code is copied, not history-preserving; each commit message in the core names the HeartMuLa commit it was taken from.

## Risks / Trade-offs

- [A subtle numerical difference slips in while unifying RoPE] → the old implementations stay in the tree until a test proves bit-for-bit equality of the new one on bf16 and fp32 inputs, both layouts of `x` and several positions, then they are deleted; the full golden-run suite (backbone teacher forcing 126 calls, decoder, sampling 168/168 tokens, vocoder 121 dB, flow matching 118 dB, two-window decode) is re-run and its measured figures are pasted in tasks.md next to the baseline.
- [Two repos drift (TorchSharp version, props)] → the version check in decision 2 and CI checking both out together.
- [The public snapshot breaks because the core is not found] → the snapshot build step already runs `dotnet build`; it now runs with the vendored core, so a missing core fails the script before committing.
- [Parameter names change by accident when classes are renamed or moved] → a test lists `named_parameters()` of every moved layer and compares with the expected checkpoint names; the real-checkpoint load tests also fail on any mismatch.
- [Checking identity with the old code doubles the work] → accepted; it is temporary and is the only proof that does not rely on the (slow) real checkpoints.
- [The extraction tempts mixing in the review's fixes] → explicitly out of scope; they are listed below as follow-up changes so they are not lost.

## Migration Plan

1. Create `~/musicgen-core` with props, solution, and CI; build empty.
2. Move leaf utilities and their tests, one group at a time (weights, audio, text/seeding, test helpers), running the no-checkpoint suite after each group.
3. Add the unified `RotaryEmbedding` next to the old classes, prove equality, switch the two models, delete the old classes.
4. Add `StrictLoader`, switch the three loaders, keep their tests unchanged.
5. Parametrize `WindowAssembler`, `FlowMlp`, `FlowAttention`, `ConvTranspose1dLayer`.
6. Update `ci.yml` and `update_public_branch.sh`; run the script with `--no-build` off against a scratch branch to prove the snapshot builds.
7. Full suite with `HEARTMULA_CKPT`, paste figures, compare with baseline, archive.

Rollback: HeartMuLa's `main` stays untouched until the whole change is green; the work is done on a branch `extract-musicgen-core` and merged at the end.

## Follow-up changes (not part of this one)

`sampling-fp32` (review 1), `codec-guidance-le-1` (2), `codec-no-grad-api` (3), `codec-fp32-explicit` (4), `codec-rng-order` (5), `llm-attention-kv-trim` (6), `codec-construction-memory` (7), the small items of review 8, and test hardening (review 9). Each has its own parity criterion.

## Open Questions

- Where the core repository is hosted (needed only for CI's second checkout; local builds do not depend on it).
- Whether `MusicGen.Core.Testing` should be a separate project or a folder inside the tests project; the first is assumed here because DiffRhythm2.Net's tests need to reference it.
