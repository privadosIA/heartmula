# Proposal

## Why

HeartMuLa.Net is the first port of a series (DiffRhythm2 is next, others may follow). About a third of its source is not specific to HeartMuLa: the safetensors reader and writer, weight-norm folding, the WAV writer, seeding, RMSNorm/Conv/MLP building blocks, a strict "checkpoint vs model" loader that exists three times with ~25 duplicated lines each, and the whole parity-test infrastructure. Copying it into the next port would fork the code and its bug fixes (the hostile-file hardening of the safetensors reader, the TorchSharp `manual_seed` CPU-only finding). Extracting it now, while only one consumer exists, is cheap and keeps both ports on the same tested base.

## What Changes

- New repository `~/musicgen-core` (sibling of `~/heartmula`) with the project `MusicGen.Core` (library) and `MusicGen.Core.Tests`, built with the same TorchSharp backend switch as HeartMuLa.Net. HeartMuLa.Net references it with a relative `ProjectReference`.
- Moved **as is** (namespaces change, behavior does not): `SafeTensors`, `CheckpointIndex`, `SafeTensorsWriter`, `WeightNorm`, `WavWriter`, `PythonText` (+ data), `Seeding`, `Conv1dLayer` (already has a non-causal mode), `PReLULayer`, and the test infrastructure (`Parity`, `Fixtures`, `RequiresCheckpoint*`, `Bf16`, `Rng`, `TinyModel`-style helpers that are not HeartMuLa-specific).
- Moved **with parametrization**:
  - `RmsNorm` and `FlowRmsNorm` become one family with the two existing variants (parameter named `scale` computed in fp32, and parameter named `weight` computed in the input dtype), both keeping their checkpoint key names.
  - `ScaledRope` and `FlowRope` become one rotary-embedding class with two layouts: *adjacent pairs* (torchtune, HeartMuLa, HeartCodec) and *rotate_half* (HuggingFace Llama, needed by DiffRhythm2). The layout is a mandatory constructor argument with no default, so sharing it cannot silently pick the wrong one.
  - `FlowMlp` (SwiGLU, SiLU) takes the hidden size as a parameter instead of deriving it only from `2·4·dim/3` rounded to 256, because the HuggingFace Llama MLP of DiffRhythm2 uses `dim · ff_mult`. The HeartCodec call site keeps the current derivation.
  - `ConvTranspose1dLayer` gains a non-causal mode with `padding = (k − s) / 2` (BigVGAN); the current causal behavior is unchanged.
  - `FlowAttention` gains an optional attention mask and optional cross-attention input; calls without them behave exactly as today.
  - `WindowAssembler` receives window, hop and overlap in samples through its constructor instead of reading statics of `DetokenizePlan`.
- New `StrictLoader(prefix, allowList, foldWeightNorm)` replacing `CodecWeights.LoadVocoder`, `FlowMatchingWeights.Load` and `HeartMuLaModel.LoadWeights` internals. Same validation (missing / unexpected / shape mismatches reported together, before any data is read), same error text structure, same sequential-read order, same memory bound.
- Stays in HeartMuLa.Net (coupled to HeartCodec or HeartMuLa): `FlowTimestepEmbedder` (512 features, cos then sin; DiffRhythm2's embedding has 256, sin then cos, so it is not shareable), `FlowMatching`, the two-stage `LlamaTransformer`, `VqCodebook`/`ResidualVq`, `ScalarVocoder`, `DetokenizePlan`, `HeartCodecDecoder`, `HeartMuLaModel`, `MusicGenerationPipeline`, the tokenizer.
- `tools/update_public_branch.sh` and `.github/workflows/ci.yml` learn about the sibling repo: CI checks out both repositories side by side; the public snapshot either vendors `MusicGen.Core` under a fixed subfolder or is blocked with a clear message until the core is published (decided in design.md).
- Scope guard: this change does **not** apply the numerical findings of the HeartMuLa review (fp32 sampling, guidance ≤ 1, `no_grad`, RNG order, memory). They are independent behavior changes with their own parity impact and are listed in design.md as follow-ups. None of them is applied here, so the extraction stays a pure refactor that can be verified by identical numbers.

Not applicable: **BREAKING** for end users. Public C# namespaces of the moved types change (`HeartMuLa.Weights` → `MusicGen.Core.Weights`, etc.); there are no external consumers.

## Capabilities

### New Capabilities
- `musicgen-core`: contract of the shared library: what it contains, that it has no dependency on HeartMuLa, that its building blocks keep the checkpoint key names, that rotary layout is explicit, that the shared strict loader reports every discrepancy before reading data, and that its test infrastructure is reusable by a consumer.

### Modified Capabilities
- `weights-loading`: the "strict binding of weights to the model" requirement is generalized to a loader that accepts a key prefix, an allow-list of ignorable keys and weight-norm folding, so that the three loaders share one implementation without changing their observable behavior.

## Impact

- Code: `src/HeartMuLa.Net` loses ~900 lines (moved) and gains a project reference; `tests/HeartMuLa.Net.Tests` loses the generic tests and helpers (moved with their code); a new `~/musicgen-core` repo.
- Build: `Directory.Build.props` (TorchSharp backend switch and Tokenizers) must be shared; the core repo gets the same props, the consumer repos reference it by relative path, so both repos must sit side by side.
- CI and public branch: see above.
- Numerical parity criterion: **not applicable (no change in behavior)**. Acceptance is that the existing suite stays identical: 252 passed / 0 failed / 36 skipped without a checkpoint before and after (counting tests moved to the core's own suite), and the full suite with `HEARTMULA_CKPT` (including the golden-run parity tests of backbone, decoder, sampling, vocoder, flow matching and two-window decode) gives the same results and the same measured numbers.
- Future: DiffRhythm2.Net will reference the same core.
