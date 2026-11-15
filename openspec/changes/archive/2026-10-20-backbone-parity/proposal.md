# Proposal

## Why

The golden run (C1) and the weight loading (C2) are done, but the numerical parity of the model has only been checked by hand: the `check` command prints `cos_sim` of `last_h` and `c0_logits` of the **first** frame and a human decides whether it is fine. It is not a test, it does not run in `dotnet test` and it covers nothing else (126 frames, growing KV cache, RoPE at distant positions, decoder). It has already bitten us: the first time `check` was run with the real checkpoint it showed a failure in `ScaledRope` (cache released after the first call) that no test detected. C4 (generation) and everything that follows rely on the backbone and decoder matching Python; without automatic tests that foundation can break unnoticed.

There is also a gap: the fixtures only store the logits of codebook 0 and `last_h` per frame, not the logits of codebooks 1–7 produced by the decoder. With teacher forcing the decoder does not influence the backbone logits, so today the decoder (300M) has no reference to validate against.

## What Changes

- `[RequiresCheckpoint]` xUnit tests that replace the manual check of `check`:
  - **Isolated RoPE** against `rope.safetensors` (torchtune): F32 at positions 0..15 and 4000..4015, and bf16.
  - **Backbone, first frame** against `reference.safetensors` (`last_h`, `c0_logits`, batch 2 with CFG).
  - **Backbone with teacher forcing over the 126 frames**: the Python frames (`frames_py.safetensors`) are re-injected and each call is compared with `frames_logits.safetensors` (cond and uncond separately, before CFG), with the KV cache accumulating as in the real loop.
  - **Decoder** (codebooks 1–7) against a new fixture `decoder_logits.safetensors`, fed with the `last_h` from Python to isolate it from the backbone.
- New fixture: `decoder` subcommand in `tools/dump_reference.py` (the only contact with Python, once) that dumps the decoder logits for a subset of frames with forced tokens. It requires setting up the golden run environment on this machine (venv with torch 2.10.0, heartlib at the pinned commit). The existing fixtures are **not** regenerated; `MANIFEST.md` is updated with the new file, its sha256 and the origin machine.
- Small code changes only to be able to test: `DecoderLogits` in `HeartMuLaModel` (decoder with forced tokens and returning logits, instead of sampling), and making the padding of an audio frame (`PadAudioToken`), which is currently private to the pipeline, reusable. The behavior of `GenerateFrame` and of the pipeline does not change.
- The device of the parity tests is chosen with `HEARTMULA_DEVICE` (`cpu` by default, `cuda` for the development machine, which has a 4070 and a 5060 Ti).
- Any architecture failure that parity uncovers is fixed in this change only if it is small and confined to one module; otherwise, work stops and the scope is reconsidered.

Out of scope: sampling (not comparable between Python and .NET) and the full generation loop (C4), tokenizer (C5), codec (C6/C7), quantization, fp32 parity of the full model (see criterion).

## Capabilities

### New Capabilities
- `model-parity`: the C# forward pass of HeartMuLa (RoPE, backbone with KV cache, codebook decoder) reproduces the tensors of the Python golden run within fixed tolerances, and that check runs as automatic tests.

### Modified Capabilities
None. (`weights-loading` does not change.)

## Impact

- **Numerical parity criterion**
  - bf16, backbone and decoder: `cos_sim ≥ 0.999` per call and `max_abs ≤ 0.5` on logits (the first frame currently measures `max_abs` 0,125 with `cos_sim` 0,99999). If any frame does not comply, the cause is investigated; the tolerances are not relaxed to make the test pass.
  - RoPE: F32 with `max_abs ≤ 1e-4`; bf16 with `max_abs` ≤ 1 bf16 ulp of the value.
  - **fp32 of the full model: does not apply**, because the backbone and decoder fixtures are bf16 from CPU (Apple M4) and comparing an fp32 forward pass against them measures nothing. The fp32 `1e-4` is only required where there is an fp32 reference (RoPE).
  - The accumulation order differences between the reference CPU and this machine (or CUDA) are what the bf16 tolerances absorb.
- Code: `HeartMuLaModel.cs` (`DecoderLogits`), `MusicGenerationPipeline.cs` (reusable frame padding), `tools/dump_reference.py` (`decoder` subcommand), new tests in `tests/HeartMuLa.Net.Tests`.
- Data: `tests/fixtures/decoder_logits.safetensors` (small, versioned like the rest) and `MANIFEST.md`.
- Environment (outside the repo): `~/.venvs/heartlib-golden` and `~/src/heartlib` on this machine.
- Documentation: README (how to run the parity tests and with which device).
