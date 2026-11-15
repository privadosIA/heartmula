# Proposal

## Why

C3 left the backbone and the decoder validated, with automated tests, using forced inputs. What is missing is the piece that joins them: the generation loop (`MusicGenerationPipeline.Generate`: prompt, frame loop, CFG, top-k, temperature, KV cache, EOS stop). Today it does not have a single test, and it is what produces the frames that C6/C7 will convert into audio.

Comparing its output with Python is not trivial because the process is stochastic. I measured, with a throwaway program (not repo code), the full loop with `manual_seed(1234)` against `frames_py.safetensors`:

- The libtorch CPU RNG matches that of Python: the first 3 frames come out **identical** (24 tokens), and in frame 3 codebooks 0–4 match.
- In codebook 5 of frame 3 the token differs (expected 7232, obtained 3908). From there on, since each token feeds back into the next step, the error propagates: only 42 of 1008 tokens are equal (4.17 %) and 3 of 126 complete frames.

A minimal logit difference in bf16 is enough to pick another token with the same noise. That is why **exact equality of the 126 frames is not a viable criterion**, and we need to define what "parity" means for this loop. We also do not yet know whether that first divergence is legitimate numerical noise or a real failure, and there is a known difference from heartlib: it runs `generate_frame` inside `torch.autocast(bf16)` and C# does not.

## What Changes

- **Sampling with the Python inputs** (no checkpoint or model): with the raw Python logits from `frames_logits.safetensors` (126 steps of codebook 0) and from `decoder_logits.safetensors` (6 frames × 7 codebooks), the same seed and the same RNG consumption as heartlib, the C# sampling (CFG + top-k + temperature + exponential noise) must choose **the same tokens as `frames_py`**. It isolates the sampling and the RNG usage from any model noise, and is deterministic.
- **Loop invariants** (no checkpoint, with the small model): output shape and type, tokens in range, both CFG rows with the same tokens, EOS stop (without including the EOS frame), maximum length, same seed → same output, different seeds → different outputs.
- **Full loop with the real model** (CPU): with seed 1234 the first frames reproduce `frames_py` (detector of gross regressions, e.g. an error in CFG, positions or cache); it reports in which frame and codebook the first divergence occurs and how many tokens match, without requiring equality beyond that.
- **Diagnosis of the first divergence** (frame 3, codebook 5): with the same Python inputs up to that point, see what probability C# assigns to the Python token and to its own, to tell numerical noise from a real failure, and note whether heartlib's `autocast` can explain it.
- Small code changes to make testing possible: the guided sampling (CFG + top-k) becomes a single function used by `GenerateFrame` and the tests, and the pipeline stops hard-coding 8 codebooks so that the small model can be used. Behavior with the real model does not change.
- Test execution stops being parallel: the libtorch RNG is global and the tests that seed it would step on each other.

Out of scope: real tokenizer (C5, the prompt still enters through `JsonIdsTokenizer` with `prompt_ids.json`), codec (C6/C7), performance optimization (the 126-frame loop takes 184 s on CPU in C# versus ~125 s in Python on the same machine; it is noted, not pursued) and regenerating fixtures with Python (none new is needed; the venv was deleted and the recipe is in `MANIFEST.md`). This change does not produce audio, so there is nothing to leave in the shared folder.

## Capabilities

### New Capabilities
- `generation-loop`: the HeartMuLa frame generation loop in C# samples and behaves like heartlib's, with exact parity where it is deterministic (sampling with the same inputs) and with invariants and a reproducible prefix where it is not.

### Modified Capabilities
None. (`model-parity` and `weights-loading` do not change.)

## Impact

- **Numerical parity criterion**
  - **Sampling with Python inputs: exact token equality.** The 126 tokens of codebook 0 and the 42 of codebooks 1–7 (6 frames) must be those of `frames_py`. It is deterministic because the inputs (Python logits and seed) are identical; if any token differs the cause is investigated and the criterion is not relaxed.
  - **Full loop with the real model, CPU, seed 1234: exact equality of the first 2 frames** (16 tokens) and, from there on, report only (frame and codebook of the first divergence, matching tokens). More is not required because a one-ulp bf16 difference can legitimately change a token and shift the rest. On CUDA token equality is not required, only the invariants.
  - **Model logits**: already covered by C3 (`cos_sim ≥ 0.999`, `max_abs ≤ 0.5`); not repeated.
- Code: `HeartMuLaModel.cs` (shared guided sampling), `MusicGenerationPipeline.cs` (number of codebooks from the config), new tests in `tests/HeartMuLa.Net.Tests`.
- Data: none new; `frames_py`, `frames_logits`, `decoder_logits` and `reference` from `tests/fixtures` are used.
- Documentation: README (how to run these tests and why exact equality of the 126 frames is not expected).
