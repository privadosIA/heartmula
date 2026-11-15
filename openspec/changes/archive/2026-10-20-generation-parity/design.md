# Design

## Context

- `MusicGenerationPipeline.Generate` replicates `_forward` from heartlib: one prefill call with the prompt (batch 2, CFG), a loop of `max_ms / 80` iterations that fills in the previous frame (`PadAudioToken`), calls `GenerateFrame` with position `last_of_prompt + i + 1`, and stops if **row 0** has any token `≥ audio_eos_id` (without saving that frame). Returns `[K, T]`.
- `GenerateFrame` samples 8 times per frame (codebook 0 and then 1–7), always with `SampleTopK` over the guided logits `[1, V]` (CFG: `uncond + (cond − uncond) · cfg`) and repeats the token in both rows. `SampleTopK` consumes the global RNG once per sampling (`exponential_` over a bf16 tensor `[1, V]`), exactly like heartlib.
- Measured (throwaway program, CPU, seed 1234, 126 frames): frames 0–2 are identical to `frames_py`, in frame 3 codebooks 0–4 match and 5 does not (7232 expected, 3908 obtained); then it diverges (42/1008 tokens, 4.17 %). Time 184 s (RTF 18).
- Known difference from heartlib: `_forward` runs `generate_frame` inside `torch.autocast(device_type, dtype=bf16)`. The C# model is already in bf16 and there is no `autocast`; on CPU almost all operations do not change, but it is a possible cause of last-bit differences.
- The libtorch RNG is global: tests that seed with `manual_seed` cannot run in parallel with others that use it.
- This change **does not touch attention, RoPE or normalization**; the torchtune vs HuggingFace differences that are already respected (RoPE over adjacent pairs with Llama3 scaling, RMSNorm in float32 with `scale`, bool mask `true = attend`, cache by absolute position) remain verified by C3 and are not modified.

### Reference fixtures and how they were generated

None new. All of them come from `tools/dump_reference.py` with heartlib `5858ca8f…` (see `MANIFEST.md`):
- `frames_py.safetensors` [8,126]: heartlib frames with seed 1234, `cfg_scale` 1.5, `topk` 50, temperature 1.0 (Apple M4, CPU, bf16).
- `frames_logits.safetensors`: `c0_logits` [126,2,8197] and `last_h` [126,2,3072], raw before CFG, from each call to `generate_frame` (hooks on `codebook0_head` and `backbone`).
- `decoder_logits.safetensors` (C3): raw logits of codebooks 1–7 for frames 0, 1, 2, 63, 124 and 125 [6,2,7,8197], with `last_h` and tokens.
- `reference.safetensors`: reference prompt (tokens, mask, positions, MuQ).
- `prompt_ids.json`: raw tokenizer ids, served by `JsonIdsTokenizer`.

## Goals / Non-Goals

**Goals:**
- Prove without a checkpoint, deterministically and exactly, that the sampling and the RNG consumption are those of heartlib.
- Prove the loop invariants (shape, range, EOS, length, reproducibility, CFG rows) without a checkpoint.
- Detect gross regressions of the full loop with the real model and record where and why it diverges from Python.

**Non-Goals:**
- Exact equality of the 126 frames (not feasible: see proposal).
- Optimizing loop performance.
- Real tokenizer, codec, audio.
- Emulating `autocast` in C#.

## Decisions

1. **Guided sampling in a single function.** The logic "CFG + top-k + temperature + repeat in both rows" is currently copied in the body of `GenerateFrame` for codebook 0 and, again, in the lambda for codebooks 1–7. It moves to `Sampling.SampleGuided(logits, cfgScale, topk, temperature)` (uses CFG if the batch is even and > 1 and `cfg > 1`, as it does now), which `GenerateFrame` and the tests use. This way the test exercises the production code and not a copy. Alternative: copy the arithmetic in the test; discarded because it gets out of sync without warning.
2. **Sampling test with Python inputs (no model).** Seeds with 1234 and goes through the 126 frames: in each one it makes 8 RNG draws. For codebook 0 of each frame it uses the logits from `frames_logits` (cast to bf16, exact because they are bf16 values stored as float); for codebooks 1–7 of the 6 frames in `decoder_logits` it uses those logits and, in the other frames, makes "filler" draws with the same shape and dtype (`exponential_` over a bf16 `[1, V]`) only to advance the RNG. It compares against `frames_py`. Since the filler draws only advance the state, the test demonstrates at the same time that the order, number and shape of draws match those of heartlib. If any token differs, the cause is investigated instead of loosening the criterion.
3. **Invariants with a small model.** The pipeline currently fixes 8 codebooks (`ParallelNumber`) and `MuqDim = 512`. They are read from `mula.Config` (identical values with the real model) so that the C3 `TinyModel` can be used, with a `GenConfig` of small ids (text BOS/EOS and `audio_eos_id` inside the vocabulary) and a toy tokenizer (one id per word). With `audio_eos_id` equal to the vocabulary size there is never an EOS (maximum-length test); with a low value it appears right away (stop test, checking that the EOS frame is not output).
4. **Non-parallel tests.** `CollectionBehavior(DisableTestParallelization = true)` at assembly level. The suite takes milliseconds except for the tests with the real model, which already run serially within the `RealModel` collection, so it costs no time.
5. **Full loop with the real model.** A single `[RequiresCpuCheckpoint]` test in the `RealModel` collection generates the 126 frames (≈ 3 min on CPU; only on CPU because on CUDA token equality is not expected) with seed 1234 from the prompt in `prompt_ids.json` and `inputs/`, first checks that the built prompt is identical to the one in `reference.safetensors` (which also validates `Preprocess`), requires the first 2 frames and writes to the output the frame and codebook of the first difference and the matching tokens. Requiring 2 frames, and not the 3 measured, is chosen to leave margin for another CPU; an error in CFG, positions or cache already breaks frame 0 or 1.
6. **Diagnosis of the first divergence.** With the backbone in teacher forcing up to frame 3 (as in C3) the `last_h` of call 3 is obtained; `DecoderLogits` with the Python tokens of frame 3 gives the logits of codebook 5; CFG, temperature and top-k are applied and the rank and probability of 7232 (Python) and 3908 (C#) are examined. The noise of that draw is also reproduced (seed 1234 advanced 29 draws) to compute the score `p / q` that decides the `argmax` of each token. Criterion: it is explainable by noise if the Python token is in the C# top-k and the difference in `log(p/q)` between the two is ≤ 2 × 0.5 (twice the C3 logit tolerance, because each of the two logits can move up to that tolerance). If this is not met, it is a sign of a real failure and work stops. This does not need the Python logits at that point, which do not exist in the fixtures (the decoder fixture does not include frame 3).
7. **No Python.** No new fixture is needed. If the diagnosis ended up requiring Python logits for frame 3, how to rebuild the venv would be looked at then (the recipe is in `MANIFEST.md`), but it is not planned.

## Risks / Trade-offs

- [Sampling with the Python logits does not match exactly] → it would be a real difference between the C# arithmetic and that of heartlib (bf16 in `softmax`/`log_softmax`, or in `autocast`); the test lists the mismatching steps and it is investigated before continuing.
- [The filler draws do not advance the RNG the same as the real ones] → it would show up in the first frame with the real decoder; the codebook 0 test over the 126 frames reveals it.
- [The full-loop test adds ~3 min on CPU] → it is what measures the divergence; it only runs with a checkpoint and on CPU.
- [Requiring 2 identical frames may fail on another CPU] → the concrete numbers of this machine are recorded; if it fails elsewhere it is reconsidered with data, not lowered blindly.
- [`autocast` explains part of the divergence and cannot be matched in C#] → it is recorded as a known difference; exact equality is not pursued.
- [Removing parallelism hides an isolation failure] → tests that use the global RNG always seed before using it.
- [The pipeline fixed 8 codebooks and 512 for MuQ] → with the real model the values are the same; the full-loop test with the real model covers it.
