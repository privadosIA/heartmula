# Design

## Context

- `HeartCodecDecoder.detokenize` (heartlib 5858ca8, `heartcodec/models/decoder.py`), step by step, for `codes [1, 8, N]`, `duration = 29.76`, 10 steps, guidance 1.25:
  1. `min = int(29.76·12.5) = 372`, `hop = min // 93 · 80 = 320`, `ovlp = 52` code frames, `ovlp_frames = 104` latent frames, `target_len = int(N / 12.5 · 48000)`.
  2. If `N < 372` the codes are repeated (`cat([codes, codes])`) and cut to 372. Then, if `(len − 52) % 320 > 0`, `len_codes = ceil((len − 52)/320)·320 + 52`, the codes are repeated and cut to that length. (Example: `N = 400` → 692, two windows; `N = 126` → 372, one window.)
  3. Windows start at `range(0, len − 320 + 1, 320)` and take `min = 372` frames each. Window 0 (or `ovlp_frames == 0`) runs `inference_codes` with `true_latents = first_latent` (a `randn` that is masked to zero) and `incontext_length = 0`. Later windows use `true_latent = cat([previous_latents[:, −104:], randn(1, 640, 256)], 1)` and `incontext_length = 104`. Every call uses `scenario = "other_seg"`, `latent_length = 744`.
  4. Each window's latents `[1, 744, 256]` are reordered to `[2, 128, 744]` (the C7 reorder), decoded by the vocoder and cut to `int(29.76·48000) = 1 428 480` samples (the whole output). Windows are joined with `ovlp_samples = 1 428 480 − 15360·80 = 199 680`: `ov = linspace(0, 1, 199 680)`, `win = cat([ov, 1 − ov])`, the last `ovlp` samples of the output become `output[-ovlp:]·win[-ovlp:] + current[:ovlp]·win[:ovlp]` (old fades out with `1 − ov`, new fades in with `ov`), and `current[ovlp:]` is appended. The result is cut to `target_len`.
- The recording wrapper in `tools/dump_reference.py codec` already captures per-call `randn` (with the name of the calling function), `codes`, `inference_codes` outputs and `scalar_model.decode` inputs; with several windows it yields `randn_0_detokenize` (first_latent), `randn_1_inference_codes`, `randn_2_detokenize` (padding of window 1), `randn_3_inference_codes`, `codes_seg{0,1}`, `latents_seg{0,1}`, `decode_in_seg{0,1}`. That naming is relied on, not assumed: the dump asserts the call order.
- C7 gives `FlowMatching.InferenceCodes(codes, latentLength, incontextLength, guidance, steps, noise, trueLatents, otherSeg)`, `FlowMatching.ReorderForVocoder`, `HeartCodecDecoder.DecodeSegment`, strict loaders, `WavWriter`, `AudioParity.Compare` and `Seeding.Apply`. `generate` is in `Program.cs` over `MusicGenerationPipeline`, and writes `frames.safetensors` with the seed in its metadata.
- Memory (measured in C7): flow matching 5.96 GB + vocoder 0.34 GB in fp32; peak VRAM 10.8 GiB on the 4070 with only the codec loaded. HeartMuLa in bf16 is 7.3 GB.

## Goals / Non-Goals

**Goals:**
- Same audio as heartlib's `detokenize` for any number of frames, validated on a 2-window reference, including the in-context path and the cross-fade.
- `generate --wav` end to end on the 12 GB GPU, with HeartMuLa released before the codec loads.
- Reproducible output for a given seed and device; GPU audio equal to the CPU's by default.

**Non-Goals:**
- Speed work (batching windows, bf16, CUDA graphs), streaming, loudness/normalisation.
- Bit-identical noise to Python's RNG: noise is injected from the fixture for parity and comes from libtorch's seeded stream in real use.
- Changes to the model, the sampler or the prompt pipeline.

## Decisions

1. **Plan the windows in a pure function.** `DetokenizePlan(N)` returns the padded length, the window starts, `target_len` and the overlap sizes; it reproduces the reference arithmetic with integer maths (including `//` and `ceil`). It is tested against values computed by hand and by the reference formula for many `N` (1, 126, 371, 372, 373, 400, 691, 692, 693, 1000), and the repetition/cut of the codes is a separate pure function tested the same way. This keeps the part with the most off-by-one risk away from the models.
2. **One decode loop, noise injected.** `Decode(flow, vocoder, codes, guidance, steps, noise provider)` where the provider yields, per window, the flow-matching noise and, for windows after the first, the padding noise for the in-context latents. Tests pass the recorded tensors; the default draws `randn` on the codec device in the same order as the reference (padding noise first, then flow noise), so the stream is deterministic under `Seeding.Apply`.
3. **Cross-fade in double then float.** The window is `np.linspace(0, 1, ovlp)` in float64 in Python and is multiplied with float32 audio (the product is float64 in torch type promotion because `ov_win` is a double tensor, then assigned into the float32 output). The C# fade builds the ramp in float64, computes in float64 and stores float32, so the rounding matches. It is tested with a naive loop and with a ramp-endpoint check (`ov[0] = 0`, `ov[-1] = 1`).
4. **Memory-light assembly.** Windows are decoded one at a time and appended to a preallocated CPU output; only the last 104 latent frames of the previous window are kept. A 10-minute song is 17 windows and never holds more than one window's activations.
5. **Unload before the codec.** `generate` stages: (a) load HeartMuLa, run the loop, keep the `[8, T]` frames on the CPU, (b) dispose the model, caches (`FreeCaches`, they are not parameters) and tensors, `GC.Collect`, (c) load the codec on `--codec-device` (default: the model's device), decode, write. With `--wav` unset the behaviour is unchanged (frames only). TorchSharp 0.107 exposes no `cuda.empty_cache()` nor memory counters (checked in the native library), so the freed blocks stay in libtorch's caching allocator, which reuses them and gives cached blocks back to the driver by itself when an allocation would fail. Tests check that the parameters, caches and given tensors are disposed and that a same-size model can then be created on the GPU; the real run checks that the peak VRAM sampled with `nvidia-smi` on the 4070 stays under its capacity.
6. **cuDNN TF32 off for the codec by default.** C7 measured that this turns a 36 dB GPU audio into the CPU's 122 dB with no speed cost, so the safe behaviour is the default and `--codec-tf32` restores heartlib's default. The setting is restored after the decode so it does not leak into other code.
7. **WAV format.** PCM16 by default, as in the C6/C7 listening files (the user listened to them and heard no difference with the reference); `--wav-format float32` writes 32-bit float. Assumption: heartlib's `torchaudio.save` output format is not a requirement of this change, only that the samples match.
8. **Fixture shape.** `codes` for the long case are the 126 reference frames tiled to 400 frames: real tokens, deterministic, no extra generation run, and 2 windows after padding (692 frames). The fixture records `codes_seg0/1`, the four `randn`s, `latents_seg0/1`, `decode_in_seg0/1` and `wav [2, 1 536 000]` (`400 / 12.5 · 48000`). The dump script gets a `codec_long` command and an assertion that the recorded call order is as designed. The venv is the one in `MANIFEST.md` (python3.11, torch 2.10.0+cpu), created for this change and deleted afterwards.

## Risks / Trade-offs

- **The vocoder quantizes its input** (`round(9·x)/9`, discontinuous). A latent error of 1e-6 (the float32 noise floor of 30 layers × 10 steps) moves one of the ~190 000 values per window across a rounding boundary now and then, which changes the audio locally by up to ~4e-3 for about 0.8 s. Found while applying (CPU, 400 frames: 1 flipped value in window 0, frame 358, and nothing else): the latents agree at 123 to 125 dB and the reference's own vocoder inputs give the reference audio at 125 dB, yet the audio from our latents shows that one local deviation (65.9 dB overall). Python on a different CPU would show the same. The whole-chain audio is therefore asserted by global agreement plus "deviations only near a flipped frame", not by a global `max_abs`; the exact checks are on the latents and on the reference inputs through the vocoder and assembler.
- **Window 1 depends on window 0.** Window 1 receives window 0's latents as in-context, so errors propagate. Mitigation: the per-window parity test feeds window 1 the *reference* window-0 latents (isolating the in-context path), and a second test runs the whole chain from the codes (the real use). Both are measured; the thresholds come from measurements.
- **GPU TF32.** With TF32 off by default for the codec, parity thresholds on GPU are the CPU ones; with `--codec-tf32` they are the looser C7 ones. Both are asserted.
- **Fixture size (~18 MB).** It is bigger than the other fixtures but needed to compare audio. Using 400 frames instead of a longer case keeps it small while still having two windows and an overlap.
- **VRAM headroom on the 4070.** C7 measured 10.8 of 12 GiB. If a real `generate` run exceeds it, the fallback is to decode on the 5060 Ti (`--codec-device`) and record that the single-GPU case needs `empty_cache` between windows; this is decided with the measurement, not before.
- **Python RNG vs libtorch RNG.** Real audio will not match Python's for the same seed, by design; parity is only claimed with injected noise.

## Open Questions

None blocking. If the real generate run shows the unload does not free enough VRAM, the design adds a per-window `empty_cache`; this is recorded as a finding when measured.
