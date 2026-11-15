# Design

## Context

- `HeartCodecDecoder.detokenize` (heartlib 5858ca8) generates the latents with flow matching and, for each window, reorders them into `[2, 128, T]` (the two stereo channels as batch) and calls `scalar_model.decode(latent)`, which returns `[2, 1, N]` → `[2, N]`, trimmed to `min_samples` and, at the end, to `target_len`. With a single segment there are no overlaps. This change implements `scalar_model.decode`; latent generation and overlaps are C7.
- `ScalarModel.decode(x)`: `x = round(9·x)/9` (`round_func9`; `torch.round` rounds half to even on ties) and the 8 layers of `decoder`:
  - `0`: `Conv1d(128 → 2048, kernel = delay_kernel_size = 5)`, with `weight_norm`. **It is not causal**: in the original code it does not receive `causal`, so the padding is symmetric, `get_padding(5) = 2` (it looks 2 frames ahead).
  - `1..5`: `ResDecoderBlock(n_in, n_out, stride, up_kernel)` with factors `5, 4, 4, 4, 3`, kernels `10, 8, 8, 8, 6` and channels `2048→1024→512→256→128→64`. Each one: `up_conv.layer` (`ConvTranspose1d` with `weight_norm`, causal: `kernel = 2·stride`, padding 0, the last `stride` samples are trimmed, no activation) and `convs.0..4`, five `ResidualUnit`s with dilations `1, 3, 5, 7, 9`: `conv1` (kernel 7, dilated, causal: left padding `dilation·(k−1)`) + `PReLU` `activation1`, `conv2` (kernel 1) + `PReLU` `activation2`, and a residual sum.
  - `6`: `PostProcessor`: repeats each sample 2 times in time (`x.repeat(1,1,2).view(B,-1,C)` over `[B,T,C]`) and then `Conv1d(64→64, kernel 7, causal)` **without** `weight_norm` (plain `weight`) + `PReLU` (`activation`).
  - `7`: `Conv1d(64 → 1, kernel 7, causal)` with `weight_norm`.
  - Total `5·4·4·4·3·2 = 1920` samples per frame (25 Hz × 1920 = 48 kHz).
- Checkpoint: 224 `scalar_model.decoder.*` keys (335 MB, F32): 57 `parametrizations.weight.original0/1` pairs (g `[C,1,1]`, v), the postprocessor's plain weight, biases and 51 single-parameter `PReLU`s (shape `[1]`). The `weight_norm` dimension is 0 in all of them (in `ConvTranspose1d` the weight is `[in, out, k]`, so g is `[in,1,1]`).
- This change **does not touch attention, RoPE or normalization** of HeartMuLa: the vocoder has no attention.
- There is an already proven strict loading pattern (`HeartMuLaModel.LoadWeights` with `CheckpointIndex`) and a header reader that does not touch data.

### Reference fixtures and how they were generated

- Existing, from the golden run (`tools/dump_reference.py codec`, heartlib `5858ca8f…`, torch 2.10.0, Apple M4, CPU, HeartCodec in fp32, see `MANIFEST.md`):
  - `codec_latents.safetensors`: `decode_in_seg0` `[2,128,744]` F32 (the exact input of `scalar_model.decode`), plus `latents_seg0`, `codes_seg0` and the noises (for C7).
  - `codec_wav.safetensors`: `wav` `[2,483840]` F32 (48 kHz stereo; peak `|x|` 0,609, RMS 0,076; the two channels differ, maximum difference 0,43), the final audio of 126 frames trimmed to `target_len = 126/12,5·48000`.
- There are no per-layer intermediate references. No new fixtures are generated and no Python is used in this change.

## Goals / Non-Goals

**Goals:**
- An fp32 vocoder that reproduces the Python audio within measured tolerances, on CPU and on CUDA.
- Strict loading, decoder only, with the `weight_norm` folded.
- Being able to write WAV and leave audio for the user to compare by ear.

**Non-Goals:**
- Flow matching, RVQ, DiT, `detokenize` and overlaps (C7), the codec encoder, bf16.
- Making `generate` produce audio (that arrives with C7).
- Processing in chunks to save memory or for performance.

## Decisions

1. **Modules with the keys of the already-folded checkpoint.** Each block derives from `Module` and directly registers `weight` and `bias` (and `weight` of `[1]` in `PReLU`), calling `F.conv1d`/`F.conv_transpose1d`/`F.prelu`, without wrapping an `nn.Conv1d` (which would add a `conv.weight` level). `ScalarVocoder` has a `decoder` field (`ModuleList` of 8 elements) and the blocks use the names `up_conv.layer`, `convs`, `conv1`, `conv2`, `activation1`, `activation2`, `conv`, `activation`. This way the checkpoint key without the `scalar_model.` prefix and with each `…parametrizations.weight.original0/1` pair replaced by `….weight` is exactly the model's, with no mapping table (project rule).
2. **Folding at load time, not in the model.** The model only has plain weights. The loader reads each pair with `CheckpointIndex.Read` (fp32), computes `w = g · v / ‖v‖` with the norm over all dimensions except 0 and copies it into the parameter. Tensor by tensor, under `no_grad`, without materializing the checkpoint; the decoder is 335 MB, but the pattern is kept for consistency with C2.
3. **Prior validation.** The expected (folded) key set is built from the checkpoint by filtering on `scalar_model.decoder.`, compared with the model's before reading data, and a single error is raised with missing keys, extra keys and mismatched shapes (with both shapes). An incomplete pair (`original0` without `original1` or vice versa) is reported naming the layer. The encoder and flow matching tensors are neither read nor counted as extra.
4. **Details that change the result and are replicated as is:** the symmetric (non-causal) first convolution; causal `ConvTranspose1d` with `kernel = 2·stride`, padding 0 and trimming of `stride` samples at the end; left padding `dilation·(k−1)` in the causal convolutions; the postprocessor's temporal duplication; `PReLU` with a single parameter; `torch.round` for the 9-level quantization.
5. **Testing strategy without intermediate references.** Since only the final input and output exist: tests of each block against naive loops written in the test itself (independent of TorchSharp); tests of shapes (`T·1920`), of causality (altering the input from frame `t` does not change the samples before `(t−2)·1920`) and of key names with a small vocoder with deterministic weights (same upsampling factors, reduced channels); and the final parity test. If the final parity fails it would be located by bisection with those tests and, only then, we would decide whether an intermediate fixture with Python is worth it (minimal venv with CPU torch and heartlib, already set up once in C3); it is not done beforehand.
6. **Tolerances.** Initial values: `cos_sim ≥ 0.9999`, `max_abs ≤ 1e-3` and signal-to-error ratio ≥ 60 dB over the 483 840 samples of both channels (peak 0,61, RMS 0,076). They are measured first on CPU and on CUDA and set with margin (≈ 4× the measured value, as in C3); they are not relaxed blindly.
   **Fixed after measuring:** CPU `cos_sim` 1,0000000, `max_abs` 1,0e-6 and 121 dB → thresholds `0.999999`, `4e-6` and 100 dB (≈ 4×). CUDA with TF32, the default of cuDNN (and of PyTorch), `max_abs` 4,1e-4 (4070) and 3,7e-4 (5060 Ti), 66 dB → `0.9999`, `1e-3` and 60 dB are kept (≈ 2,4× margin). The GPU error is about 450 times larger than on CPU **because of TF32**: with `backends.cudnn.allow_tf32 = false` both GPUs give `max_abs` 8,2e-7 and 122 dB, the same as the CPU, and the test requires it. No intermediate fixture with Python was needed.
7. **Device and memory.** A dedicated collection fixture loads only the decoder (335 MB) on `HEARTMULA_DEVICE` (CPU by default), without touching the 7 GiB HeartMuLa model. The largest activation is `[2, 64, ~1,43 M]` floats (~0,73 GB) and there are several live copies (padding, convolution, residual), so the real peak is measured on CPU and on each GPU and noted. The 2-channel batch is run at once, like Python.
8. **WAV writer.** `WavWriter.Write(path, audio [C, N], sampleRate, format)` with 16-bit PCM (saturation to `[-1, 1]`, rounding to nearest) or 32-bit float, interleaved channels. The tests read the file with a minimal reader written in the test itself so as not to check the writer against itself.
9. **Audio only on request.** A test marked with an attribute that skips it if there is no checkpoint or if `HEARTMULA_AUDIO_DIR` is not set writes `c6_vocoder_from_reference_latents.wav` and `c6_reference_python.wav` (both 16-bit PCM, same format, trimmed to 483 840 samples) in that directory. The listening task runs it pointing at `<audio-dir>/` (the user's Samba share) and asks the user to compare the two by ear.

## Risks / Trade-offs

- [The initial tolerances are not met because of accumulation order between the Apple M4 and this machine or CUDA] → measure and decide with data; they are not relaxed blindly. The vocoder is few multiplications per sample but 25 consecutive residual units, so the error accumulates and it is worth knowing how much.
- [A wrongly replicated layer (padding, trimming, duplication) gives plausible but different audio] → the shape and causality tests bound it, and the final parity detects it; bisection would be done with those tests.
- [High peak memory in the last stage] → measured; on the 4070 (12 GB) it should fit; if not, chunked processing would be considered in a separate change (out of scope).
- [CPU time] → a full window may take minutes; it is measured and noted, and optimizing is not a goal.
- [TorchSharp's `ConvTranspose1d` or padding behave differently from PyTorch] → the tests against naive loops detect it in small blocks.
- [The user hears a difference that the numbers do not see] → that is why the user is asked to compare by ear; if there is one, it is a result to investigate, not to dismiss.
- [The GPUs are busy with ollama] → before using them, check with `nvidia-smi` and `ollama ps`, and ask if they are not free.
