# Proposal

## Why

HeartMuLa already generates the music tokens (C4) with the correct prompt (C5) and reproducibly, but none of that can be heard: the codec that turns them into audio is missing. HeartCodec is large, and is split into two halves that can be tested separately with the fixtures that already exist:

- the **flow matching** (RVQ, 30-layer DiT transformer, Euler solver with CFG) generates the latents, and
- the **vocoder** (`scalar_model.decoder`) turns the latents into 48 kHz stereo audio.

This change is the second half, the vocoder, because it is self-contained (convolutions, no attention), has exact reference input and output in `tests/fixtures` and produces **the project's first audible audio**. The flow matching and the `detokenize` loop with overlapping windows are the next change (C7).

What I measured by reading heartlib and the checkpoint (not repo code):

- HeartCodec-oss is 818 F32 tensors (6,2 GB, 2 shards): `flow_matching` 365, `scalar_model.encoder` 229 (not needed for decoding) and `scalar_model.decoder` **224 tensors, 335 MB**.
- The vocoder does `round(9·x)/9` and 8 layers: an input convolution (128→2048, kernel 5, **non-causal**), five upsampling blocks (factors 5, 4, 4, 4, 3; channels 2048→64) with causal `ConvTranspose1d` and five dilated residual units (1, 3, 5, 7, 9) with `PReLU`, a postprocessor that duplicates each sample and a final 64→1 convolution. In total 5·4·4·4·3·2 = **1920 samples per latent frame** (25 Hz × 1920 = 48 kHz).
- The weights with `weight_norm` come as `parametrizations.weight.original0` (g, shape `[C,1,1]`) and `original1` (v): 57 pairs in the decoder. They must be folded (`w = g·v/‖v‖`, norm over all dimensions except 0) when loading.
- The fixtures `codec_latents.safetensors` (`decode_in_seg0` `[2,128,744]`, the exact vocoder input) and `codec_wav.safetensors` (`wav` `[2,483840]`, peak 0,609, RMS 0,076, different channels) are enough: with a single segment, `detokenize` returns the vocoder trimmed to 483 840 samples, so the valid comparison is `vocoder(decode_in_seg0)[:, :483840]` against `wav`.

## What Changes

- **Loading the vocoder decoder** from `HeartCodec-oss`: only the `scalar_model.decoder.*` keys (without reading the encoder or the flow matching), with strict validation of keys and shapes against the model (a single error with all discrepancies, as in HeartMuLa's `LoadWeights`) and **`weight_norm` folding in fp32**.
- **Vocoder in C#/TorchSharp**: causal and non-causal convolution, causal `ConvTranspose1d` with trimming, `PReLU`, residual unit, upsampling block, postprocessor, 9-level quantization and `Decode`, with names that replicate the keys of the already-folded checkpoint so that the dump fits without mappings.
- **WAV writer** (16-bit PCM and 32-bit float, stereo, 48 kHz) to be able to hear the result.
- **Audio to listen to**: `c6_vocoder_from_reference_latents.wav` (our vocoder applied to `decode_in_seg0`) and `c6_reference_python.wav` (Python's `wav`) in `<audio-dir>/`, to compare them by ear, and the user is asked to listen to them. By default the tests do not write outside the repo: audio is only written if `HEARTMULA_AUDIO_DIR` is defined.
- Time and peak memory are measured and noted on CPU and on CUDA. The codec runs in **fp32** (project rule: in bf16 the audio degrades).

Out of scope: flow matching, RVQ, DiT, `detokenize` and the overlaps (C7), the codec encoder, bf16, and changing `generate` (which will still not produce end-to-end audio until C7 is finished).

## Capabilities

### New Capabilities
- `codec-vocoder`: the C# HeartCodec vocoder strictly loads its weights from the checkpoint, folds the `weight_norm`, converts latents into 48 kHz stereo audio with parity against the Python reference, and the result can be written as WAV.

### Modified Capabilities
None.

## Impact

- **Numerical parity criterion** (fp32 audio, peak 0,61, comparison over the 483 840 samples of the two channels): `cos_sim ≥ 0.9999`, `max_abs ≤ 1e-3` and signal-to-error ratio ≥ 60 dB. These are reasonable initial values that are **measured first** on CPU and on CUDA and set with margin; if they are not met, the cause is investigated (they are not relaxed blindly) and, if it is a legitimate divergence in accumulation order, it is documented with data. The fixtures come from an Apple M4 (CPU, fp32): on this x86 machine and on CUDA bit-for-bit equality is not expected.
- Code: new `HeartMuLa.Codec` namespace in `src/HeartMuLa.Net/` (vocoder, weight folding, WAV writer) and new tests.
- Data: none new (`codec_latents` and `codec_wav` are used); Python is not needed.
- Dependencies: none.
- Audio: it does produce audio; it is left in the Samba shared folder and the user is asked to listen to it.
- Documentation: README.
