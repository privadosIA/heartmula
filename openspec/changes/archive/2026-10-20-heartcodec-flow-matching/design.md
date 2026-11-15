# Design

## Context

- `FlowMatching.inference_codes` (heartlib 5858ca8, `heartcodec/models/flow_matching.py`) for one segment:
  1. `quantized = vq_embed.get_output_from_indices(codes.transpose(1, 2))`, which in vector-quantize-pytorch sums the codebook vector selected by each of the 8 layers (dimension 32) and applies `project_out` (32→512). Then `cond_feature_emb` (Linear 512→512) and a nearest ×2 upsampling in time (372 → 744 frames at 25 Hz).
  2. `latent_masks` is zeros with `[:, :latent_length] = 2` (`latent_length = int(29.76·25) = 744`, so every frame is conditioned) and, for `scenario = "other_seg"`, `[:, :incontext_length] = 1`. Conditioning is `(mask > 0.5)·quantized + (mask < 0.5)·zero_cond_embedding1`. In-context latents are `true_latents · ((mask > 0.5)·(mask < 1.5))` and `incontext_length` is the number of frames with mask 1. In the reference fixture it is 0.
  3. `solve_euler` with `num_steps = 10` (`t_span = linspace(0, 1, 11)`), `guidance_scale = 1.25` in `detokenize` (2.0 is only the default of `inference_codes`). At each step `x[:, :incontext_length] = (1 − (1 − 1e-6)·t)·noise + t·incontext_x`, the estimator runs on a doubled batch with input `cat([cat([x, x]), cat([incontext_x, incontext_x]), cat([zeros_like(mu), mu])], dim = 2)` (256 + 256 + 512 = 1024 channels) and timestep `t` repeated for the batch, `dphi = uncond + g·(cond − uncond)`, `x = x + dt·dphi`, and finally `latents[:, :incontext_length] = incontext_latents`. `t` and `dt` are updated with the same floating-point operations as the reference (`t = t + dt; dt = t_span[step + 1] − t`).
- The estimator (`transformer.py`, `LlamaTransformer`): `inner_dim = 24·64 = 1536`, `inner_dim_2 = 3072`.
  - `proj_in = ProjectLayer(1024 → 1536, k = 3)`: `Conv1d(k = 3, padding 1)` over the transposed input, times `k^−0.5`, then `Linear`.
  - `adaln_single`: sinusoidal embedding of 512 features (`freqs = exp(−ln(10000)·arange(256)/256)`, `args = t·freqs·1000`, `cat([cos, sin])`), `TimestepEmbedding` (Linear 512→dim, SiLU, Linear dim→dim), then `Linear(SiLU(emb))` to `6·dim`; it returns the modulation and the embedded timestep.
  - 24 blocks at width 1536 (24 heads × 64) and 6 blocks at width 3072 (**24 heads × 128**, with RoPE over the full 128). A block with AdaLN-single adds its `scale_shift_table [6, dim]` to the modulation reshaped to `[B, 6, dim]`, chunks it into `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`, and does `x += gate_msa · attn(RMSNorm(x)·(1 + scale_msa) + shift_msa)` then `x += gate_mlp · mlp(RMSNorm(x)·(1 + scale_mlp) + shift_mlp)`.
  - `RMSNorm` has eps 1e-6, a `weight`, and no cast to fp32. Attention has bias-free q/k/v/o, RoPE over adjacent pairs with base 10000 and positions `0..T−1` computed in the input dtype, and non-causal SDPA with no mask. The MLP is `down(silu(gate(x))·up(x))` with hidden size `int(2·4·dim/3)` rounded up to a multiple of 256 (4096 and 8192).
  - After stage 1: `shift, scale = (scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim = 1)`, `s = LayerNorm(no affine, 1e-6)(s)·(1 + scale) + shift`, `x = cat([hidden_states, s], −1)` (1024 + 1536 = 2560), `connection_proj = ProjectLayer(2560 → 3072, k = 3)`. Stage 2 repeats the pattern with `adaln_single_2`, `scale_shift_table_2` and `norm_out_2`, then `proj_out = ProjectLayer(3072 → 256, k = 3)`.
- Checkpoint: `flow_matching.*` is 365 tensors, 5.97 GB, F32: `cond_feature_emb` 2, `zero_cond_embedding1` 1, `estimator` 326, `vq_embed` 36 (8 layers × {`_codebook.embed [1, 8192, 32]`, `embed_avg`, `cluster_size`, `initted`} plus `project_in` and `project_out`). Inference needs 8 + 2 of the `vq_embed` tensors, so 339 in total; the other 26 are training state.
- This change touches no HeartMuLa attention, RoPE or normalization. The torchtune-vs-HuggingFace differences recorded for HeartMuLa do not apply here: this is a separate DiT with its own RoPE (adjacent pairs, base 10000, no Llama-3 scaling) and a plain RMSNorm.
- C6 already provides `CheckpointIndex`-based strict loading (`CodecWeights`), `ScalarVocoder`, `WavWriter`, the audio-for-listening pattern and the per-device parity helpers.

### Reference fixtures and how they were generated

Existing, from the golden run (`tools/dump_reference.py codec`, heartlib 5858ca8, torch 2.10.0, Apple M4 CPU, HeartCodec in fp32, see `MANIFEST.md`); no new fixtures and no Python in this change:
- `codec_latents.safetensors`: `codes_seg0` `[1, 8, 372]` int64 (the 126 reference frames repeated to 372), `randn_0_detokenize` `[1, 744, 256]` (used as `true_latents` but masked to zero), `randn_1_inference_codes` `[1, 744, 256]` (the noise that drives flow matching, recorded by wrapping `torch.randn` inside `inference_codes`), `latents_seg0` `[1, 744, 256]` (the flow-matching output) and `decode_in_seg0` `[2, 128, 744]`.
- `codec_wav.safetensors`: `wav` `[2, 483840]`, already matched by the C6 vocoder.
- There are **no intermediate references** (quantized features, `mu`, per-step estimator outputs).

## Goals / Non-Goals

**Goals:**
- Codes → latents with measured parity, and codes → audio for one segment, in fp32 on CPU and GPU.
- Strict loading that reads only what inference needs.
- Faithful implementation of the whole solver, including the in-context path, with that path clearly marked as validated later.

**Non-Goals:**
- The multi-segment `detokenize` loop, cross-fades, code repetition and trimming (C8).
- Wiring `generate` to audio, and sharing a 12 GB GPU with HeartMuLa (C8).
- bf16, or any speed optimisation.

## Decisions

1. **Names replicate the checkpoint keys.** `FlowMatching` has `vq_embed` (with `layers`, a `ModuleList` of 8 layers each holding `_codebook` with a parameter `embed` of shape `[1, 8192, 32]`, and `project_out`), `cond_feature_emb` (`nn.Linear`), `zero_cond_embedding1` (`Parameter [512]`) and `estimator`. The estimator mirrors `LlamaTransformer`: `proj_in`, `adaln_single` (`emb.timestep_embedder.linear_1/linear_2`, `linear`), `transformer_blocks` and `transformer_blocks_2` (`attn.{q,k,v,o}_proj`, `attn_norm`, `mlp.{gate,up,down}`, `mlp_norm`, `scale_shift_table`), `scale_shift_table`, `connection_proj`, `adaln_single_2`, `scale_shift_table_2` and `proj_out`; a `ProjectLayer` has `ffn_1` (`nn.Conv1d`) and `ffn_2` (`nn.Linear`). `project_in` and the RVQ EMA tensors are not part of the model. The expected model has 339 parameters.
2. **Loader.** `FlowMatchingWeights.Load(model, index, device)` follows `CodecWeights`: filter keys under `flow_matching.`, compare with the model using headers only, collect every discrepancy into one `InvalidDataException`, then read tensor by tensor under `no_grad`. Keys matching the allow-list `vq_embed.layers.N._codebook.(embed_avg|cluster_size|initted)` and `vq_embed.project_in.(weight|bias)` are skipped without being read; `scalar_model.*` is ignored. The report states tensors read, tensors ignored and bytes.
3. **RVQ semantics.** The code lookup is `Σ_q embed_q[0][code_q]` followed by `project_out`, following vector-quantize-pytorch's `get_output_from_indices`. That library is not installed here, so this is the one point not checked against Python source or an intermediate fixture: unit tests check my definition against a naive loop, and the final latent parity validates it indirectly. If the parity fails, it is the first suspect, and the fallback is to rebuild a minimal Python venv with that library and dump the quantized features.
4. **RoPE.** Cosines and sines are computed in fp32 on the device from `inv_freq = 1 / base^(2i/dim)` and positions `0..T−1`, cached by length; each adjacent pair `(x[2i], x[2i + 1])` is rotated by `position · inv_freq[i]`, over the full head dimension (64 in stage 1, 128 in stage 2).
5. **Solver.** The estimator is called through a small interface (`IFlowEstimator`: input and timestep in, `dphi` out) so tests can inject stubs. `Solve` takes `noise`, `incontextLatents` and `incontextLength` explicitly, and `InferenceCodes` builds the conditioning and masks exactly as the reference. `t_span` is fp32 and `t`, `dt` follow the reference's operations.
6. **Memory discipline.** Parameters have `requires_grad = false`; everything runs under `no_grad`; each block disposes its temporaries; the estimator input is built once per step. Weights are created on the CPU and moved to the device (a 6 GB transient copy on CPU), so free RAM is checked before loading.
7. **Fixture of loaded models.** A collection fixture loads flow matching and the vocoder once per device (6.3 GB). Collections run serially, so the HeartMuLa fixture (7.3 GB) is disposed before it is created.
8. **Tolerances are measured, not guessed.** A starting bound of `cos_sim ≥ 0.9999` is used only as the first check; the final `max_abs` and signal-to-error thresholds per device are fixed with about 4× margin after measuring (CPU, RTX 4070, RTX 5060 Ti), and recorded in `tasks.md` and here.
9. **TF32.** The DiT is dominated by matmuls and the projection layers use cuDNN convolutions. Tests run with the library defaults (cuDNN TF32 on, as PyTorch), and the parity run on GPU is repeated with `backends.cudnn.allow_tf32` and the cuBLAS TF32 switch (`THSBackend_cublas_*_allow_tf32` exists in TorchSharp 0.107) turned off to see which one matters and to document how to get CPU-level exactness.
10. **Probe before scaling.** Heavy runs start with a cheap probe (one Euler step, then the full 10) to measure time and memory before committing, and free RAM and VRAM are checked first.
11. **Audio.** A test enabled by `HEARTMULA_AUDIO_DIR` (same attribute as C6) writes `c7_codes_to_wav_csharp.wav` (PCM16, 483 840 samples) next to the Python reference, and the task asks the user to listen.

## Risks / Trade-offs

- [The RVQ lookup semantics differ from my reading of vector-quantize-pytorch (for example the projection per layer rather than after the sum)] → no Python source or fixture to confirm, so the final parity is the check; if it fails, bisect with a minimal Python venv.
- [fp32 error accumulates over 30 layers × 10 Euler steps and exceeds the starting bound] → measure, investigate the dominant cause (RoPE precision, attention accumulation order, convolution) before relaxing; do not guess a threshold.
- [GPU TF32 error is larger than on CPU] → measured and documented, with the switches to turn it off.
- [Loading 5.97 GB uses a lot of RAM and time] → check free RAM first, report load time and bytes read; the model is created once per test collection.
- [The 12 GB GPU cannot hold HeartMuLa and the codec together] → out of scope here; recorded for C8, which will load them one after the other.
- [Only the no-in-context path can be checked] → the in-context path is implemented from the reference and unit-tested with stubs; its validation against real data is explicitly deferred to C8.
- [`/tmp` is RAM-backed] → tests do not write there, and scratch projects are built on disk.
