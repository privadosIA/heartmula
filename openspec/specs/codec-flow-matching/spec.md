# codec-flow-matching Specification

## Purpose

Define how HeartCodec's flow matching runs in C#: it loads its weights strictly, turns the 8 RVQ code streams of one segment into the latents the vocoder consumes with parity against the Python reference, and, together with the vocoder, decodes one segment of codes into stereo audio.

## Requirements

### Requirement: Strict loading of the flow-matching weights
The system SHALL load the `flow_matching.*` tensors needed for inference from the HeartCodec checkpoint, verify before copying any data that their keys and shapes match the model exactly, and fail with a single error listing every discrepancy. The RVQ training-state tensors SHALL be ignored through an explicit allow-list, and `scalar_model.*` SHALL not be read.

#### Scenario: Complete checkpoint
- **WHEN** HeartCodec-oss is loaded into the flow-matching model
- **THEN** the 339 inference tensors are read, none is missing or unexpected, and the 26 RVQ training-state tensors (`embed_avg`, `cluster_size` and `initted` of each of the 8 layers, and `project_in`) are ignored without being reported

#### Scenario: Tensors of other modules
- **WHEN** the checkpoint contains `scalar_model.*` tensors
- **THEN** they are neither read nor counted as unexpected

#### Scenario: Missing key, unexpected key or different shape
- **WHEN** an inference tensor is missing, a `flow_matching.*` tensor that is neither expected nor allow-listed is present, or a tensor has a different shape
- **THEN** loading fails before reading tensor data, with one error that lists all of them and, for shapes, both shapes

#### Scenario: Model left untouched on failure
- **WHEN** validation fails
- **THEN** no parameter of the model has been modified

### Requirement: RVQ code lookup and conditioning
The system SHALL turn the codes `[B, 8, T]` into conditioning features by summing, over the 8 quantizer layers, the codebook vector selected by each code, applying the quantizer's output projection, applying `cond_feature_emb`, and repeating each frame twice in time (nearest-neighbour ×2); frames outside the conditioned region SHALL use the learned `zero_cond_embedding1` instead.

#### Scenario: Lookup
- **WHEN** codes are looked up in a codebook with known values
- **THEN** the result equals the per-layer sum of the selected vectors followed by the output projection, computed independently in the test

#### Scenario: Time upsampling
- **WHEN** the features of `T` frames are upsampled
- **THEN** the result has `2T` frames and frames `2t` and `2t + 1` both equal frame `t`

#### Scenario: Frames outside the conditioned region
- **WHEN** the latent length is shorter than the number of frames
- **THEN** the frames at or beyond the latent length use `zero_cond_embedding1` and the earlier ones use the quantized features

### Requirement: DiT estimator
The system SHALL implement the two-stage DiT estimator exactly as specified by the reference: width 1536 with 24 heads of 64 for the first 24 blocks and width 3072 with 24 heads of 128 for the last 6, RMSNorm without casting to fp32, rotary embeddings over adjacent pairs of the full head dimension with base 10000, non-causal scaled dot-product attention without mask, SwiGLU feed-forward with hidden size `int(2·4·dim/3)` rounded up to a multiple of 256, AdaLN-single modulation with a per-block `scale_shift_table`, kernel-3 projection layers, and a timestep embedding of 512 sinusoidal features.

#### Scenario: Output shape
- **WHEN** the estimator receives `[B, T, 1024]` features and `B` timesteps
- **THEN** it returns `[B, T, 256]`

#### Scenario: Rotary embeddings
- **WHEN** rotary embeddings are applied to a tensor
- **THEN** each adjacent pair `(x[2i], x[2i + 1])` is rotated by the angle `position · base^(−2i/dim)`, matching a naive loop written in the test, and position 0 leaves the tensor unchanged

#### Scenario: Timestep embedding and modulation
- **WHEN** a timestep `t` is embedded
- **THEN** the sinusoidal features are `cos` and `sin` of `t · 1000 · exp(−ln(10000)·i/256)` for `i` in `0..255`, and the six modulation vectors of a block are the sum of its `scale_shift_table` and the projected embedding

#### Scenario: Feed-forward size
- **WHEN** the feed-forward width is computed for dimensions 1536 and 3072
- **THEN** the hidden sizes are 4096 and 8192

#### Scenario: Parameter names
- **WHEN** the estimator is built
- **THEN** it has exactly 326 parameters whose names are the checkpoint keys under `flow_matching.estimator.`

### Requirement: Euler solver with classifier-free guidance
The system SHALL generate latents with a fixed-step Euler solver over `t` from 0 to 1, running the estimator on a doubled batch (unconditional with zeroed conditioning, then conditional) and combining the two as `uncond + guidance · (cond − uncond)`; the in-context frames SHALL be blended into the state at every step and restored exactly in the result. The noise SHALL be an explicit parameter so that it can be injected; when omitted it comes from the seeded random generator.

#### Scenario: Guidance formula
- **WHEN** the estimator is replaced by a stub that returns known conditional and unconditional values
- **THEN** each step advances the state by `dt · (uncond + guidance · (cond − uncond))`

#### Scenario: Injected noise is deterministic
- **WHEN** the same noise, codes and settings are used twice
- **THEN** the two results are identical

#### Scenario: In-context frames
- **WHEN** `incontext_length` is greater than 0
- **THEN** at every step the first `incontext_length` frames of the state equal `(1 − (1 − 1e-6)·t)·noise + t·incontext_latents`, and in the result they equal the in-context latents exactly

#### Scenario: No in-context frames
- **WHEN** `incontext_length` is 0
- **THEN** the state is never blended and the result is the Euler solution from the noise

### Requirement: Latent parity with the Python reference
The system SHALL reproduce the flow-matching output of the reference run: with `codes_seg0` and the recorded noise, the latents match `latents_seg0` within the thresholds fixed in the design after measuring them.

#### Scenario: Reference latents on CPU
- **WHEN** `codes_seg0` and `randn_1_inference_codes` are fed to the solver with 10 steps and guidance 1.25 on CPU in fp32
- **THEN** the output has shape `[1, 744, 256]` and its `cos_sim`, `max_abs` and signal-to-error ratio against `latents_seg0` meet the measured thresholds

#### Scenario: Reference latents on CUDA
- **WHEN** the same run is repeated on a GPU
- **THEN** the thresholds fixed for that device are met

### Requirement: Latent reordering for the vocoder
The system SHALL reorder `[B, T, 256]` latents into the `[2B, 128, T]` layout the vocoder consumes by splitting the 256 channels into two stereo channels of 128.

#### Scenario: Reference reordering
- **WHEN** `latents_seg0` is reordered
- **THEN** the result is exactly `decode_in_seg0`

### Requirement: Codes to audio for one segment
The system SHALL decode one segment of codes into stereo audio by chaining the code lookup, the solver, the reordering and the vocoder.

#### Scenario: Reference audio
- **WHEN** `codes_seg0` and the recorded noise are decoded and the first 483 840 samples of both channels are taken
- **THEN** they match `wav` within the thresholds fixed in the design after measuring them, on CPU and on CUDA

### Requirement: Audio for listening only on request
Tests SHALL write audio files only when `HEARTMULA_AUDIO_DIR` is set, and never outside the repository by default.

#### Scenario: Without an output directory
- **WHEN** `HEARTMULA_AUDIO_DIR` is not set
- **THEN** no test writes audio

#### Scenario: With an output directory
- **WHEN** `HEARTMULA_AUDIO_DIR` points to a directory
- **THEN** the audio decoded from codes is written there as 16-bit PCM with the same format as the Python reference, under a name that identifies it
