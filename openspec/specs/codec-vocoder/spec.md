# codec-vocoder Specification

## Purpose

Define that HeartCodec's vocoder in C# loads its weights from the checkpoint strictly and with the `weight_norm` folded, turns latents into 48 kHz stereo audio with parity against the Python reference, and that the result can be written as a WAV file to be listened to.

## Requirements

### Requirement: Strict loading of the vocoder's decoder
The system SHALL load from the HeartCodec checkpoint only the `scalar_model.decoder.*` tensors, verifying before copying data that the keys and shapes match the model's exactly, and SHALL fail with a single error that lists all the discrepancies.

#### Scenario: Correct load
- **WHEN** HeartCodec-oss is loaded into the vocoder
- **THEN** the 224 decoder tensors are read, no model key is left unloaded or extra, and loading finishes without error

#### Scenario: Tensors outside the decoder
- **WHEN** the checkpoint contains tensors of the `scalar_model` encoder or of the flow matching
- **THEN** they are ignored: their data is not read and they do not count as extra keys

#### Scenario: Missing key, extra key or different shape
- **WHEN** a decoder key is missing, there is a decoder key that the model does not have, or a key has another shape
- **THEN** loading fails before reading tensor data, with a single error that lists all the discrepancies and, for shapes, both forms

### Requirement: Folding the weight_norm
The system SHALL fold each pair `parametrizations.weight.original0` (g) and `original1` (v) into a single weight `w = g · v / ‖v‖`, with the norm computed over all dimensions except 0, in fp32, for both convolutions and transposed convolutions.

#### Scenario: Identity with g equal to the norm
- **WHEN** g is the per-channel norm of v
- **THEN** the folded weight equals v

#### Scenario: General formula
- **WHEN** a pair with arbitrary g and v is folded, in a convolution and in a transposed convolution
- **THEN** each row of the folded weight has a norm equal to its g and the same direction as its v

#### Scenario: Incomplete pair
- **WHEN** `original0` or `original1` of a pair is missing
- **THEN** loading fails naming the layer

### Requirement: Output shape and causality
The system SHALL produce 1920 audio samples for each input latent frame, and the upsampling and residual layers SHALL be causal, so that the output only depends on the current and past input except for the input convolution, which looks two frames ahead.

#### Scenario: Output length
- **WHEN** an input of `T` frames with `B` channels is decoded
- **THEN** the output has shape `[B, T · 1920]`

#### Scenario: Causality
- **WHEN** the input is altered from frame `t` onwards
- **THEN** the output samples before `(t − 2) · 1920` do not change

#### Scenario: Quantization to 9 levels
- **WHEN** the input is quantized before the layers
- **THEN** each value `x` is replaced by `round(9 · x) / 9`, rounding to the nearest even on ties

### Requirement: Parity with the Python reference
The system SHALL reproduce the reference audio: with the vocoder applied to the reference input and trimmed to the final length, the result matches Python's audio within the fixed tolerances.

#### Scenario: Reference audio on CPU
- **WHEN** `decode_in_seg0` is decoded with HeartCodec-oss in fp32 and the first 483 840 samples of both channels are taken
- **THEN** the shape matches `wav`, and `cos_sim`, `max_abs` and the signal-to-error ratio meet the thresholds fixed in the design after measuring them

#### Scenario: Reference audio on CUDA
- **WHEN** it is repeated on a GPU with the same inputs
- **THEN** the same thresholds are met

### Requirement: WAV writing
The system SHALL write stereo audio as a valid WAV file, in 16-bit PCM or 32-bit floating point, at the indicated sample rate.

#### Scenario: Header and data
- **WHEN** audio of `C` channels, `N` samples and 48 kHz is written
- **THEN** the file has a valid RIFF/WAVE header with those channels, rate and data size, and the data interleaves the channels sample by sample

#### Scenario: Round trip
- **WHEN** audio is written and read back
- **THEN** in floating point the values are identical, and in 16-bit PCM each value differs by at most half a quantization step

#### Scenario: Out-of-range values
- **WHEN** there are samples outside `[-1, 1]` and it is written in 16-bit PCM
- **THEN** they are clipped to the extremes without overflowing

### Requirement: Test audio only on request
The tests SHALL write audio files only when an output directory is given through `HEARTMULA_AUDIO_DIR`, and never outside the repository by default.

#### Scenario: Without an output directory
- **WHEN** `HEARTMULA_AUDIO_DIR` is not defined
- **THEN** no test writes audio

#### Scenario: With an output directory
- **WHEN** `HEARTMULA_AUDIO_DIR` points to a directory
- **THEN** the vocoder's audio and the reference audio are written to it, in the same format, with names that identify them
