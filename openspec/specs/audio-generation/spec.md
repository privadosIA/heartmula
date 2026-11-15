# audio-generation Specification

## Purpose

Define how the `generate` command turns the generated tokens into a WAV file: the model is released before the codec loads so a 12 GB GPU suffices, the codec precision defaults make GPU audio match the CPU, and the output is reproducible for a given seed.

## Requirements

### Requirement: Generate writes a WAV
When `--wav <path>` is given, `generate` SHALL decode the generated frames with the codec after the token loop and write the audio as a stereo 48 kHz WAV, PCM16 by default or 32-bit float with `--wav-format float32`. `frames.safetensors` SHALL still be written, and without `--wav` the behaviour SHALL be unchanged.

#### Scenario: WAV produced
- **WHEN** `generate --wav song.wav` finishes
- **THEN** `song.wav` exists, is a valid 2-channel 48 kHz WAV whose length is `int(frames / 12.5 · 48000)` samples, and `frames.safetensors` exists next to the usual output

#### Scenario: No WAV requested
- **WHEN** `generate` runs without `--wav`
- **THEN** the codec is neither loaded nor read and only the frames are written

#### Scenario: Invalid option
- **WHEN** `--wav-format` has a value other than `pcm16` or `float32`
- **THEN** the command fails before loading any model, naming the accepted values

### Requirement: HeartMuLa is released before the codec loads
Before reading the codec, `generate` SHALL dispose the HeartMuLa model, its KV caches and masks and its prompt tensors, so that the device memory they used is available to the codec. (TorchSharp 0.107 has no `cuda.empty_cache()`: freed blocks stay in libtorch's caching allocator, which reuses them and returns its cached blocks to the driver by itself when an allocation would otherwise fail.)

#### Scenario: Model, caches and tensors are gone
- **WHEN** the model has been unloaded
- **THEN** its parameters, its KV caches and masks and the given tensors are disposed, and a model of the same size can be created on the same device

#### Scenario: Whole run fits a 12 GB GPU
- **WHEN** `generate --wav` runs on the 12 GB GPU with the real checkpoints
- **THEN** it completes without an out-of-memory error and the sampled peak VRAM is below the GPU's capacity

### Requirement: A model can be reused for another generation
`AudioStage.ReleaseGeneration` SHALL dispose the KV caches, the causal masks and the given tensors of a generation while keeping the model's parameters, so that the same model can generate again (the next `Generate` rebuilds the caches). It is the alternative to unloading when the codec runs on another device or both fit together.

#### Scenario: Caches gone, model alive
- **WHEN** a generation has been released with `ReleaseGeneration`
- **THEN** the caches and the given tensors are disposed, the parameters are intact and `SetupCaches` enables the caches again

### Requirement: Codec device and precision
The codec SHALL run in fp32 on the device given by `--codec-device` (default: the model's device), with cuDNN TF32 disabled unless `--codec-tf32` is given, and the TF32 setting SHALL be restored after the decode.

#### Scenario: Different device for the codec
- **WHEN** the model runs on one GPU and `--codec-device` names another device
- **THEN** the codec is loaded and run on that device and the audio is produced

### Requirement: Reproducible audio
For the same checkpoint, inputs, seed and device, `generate --wav` SHALL produce the same audio file, and the seed already printed and stored for the frames SHALL govern the codec noise too.

#### Scenario: Same seed twice
- **WHEN** `generate --seed 7 --wav` is run twice with the same inputs on the same device
- **THEN** the two WAV files are byte-identical

### Requirement: Audio for listening
The test suite SHALL be able to leave a real generated clip in `HEARTMULA_AUDIO_DIR` and SHALL NOT write any audio when that variable is unset.

#### Scenario: Variable unset
- **WHEN** the suite runs without `HEARTMULA_AUDIO_DIR`
- **THEN** the audio test is skipped with an explicit reason and no file is written
