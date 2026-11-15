# Spec Delta

## Purpose

Defines the shared library that holds the model-agnostic building blocks of the music-generation ports (checkpoint reading, loading, rotary embeddings, basic layers, audio output, seeding, test infrastructure), so that several ports reuse one tested implementation instead of copies.

## ADDED Requirements

### Requirement: Independence from any model
The shared library SHALL NOT depend on any model-specific project, and a port SHALL be able to consume it through a relative project reference to a sibling repository, building with the same native backend switch as the port itself.

#### Scenario: Core builds alone
- **WHEN** the library and its tests are built from their own repository with the CPU backend
- **THEN** the build succeeds without HeartMuLa.Net or any other port present

#### Scenario: Port builds against the core
- **WHEN** HeartMuLa.Net is built with the core repository checked out next to it
- **THEN** it builds without copying core sources into its own tree, with the backend chosen by the same `TorchBackend` property

### Requirement: Unchanged behavior of moved components
A component moved into the shared library SHALL behave identically to its previous version in HeartMuLa.Net: same results for the same inputs, same error conditions, same checkpoint key names, same dtype handling.

#### Scenario: Existing suite stays green
- **WHEN** the tests of HeartMuLa.Net and of the shared library are run without a checkpoint
- **THEN** the total of passed tests equals the 252 passed before the extraction, with 0 failures, and the total of skipped tests equals the 36 skipped before

#### Scenario: Golden-run parity unchanged
- **WHEN** the suite is run with the HeartMuLa checkpoint available
- **THEN** every golden-run parity test (backbone, decoder, sampling, vocoder, flow matching, windowed decode) passes with the same measured figures as before the extraction

#### Scenario: Checkpoint key names preserved
- **WHEN** the parameter names of a moved layer are listed
- **THEN** they are exactly the names that the checkpoints of the supported models use (`scale` for the torchtune RMSNorm, `weight` for the flow RMSNorm, and so on)

### Requirement: Explicit rotary layout
The shared rotary embedding SHALL require the caller to choose its layout, either rotation over adjacent pairs `(x[2i], x[2i+1])` or rotation of the two halves of the head dimension, and SHALL NOT provide a default.

#### Scenario: Adjacent pairs matches the existing models
- **WHEN** the adjacent-pairs layout is applied with the same base, scaling and positions that HeartMuLa's backbone and HeartCodec use today
- **THEN** the output equals the previous implementation's output bit for bit on the same device and dtype

#### Scenario: Half-rotation layout matches the HuggingFace convention
- **WHEN** the half-rotation layout is applied to a fixed input with known positions
- **THEN** the output matches a reference computed with the HuggingFace Llama formulation (`x * cos + rotate_half(x) * sin`) within 1e-6 in fp32

#### Scenario: No layout chosen
- **WHEN** code tries to create the rotary embedding without choosing a layout
- **THEN** it does not compile (the layout is a required argument)

### Requirement: Layers accept variants without changing existing call sites
The shared layers SHALL offer the variants needed by more than one model (a non-causal transposed convolution with symmetric padding, a SwiGLU MLP with a configurable hidden size, attention with an optional padding mask and optional cross-attention input) such that calls that do not use the variants behave as before.

#### Scenario: Non-causal transposed convolution
- **WHEN** a transposed convolution is created in non-causal mode with kernel `k` and stride `s`
- **THEN** its output length is `(T − 1)·s + k − 2·p` with `p = (k − s) / 2`, and it matches the reference convolution computed with naive loops

#### Scenario: Default call sites unchanged
- **WHEN** the existing causal transposed convolution, the SwiGLU MLP with its derived hidden size and mask-free attention are used as HeartMuLa.Net uses them
- **THEN** their outputs equal the previous outputs

#### Scenario: Padding mask
- **WHEN** attention receives a mask that hides the last `m` positions of a sequence
- **THEN** the output for the visible positions equals the output computed on the sequence truncated to its visible length

### Requirement: Window assembly is parametrized
The assembler that joins consecutive audio windows with a linear cross-fade SHALL receive the window length, hop and overlap, in samples, from its caller.

#### Scenario: HeartCodec window sizes
- **WHEN** the assembler is created with the HeartCodec values (window 1 428 480 samples, hop 1 228 800, overlap 199 680)
- **THEN** its output is identical to the previous assembler's on the existing test inputs

#### Scenario: Different sizes
- **WHEN** the assembler is created with a window of 1000 samples, a hop of 750 and an overlap of 250
- **THEN** joining two windows produces 1750 samples whose last 250 of the first window are the linear cross-fade of both windows

### Requirement: Reusable test infrastructure
The shared library SHALL publish the test helpers that every port needs: numerical parity comparison (cosine similarity and maximum absolute error), reference-fixture reading with checksum verification, checkpoint-gated test attributes parametrized by environment variables, bf16 helpers and a deterministic random-number helper.

#### Scenario: Fixture checksum mismatch
- **WHEN** a fixture file does not match its recorded SHA-256
- **THEN** the helper fails the test with a message naming the file, the expected and the actual checksum

#### Scenario: Gated test without checkpoint
- **WHEN** a test is gated on a checkpoint and the environment variable that points to it is not set
- **THEN** the test is reported as skipped with a reason, not as passed

### Requirement: Documented TorchSharp findings
The repository of the shared library SHALL document the TorchSharp behaviors that every port must know, starting with the fact that `manual_seed` seeds only the CPU generator.

#### Scenario: Findings available to the next port
- **WHEN** a developer starts a new port
- **THEN** the core's documentation lists the known TorchSharp pitfalls with the evidence that was measured for each
