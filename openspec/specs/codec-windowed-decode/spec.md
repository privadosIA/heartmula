# codec-windowed-decode Specification

## Purpose

Define how HeartCodec codes of any length are decoded to stereo audio in C#: overlapping windows with in-context latents, linear cross-fades and trimming, with parity against heartlib's `detokenize`, including the in-context path checked against real reference data.

## Requirements

### Requirement: Window planning matches the reference
The system SHALL plan the decoding of `N` code frames exactly as heartlib's `detokenize` does: windows of 372 frames with a hop of 320 and an overlap of 52 frames, codes shorter than 372 repeated up to 372, codes whose length minus 52 is not a multiple of 320 repeated and cut up to the next such length, and a target length of `int(N / 12.5 · 48000)` samples.

#### Scenario: Reference lengths
- **WHEN** `N` is 1, 126, 371, 372, 373, 400, 691, 692, 693 or 1000
- **THEN** the padded length, the number of windows and the target length equal those the reference formulas give (for example 126 → 372 and 1 window, 400 → 692 and 2 windows)

#### Scenario: Repetition keeps the original codes first
- **WHEN** codes are repeated to reach the padded length
- **THEN** the first `N` frames are the original codes and the following frames repeat them in order, as `cat([codes, codes])` followed by a cut does

### Requirement: In-context hand-over between windows
For every window after the first, the system SHALL take the last 104 latent frames of the previous window as in-context latents, complete them to 744 frames with fresh noise, and run flow matching with `incontext_length = 104`; the first window SHALL run with no in-context frames.

#### Scenario: Second window against real data
- **WHEN** window 1 of the multi-window reference is run with the recorded codes, the recorded noise and the reference window-0 latents
- **THEN** its latents match the reference `latents_seg1` within the measured thresholds, and its first 104 frames equal the reference window-0 tail exactly

#### Scenario: First window has no in-context
- **WHEN** window 0 runs
- **THEN** no in-context blending is applied and its latents match `latents_seg0` within the measured thresholds

### Requirement: Cross-faded assembly and trimming
The system SHALL decode each window's latents with the vocoder, join consecutive windows with a linear cross-fade over 199 680 samples (the previous window fading out with `1 − ramp` and the next fading in with `ramp`, `ramp = linspace(0, 1, 199 680)`), and cut the result to the target length.

#### Scenario: Cross-fade arithmetic
- **WHEN** two known signals are joined
- **THEN** the overlap equals `previous·(1 − ramp) + next·ramp` computed in double precision, the samples before and after it are untouched, and the output length is `len(previous) + len(next) − 199 680`

#### Scenario: Reference inputs give the reference audio
- **WHEN** the vocoder inputs recorded for both windows of the multi-window reference are decoded and assembled
- **THEN** the output has shape `[2, 1 536 000]` and matches the reference wav within the measured thresholds, cross-fade region included

#### Scenario: Full decode from the codes
- **WHEN** the 400-frame reference codes are decoded with the recorded noise
- **THEN** the latents of both windows match the reference within the measured thresholds, the output has shape `[2, 1 536 000]`, and its global agreement with the reference wav (`cos_sim`, signal-to-error ratio) meets the measured thresholds

#### Scenario: Quantization flips are the only source of local deviations
- **WHEN** the full decode deviates from the reference audio by more than the float32 noise floor
- **THEN** the deviation lies within the neighbourhood of a latent frame whose value after the vocoder's `round(9·x)/9` quantization differs from the reference's, and the number of such flipped values is small (the vocoder is discontinuous, so a latent error of 1e-6 can change one quantization step)

#### Scenario: Single window is unchanged
- **WHEN** 126 frames are decoded
- **THEN** the result equals what `DecodeSegment` of the single-segment decode gives, trimmed to 483 840 samples

### Requirement: Reproducible noise
When no noise is injected, the system SHALL draw it from the seeded generator in the order the reference draws it (padding noise of the in-context latents, then flow-matching noise, window by window), so that the same seed and the same codes give the same audio on the same device.

#### Scenario: Same seed twice
- **WHEN** the same codes are decoded twice after seeding with the same value
- **THEN** the two outputs are identical

### Requirement: GPU precision defaults
The system SHALL decode on GPU with cuDNN TF32 disabled unless explicitly allowed, restoring the previous setting afterwards.

#### Scenario: GPU equals CPU by default
- **WHEN** the multi-window reference is decoded on each available GPU with the defaults
- **THEN** the output meets the CPU thresholds, and `cudnn.allow_tf32` has its previous value after the call

#### Scenario: TF32 allowed
- **WHEN** TF32 is explicitly allowed
- **THEN** the output meets the looser thresholds measured for that mode
