# generation-loop Specification

## Purpose

Define how the HeartMuLa frame generation loop behaves in C# compared with heartlib: sampling (CFG, top-k, temperature and random noise) picks the same tokens as Python when it receives the same inputs, and the full loop meets its invariants and reproduces the beginning of Python's generation, without demanding an equality that a stochastic process in bf16 cannot guarantee.

## Requirements

### Requirement: Sampling equivalent to heartlib with the same inputs
Starting from Python's raw logits and with the same seed, the system SHALL pick the same tokens as heartlib: applying CFG over the conditional and unconditional rows, filtering by top-k, applying the temperature and consuming the RNG with the same order, number and shape of draws per frame.

#### Scenario: Codebook 0 over the 126 steps
- **WHEN** it is sampled, with seed 1234 and the codebook 0 logits from `frames_logits` of each of the 126 calls, advancing the RNG as a full frame does
- **THEN** the 126 chosen tokens equal those of row 0 of `frames_py`

#### Scenario: Codebooks 1 to 7 on the decoder fixture's frames
- **WHEN** the seven codebooks of frames 0, 1, 2, 63, 124 and 125 are sampled, with the logits from `decoder_logits` and the RNG at the position that corresponds to each frame
- **THEN** the 42 chosen tokens equal those of `frames_py`

#### Scenario: Different seed
- **WHEN** the codebook 0 sampling is repeated with another seed
- **THEN** the sequence of chosen tokens differs from `frames_py`'s

### Requirement: Generation loop invariants
The system SHALL return the frames as a 64-bit integer tensor with shape `[codebooks, frames]` and tokens within the audio vocabulary, use the same tokens in the two CFG rows, be reproducible with the same seed, and stop at EOS or at the maximum length.

#### Scenario: Shape, type and range
- **WHEN** it generates with a small model and a maximum length of N frames without any EOS appearing
- **THEN** the output has shape `[codebooks, N + 1]`, type Int64 and all the tokens are in `[0, audio_vocab_size)`

#### Scenario: Same seed, same output
- **WHEN** it generates twice with the same seed and the same inputs
- **THEN** the two outputs are identical, and with another seed they differ

#### Scenario: Stop at EOS
- **WHEN** in any frame a codebook of the conditional row is at least `audio_eos_id`
- **THEN** the loop stops and that frame is not part of the output

#### Scenario: Maximum length
- **WHEN** an EOS token never appears
- **THEN** exactly the prompt's frame plus `max_length_ms / 80` frames are generated and the loop ends

#### Scenario: Aligned CFG rows
- **WHEN** it generates with CFG (batch 2)
- **THEN** in each frame the two rows receive the same tokens, so that both keep the same position and audio context

### Requirement: Reproducible prefix against Python
The system SHALL reproduce heartlib's first frames on CPU with the same seed and inputs, and SHALL report, without demanding equality beyond the prefix, where the generation diverges.

#### Scenario: Identical first frames
- **WHEN** it generates on CPU with HeartMuLa-oss-3B in bf16, seed 1234, CFG 1.5, top-k 50 and temperature 1.0, from the reference prompt
- **THEN** the first two frames (16 tokens) equal those of `frames_py`

#### Scenario: Divergence reported
- **WHEN** the full 126-frame generation is compared with `frames_py`
- **THEN** the result states the frame and codebook of the first difference and the number of equal tokens, and the 126 frames are not required to match

### Requirement: First divergence explained
The system SHALL be able to explain the first divergence from Python: given the same Python inputs up to that point, the probabilities that C# assigns to Python's token and to its own indicate whether it is a numerical difference compatible with bf16.

#### Scenario: Frame 3, codebook 5
- **WHEN** the decoder is evaluated with the backbone state and Python's tokens of frame 3 up to codebook 4
- **THEN** the rank and probability, under C#'s guided distribution, of Python's token and of the token C# chose are obtained, and it is noted whether the difference is explainable by numerical noise
