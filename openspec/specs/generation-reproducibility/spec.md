# generation-reproducibility Specification

## Purpose

Define that a HeartMuLa generation can be repeated: the command accepts a seed (or picks one and shows it), seeds effectively on CPU and CUDA right before generating, and leaves in the output file the seed and the parameters needed to run it again.

## Requirements

### Requirement: Seed of the generation command
The `generate` command SHALL accept a seed with `--seed`, and SHALL pick a random seed when none is given, always showing it in the output so that the run can be repeated.

#### Scenario: Seed given
- **WHEN** `generate --seed 1234` is run
- **THEN** the run uses seed 1234 and shows it as `seed=1234`

#### Scenario: Seed omitted
- **WHEN** `generate` is run without `--seed`
- **THEN** a random seed is picked, shown as `seed=N`, and running again with that `--seed N` repeats the generation

#### Scenario: Invalid seed
- **WHEN** `--seed` receives a value that is not a 64-bit integer
- **THEN** the command fails with a clear message before loading the model

### Requirement: Effective seeding on CPU and CUDA
The system SHALL seed the CPU random generator and, if a CUDA GPU is available, the CUDA one too, so that the same seed gives the same sequence on each device and another seed gives another one.

#### Scenario: CPU
- **WHEN** it is seeded with seed S, a random sequence is generated on CPU, it is seeded again with S and another one is generated
- **THEN** the two sequences are identical, and with another seed the sequence differs

#### Scenario: CUDA
- **WHEN** a CUDA GPU is available and the previous scenario is repeated with CUDA tensors
- **THEN** the two sequences with the same seed are identical and with another seed they differ

#### Scenario: No GPU
- **WHEN** no CUDA GPU is available
- **THEN** seeding does not fail and the CUDA test is skipped with an explicit reason

#### Scenario: Moment of seeding
- **WHEN** `generate` loads the model, prepares the prompt and starts generating
- **THEN** the seed is applied after preparing the prompt and right before generation, so that neither loading nor preparation consumes random numbers that affect the frames

### Requirement: Reproducible generation
With the same seed, the same device and the same inputs, the system SHALL generate exactly the same frames, and with another seed, different ones.

#### Scenario: Same seed
- **WHEN** the `generate` command is run twice with the same seed, inputs and device
- **THEN** the two output files are byte-identical

#### Scenario: Another seed
- **WHEN** `generate` is run with another seed
- **THEN** the frames of the output file differ

#### Scenario: Reproducing Python on CPU
- **WHEN** `generate --seed 1234` is run on CPU with the golden run's ids and a length of at least 3 frames
- **THEN** the first 3 frames of the saved file equal those of `frames_py`

### Requirement: Reproducibility metadata in the output
The frames file SHALL store, as metadata, the seed and the generation parameters (`cfg_scale`, maximum length, temperature, top-k, dtype, device and prompt origin), without altering the tensors or the result of callers that do not ask for them.

#### Scenario: Metadata present
- **WHEN** `generate` saves `frames.safetensors`
- **THEN** the file contains the metadata with the seed used and the parameters, as strings, and they can be read with Python's `safetensors` library

#### Scenario: Tensors do not change
- **WHEN** the file is read with the project's reader
- **THEN** the `frames` tensor has the generated shape and values, and the metadata does not appear as tensors

#### Scenario: Writer without metadata
- **WHEN** a file is saved without giving metadata
- **THEN** the content is identical to what the writer produced before this change
