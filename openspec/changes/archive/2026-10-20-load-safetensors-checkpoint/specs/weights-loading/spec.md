# Spec Delta

## Purpose

Defines how the port reads safetensors checkpoints (sharded or single-file) and links them to a model, preserving the exact values of the file and failing explicitly on any key or shape mismatch.

## ADDED Requirements

### Requirement: Resolution of sharded checkpoints
The system SHALL resolve a checkpoint directory through `model.safetensors.index.json` and expose exactly the set of keys of the index, each associated with its file.

#### Scenario: Index keys
- **WHEN** the HeartMuLa-oss-3B directory is opened
- **THEN** the resulting set of keys is identical to that of the index's `weight_map` (289 keys in 4 shards)

#### Scenario: Shard inconsistent with the index
- **WHEN** a shard does not contain a key that the index attributes to it, or contains one that the index does not list
- **THEN** the opening fails with an error that names the shard and the mismatched keys

#### Scenario: Single file
- **WHEN** the directory has no index but has `model.safetensors`
- **THEN** the keys of that file are loaded

### Requirement: Fidelity of dtype, shape and bytes
When loading a tensor without conversion, the system SHALL preserve the dtype, the shape declared in the header and the bytes of the file.

#### Scenario: BF16 tensor
- **WHEN** a tensor in the file is BF16
- **THEN** the resulting tensor is `BFloat16` and its bytes match those of the file

#### Scenario: F32 tensor from the real checkpoint
- **WHEN** an F32 tensor of HeartMuLa-oss-3B is loaded without conversion
- **THEN** the resulting tensor is `Float32`, with the header's shape and bytes identical to those of the file

#### Scenario: Unsupported dtype
- **WHEN** the header declares a dtype that the system does not support
- **THEN** the load fails with an error that names the tensor and the dtype

### Requirement: Dtype conversion on load
The system SHALL be able to deliver the weights in a dtype different from the stored one, applying round-to-nearest with ties-to-even in the conversion from F32 to BF16.

#### Scenario: F32 to BF16
- **WHEN** HeartMuLa-oss-3B is loaded requesting bf16
- **THEN** all model parameters are `BFloat16` and each value matches the round-to-nearest-even of the F32 value in the file

### Requirement: Bounded memory during load
The system SHALL load the weights without keeping the whole checkpoint in memory at once, so that the resident memory peak does not exceed 1,5 times the size of the model in the target dtype.

#### Scenario: Load on a 16 GB machine
- **WHEN** HeartMuLa-oss-3B (15 GB in F32 on disk) is loaded in bf16 on CPU
- **THEN** the load finishes and the resident memory peak of the process does not exceed 1,5 times the size of the resulting bf16 weights

### Requirement: Strict linking of weights to the model
The system SHALL verify, before reading tensor data, that the model's keys and shapes match exactly those of the checkpoint, and SHALL fail listing all the discrepancies.

#### Scenario: Correct load
- **WHEN** HeartMuLa-oss-3B is loaded into `HeartMuLaModel`
- **THEN** there are no missing or extra keys, no shape mismatches and the load finishes without error

#### Scenario: Missing or extra key
- **WHEN** the checkpoint lacks a key that the model expects, or contains one that the model does not have
- **THEN** the load fails before reading tensor data, with an error that lists all the missing and extra keys

#### Scenario: Different shape
- **WHEN** a key exists in both but with a different shape
- **THEN** the load fails with an error that names the key and both shapes
