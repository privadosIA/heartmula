# weights-loading Specification

## Purpose

Define how the port reads safetensors checkpoints (sharded or single-file) and binds them to a model, keeping the exact values of the file and failing explicitly on any discrepancy in keys or shapes.

## Requirements

### Requirement: Resolution of sharded checkpoints
The system SHALL resolve a checkpoint directory through `model.safetensors.index.json` and expose exactly the set of keys of the index, each one associated with its file.

#### Scenario: Index keys
- **WHEN** the HeartMuLa-oss-3B directory is opened
- **THEN** the resulting set of keys is identical to the index's `weight_map` (289 keys in 4 shards)

#### Scenario: Shard inconsistent with the index
- **WHEN** a shard does not contain a key that the index attributes to it, or contains one that the index does not list
- **THEN** opening fails with an error that names the shard and the discrepant keys

#### Scenario: Single file
- **WHEN** the directory has no index but does have `model.safetensors`
- **THEN** the keys of that file are loaded

### Requirement: Fidelity of dtype, shape and bytes
When loading a tensor without conversion, the system SHALL keep the dtype, the shape declared in the header and the bytes of the file.

#### Scenario: BF16 tensor
- **WHEN** a tensor in the file is BF16
- **THEN** the resulting tensor is `BFloat16` and its bytes match those of the file

#### Scenario: F32 tensor of the real checkpoint
- **WHEN** an F32 tensor of HeartMuLa-oss-3B is loaded without conversion
- **THEN** the resulting tensor is `Float32`, with the header's shape and bytes identical to those of the file

#### Scenario: Unsupported dtype
- **WHEN** the header declares a dtype that the system does not support
- **THEN** loading fails with an error that names the tensor and the dtype

### Requirement: Dtype conversion on load
The system SHALL be able to deliver the weights in a dtype other than the stored one, applying round-to-nearest with ties to even in the conversion from F32 to BF16.

#### Scenario: F32 to BF16
- **WHEN** HeartMuLa-oss-3B is loaded asking for bf16
- **THEN** all the model's parameters are `BFloat16` and each value matches the round-to-nearest-even of the F32 value in the file

### Requirement: Bounded memory during loading
The system SHALL load the weights without keeping the whole checkpoint in memory at once, so that the peak resident memory does not exceed 1.5 times the size of the model in the target dtype.

#### Scenario: Loading on a 16 GB machine
- **WHEN** HeartMuLa-oss-3B (15 GB in F32 on disk) is loaded in bf16 on CPU
- **THEN** loading finishes and the process's peak resident memory does not exceed 1.5 times the size of the resulting bf16 weights

### Requirement: Strict binding of weights to the model
The system SHALL verify, before reading tensor data, that the model's keys and shapes match the checkpoint's exactly, and SHALL fail listing all the discrepancies. The check SHALL work on a subset of the checkpoint chosen by a key prefix, SHALL be able to ignore an explicit allow-list of keys that only exist for training, and SHALL be able to fold weight-normalization pairs into plain weights before comparing; anything outside the prefix is not read and is not reported as extra, and anything inside it that is neither in the model nor in the allow-list is reported as extra. The same implementation SHALL serve every model the system loads.

#### Scenario: Correct load
- **WHEN** HeartMuLa-oss-3B is loaded into `HeartMuLaModel`
- **THEN** there are no missing or extra keys, no shape differs and loading finishes without error

#### Scenario: Missing or extra key
- **WHEN** the checkpoint lacks a key that the model expects, or contains one that the model does not have
- **THEN** loading fails before reading tensor data, with an error that lists all the missing and extra keys

#### Scenario: Different shape
- **WHEN** a key exists in both but with a different shape
- **THEN** loading fails with an error that names the key and both shapes

#### Scenario: Prefix selects a subset
- **WHEN** a model is loaded from the keys that start with its prefix in a checkpoint that holds several models
- **THEN** the keys outside the prefix are neither read nor reported, and the report counts them as ignored

#### Scenario: Allow-listed training state
- **WHEN** the checkpoint contains keys that match the allow-list (statistics that only exist for training)
- **THEN** they are skipped and counted in the report, while any other key the model does not have is still reported as extra

#### Scenario: Weight-norm pairs
- **WHEN** the checkpoint stores a layer as an `original0`/`original1` pair
- **THEN** the pair is folded into one weight that is compared against the model's key, and an incomplete pair is reported as a discrepancy
