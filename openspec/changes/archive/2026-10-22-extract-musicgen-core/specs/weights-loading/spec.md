# Spec Delta

## MODIFIED Requirements

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
