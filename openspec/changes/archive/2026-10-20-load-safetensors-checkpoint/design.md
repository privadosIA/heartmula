# Design

## Context

`SafeTensors.LoadCheckpoint` returns a `Dictionary<string, Tensor>` with the whole checkpoint on CPU (15 GB in F32 for HeartMuLa-oss-3B, measured in C1: 289 F32 tensors; the codec is 818 F32 tensors, 6,2 GB). `HeartMuLaModel.LoadWeights` then converts to bf16 and calls `load_state_dict(strict: false)`; moreover the model constructor allocates the parameters in fp32 (~13 GB). The development machine has 16 GB, so both things must be avoided. The largest single tensor is `audio_embeddings.weight` [65576, 3072] (~805 MB in F32). See proposal.md for the motivation and the `weights-loading` spec for the requirements.

## Goals / Non-Goals

**Goals:**
- Memory peak ≈ model in bf16 + one tensor in transit.
- Fail fast (with headers only) on key or shape mismatches.
- API reusable by C6 for the codec (list headers, iterate tensors in streaming).

**Non-Goals:**
- Folding of `weight_norm` and any weight transformation (C6).
- Lazy per-layer loading, mmap or quantization.
- Changing model field names (they already replicate the checkpoint keys).

## Decisions

1. **Headers-only index.** A function that, given a directory, reads `model.safetensors.index.json` (or `model.safetensors`) and the header of each shard, and returns `key → (shard, dtype, shape, offsets)` without touching data. It validates that the keys of each header match the index. `Describe` already reads headers; its parser is reused. Alternative: trust only the JSON index; rejected because an index out of sync with the shards would go unnoticed.
2. **Tensor-by-tensor streaming with in-place copy.** For each tensor: read its bytes into a temporary CPU tensor of the file's dtype, convert (`.to(dtype)`, which in libtorch rounds to nearest-even), `copy_` into the corresponding parameter under `no_grad`, and release the temporary. `load_state_dict` is not used, since it requires the full dictionary. Ordered by shard and offset for sequential reading. Alternative: load per shard (≈5 GB) and then convert; rejected: larger peak with no advantage.
3. **Prior validation against the model.** Before reading data, the set of keys and shapes of the model's `named_parameters()` + persistent `named_buffers()` is compared with the index; all discrepancies (missing, extra, shapes) are accumulated and a single exception is thrown. The KV caches and masks are not part of the state (they are already unregistered fields).
4. **Construction in the target dtype.** The initial fp32 allocation must be avoided. Preferred option: set `torch.set_default_dtype(dtype)` while the model is built and restore it afterwards (a `using`/`try-finally`). If TorchSharp ignores the default dtype in some module, or the random initialization of 3B parameters is too slow, plan B: build on `meta`/skipping initialization if the version allows it, or create the parameters with an explicit `dtype:`. The corresponding task measures memory and time and decides; the spec's memory requirement is the criterion.
5. **Independent oracle for bf16.** The test does not compare `.to(bf16)` with itself: it converts with bit arithmetic (add `0x7FFF + bit16` to the F32 pattern and truncate), which is round-to-nearest-even, on a sample of values from the real checkpoint and on synthetic values at the edges (exact ties, subnormals, ±0, Inf).
6. **Tests without checkpoint with synthetic shards.** They are generated in the test with `SafeTensorsWriter` plus a hand-written `model.safetensors.index.json`; this way the scenarios of index, inconsistent shard, dtype and strict errors run in CI. The scenarios with HeartMuLa-oss-3B go with `[RequiresCheckpoint]` and `HEARTMULA_CKPT`.
7. **Memory peak.** It is measured with `Process.PeakWorkingSet64` in a separate test with the real checkpoint; the limit (1,5 × bf16 size ≈ 9,8 GB) leaves room for the tensor in transit and the runtime.
8. **TorchSharp 0.107.** It is upgraded before writing the rest: it is the first task because it changes the dependency on which everything is built. Alternative: stay on 0.105 and use only the 4070; rejected because it rules out the 5060 Ti (Blackwell, sm_120) and leaves libtorch 2.5.1 versus the golden run's torch 2.10.0. The CPU part is validated on macOS and the CUDA part on the CachyOS machine (`cuda-linux`).
9. **`RollForward=Major` in the app**, same as in the test project, to run on machines with only a newer runtime.

On torchtune vs HuggingFace: attention, RoPE and normalization are not touched; weights are only linked by name. Reference fixture: none from `tests/fixtures` is used; the truth is the checkpoint itself (headers and bytes), whose revisions are recorded in `tests/fixtures/MANIFEST.md`.

## Risks / Trade-offs

- [0.107 may bring API changes with respect to 0.105] → task 1.0 compiles and runs the tests before continuing and notes any adjustment; if the jump were costly, it is split into its own change.
- [TorchSharp may ignore `set_default_dtype` or be slow to initialize 3B parameters] → measure in task 3.1 and apply plan B of decision 4; the load is not considered good until it meets the spec's memory limit.
- [Converting tensor by tensor is slower than loading in bulk] → acceptable (sequential reading per shard); the total time is recorded.
- [`copy_` on parameters with `requires_grad` requires `no_grad`] → the whole load is wrapped in `torch.no_grad()`.
- [Measuring the memory peak with `PeakWorkingSet64` includes the memory of other tests in the same process] → the memory test runs isolated (filter) and the limit has margin.
- [The spec scenario "Tensor BF16" cannot be tested with the real checkpoint, which is F32] → it is tested with a synthetic file and with real F32; both scenarios are separated in the spec.
