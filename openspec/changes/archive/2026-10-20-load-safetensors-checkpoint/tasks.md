# Tasks

## 1. Header index and validation

- [x] 1.0 Raise `TorchSharpVersion` to `0.107.*` in `Directory.Build.props` (libtorch 2.10.0 with CUDA 12.8, the same torch 2.10.0 as the golden run and compatible with Blackwell); verify that `dotnet build` and `dotnet test -p:TorchBackend=cpu` remain green without new warnings, note here any API change that forces code changes, and confirm on the CUDA machine (4070/5060 Ti) that `-p:TorchBackend=cuda-linux` resolves the package and `torch.cuda.is_available()` is true
  - Result (2026-10-20, CachyOS, .NET SDK 10.0.111 compiling net8.0): no API or code changes; `dotnet build` 0 warnings (same as with 0.105); `dotnet test -p:TorchBackend=cpu` 9/9 with `HEARTMULA_CKPT`. With `cuda-linux` libtorch-cuda-12.8 2.10.0 is resolved, `torch.cuda.is_available()` = true with 2 devices (4070 and 5060 Ti) and bf16 matmul correct on both.
- [x] 1.1 Add in `Weights/` a headers-only checkpoint index (JSON index or single file → key, shard, dtype, shape) that validates each shard against the index; verify with xUnit tests on synthetic shards: exact keys, shard with missing/extra key fails naming shard and keys, and single-file case
- [x] 1.2 Cover unsupported dtype with an error that names tensor and dtype; verify with a test that fabricates a header with an unknown dtype

## 2. Faithful reading and streaming conversion

- [x] 2.1 Add reading of a specific tensor by key (without loading the rest) preserving dtype, shape and bytes; verify with synthetic BF16, F32 and I64 tests comparing bytes with those of the written file
- [x] 2.2 Add per-tensor dtype conversion and a bitwise F32→BF16 rounding oracle in the test (exact ties, subnormals, ±0, Inf); verify that `.to(BFloat16)` matches the oracle on synthetic edge values and on a random sample of 1M values

## 3. Strict load with bounded memory

- [x] 3.1 Investigate and fix the construction of `HeartMuLaModel` in the target dtype without allocating fp32 (default dtype, or plan B of the design); verify by measuring resident memory and construction time on CPU bf16 and leaving the result noted in this file
  - Result (CPU, 3,94 B parameters, Release): constructing with libtorch's default dtype set to bf16 (`HeartMuLaModel.Create`) works in all modules, without plan B. RSS peak 8,18 GiB (1,11 × the 7,34 GiB of bf16 parameters) in 9,2 s; the current fp32 construction peaks at 14,16 GiB in 7,1 s.
- [x] 3.2 Rewrite `HeartMuLaModel.LoadWeights` with prior validation of keys and shapes (single error with all the discrepancies) and in-place tensor-by-tensor copy under `no_grad`; verify with a small synthetic model (reduced config) that it covers correct load, missing key, extra key and different shape
- [x] 3.3 Update `Program.cs` to use the new load and remove the use of `LoadCheckpoint` that materializes everything; verify that `dotnet build -p:TorchBackend=cpu` compiles without new warnings and that `check`/`inspect` still start
  - Result: `Program.cs` uses `HeartMuLaModel.Create` + strict `LoadWeights` (no use of `SafeTensors.LoadCheckpoint` remains). Build 0 warnings; `inspect` starts; `check --dtype bf16` with the real checkpoint loads and validates the 289 tensors without errors (72 s, peak 9,69 GiB), but then fails in the first forward pass with `Tensor invalid -- empty handle` in `ScaledRope.Apply`. It is a prior failure, independent of this change: `GetCache` moves the cache to the scope of `Apply`, which releases it on exit, and the second call finds it released (reproduces the same with TorchSharp 0.105 and 0.107). Fixed separately from this change's spec (`DetachFromDisposeScope` in `ScaledRope.GetCache`, with test `ScaledRopeTests`); after the fix `check --dtype bf16` completes: `last_h` cos_sim=0,999987 (max_abs 6,25E-2) and `c0_logits` cos_sim=0,999997 (max_abs 1,25E-1).
- [x] 3.4 Add `RollForward=Major` to the app's csproj; verify that `dotnet run --project src/HeartMuLa.Net -- inspect <shard>` starts without `DOTNET_ROLL_FORWARD`

## 4. Documentation

- [x] 4.1 Document in the README how to run the tests with checkpoint (`HEARTMULA_CKPT`) and the strict behavior of the load; verify that the documented command runs as written

## 5. Verification with the real checkpoint

- [x] 5.1 `[RequiresCheckpoint]` tests with HeartMuLa-oss-3B: keys == index (289), shapes == headers, no missing/extra, bytes of a sample of F32 tensors identical to the file, and bf16 parameters equal to the oracle on a sample; verify by running them and pasting the result here
  - Result (`HEARTMULA_CKPT=$HOME/models/heartmula dotnet test`, `RealCheckpointTests`, 5/5 passed): 289 keys = `weight_map` (4 shards) and no missing/extra in the model; model shapes = headers and all parameters `BFloat16`; 28 F32 tensors (every 8th in alphabetical order of those ≤ 64 MiB, plus `audio_head`, `muq_linear.bias`, `backbone.norm.scale`, `q_proj` of layer 0 and `projection.weight`) with bytes identical to those of the file read with an independent parser; those same 28 tensors (297.406.464 values) bit-for-bit equal to the F32→BF16 rounding oracle.
- [x] 5.2 Isolated memory test: load HeartMuLa-oss-3B in bf16 on CPU and check that `PeakWorkingSet64` ≤ 1,5 × bf16 size; verify by running it and pasting the measured peak here (last task: run the parity test and paste the result)
  - Result (isolated with `--filter "FullyQualifiedName~PeakMemory"`): bf16 load on CPU in 16 s, weights 7,34 GiB, `PeakWorkingSet64` = 9,78 GiB (1,33 ×), limit 1,5 × = 11,00 GiB. Full suite with checkpoint: `Correctas! - Con error: 0, Superado: 34, Omitido: 0, Total: 34`. Without checkpoint: the 5 of `RealCheckpointTests` are skipped and the rest passes.

## Workflow follow-up

- Archive after review (`/opsx:archive`); syncs the `weights-loading` spec to `openspec/specs/`.
