# Proposal

## Why

Loading the checkpoint is the first piece that the whole model depends on, and the current code does not work with the real checkpoint on this machine. `SafeTensors.LoadCheckpoint` loads the ~15 GB of F32 weights into RAM at once, and `new HeartMuLaModel` additionally allocates ~13 GB of fp32 parameters before converting to bf16; with 16 GB of RAM that does not fit. Moreover, `LoadWeights` only prints the `UNEXPECTED` keys and uses `strict: false`, so a name or shape failure would go unnoticed until the C3 parity test. The golden run (C1) revealed that the checkpoint is F32 on disk (289 tensors), not bf16, so dtype conversion on load is an essential part of the behavior.

## What Changes

- Resolve the checkpoint through `model.safetensors.index.json` reading only headers, and validate before reading data that the model's set of keys and shapes match the checkpoint's exactly.
- Load tensor by tensor in streaming, converting to the requested dtype and copying directly into the model parameters, without materializing the whole checkpoint.
- Build `HeartMuLaModel` without first allocating the parameters in fp32 (directly in the target dtype).
- Any key that is missing, extra or has a different shape is an error that lists all the affected keys (no more silent `strict: false`).
- Upgrade TorchSharp from 0.105 to 0.107 (libtorch 2.10.0, CUDA 12.8): 0.105 ships libtorch 2.5.1 with CUDA 12.1, which does not support the Blackwell GPU (RTX 5060 Ti) of the CUDA machine, and 2.10.0 matches the golden run's torch.
- Add `RollForward=Major` to the app so that it starts with the .NET runtime available on this machine.
- Tests: synthetic sharded checkpoints (without the real checkpoint) and `[RequiresCheckpoint]` tests against HeartMuLa-oss-3B.

Out of scope: folding of `weight_norm` and loading of the codec (C6), any change in the architecture or the forward pass of the model, quantization.

## Capabilities

### New Capabilities
- `weights-loading`: reading of sharded safetensors checkpoints with faithful dtype and bytes, dtype conversion on load with bounded memory, and strict linking of the weights to the model.

### Modified Capabilities
None.

## Impact

- Numerical parity criterion: exact (bit-for-bit) equality between the loaded tensors and the file bytes when there is no conversion, and agreement with an independent round-to-nearest-even oracle (F32→BF16) when there is. There is no forward computation in this change, so cos_sim is not used.
- Dependencies: `TorchSharp` and the backends `TorchSharp-cpu|cuda-windows|cuda-linux` go from `0.105.*` to `0.107.*`.
- Code: `Weights/SafeTensors.cs`, `Model/HeartMuLaModel.cs` (`LoadWeights` and construction), `Program.cs`, `HeartMuLa.Net.csproj`.
- Tests: new ones in `tests/HeartMuLa.Net.Tests`; those that use the real checkpoint are marked `[RequiresCheckpoint]` (`HEARTMULA_CKPT=~/models/heartmula/HeartMuLa-oss-3B`).
- Documentation: README (how to run the tests with checkpoint).
