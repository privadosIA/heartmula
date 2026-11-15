# Proposal

## Why

`HeartMuLa.Net generate` takes no seed and seeds nothing: two runs with the same text give different frames and there is no way to repeat one that turned out well. It is the bare minimum needed to work with a stochastic generator (comparing changes, reproducing a failure, going back to a song).

What we already know:

- **Seeding has a trap in TorchSharp.** `torch.manual_seed` only seeds the CPU generator, unlike PyTorch in Python, which also seeds CUDA. In C4 we discovered it because, without seeding CUDA, two generations with the same seed gave 126 of 128 different tokens from frame 0; seeding `torch.cuda.manual_seed_all` as well brought it to 0 of 128 on the 4070 and on the 5060 Ti. Today that logic lives only in the test helper `Rng.Seed`, that is, the production command does not have it.
- **heartlib does not offer it either.** The `run_music_generation.py` example has no seed option (only the reconstruction one has `--seed`, default 42) and the generation code does not seed. The golden run seeded 1234 by hand before `_forward`. With that seed, on CPU, our loop reproduces the first 3 frames of `frames_py` (measured in C4).

## What Changes

- **`--seed N`** in `generate`. If not given, a random seed is chosen and **always printed** (`seed=N`), so that any run can be repeated afterwards.
- **A single seeding function** in the library (CPU always, and CUDA if there is a GPU) used by the command and by the test helper, instead of two copies.
- **Seed right before generating**, after loading the model and preparing the prompt, as the golden run did before `_forward`. `MusicGenerationPipeline.Generate` does not change signature: the seed is applied outside.
- **Reproducibility metadata** in `frames.safetensors` (`seed`, `cfg_scale`, `max_audio_length_ms`, `temperature`, `topk`, `dtype`, `device` and the origin of the prompt). `SafeTensorsWriter` gains an optional metadata parameter and the current callers keep producing the same file. The project's reader and Python's `load_file` already ignore `__metadata__`.
- Documentation in the README.

Out of scope: `--topk` and `--temperature` (heartlib's example exposes them, plus `--cfg_scale` with 3.0 by default and 240 000 ms by default, whereas our `generate` only has `--cfg` with 1.5 and `--ms` with 30 000; left as a possible separate improvement), equality between CPU and CUDA (not expected), bit-for-bit determinism between different machines (not guaranteed; only the same machine and device) and decoding to audio. This change does not produce audio.

## Capabilities

### New Capabilities
- `generation-reproducibility`: a generation can be repeated with its seed and leaves in the output what is needed to repeat it.

### Modified Capabilities
None. (`generation-loop` does not change: the loop still receives the RNG state already seeded.)

## Impact

- **Numerical parity criterion: `cos_sim` does not apply; exact equality.** Same seed, same device and same inputs must give an identical frames file between two real runs of the command; another seed, a different one. On CPU with `--seed 1234` and the golden run ids, the first 3 frames of the saved file must equal `frames_py.safetensors` (end-to-end proof that the command's path reproduces Python). On CUDA only reproducibility with itself is required, not equality with CPU or with Python.
- Code: new seeding function in `src/HeartMuLa.Net/`, `Program.cs` (`generate`), `Weights/SafeTensorsWriter.cs` (optional metadata), test helper `Rng.cs` (delegates to the production function), new tests.
- Dependencies: none.
- Data: nothing new.
- Coordination: `real-tokenizer` (C5) also touches `generate` (default tokenizer and file reading). The two changes are independent but overlap in `Program.cs`; whichever is applied second adjusts the value recorded for the prompt origin.
- Documentation: README.
