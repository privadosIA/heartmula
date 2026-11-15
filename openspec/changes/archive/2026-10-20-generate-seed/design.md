# Design

## Context

- `Program.cs generate` loads the model, reads `--tags`/`--lyrics`, prepares the prompt with `Preprocess`, calls `pipe.Generate(inputs, maxAudioLengthMs, cfgScale, progress)` (temperature 1,0 and top-k 50 by default in `Generate`) and saves `frames` with `SafeTensorsWriter.Save`. It seeds nothing.
- `Sampling.SampleTopK` consumes the global RNG (`exponential_`) once per sampling; the Python golden run seeded 1234 before `_forward`. In C4 it was verified that with that seed and on CPU the loop reproduces the first 3 frames of `frames_py`.
- TorchSharp's `torch.manual_seed` only seeds the CPU; `torch.cuda.manual_seed_all` seeds all GPUs. That logic currently lives only in `tests/HeartMuLa.Net.Tests/Rng.cs` (`Rng.Seed`).
- `SafeTensorsWriter.Save(path, tensors)` writes the header with the tensors sorted; it does not support `__metadata__`. It has one production caller (`generate`) and several test callers. The reader (`SafeTensors.ReadHeader`) already skips `__metadata__`. In safetensors the metadata is a dictionary of strings to strings.
- heartlib does not offer a seed in `run_music_generation.py`; its example exposes `--topk`, `--temperature`, `--cfg_scale` (3,0) and `--max_audio_length_ms` (240 000). Our `generate` only exposes `--cfg` (1,5) and `--ms` (30 000).
- This change does not touch attention, RoPE or normalization, nor does it use new reference fixtures: only the existing `frames_py.safetensors` (golden run with seed 1234, see `MANIFEST.md`) for the end-to-end test.

## Goals / Non-Goals

**Goals:**
- Be able to repeat a generation with its seed, on CPU and on CUDA.
- Have the effective seed and the parameters recorded in the output file.
- A single, tested implementation of the seeding.

**Non-Goals:**
- `--topk` and `--temperature` (separate improvement).
- Equality between CPU and CUDA, or between different machines.
- Changing the signature of `MusicGenerationPipeline.Generate`.

## Decisions

1. **`Seeding` in the library.** A small static class with `Apply(long seed)` (seeds the CPU and, if `torch.cuda.is_available()`, all GPUs with `manual_seed_all`, because `--device cuda:N` can be any of them) and `NewSeed()` (non-negative 32-bit random seed, so that it can be read and typed). The test project's `Rng.Seed` now calls it, so there is a single implementation. With the CPU backend, `cuda.is_available()` is false and seeding does not fail.
2. **Order in the command's `Generate`.** `--seed` is parsed at the start and before loading anything (an invalid value fails in milliseconds, not after loading 7 GiB) and `seed=N` is printed at that moment, so that it stays in the log even if the run fails later. The seeding (`Seeding.Apply`) is done after `Preprocess` and right before `pipe.Generate`, like the golden run before `_forward`. The signature of `Generate` does not change.
3. **`--seed` as a 64-bit integer** (`long.TryParse` with invariant culture, accepts negatives, just like `torch.manual_seed`). Without `--seed`, `Seeding.NewSeed()`.
4. **Optional metadata in the writer.** `Save(path, tensors, metadata = null)`. With metadata, the header carries `__metadata__` (strings, keys sorted ordinally) and then the tensors in the same order as before; without metadata the header is the same as before, byte for byte. The values include nothing that changes between runs (neither dates nor absolute user paths), so that two identical runs give the same file byte for byte. A function `SafeTensors.ReadMetadata(file)` reads them, for the tests and for anyone who needs it. Keys: `seed`, `cfg_scale`, `max_audio_length_ms`, `temperature`, `topk`, `dtype`, `device`, `prompt` (today `ids:<file name>`; `real-tokenizer` will change it to `tokenizer.json` if applied afterwards).
5. **Tests.** Unit tests of `Seeding`: CPU (same seed → same sequence with `rand`, another → another) always; CUDA with an attribute that skips with an explicit reason if there is no GPU. Of the writer: without metadata it produces bytes identical to the expected ones written by hand, with metadata the round trip returns the strings and the tensors intact. End to end (real runs of the command, manual verification with a throwaway program to read the files): same seed twice → identical `cmp`; another seed → different; `--seed 1234 --ids …` → first 3 frames equal to `frames_py`; and on CUDA the same with two runs. To check that Python reads the metadata, a minimal temporary venv with only `safetensors` and `numpy` is used, deleted afterwards.
6. **The CUDA test must be able to fail.** It is checked once, by temporarily removing the CUDA seeding from `Seeding.Apply`, that the CUDA test fails; it is the regression that already bit us.

## Risks / Trade-offs

- [The GPUs are busy with other processes (ollama with a model loaded "forever")] → before using CUDA, check with `nvidia-smi` and `ollama ps`; if they are not free, ask before touching anything.
- [The same seed on CUDA may not be bit-for-bit reproducible on other GPUs, drivers or versions] → the spec only requires the same machine and device; in C4 it was 0 of 128 different tokens on both GPUs.
- [`Program.cs generate` is also touched by `real-tokenizer`] → small, independent changes; whichever is applied second resolves the overlap and updates the value recorded for `prompt`.
- [Adding metadata to a format that other tools read] → they are strings in `__metadata__`, which safetensors defines; it is verified with the Python library.
- [A random default seed changes the behavior of anyone who relied on not seeding] → it was not reproducible anyway; now at least it is printed and saved.
