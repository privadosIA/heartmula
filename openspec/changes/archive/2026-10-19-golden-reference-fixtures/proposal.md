# Proposal

## Why

Every change with model code (C3 backbone, C4 generation, C5 tokenizer, C6/C7 codec) ends in a parity test against reference tensors, and right now none exists. `tools/dump_reference.py` is written but has never been run, does not cover the codec, the unit RoPE, the per-frame logits or the non-ASCII cases, and there is neither a Python environment nor a checkpoint on this machine (macOS arm64). Without reproducible fixtures there is no objective way to say that the port is correct.

## What Changes

- Set up an isolated Python environment (`uv`, outside the repo) with heartlib pinned to a commit, and download the checkpoints outside the repo.
- Review and extend `tools/dump_reference.py` (the only piece with Python) to dump into `tests/fixtures/`:
  - `prompt_ids.json` and `reference.safetensors` (prompt + `last_h` + `c0_logits`) — already planned.
  - `rope.safetensors`: torchtune RoPE over a fixed tensor (input, positions, output).
  - `frames_py.safetensors` with a fixed seed and `frames_logits.safetensors` with the codebook 0 logits per frame (for teacher forcing in C4).
  - `tokenizer_cases.json`: ids for texts with ñ, accents and CJK, besides tags/lyrics.
  - `codec_latents.safetensors` (frames → latents after flow matching, including `first_latent`, the initial noise) and `codec_wav.safetensors` (latents → wav).
- Document in `tests/fixtures/MANIFEST.md` the checkpoint version, the heartlib commit, the commands and the sha256 of each file.
- An xUnit test without a checkpoint that reads the fixtures with the C# safetensors reader and checks keys, dtypes and shapes.
- Version the fixtures in plain git (no LFS) if the total stays below ~50 MB.

Out of scope: any parity test (C3–C7 do those), changes to the model's C# code, running any Python at runtime.

## Capabilities

### New Capabilities
None. It is tooling and test data; the change declares `skip_specs: true`.

### Modified Capabilities
None.

## Impact

- Numerical parity: not applicable (no behavior change in the port). The criterion of this change itself is that the dumps are deterministic: two runs give the same sha256.
- Code: `tools/dump_reference.py` (rewrite/extension), `tools/decode_frames.py` (keep), a new test in `tests/HeartMuLa.Net.Tests`, `.gitignore`.
- Data: `tests/fixtures/*` (tens of MB at most).
- Environment: Python 3.11 + torch/torchtune/heartlib in a local venv; ~7 GB of checkpoints in `~/models/heartmula` (outside the repo, `HEARTMULA_CKPT`).
