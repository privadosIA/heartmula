# Tasks

## 1. Environment and checkpoints (outside the repo)

- [x] 1.1 Create the venv with `uv` (Python 3.11) in `~/.venvs/heartlib-golden`, clone heartlib into `~/src/heartlib` pinned to `5858ca8` and install it; verify with `python -c "import heartlib, torch, torchtune"` and note the versions and any skipped dependency
- [x] 1.2 Read, at the pinned commit, `src/heartlib/pipelines`, `heartmula` and `heartcodec` (and the README) to confirm the HF repo names, the API of `from_pretrained`/`preprocess`/`_forward`, and where the initial flow matching noise is generated; leave the findings in a short note at the top of `tools/dump_reference.py`
- [x] 1.3 Download HeartMuLa-oss-3B, HeartCodec-oss and the tokenizer/gen_config files into `~/models/heartmula`; verify that the directory has the 4 shards with `model.safetensors.index.json` and that `dotnet run --project src/HeartMuLa.Net -- inspect <shard>` lists keys

## 2. Dump script

- [x] 2.1 Restructure `tools/dump_reference.py` into subcommands with CPU by default, a fixed seed and the default output in `tests/fixtures/`; verify with `python tools/dump_reference.py --help`
- [x] 2.2 `prompt` subcommand: generate `prompt_ids.json` and `reference.safetensors` with tags/lyrics from the heartlib assets; verify that they exist, that `last_h` and `c0_logits` are finite and that their shapes are [2, 3072] and [2, 8197] (CFG active)
- [x] 2.3 `rope` subcommand: generate `rope.safetensors` (fixed input with a seed, positions, `Llama3ScaledRoPE` output); verify that the output differs from applying HuggingFace-style `rotate_half` to the same input (check done in the script itself)
- [x] 2.4 `frames` subcommand: generate `frames_py.safetensors` (≈125 frames, shape [8, T]) and `frames_logits.safetensors` (raw codebook 0 logits per frame via hook); verify that the number of logits matches the number of frames and that the rows of `frames_py` are within the vocabulary range
- [x] 2.5 `tokenizer` subcommand: generate `tokenizer_cases.json` with tags, lyrics and texts with ñ, accents and CJK, including `<tag>`/`</tag>`; verify that the ids can be rebuilt with a `decode` round trip
- [x] 2.6 `codec` subcommand: generate `codec_latents.safetensors` (initial noise `noise_seg{i}` of each window captured in the `torch.randn` of `FlowMatching.inference_codes`, the codes after the repetition up to 372 frames, and the final latents) and `codec_wav.safetensors` in fp32 from `frames_py`; verify a 48 kHz stereo wav with the expected duration (T × 80 ms, after trimming to `target_len`) and no NaN

## 3. Determinism, documentation and C# test

- [x] 3.1 Run all the subcommands twice and compare sha256; verify that they match or document which files are not bit-for-bit deterministic and why
- [x] 3.2 Write `tests/fixtures/MANIFEST.md` with the checkpoint revision, the heartlib commit, the Python/torch/torchtune versions, the exact command of each dump, and the size and sha256 of each file; verify that the sha256 values in the document match `shasum -a 256 tests/fixtures/*`
- [x] 3.3 Adjust `.gitignore` to version `tests/fixtures/**/*.safetensors`, check the total size (≤ ~50 MB) and update `tools/README`/README with the regeneration command; verify with `git check-ignore` and `du -sh tests/fixtures`
- [x] 3.4 Add an xUnit test without a checkpoint that loads every `.safetensors` and `.json` in `tests/fixtures` with the C# reader and checks the expected keys, dtypes and shapes; verify with `dotnet test -p:TorchBackend=cpu` green

## Workflow follow-up

- Archive after reviewing the fixtures (`/opsx:archive`); no specs to sync.
