# Design

## Context

Development machine: macOS arm64, no CUDA. `uv` and Python 3.11 are available; there is no torch and no git-lfs. heartlib (commit `5858ca8`, 2026-10-08) requires `torch>=2.4,<2.11`, `torchtune==0.4.0`, `transformers==4.57.0`, `numpy==2.0.2`, `bitsandbytes`, `torchao`, among others; several of them are only imported on quantization or CUDA paths. The existing script assumes `--device cuda` and `torch.autocast`, and copies the first half of `generate_frame` by hand, so it can diverge from the real library. See proposal.md for the motivation.

## Goals / Non-Goals

**Goals:**
- Reproducible, deterministic fixtures, generated with the original implementation without modifying it.
- Cover what the C3–C7 scenarios ask for, without having to go back to Python afterwards.

**Non-Goals:**
- Performance of the golden run; it is enough that it finishes in a reasonable time.
- Validating the port; the only check is that the files can be read from C#.

## Decisions

1. **`uv` venv outside the repo** (`~/.venvs/heartlib-golden`, Python 3.11) and heartlib cloned into `~/src/heartlib` pinned to commit `5858ca8`, installed with `uv pip install -e`. Alternative: a venv inside the repo; discarded so as not to dirty the tree. If a package (e.g. `bitsandbytes`) does not install on macOS, it is installed with `--no-deps` or skipped only if the import is never executed; any deviation is recorded in the MANIFEST.
2. **CPU device, bf16 dtype for HeartMuLa and fp32 for the codec**, like the original. MPS is discarded as a source of truth because of its patchy bf16 coverage and non-determinism. If the RAM is not enough for the fp32 dump of the backbone (needed for the 1e-4 tolerance), that dump is marked optional.
3. **Checkpoints in `~/models/heartmula`**, downloaded with `huggingface_hub` (HeartMuLa-oss-3B, HeartCodec-oss and the tokenizer/gen_config files). The exact HF repo name and revision are read from the heartlib README and recorded in the MANIFEST. `HEARTMULA_CKPT` points there.
4. **Capture without patching the library**: the codebook 0 logits per frame are obtained with a `register_forward_hook` on `codebook0_head` (raw output per row, before CFG); the codec noise by wrapping `torch.randn` only during the call to `detokenize` (the `first_latent` of `detokenize` is overridden by the mask in the first segment; the effective noise is that of `FlowMatching.inference_codes`, one per 29.76 s window). Alternative: copy the generation loop (as the first half is today); discarded because it drifts out of sync with the library.
5. **Determinism**: a fixed `torch.manual_seed` before each dump, `topk=50`, `temperature=1.0`, a small fixed number of frames (≈125, 10 s) to limit the size of the wav. The script is run twice and the sha256 values are compared; differences in non-deterministic tensors are documented and replaced by a tolerance in the corresponding parity test.
6. **Unit RoPE** from `torchtune.modules.Llama3ScaledRoPE` with the parameters read from the loaded model (head_dim, max_seq_len, base, scale_factor), not written by hand.
7. **Format**: every `.safetensors` in fp32 except integer tokens/positions; JSON with `ensure_ascii=False`. `.gitignore` lets `tests/fixtures/**/*.safetensors` through; no LFS while the total is ≤ ~50 MB, and if it is exceeded the number of frames is reduced instead of adopting LFS.
8. **A single script with subcommands** (`prompt`, `rope`, `frames`, `tokenizer`, `codec`) so a group can be regenerated without repeating the model load when it is not needed; each one writes its own file.

On torchtune vs HuggingFace: this change does not implement attention/RoPE/norm, but the RoPE fixture is generated with torchtune (adjacent pairs + Llama3 scaling), which is the reference C3 must match.

## Risks / Trade-offs

- [The current script has never been run and may not match the real heartlib API] → read `src/heartlib/pipelines` and `heartmula` at the pinned commit before modifying it; the first dump of the `prompt` group is the test.
- [CUDA/quantization dependencies cannot be installed on macOS] → see decision 1; no path that uses them is executed.
- [Generating 125 frames on CPU with 3B may take long] → acceptable once; if it is excessive, lower the number of frames and document it.
- [bf16 on CPU may differ from bf16 on CUDA] → the fixtures are the truth of this port; the bf16 tolerances already assume a different accumulation order.
- [The size of the fixtures grows] → ~50 MB limit and the policy of decision 7.

- [`detokenize` loops the codes up to 372 frames and decodes 29.76 s windows with overlap] → the codec fixture starts from fewer frames on purpose, so that C7 has to replicate that repetition and the final trim to `target_len`; the command is in the MANIFEST.
