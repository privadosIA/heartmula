# Design

## Context

- `ITextTokenizer.Encode(string) → IReadOnlyList<long>` is the seam used by `MusicGenerationPipeline`; today there is only `JsonIdsTokenizer` (ids precomputed by Python). `Preprocess` already replicates the rest of heartlib's `preprocess`: `<tag>` wrapping, BOS/EOS, text column, mask and MuQ zeroed. Its validation against `reference.safetensors` already exists (`GenerationParityTests`, with the ids from the JSON).
- The checkpoint's `tokenizer.json`: Llama 3 BPE (128 000 + 256 special tokens; `<tag>`/`</tag>` are not special), no normalizer, regex `Split` + `ByteLevel`, `ignore_merges`, a post-processor that only prepends BOS. It is not in the repo (9 MB, it lives in the checkpoint directory), so its tests carry `[RequiresCheckpoint]`.
- Spike (scratchpad): `Tokenizers.DotNet` 1.4.1 + `Tokenizers.DotNet.runtime.linux-x64` 1.4.1 → loads in 102 ms and 12/12 cases of `tokenizer_cases.json` identical. Without the runtime package it fails with `DllNotFoundException: hf_tokenizers`.
- Measured: `ToLowerInvariant()` differs from `str.lower()` in the final sigma (`ΑΣ`: `ασ` vs `ας`) and in `İ` (U+0130: it stays the same vs `i̇`). They match on `ÀÉÎ`, `ǅ`, `ẞ` and `Ａ`.
- heartlib opens files with `open(path, encoding="utf-8")` (universal newlines, BOM kept as U+FEFF) and uses ordinal `startswith`/`endswith`. Python fails on invalid UTF-8.
- This change **does not touch attention, RoPE or normalization** of the model (only input text); the torchtune vs HuggingFace differences already verified by C3 are not modified. "HuggingFace" here is only the tokenizer.

### Reference fixtures and how they were generated

- Existing (golden run, `tools/dump_reference.py`, heartlib `5858ca8f…`; see `MANIFEST.md`): `tokenizer_cases.json` (12 cases: `name`, `text`, `ids` with BOS, `tokens`; the text of `tags` and `lyrics` already lowercased), `prompt_ids.json` (raw tokenizer ids for the two reference inputs, produced by the real pipeline's `text_tokenizer`) and `reference.safetensors` (prompt of 227 positions). They are not regenerated.
- New (this change): `text_cases.json` and `lower_reference.json`, generated once by `tools/dump_text_reference.py` with `tokenizers==0.22.1` (the golden run version) on Python 3.11 (Unicode 14.0, heartlib's). See decision 4.

## Goals / Non-Goals

**Goals:**
- Same ids as Python for any text, not just the two golden run inputs.
- That the lowercasing and file-reading differences do not change the prompt across systems.
- That the app and the tests work on Linux, Windows and macOS without manual steps.

**Non-Goals:**
- Decoding ids, truncating or padding, text normalization beyond what heartlib does.
- Emulating other Python string functions (`strip`, `split`…) that heartlib does not use on this path.
- Implementing a custom BPE (plan B only).

## Decisions

1. **HuggingFace library via `Tokenizers.DotNet`.** It is the exact reference (the same Rust library Python uses) and the spike gives 12/12. The runtime packages are added in `Directory.Build.props` (shared by the app and the tests, like `TorchBackend`) with conditions per OS and architecture (`IsOSPlatform` and `OSArchitecture`) and a `TokenizerRuntime` property to force a RID when building for another machine. Version 1.4.1 is pinned. **Plan B** (own byte-level BPE in C#: .NET Llama 3 regex + merges from `tokenizer.json`, with `ignore_merges` and added tokens) only if the restore or the load fails verifiably on a supported platform. Discarded alternative: `Microsoft.ML.Tokenizers`, which does not load a generic `tokenizer.json` and would force converting vocab and merges.
2. **Class `HeartMuLaTokenizer : ITextTokenizer`.** It loads `<ckpt>/tokenizer.json` and returns the ids of `Encode` (which include BOS through the post-processor) as `long`. Behavior with text that looks like a special token: that of the library, which is that of Python; the fixture pins it. Access is wrapped with a `lock` unless the API check shows it is safe in parallel.
3. **Python lowercasing (`PythonText.Lower`) with embedded tables.** It walks the text by code points: `U+0130` → `i` + `U+0307`; `U+03A3` → `ς` or `σ` according to the Final_Sigma rule implemented by CPython (there is a cased letter before, skipping case-ignorable characters, and there is no cased letter after, skipping case-ignorable); the rest, Python's simple lowercase. The lowercase mappings and the two properties (`Cased` and `Case_Ignorable`) come from **tables generated from `lower_reference.json`** (`tools/gen_python_text_data.py` → `PythonTextData.cs`, standard library only), not from .NET's categories or `Rune.ToLowerInvariant`. **Why (measured while implementing):** with .NET's categories and no tables, the exhaustive test gave 55 lowercase mappings, 170 `Cased` and 105 `Case_Ignorable` different from Python; the 55 lowercase mappings are Unicode 16 characters that Python 3.11 (Unicode 14) does not change, and since `ToLowerInvariant` uses ICU on Linux, macOS and Windows, the result would depend on each machine's ICU version. The tables give the same result on all three platforms and exactly that of Python 3.11. Trade-off: characters added after Unicode 14 have no lowercase, just like in Python 3.11 (the golden run's). Discarded alternative: .NET categories plus correction lists; it would still depend on ICU.
4. **Exhaustive verification of lowercasing with a small fixture.** `dump_text_reference.py` walks all code points (except surrogates) and stores in `lower_reference.json`: those that `str.lower()` changes (≈1 400, with their full result), and, through probes with the sigma (`c + "Σ"` and `"a" + c + "Σ"`), the ranges of `Cased` and `Case_Ignorable` as CPython uses them. A test walks all of Unicode and requires zero differences in single-character `Lower` and in the probes. With the embedded tables the test also checks that `PythonTextData.cs` is up to date with the fixture and that the logic (final sigma, `İ`, lone surrogates, skipping over case-ignorable) is correct for every character.
5. **Cases with their ids.** `text_cases.json` stores, per case: name, input bytes (base64), text after reading the file with Python, lowercased text and `encode` ids. It includes: final sigma and `İ`, empty string, multiple spaces and newlines, numbers of 1 to 5 digits, apostrophes and contractions, `<|end_of_text|>` and `<|begin_of_text|>` inside the text, characters outside the BMP, Vithkuqi (cased since Unicode 14), long text, and inputs with CRLF, with lone `\r` and with BOM. It also stores tags+lyrics pairs with heartlib's final ids (`tags_ids`, `lyrics_ids` with BOS and EOS). The script reproduces heartlib's `preprocess` logic (without loading the model: what matters is the tokenizer) and **checks itself** against `prompt_ids.json`, which did come from the real pipeline, for the reference pair. Alternative: call `HeartMuLaGenPipeline.preprocess`; discarded because it requires installing torch and loading the model for something that only depends on the tokenizer.
6. **Separate script and venv.** `tools/dump_text_reference.py` is new, imports only `tokenizers` and the standard library, and does not touch `dump_reference.py` (which imports torch on load). The minimal venv `~/.venvs/tokenizers-ref` with `/usr/bin/python3.11` and `tokenizers==0.22.1` is created, used once and deleted; the versions are recorded in `MANIFEST.md`.
7. **Python-style reading (`PythonText.ReadAllText`).** It reads the bytes, decodes strict UTF-8 without discarding the BOM (which stays as `U+FEFF`) and applies universal newlines (`\r\n` and `\r` → `\n`). `Program.cs generate` uses it for the tags and lyrics files.
8. **`Preprocess`.** It uses `PythonText.Lower` and `StartsWith`/`EndsWith` with `StringComparison.Ordinal`. Its signature does not change.
9. **`generate`.** Without `--ids` it uses `HeartMuLaTokenizer` with `<ckpt>/tokenizer.json`; with `--ids`, `JsonIdsTokenizer` as until now.
10. **Platform check.** It cannot be run on Windows or macOS from here; it is checked that `dotnet restore -r <rid>` resolves the package for each supported RID and that it contains the native library (`libhf_tokenizers.so`, `.dll`, `.dylib`), and it is noted that execution is only verified on linux-x64.

## Risks / Trade-offs

- [The native library is third-party and I have only run it on linux-x64] → the version is pinned, the restore and the contents of the packages for the other platforms are checked, and what is verified and what is not is documented. Plan B exists if it fails.
- [The tables fix Unicode 14.0 (that of Python 3.11, the golden run's)] → characters added later do not change to lowercase, as in that Python version; heartlib run with another Python version could differ only in those characters, and this is documented. In exchange it does not depend on each system's ICU version.
- [`PythonTextData.cs` becomes out of date with respect to the fixture] → the exhaustive test detects it; it is regenerated with `tools/gen_python_text_data.py`.
- [The Python venv is needed again] → it is minimal (`tokenizers` only) and temporary; the original golden run is not regenerated.
- [Text that looks like a special token may give surprises] → the fixture pins it with Python's ids.
- [Strict `ReadAllText` rejects invalid UTF-8] → same as Python; it is an explicit failure and not a silent substitution.
- [Concurrent access to the native library might not be safe] → `lock` unless proven otherwise.
