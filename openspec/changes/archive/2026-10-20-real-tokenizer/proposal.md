# Proposal

## Why

Today the prompt enters with ids precomputed in Python (`JsonIdsTokenizer` + `prompt_ids.json`): the pipeline only works with the two golden run inputs, and no other lyrics or tags can be used. To generate music from your own text the real tokenizer in .NET is needed. It is the next item in the plan (C5) and does not depend on the codec.

I measured three things before proposing (throwaway programs in the scratchpad, not repo code):

1. **The HuggingFace library works as is.** The checkpoint's `tokenizer.json` is a Llama 3-style BPE (128 000 tokens + 256 special, no normalizer, split regex + ByteLevel, `ignore_merges`, the post-processor only prepends BOS). The NuGet package `Tokenizers.DotNet` 1.4.1 wraps the original Rust library: with its native Linux runtime package it loads the file in 102 ms and reproduces **12 of 12** cases of `tokenizer_cases.json`. The main package does not ship the native library; there are runtime packages for linux-x64/arm64, win-x64/arm64 and osx-x64/arm64.
2. **The text that reaches the tokenizer no longer matches Python's.** `Preprocess` lowercases with `ToLowerInvariant()` and heartlib uses `str.lower()`: `ΑΣ` gives `ασ` in .NET and `ας` in Python (final sigma), `ΟΔΥΣΣΕΥΣ` gives `οδυσσευσ` versus `οδυσσευς`, and `İstanbul` stays the same in .NET while Python gives `i̇stanbul` (i + U+0307). Different ids for letters with Greek or Turkish capitals.
3. **Two other details of the text preparation differ**, also due to reading the .NET way: heartlib opens files with `open(..., encoding="utf-8")` (universal newlines: `\r\n` becomes `\n`, and a BOM is not removed) and checks `<tag>` with an ordinal `startswith`; in C# `File.ReadAllText` keeps `\r\n` and discards the BOM, and `StartsWith(string)` depends on the culture. On Windows, with lyrics saved with CRLF, the ids would change without anything warning.

## What Changes

- **Real tokenizer**: a class that implements `ITextTokenizer` on top of `Tokenizers.DotNet`, loading `<ckpt>/tokenizer.json`; `Encode` returns the ids with BOS included, like HuggingFace. The native runtime packages are resolved by operating system and architecture with MSBuild conditions (analogous to `TorchBackend`) for the app and the tests.
- **Python-compatible lowercasing** (`str.lower()`): `İ` becomes `i̇` and the final sigma follows the Unicode Final_Sigma rule. `Preprocess` uses it for tags and lyrics.
- **Reading and comparisons as in heartlib**: reading the tags and lyrics files with universal newlines and without removing the BOM, and ordinal `StartsWith`/`EndsWith` for `<tag>`/`</tag>`.
- **`generate`** uses the checkpoint's real tokenizer by default; `--ids` (with `JsonIdsTokenizer`) remains as an option to reproduce the golden run.
- **New fixture** of text cases, generated once with a new, minimal script (`tools/dump_text_reference.py`) that only needs `tokenizers==0.22.1`, in a venv of a few MB that is deleted afterwards. The golden run script is not touched and no existing fixture is regenerated. It covers: special lowercasing, empty string, multiple spaces and newlines, numbers, apostrophes, text that looks like a special token (`<|end_of_text|>`) inside the lyrics, characters outside the BMP, long text and inputs with CRLF and with BOM.
- The end-to-end test already exists at no cost: with the real tokenizer, the tags/lyrics in `tests/fixtures/inputs/` must give tokens, mask and positions identical to `reference.safetensors`.

Out of scope: decoding ids to text, truncation and padding, the codec (C6/C7) and the `generate` seed (separate change). It does not produce audio.

## Capabilities

### New Capabilities
- `text-tokenization`: the tags and lyrics text is converted into the same ids that heartlib produces (reading, lowercasing, `<tag>` wrapping, tokenization with BOS/EOS).

### Modified Capabilities
None. (`generation-loop` does not change: the loop still receives the same prompt.)

## Impact

- **Numerical parity criterion: `cos_sim` does not apply; exact equality of ids with Python.** The 12 existing cases plus the new ones must give exactly the same ids, the lowercased text must be exactly what `str.lower()` gives, and the prompt built with the real tokenizer must be identical (tokens, mask, positions) to that of `reference.safetensors`.
- Code: new tokenizer class and text utilities in `src/HeartMuLa.Net/Pipeline/`, `MusicGenerationPipeline.cs` (`Preprocess`), `Program.cs` (`generate`), `Directory.Build.props` (per-platform packages).
- New dependencies: `Tokenizers.DotNet` 1.4.1 and `Tokenizers.DotNet.runtime.<rid>` 1.4.1 (native, third-party, per platform).
- Data: `tests/fixtures/text_cases.json` (small, versioned) and `MANIFEST.md`. The tests that use `tokenizer.json` carry `[RequiresCheckpoint]` (it is not in the repo).
- Environment (outside the repo): minimal temporary venv with `tokenizers==0.22.1`.
- Documentation: README.
