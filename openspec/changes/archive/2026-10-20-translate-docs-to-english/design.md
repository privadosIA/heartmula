# Design

## Context

- The project convention is English for README, docs, OpenSpec artifacts, new comments, assertion messages and manifests. Spanish remained in what was written before that decision. Archived changes are not touched: they are the history of what was decided and when.
- Spanish text is not only prose. Some of it is **data or protocol**: fixture contents (`text_cases.json` has Spanish, Korean, CJK and emoji inputs; `tokenizer_cases.json`; `prompt_ids.json`), test inputs that are deliberately accented (`ñ`, accents) to exercise the tokenizer and text reading, tag and lyric samples, key names, file names, option names and metadata keys. Those must not be translated, and the verification has to be able to tell them apart from messages.
- Fourteen assertions match substrings of Spanish exception messages (`faltan`, `sobran`, `shapes distintas`, `faltan en el checkpoint N claves`, `índice 1`, ...) in `CheckpointIndexTests`, `CodecWeightsTests`, `LoadWeightsTests` and `WavWriterTests`. Newer loaders (`FlowMatchingWeights`) already use English messages (`are missing from the checkpoint`, `not in the model`, `shape mismatches`), so the translated messages should converge on the same wording.
- The main specs are normative. `openspec archive` normally merges deltas into main specs. A translation has no new requirement text to merge.
- The OpenSpec `config.yaml` `context` and `rules` are Spanish and are injected into the instructions of every future artifact, so they matter for the language of everything generated afterwards.

## Goals / Non-Goals

**Goals:**
- No Spanish prose left in the repository outside the archive and the named test data.
- Zero behaviour change, demonstrated mechanically rather than by reading the diff.
- The same wording for the same concept across files (for example "missing from the checkpoint", "shape mismatch", "not in the model").

**Non-Goals:**
- Translating archived changes, the `.claude/` files, the memory notes outside the repository, or fixture data.
- Rewording or restructuring documentation beyond what translation needs, fixing stale text, or adding new content.
- Changing identifiers other than the recorded test method names.

## Decisions

1. **Translate the main specs in place and declare `skip_specs: true`.** Expressing the translation as `MODIFIED` deltas would require copying every requirement of six specs (about 430 lines) into delta files, to be merged back unchanged in meaning; that duplicates content and gives the archive step nothing to decide. The change declares no capability and edits `openspec/specs/*/spec.md` directly as documentation. The guards are: every file keeps its `# <capability> Specification` title, `## Purpose` and `## Requirements` structure, the number of `### Requirement:` and `#### Scenario:` headings per file is the same before and after, every requirement keeps its `SHALL`/`MUST` wording, and `openspec validate --specs --strict` passes. The requirement and scenario counts are recorded before translating (the task records them) and compared after.
2. **One language per concept.** Translations follow the English already used in C7/C8 code and specs. Terms are fixed once: checkpoint, shard, header, weights, tensor, key, shape, window, in-context, cross-fade, golden run, fixture, parity, fp32/bf16. Spanish technical borrowings (for example "golden run") stay as they are in the English text.
3. **Messages and the assertions that match them change in the same step.** For each of the 14 assertions the message and the test are edited together, and the exact new text is chosen to match the English of `FlowMatchingWeights` where the concept is the same. A test that matches message text must match the new text exactly; no assertion is weakened (for example to a shorter substring) to make it pass.
4. **Mechanical proof that the C# edits are cosmetic.** A script (`verify_cs_equivalence.py`, kept in this change folder and removed with the archive) compares, for every changed `.cs` file, the version in `HEAD` with the working tree after a lexer has removed all comments (`//`, `///`, `/* */`) and replaced the text parts of string literals by a placeholder. Interpolated strings (`$"..."`, `$@"..."`) keep their `{expression}` holes verbatim, because those are code. The token streams must be identical except for identifiers listed in the rename file (the test method names). The script also prints every string literal that changed, one per line, and the task reviews that list by hand to confirm none is a key, file name, option name, metadata key, format string or protocol value (those would be behaviour). A second, independent check compiles the library before and after and compares the **IL of every method body with the string operands of `ldstr` masked**, which catches anything the lexer misses.
5. **Data is protected by hash.** Before starting, the sha256 of every file under `tests/fixtures/` is recorded; after each area it is compared. Files are never rewritten by a tool, so a mismatch means an edit went to the wrong place. `MANIFEST.md` itself changes (it is prose) but the sha256 values it lists do not.
6. **Intentional Spanish is named, not hidden.** After the translation, a search for accented characters and for a list of Spanish stop-words (`el`, `la`, `los`, `las`, `una`, `para`, `con`, `que`, `del`, `por`, `como`, `sin`, `cuando`) runs over `src`, `tests`, `tools`, `README.md`, `MANIFEST.md`, the specs and the props. The expected remaining matches are the accented test inputs of the tokenizer and text tests; each remaining file is listed in the task with the reason, and nothing else may match.
7. **Order by area, one code commit and one archive commit.** Work goes README and manifests, tools, specs and `config.yaml`, `src`, `tests`, because the later areas depend on the earlier ones (messages in `src` before the tests that match them). After each area the build and a selection of fast tests run; the full suite runs once at the end. Commits follow the project convention (the code and documentation commit, then the archive commit); there is no commit per area.
8. **CLI text that changes language is documented.** The README gets a short list of the user-visible strings that change (usage text of `check` and `generate`, argument errors such as `--ckpt requerido`, progress lines such as `frames guardados en`, `device=`, `frame i/n`), so nobody who parsed them is surprised. Option names, exit codes and the `seed=N` line do not change.
9. **Test renames are listed.** The Spanish test method names are renamed to English; the mapping is recorded in the task. A rename is a behaviour-neutral change but it changes test IDs in reports and in any filter that used them; the README commands that filter by `FullyQualifiedName~` are checked after the renames.

## Risks / Trade-offs

- **A translated string that was really a key.** A string such as a metadata name, a JSON property or a file name could be mistaken for prose. Mitigation: the changed-literals list is reviewed by hand, the IL comparison and the full suite catch what the review misses, and the fixtures are protected by hash.
- **Specs drifting in meaning.** A translation can subtly change a requirement. Mitigation: the counts, the `SHALL`/`MUST` check and a read of each translated spec side by side with the original (the original stays in `git` and in the previous commit); no requirement is reworded.
- **Size.** About 260 lines in `src`/`tests` plus 600 lines of documents. The risk is tedium, not complexity; the mechanical checks make it safe to do quickly.
- **Test names in reports.** Anyone with a saved filter on a Spanish test name will find nothing; the README commands are rechecked.
- **Wording that sounds literal.** Prose is translated for meaning, not word by word; where the Spanish used a shorthand that has no good English twin, the translation is a short phrase rather than a calque.
