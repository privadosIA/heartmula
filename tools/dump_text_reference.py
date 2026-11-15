"""
Text fixtures for the real tokenizer (C5). It complements dump_reference.py (golden run) WITHOUT touching it: this script only imports
`tokenizers` and the standard library, so it runs in a minimal venv (`pip install tokenizers==0.22.1`, the golden run's version)
with the same Python version as the golden run (3.11 -> Unicode 14.0, which is heartlib's).

Writes to --out (by default tests/fixtures/):
    lower_reference.json   For ALL code points (except surrogates): what `str.lower()` changes, and the `Cased` and `Case_Ignorable`
                           ranges as CPython uses them in the final-sigma rule, obtained with probes:
                             A(c) = (c + "Σ").lower() ends in "ς"       <=> c is a cased letter and NOT case-ignorable
                             B(c) = ("a" + c + "Σ").lower() ends in "ς"  <=> c is case-ignorable or (cased and not ignorable)
                           => cased_nonignorable = A ; case_ignorable = B and not A.
    text_cases.json        Text cases: input bytes, text after reading them with `open(..., encoding="utf-8").read()` (universal newlines,
                           BOM kept), Python's lowercase and the ids of `encode`; and tags+lyrics pairs with the final ids of heartlib's
                           `preprocess` logic (<tag> wrapping, BOS and EOS).

The `preprocess` logic is reproduced here without loading the model (it only depends on the tokenizer) and CHECKED against
tests/fixtures/prompt_ids.json, which did come from heartlib's real pipeline, for the reference pair.

Uso:
    ~/.venvs/tokenizers-ref/bin/python tools/dump_text_reference.py --model_path "$HEARTMULA_CKPT"
"""
import argparse
import base64
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer
import tokenizers

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "tests" / "fixtures"
SIGMA = "Σ"
FINAL_SIGMA = "ς"


def read_like_heartlib(raw: bytes) -> str:
    """heartlib: open(path, encoding="utf-8").read() -> universal newlines, BOM kept as U+FEFF."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "input.txt"
        p.write_bytes(raw)
        with open(p, encoding="utf-8") as fp:
            return fp.read()


def ranges(cps):
    out = []
    for cp in cps:
        if out and cp == out[-1][1] + 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def dump_lower(out):
    lower_diff, cased_nonign, ign = [], [], []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        low = c.lower()
        if low != c:
            lower_diff.append([cp, [ord(x) for x in low]])
        a = (c + SIGMA).lower()[-1] == FINAL_SIGMA
        b = ("a" + c + SIGMA).lower()[-1] == FINAL_SIGMA
        if a:
            cased_nonign.append(cp)
        elif b:
            ign.append(cp)
    data = {
        "python": sys.version.split()[0],
        "unicode": unicodedata.unidata_version,
        "code_points_checked": 0x110000 - 0x800,
        "lower_diff": lower_diff,
        "cased_nonignorable_ranges": ranges(cased_nonign),
        "case_ignorable_ranges": ranges(ign),
    }
    with open(out / "lower_reference.json", "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=True, separators=(",", ":"))
    print(f"lower_reference: {len(lower_diff)} lowercase changes, {len(data['cased_nonignorable_ranges'])} Cased(non-ignorable) ranges, "
          f"{len(data['case_ignorable_ranges'])} rangos Case_Ignorable (Python {data['python']}, Unicode {data['unicode']})")


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def build_cases(lyrics_ref: str):
    u = lambda s: s.encode("utf-8")
    return [
        ("sigma_final_words", u("ΟΔΥΣΣΕΥΣ ΑΣ. ΣΑΣ")),
        ("sigma_isolated", u("Σ")),
        ("sigma_middle", u("ΑΣΑ ΑΣΣΑ")),
        ("sigma_with_ignorables", u("ΑΣ.Α ΑΣ. Α.Σ ΑΣ'Α ΑΣ́ ΑΣ​")),
        ("sigma_polytonic", u("Ὀδυσσεύς ΟΔΥΣΣΕΎΣ")),
        ("turkish_dotted_i", u("İstanbul İZMİR iI")),
        ("latin_special_cases", u("ß ẞ ǅ Ǆ ǈ Ĳ ŉ ǰ ΐ ῼ")),
        ("empty", b""),
        ("whitespace_runs", u("  hello   world  \n\n\n  next\t\tline \n")),
        ("numbers", u("1 12 123 1234 12345 123456 3.14 1,000 2024-10-20")),
        ("apostrophes", u("don't I'm they're we've you'll he'd it's 'quoted' O'Neil")),
        ("special_end_of_text", u("hello <|end_of_text|> world")),
        ("special_begin_of_text", u("<|begin_of_text|>hello")),
        ("special_header_ids", u("<|start_header_id|>user<|end_header_id|>")),
        ("astral_plane", u("𝔘𝔫𝔦𝔠𝔬𝔡𝔢 😀 𐐀 𠀀")),
        ("vithkuqi_cased_in_unicode14", u("\U00010570\U00010571 \U00010580")),
        ("long_text", u(("\n".join([lyrics_ref] * 6)))),
        ("crlf", u("[verse]\r\nline one\r\n\r\n[chorus]\r\nline two\r\n")),
        ("cr_only", u("line one\rline two\r")),
        ("mixed_newlines", u("a\r\nb\rc\nd")),
        ("bom_utf8", b"\xef\xbb\xbfhello world"),
        ("tag_wrapped", u("<tag>piano</tag>")),
        ("emoji_zwj_flags", u("👩‍👩‍👧‍👦 and 🇪🇸")),
        ("cjk_punctuation", u("你好，世界！「テスト」")),
        ("combining_marks", u("é café ñ")),
        ("fullwidth_and_ideographic_space", u("ＡＢＣ　ｄｅｆ")),
        ("unusual_spaces", u("a b c​d")),
        ("uppercase_ascii", u("HELLO WORLD THE QUICK BROWN FOX")),
    ]


def build_prompts(inputs_dir: Path):
    ref_tags = (inputs_dir / "tags.txt").read_bytes()
    ref_lyrics = (inputs_dir / "lyrics.txt").read_bytes()
    u = lambda s: s.encode("utf-8")
    return [
        ("reference", ref_tags, ref_lyrics),
        ("greek_final_sigma", u("Rock, ΕΛΛΗΝΙΚΆ"), u("ΟΔΥΣΣΕΥΣ\nΑΣ ΣΑΣ\n")),
        ("turkish_dotted_i", u("İstanbul pop"), u("İZMİR İÇİN\n")),
        ("crlf_lyrics", u("piano,happy"), u("[verse]\r\nline one\r\n\r\n[chorus]\r\nline two\r\n")),
        ("tags_already_wrapped", u("<tag>rock</tag>"), u("")),
        ("tags_half_wrapped", u("<tag>rock"), u("la la la")),
        ("bom_tags_and_lyrics", b"\xef\xbb\xbfpiano", b"\xef\xbb\xbfhello"),
        ("empty_both", b"", b""),
    ]


def preprocess_ids(tok, bos, eos, tags_raw: bytes, lyrics_raw: bytes):
    """Same logic as HeartMuLaGenPipeline.preprocess (heartlib 5858ca8) for the tags and the lyrics."""
    tags = read_like_heartlib(tags_raw).lower()
    if not tags.startswith("<tag>"):
        tags = f"<tag>{tags}"
    if not tags.endswith("</tag>"):
        tags = f"{tags}</tag>"
    tags_ids = tok.encode(tags).ids
    if tags_ids[0] != bos:
        tags_ids = [bos] + tags_ids
    if tags_ids[-1] != eos:
        tags_ids = tags_ids + [eos]

    lyrics = read_like_heartlib(lyrics_raw).lower()
    lyrics_ids = tok.encode(lyrics).ids
    if lyrics_ids[0] != bos:
        lyrics_ids = [bos] + lyrics_ids
    if lyrics_ids[-1] != eos:
        lyrics_ids = lyrics_ids + [eos]
    return tags, tags_ids, lyrics, lyrics_ids


def dump_cases(out, model_path):
    tok = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
    gen = json.load(open(os.path.join(model_path, "gen_config.json"), encoding="utf-8"))
    bos, eos = gen["text_bos_id"], gen["text_eos_id"]
    inputs_dir = out / "inputs"
    lyrics_ref = read_like_heartlib((inputs_dir / "lyrics.txt").read_bytes())

    cases = []
    for name, raw in build_cases(lyrics_ref):
        text = read_like_heartlib(raw)
        lower = text.lower()
        cases.append({
            "name": name, "bytes_b64": b64(raw), "text": text, "lower": lower,
            "ids_raw": tok.encode(text).ids,       # tokenizer only, no lowercasing
            "ids": tok.encode(lower).ids,          # what the model receives
        })

    prompts = []
    for name, tags_raw, lyrics_raw in build_prompts(inputs_dir):
        tags, tags_ids, lyrics, lyrics_ids = preprocess_ids(tok, bos, eos, tags_raw, lyrics_raw)
        prompts.append({
            "name": name, "tags_b64": b64(tags_raw), "lyrics_b64": b64(lyrics_raw),
            "tags_wrapped": tags, "lyrics_lower": lyrics, "tags_ids": tags_ids, "lyrics_ids": lyrics_ids,
        })

    # Self-check: the reference pair must give the ids that heartlib's REAL pipeline dumped to prompt_ids.json.
    real = json.load(open(out / "prompt_ids.json", encoding="utf-8"))
    ref = next(p for p in prompts if p["name"] == "reference")
    # prompt_ids.json stores the raw encode (with BOS, without EOS); the final ids are those plus the EOS.
    assert real[ref["tags_wrapped"]] == ref["tags_ids"][:-1], "tags: does not match prompt_ids.json"
    assert real[ref["lyrics_lower"]] == ref["lyrics_ids"][:-1], "lyrics: does not match prompt_ids.json"
    print("self-check OK: the preprocess logic reproduces prompt_ids.json (from the real pipeline) for the reference pair")

    data = {
        "python": sys.version.split()[0], "tokenizers": tokenizers.__version__,
        "text_bos_id": bos, "text_eos_id": eos, "cases": cases, "prompts": prompts,
    }
    with open(out / "text_cases.json", "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=True, indent=1)
    print(f"text_cases: {len(cases)} cases and {len(prompts)} tags+lyrics pairs (tokenizers {tokenizers.__version__})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default=os.environ.get("HEARTMULA_CKPT", os.path.expanduser("~/Proyectos/heartmula-models")))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_lower(out)
    dump_cases(out, args.model_path)
    print("done")


if __name__ == "__main__":
    main()
