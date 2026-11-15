"""
Golden run: the ONLY contact with Python in the project. It is run once (or when regenerating fixtures)
with heartlib's ORIGINAL implementation, unpatched, and dumps to tests/fixtures/ what the C#
parity tests need.

Reading notes on heartlib @ 5858ca8 (see tests/fixtures/MANIFEST.md for versions):
  - Checkpoints: HeartMuLa/HeartMuLaGen (config.json, gen_config.json, tokenizer.json) at the root,
    HeartMuLa/HeartMuLa-oss-3B-happy-new-year -> HeartMuLa-oss-3B/, HeartMuLa/HeartCodec-oss-20260123 -> HeartCodec-oss/.
  - HeartMuLaGenPipeline.preprocess() adds BOS/EOS if they are missing; prompt_ids.json stores the RAW ids of the
    tokenizer (key = text already lowercased and with <tag>), which is what JsonIdsTokenizer serves.
  - _forward() calls self._unload(), which touches torch.cuda.*: on CPU it has to be disabled.
  - torch.compile (_apply_compile) is disabled with TORCHDYNAMO_DISABLE=1 (this script sets it).
  - Logits/hidden are captured with forward hooks: the generation loop is neither copied nor modified.
  - `decoder` is the only exception to "unpatched": it replaces IN MEMORY `sample_topk` (a module function that
    generate_frame resolves as a global) with one that returns the tokens of frames_py (teacher forcing). The rest of
    generate_frame (backbone, projection, decoder, embeddings) is the original; heartlib on disk is not touched.
  - HeartCodec.detokenize: detokenize's `first_latent` is zeroed by the mask in the first segment; the effective
    flow-matching NOISE is the torch.randn of FlowMatching.inference_codes (one per 29.76 s window).
    With <372 frames the codes are REPEATED in a loop up to 372 and trimmed to target_len at the end.
    Here torch.randn is wrapped only during the call, to store it.

Subcommands (all write to --out, by default tests/fixtures/):
    prompt     prompt_ids.json + reference.safetensors (prompt, last_h, c0_logits of the first frame)
    rope       rope.safetensors (torchtune RoPE over fixed tensors)
    frames     frames_py.safetensors + frames_logits.safetensors (raw codebook 0 logits per call)
    tokenizer  tokenizer_cases.json
    codec      codec_latents.safetensors + codec_wav.safetensors (from frames_py.safetensors)
    codec_long codec_long_latents.safetensors + codec_long_wav.safetensors: frames_py repeated up to 400 frames = 2 windows (C8)
    decoder    decoder_logits.safetensors: raw decoder logits (codebooks 1-7) with forced tokens (from frames_py.safetensors)

Usage:
    export TORCHDYNAMO_DISABLE=1
    ~/.venvs/heartlib-golden/bin/python tools/dump_reference.py prompt --model_path "$HEARTMULA_CKPT"
"""
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "tests" / "fixtures"
SEED = 1234
CFG_SCALE = 1.5
TOPK = 50
TEMPERATURE = 1.0


def _contig(d):
    return {k: v.detach().cpu().contiguous() for k, v in d.items()}


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=1)


def _inputs_dir(out):
    return Path(out) / "inputs"


def _load_pipeline(model_path):
    from heartlib import HeartMuLaGenPipeline

    pipe = HeartMuLaGenPipeline.from_pretrained(
        model_path,
        device=torch.device("cpu"),
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B",
        lazy_load=True,  # do not load the codec until it is asked for
    )
    pipe._unload = lambda: None  # the original calls torch.cuda.* and fails on CPU
    return pipe


def _attach_hooks(mula):
    """Hooks on backbone (output [B,S,D]) and codebook0_head (output [B,V]); one entry per generate_frame."""
    rec = {"last_h": [], "c0_logits": []}

    def on_backbone(_m, _i, out):
        rec["last_h"].append(out[:, -1, :].detach().float().cpu())

    def on_head(_m, _i, out):
        rec["c0_logits"].append(out.detach().float().cpu())

    h1 = mula.backbone.register_forward_hook(on_backbone)
    h2 = mula.codebook0_head.register_forward_hook(on_head)
    return rec, (h1, h2)


def _run(pipe, max_audio_length_ms):
    inputs_dir = _inputs_dir(DEFAULT_OUT)
    inputs = pipe.preprocess(
        {"tags": str(inputs_dir / "tags.txt"), "lyrics": str(inputs_dir / "lyrics.txt")}, cfg_scale=CFG_SCALE
    )
    rec, hooks = _attach_hooks(pipe.mula)
    torch.manual_seed(SEED)
    out = pipe._forward(
        inputs, max_audio_length_ms=max_audio_length_ms, temperature=TEMPERATURE, topk=TOPK, cfg_scale=CFG_SCALE
    )
    for h in hooks:
        h.remove()
    return inputs, rec, out["frames"]


# ---------------------------------------------------------------------------------------------- subcomandos
def cmd_prompt(args):
    out = Path(args.out)
    pipe = _load_pipeline(args.model_path)

    tags = (_inputs_dir(out) / "tags.txt").read_text(encoding="utf-8").lower()
    if not tags.startswith("<tag>"):
        tags = f"<tag>{tags}"
    if not tags.endswith("</tag>"):
        tags = f"{tags}</tag>"
    lyrics = (_inputs_dir(out) / "lyrics.txt").read_text(encoding="utf-8").lower()
    _write_json(out / "prompt_ids.json", {tags: pipe.text_tokenizer.encode(tags).ids, lyrics: pipe.text_tokenizer.encode(lyrics).ids})

    # A single generate_frame (max 0 ms -> no loop): prompt + first frame.
    inputs, rec, frames = _run(pipe, max_audio_length_ms=0)
    ref = {
        "tokens": inputs["tokens"],
        "tokens_mask": inputs["tokens_mask"],
        "pos": inputs["pos"],
        "muq_embed": inputs["muq_embed"].float(),
        "muq_idx": torch.tensor(inputs["muq_idx"][0], dtype=torch.long),
        "cfg_scale": torch.tensor(CFG_SCALE, dtype=torch.float32),
        "last_h": rec["last_h"][0],
        "c0_logits": rec["c0_logits"][0],
    }
    save_file(_contig(ref), str(out / "reference.safetensors"))
    for k in ("tokens", "last_h", "c0_logits"):
        t = ref[k]
        print(f"{k}: {tuple(t.shape)} finite={bool(torch.isfinite(t.float()).all())}")


def cmd_rope(args):
    from torchtune.modules import RotaryPositionalEmbeddings  # noqa: F401  (only to check the import)
    from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
    from heartlib.heartmula.modeling_heartmula import llama3_2_3B

    with torch.device("meta"):
        ref_model = llama3_2_3B()
    rope_meta = ref_model.layers[0].attn.pos_embeddings
    dim, base, max_seq = rope_meta.dim, rope_meta.base, rope_meta.max_seq_len
    scale_factor = rope_meta.scale_factor
    extra = [rope_meta.low_freq_factor, rope_meta.high_freq_factor, rope_meta.old_context_len]
    print(f"RoPE params: dim={dim} base={base} max_seq_len={max_seq} scale_factor={scale_factor}")
    rope = Llama3ScaledRoPE(dim=dim, max_seq_len=max_seq, base=base, scale_factor=scale_factor)
    assert [rope.low_freq_factor, rope.high_freq_factor, rope.old_context_len] == extra

    g = torch.Generator().manual_seed(SEED)
    b, s, n_h = 2, 16, 4
    x = torch.randn(b, s, n_h, dim, generator=g)
    pos_small = torch.arange(s).unsqueeze(0).repeat(b, 1)
    pos_large = pos_small + 4000  # low-frequency zone where the Llama3 scaling acts
    res = {
        "x": x,
        "pos_small": pos_small,
        "out_small": rope(x, input_pos=pos_small),
        "pos_large": pos_large,
        "out_large": rope(x, input_pos=pos_large),
        "x_bf16": x.to(torch.bfloat16),
        "out_bf16": rope(x.to(torch.bfloat16), input_pos=pos_small),
        "theta": rope.theta.float(),
        # [dim, base, max_seq_len, scale_factor, low_freq_factor, high_freq_factor, old_context_len]
        "params": torch.tensor([dim, base, max_seq, scale_factor, *extra], dtype=torch.float32),
    }

    # Check: it is NOT HuggingFace's rotate_half (adjacent pairs in torchtune).
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    ang = pos_small[..., None].float() * inv  # no scaling, only to compare the layout
    cos, sin = ang.cos()[:, :, None, :], ang.sin()[:, :, None, :]
    x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
    hf = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    diff = (hf - res["out_small"]).abs().max().item()
    assert diff > 1e-3, "torchtune's RoPE matches HF's rotate_half: something is wrong"
    print(f"RoPE torchtune vs rotate_half(HF) max_abs diff = {diff:.3f} (expected: large)")
    save_file(_contig(res), str(Path(args.out) / "rope.safetensors"))


def cmd_frames(args):
    out = Path(args.out)
    pipe = _load_pipeline(args.model_path)
    _, rec, frames = _run(pipe, max_audio_length_ms=args.frames * 80)
    logits = torch.stack(rec["c0_logits"])  # [calls, B, V]; call 0 is the prompt's
    save_file(_contig({"frames": frames.to(torch.int64)}), str(out / "frames_py.safetensors"))
    save_file(_contig({"c0_logits": logits, "last_h": torch.stack(rec["last_h"])}), str(out / "frames_logits.safetensors"))
    print(f"frames {tuple(frames.shape)} min={frames.min().item()} max={frames.max().item()}; logits {tuple(logits.shape)}")
    assert logits.shape[0] in (frames.shape[1], frames.shape[1] + 1)


def cmd_tokenizer(args):
    from tokenizers import Tokenizer

    out = Path(args.out)
    tok = Tokenizer.from_file(os.path.join(args.model_path, "tokenizer.json"))
    cases = {
        "tags": (_inputs_dir(out) / "tags.txt").read_text(encoding="utf-8").lower(),
        "lyrics": (_inputs_dir(out) / "lyrics.txt").read_text(encoding="utf-8").lower(),
        "tag_wrapped": "<tag>piano,happy</tag>",
        "ascii_simple": "hello world",
        "spanish_ene": "el niño soñó con la mañana",
        "accents": "canción de amor, café y corazón; àèìòù üöä",
        "cjk_chinese": "月亮代表我的心 你问我爱你有多深",
        "cjk_japanese": "夜に駆ける 君の声が聞こえる",
        "korean": "사랑해요 노래",
        "emoji": "love 🎵 music 🎶",
        "newlines_sections": "[verse]\nline one\n\n[chorus]\nline two",
        "mixed": "<tag>rock,中文,español</tag>",
    }
    res = []
    for name, text in cases.items():
        enc = tok.encode(text)
        assert tok.decode(enc.ids, skip_special_tokens=False) is not None
        res.append({"name": name, "text": text, "ids": enc.ids, "tokens": enc.tokens})
    _write_json(out / "tokenizer_cases.json", {"cases": res})
    print(f"{len(res)} cases; ids of 'spanish_ene': {res[4]['ids']}")


def cmd_codec(args):
    from safetensors.torch import load_file

    out = Path(args.out)
    frames = load_file(str(out / "frames_py.safetensors"))["frames"]  # [8, T]
    _codec_dump(args, frames, "codec_latents", "codec_wav", expected_callers=None)


LONG_FRAMES = 400  # two windows after padding to 692 frames (see heartcodec/models/decoder.py)


def cmd_codec_long(args):
    """Multi-window case (C8): the reference frames tiled to 400 frames -> 2 windows, so the in-context path runs on real data."""
    from safetensors.torch import load_file

    out = Path(args.out)
    base = load_file(str(out / "frames_py.safetensors"))["frames"]  # [8, 126]
    reps = -(-LONG_FRAMES // base.shape[1])
    frames = base.repeat(1, reps)[:, :LONG_FRAMES].contiguous()
    # first_latent, flow noise of window 0, in-context padding of window 1, flow noise of window 1
    _codec_dump(args, frames, "codec_long_latents", "codec_long_wav",
                expected_callers=["detokenize", "inference_codes", "detokenize", "inference_codes"])


def _codec_dump(args, frames, latents_name, wav_name, expected_callers):
    import soundfile as sf
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec

    out = Path(args.out)
    codec = HeartCodec.from_pretrained(
        os.path.join(args.model_path, "HeartCodec-oss"), device_map="cpu", dtype=torch.float32
    ).eval()
    dec = codec.decoder

    rec = {"randn": [], "codes_in": [], "flow_out": [], "decode_in": []}
    real_randn = torch.randn

    def randn_spy(*a, **kw):
        t = real_randn(*a, **kw)
        import inspect

        rec["randn"].append((inspect.stack()[1].function, t.detach().clone()))
        return t

    real_inf = dec.flow_matching.inference_codes

    def inf_spy(codes, *a, **kw):
        rec["codes_in"].append(codes[0].detach().clone())
        r = real_inf(codes, *a, **kw)
        rec["flow_out"].append(r.detach().clone())
        return r

    real_dec = dec.scalar_model.decode

    def dec_spy(x, *a, **kw):
        rec["decode_in"].append(x.detach().clone())
        return real_dec(x, *a, **kw)

    torch.manual_seed(SEED)
    torch.randn = randn_spy
    dec.flow_matching.inference_codes = inf_spy
    dec.scalar_model.decode = dec_spy
    try:
        wav = codec.detokenize(frames, disable_progress=True)  # [2, samples]
    finally:
        torch.randn = real_randn
        dec.flow_matching.inference_codes = real_inf
        dec.scalar_model.decode = real_dec

    if expected_callers is not None:
        callers = [c for c, _ in rec["randn"]]
        assert callers == expected_callers, f"unexpected torch.randn call order: {callers}"
        assert len(rec["codes_in"]) == len(rec["flow_out"]) == len(rec["decode_in"]) == 2, "expected 2 windows"

    lat = {"frames": frames.to(torch.int64)}
    for i, (caller, t) in enumerate(rec["randn"]):
        lat[f"randn_{i}_{caller}"] = t
    for i, t in enumerate(rec["codes_in"]):
        lat[f"codes_seg{i}"] = t
    for i, t in enumerate(rec["flow_out"]):
        lat[f"latents_seg{i}"] = t
    for i, t in enumerate(rec["decode_in"]):
        lat[f"decode_in_seg{i}"] = t
    save_file(_contig(lat), str(out / f"{latents_name}.safetensors"))

    wav = wav.float()
    save_file(_contig({"wav": wav}), str(out / f"{wav_name}.safetensors"))
    expected = int(frames.shape[1] / 12.5 * 48000)
    print(f"wav {tuple(wav.shape)} expected samples={expected} finite={bool(torch.isfinite(wav).all())}")
    print("latents keys:", sorted(lat))
    listen = out.parent / "_listen"
    listen.mkdir(exist_ok=True)
    sf.write(str(listen / f"{wav_name}.wav"), wav.T.numpy(), 48000)  # folder ignored by git


DECODER_FRAMES = (0, 1, 2, 63, 124, 125)  # the decoder resets its cache on every frame: they do not depend on each other


def cmd_decoder(args):
    """Decoder logits (codebooks 1-7) with forced tokens: runs heartlib's ORIGINAL generate_frame
    over the frames of frames_py.safetensors, replacing only the sampling (see the note in the header)."""
    from safetensors.torch import load_file
    import heartlib.heartmula.modeling_heartmula as mm

    out = Path(args.out)
    frames = load_file(str(out / "frames_py.safetensors"))["frames"]  # [8, T] int64
    n_cb, n_frames = frames.shape
    assert n_cb == 8
    pipe = _load_pipeline(args.model_path)

    # Forced sampling: generate_frame calls sample_topk 8 times per frame (c0 and then codebooks 1..7).
    calls = {"sample": 0}

    def forced_sample(logits, topk, temperature):
        t, k = divmod(calls["sample"], n_cb)
        calls["sample"] += 1
        return frames[k, t].reshape(1, 1).to(torch.int)

    # Raw logits per decoder row: the same ones heartlib computes before CFG (torch.mm(decoder_h[:, -1], audio_head[i-1])).
    mula = pipe.mula
    dec = {"calls": 0, "ci": {}}

    def on_decoder(_m, _i, o):
        t, i = divmod(dec["calls"], n_cb - 1)
        dec["calls"] += 1
        if t in DECODER_FRAMES:
            dec["ci"].setdefault(t, []).append(torch.mm(o[:, -1, :], mula.audio_head[i]).detach().float().cpu())

    rec, hooks = _attach_hooks(mula)
    h3 = mula.decoder.register_forward_hook(on_decoder)
    original = mm.sample_topk
    mm.sample_topk = forced_sample
    try:
        inputs_dir = _inputs_dir(DEFAULT_OUT)
        inputs = pipe.preprocess(
            {"tags": str(inputs_dir / "tags.txt"), "lyrics": str(inputs_dir / "lyrics.txt")}, cfg_scale=CFG_SCALE
        )
        torch.manual_seed(SEED)
        res = pipe._forward(inputs, max_audio_length_ms=(n_frames - 1) * 80, temperature=TEMPERATURE, topk=TOPK, cfg_scale=CFG_SCALE)
    finally:
        mm.sample_topk = original
        h3.remove()
        for h in hooks:
            h.remove()

    got = res["frames"].to(torch.int64)
    assert got.shape == frames.shape and torch.equal(got, frames), "the forced run did not reproduce frames_py"
    assert calls["sample"] == n_frames * n_cb and dec["calls"] == n_frames * (n_cb - 1)
    assert len(rec["last_h"]) == n_frames

    idx = list(DECODER_FRAMES)
    ci = torch.stack([torch.stack(dec["ci"][t], dim=1) for t in idx])  # [n, 2, 7, V]
    last_h = torch.stack([rec["last_h"][t] for t in idx])               # [n, 2, D]
    tokens = torch.stack([frames[:, t] for t in idx])                  # [n, 8]
    save_file(_contig({"frame_idx": torch.tensor(idx, dtype=torch.int64), "last_h": last_h, "tokens": tokens, "ci_logits": ci}),
              str(out / "decoder_logits.safetensors"))
    print(f"ci_logits {tuple(ci.shape)} finite={bool(torch.isfinite(ci).all())} last_h {tuple(last_h.shape)} tokens {tuple(tokens.shape)}")

    # Informative: this machine's last_h against the Mac's that is already in frames_logits.safetensors.
    ref_h = load_file(str(out / "frames_logits.safetensors"))["last_h"]
    for j, t in enumerate(idx):
        cs = torch.nn.functional.cosine_similarity(last_h[j].flatten(), ref_h[t].flatten(), dim=0).item()
        print(f"  frame {t}: cos_sim(last_h this machine vs frames_logits) = {cs:.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("prompt", cmd_prompt), ("rope", cmd_rope), ("frames", cmd_frames), ("tokenizer", cmd_tokenizer), ("codec", cmd_codec), ("codec_long", cmd_codec_long), ("decoder", cmd_decoder)):
        p = sub.add_parser(name)
        p.add_argument("--model_path", default=os.environ.get("HEARTMULA_CKPT", os.path.expanduser("~/Proyectos/heartmula-models")))
        p.add_argument("--out", default=str(DEFAULT_OUT))
        if name == "frames":
            p.add_argument("--frames", type=int, default=125, help="number of frames to generate (80 ms each)")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    args.fn(args)
    print("done")


if __name__ == "__main__":
    main()
