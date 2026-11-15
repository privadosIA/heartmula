using MusicGen.Core;
using MusicGen.Core.Text;
using HeartMuLa.Model;
using HeartMuLa.Pipeline;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

// Usage:
//   HeartMuLa.Net inspect  <file.safetensors>
//   HeartMuLa.Net check    --ckpt ./ckpt [--device cuda|cpu] [--dtype bf16|fp32] --reference ./reference
//   HeartMuLa.Net generate --ckpt ./ckpt --tags tags.txt --lyrics lyrics.txt [--ids prompt_ids.json] [--out frames.safetensors] [--ms 30000] [--seed N]
//     Tokenizes with <ckpt>/tokenizer.json (the HuggingFace library). --ids uses the golden run's precomputed ids instead (JsonIdsTokenizer).
//     --seed: 64-bit integer; if omitted, a random one is chosen. It is always printed (seed=N) and stored in the output metadata.
//     [--wav song.wav [--wav-format pcm16|float32] [--codec-device cpu|cuda:N] [--codec-tf32]]
//     --wav: after generating, HeartMuLa is released, HeartCodec is loaded (fp32) and the frames are decoded to a stereo 48 kHz WAV (PCM16 by default).
//     The codec runs on the model's device unless --codec-device says otherwise, with cuDNN TF32 off unless --codec-tf32 is given.
//
// "check" is step 2 of the plan: load the weights and compare logits with the Python golden run.

var args_ = Args.Parse(args);
var cmd = args_.Positional.ElementAtOrDefault(0) ?? "help";

switch (cmd)
{
    case "inspect":
        foreach (var (name, dtype, shape) in SafeTensors.Describe(args_.Positional[1]))
            Console.WriteLine($"{name,-60} {dtype,-5} [{string.Join(", ", shape)}]");
        break;

    case "check":
        Check(args_);
        break;

    case "generate":
        Generate(args_);
        break;

    default:
        Console.WriteLine("commands: inspect | check | generate (see Program.cs)");
        break;
}

static (HeartMuLaModel model, Device device, ScalarType dtype, string ckpt) LoadModel(Args a)
{
    var ckpt = a.Get("ckpt") ?? throw new ArgumentException("--ckpt is required");
    var device = new Device(a.Get("device") ?? (torch.cuda.is_available() ? "cuda" : "cpu"));
    var dtype = (a.Get("dtype") ?? "bf16") switch
    {
        "bf16" => ScalarType.BFloat16,
        "fp16" => ScalarType.Float16,
        "fp32" => ScalarType.Float32,
        var d => throw new ArgumentException($"dtype {d}")
    };
    Console.WriteLine($"device={device} dtype={dtype}");

    var mulaDir = Path.Combine(ckpt, $"HeartMuLa-oss-{a.Get("version") ?? "3B"}");
    var config = HeartMuLaConfig.FromFile(Path.Combine(mulaDir, "config.json"));
    var model = HeartMuLaModel.Create(config, dtype);
    model.eval();
    model.LoadWeights(mulaDir, dtype, device);
    return (model, device, dtype, ckpt);
}

static void Check(Args a)
{
    var refDir = a.Get("reference") ?? "reference";
    var (model, device, dtype, _) = LoadModel(a);
    var r = SafeTensors.LoadFile(Path.Combine(refDir, "reference.safetensors"));

    using var _ = torch.no_grad();
    var tokens = r["tokens"].to(device);
    var mask = r["tokens_mask"].to(device);
    var pos = r["pos"].to(device);
    var muq = r["muq_embed"].to(dtype).to(device);
    int muqIdx = (int)r["muq_idx"].item<long>();
    int bs = (int)tokens.shape[0];

    model.SetupCaches(bs);

    // We compare last_h and the codebook 0 logits of the FIRST frame, before sampling anything
    // (sampling is random and not comparable between Python and .NET).
    var (lastH, c0Logits) = model.BackboneLogits(tokens, mask, pos, cfgScale: (double)r["cfg_scale"].item<float>(), muq, Enumerable.Repeat(muqIdx, bs).ToArray());

    Compare("last_h", lastH, r["last_h"].to(device));
    Compare("c0_logits", c0Logits, r["c0_logits"].to(device));

    Console.WriteLine("If cos_sim > 0.999 and max_abs is small (bf16: ~1e-1 on logits is normal), the backbone is fine.");
}

static void Compare(string name, Tensor got, Tensor expected)
{
    var g = got.to(ScalarType.Float32).flatten();
    var e = expected.to(ScalarType.Float32).flatten();
    if (g.shape[0] != e.shape[0])
    {
        Console.WriteLine($"[{name}] SHAPE MISMATCH got {string.Join("x", got.shape)} expected {string.Join("x", expected.shape)}");
        return;
    }
    var maxAbs = (g - e).abs().max().item<float>();
    var cos = torch.nn.functional.cosine_similarity(g.unsqueeze(0), e.unsqueeze(0)).item<float>();
    Console.WriteLine($"[{name}] shape={string.Join("x", got.shape)} max_abs={maxAbs:E3} cos_sim={cos:F6}");
}

static void Generate(Args a)
{
    // The seed is parsed before loading anything: an invalid value fails immediately, and it is printed up front
    // so it is on record even if the run fails later.
    long seed;
    if (a.Get("seed") is { } seedText)
    {
        if (!long.TryParse(seedText, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out seed))
        {
            Console.Error.WriteLine($"--seed must be a 64-bit integer, got: '{seedText}'");
            Environment.Exit(2);
            return;
        }
    }
    else seed = Seeding.NewSeed();
    Console.WriteLine($"seed={seed}");

    // The audio options are checked before anything is loaded too.
    AudioOptions? audio;
    try { audio = AudioOptions.Parse(a.Get); }
    catch (ArgumentException e)
    {
        Console.Error.WriteLine(e.Message);
        Environment.Exit(2);
        return;
    }

    // The tokenizer is built before loading the model: if tokenizer.json is missing it fails in milliseconds, not after loading 7 GiB.
    var ckptDir = a.Get("ckpt") ?? throw new ArgumentException("--ckpt is required");
    var idsPath = a.Get("ids");
    ITextTokenizer tok = idsPath is null
        ? new HeartMuLaTokenizer(Path.Combine(ckptDir, "tokenizer.json"))
        : new JsonIdsTokenizer(idsPath);
    using var tokDisposer = tok as IDisposable;

    var (model, device, dtype, ckpt) = LoadModel(a);
    var gen = GenConfig.FromFile(Path.Combine(ckpt, "gen_config.json"));
    var pipe = new MusicGenerationPipeline(model, tok, gen, device, dtype);

    // Files are read like heartlib does (open(..., encoding="utf-8").read()): universal newlines and BOM preserved.
    var tags = PythonText.ReadAllText(a.Get("tags") ?? "tags.txt");
    var lyrics = PythonText.ReadAllText(a.Get("lyrics") ?? "lyrics.txt");
    double cfgScale = double.Parse(a.Get("cfg") ?? "1.5", System.Globalization.CultureInfo.InvariantCulture);

    var inputs = pipe.Preprocess(tags, lyrics, cfgScale);
    int maxMs = int.Parse(a.Get("ms") ?? "30000");

    // Right before generating, like the golden run before _forward: neither loading nor the prompt consumes random numbers from the generation.
    Seeding.Apply(seed);
    var frames = pipe.Generate(inputs, maxAudioLengthMs: maxMs, cfgScale: cfgScale,
        progress: (i, n) => { if (i % 25 == 0) Console.Write($"\r  frame {i}/{n}"); });
    Console.WriteLine();

    // The [K, T] frames are saved to be decoded with HeartCodec (step 3) — in the meantime,
    // you can decode them with the reference Python to check that they sound like music.
    var outPath = a.Get("out") ?? "frames.safetensors";
    // Everything needed to repeat the generation; nothing that changes between identical runs (no dates or user paths).
    var metadata = new Dictionary<string, string>
    {
        ["seed"] = seed.ToString(System.Globalization.CultureInfo.InvariantCulture),
        ["cfg_scale"] = cfgScale.ToString(System.Globalization.CultureInfo.InvariantCulture),
        ["max_audio_length_ms"] = maxMs.ToString(System.Globalization.CultureInfo.InvariantCulture),
        ["temperature"] = MusicGenerationPipeline.DefaultTemperature.ToString(System.Globalization.CultureInfo.InvariantCulture),
        ["topk"] = MusicGenerationPipeline.DefaultTopK.ToString(System.Globalization.CultureInfo.InvariantCulture),
        ["dtype"] = dtype.ToString(),
        ["device"] = device.ToString(),
        ["prompt"] = idsPath is null ? "tokenizer.json" : $"ids:{Path.GetFileName(idsPath)}",
    };
    SafeTensorsWriter.Save(outPath, new Dictionary<string, Tensor> { ["frames"] = frames.cpu() }, metadata);
    Console.WriteLine($"frames saved to {outPath} ({string.Join("x", frames.shape)})");

    if (audio is null) return;

    // HeartMuLa (7.3 GB in bf16) and the codec (6.3 GB in fp32) do not fit together on a 12 GB GPU: everything of the model goes before the codec is read.
    var framesCpu = frames.cpu().clone();                      // clone: on a CPU run cpu() is the same tensor and Unload disposes it
    AudioStage.Unload(model, frames, inputs.Tokens, inputs.TokensMask, inputs.MuqEmbed, inputs.Pos);
    var codecDevice = new Device(audio.CodecDevice ?? device.ToString());
    var samples = AudioStage.DecodeToWav(framesCpu, ckpt, codecDevice, audio, Console.WriteLine,
        (w, n) => Console.WriteLine($"  codec window {w + 1}/{n}"));
    Console.WriteLine($"wav written to {audio.WavPath} ({samples} samples, {samples / 48000.0:F2} s, {audio.Format})");
}

/// <summary>Minimal argument parser: --k v and positionals.</summary>
sealed class Args
{
    public List<string> Positional { get; } = new();
    private readonly Dictionary<string, string> named = new();
    public string? Get(string k) => named.TryGetValue(k, out var v) ? v : null;

    public static Args Parse(string[] argv)
    {
        var a = new Args();
        for (int i = 0; i < argv.Length; i++)
        {
            if (argv[i].StartsWith("--"))
            {
                var key = argv[i][2..];
                var val = i + 1 < argv.Length && !argv[i + 1].StartsWith("--") ? argv[++i] : "true";
                a.named[key] = val;
            }
            else a.Positional.Add(argv[i]);
        }
        return a;
    }
}
