using MusicGen.Core.Text;
using System.Diagnostics;
using System.Text.Json;
using HeartMuLa.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Pipeline;

/// <summary>gen_config.json at the root of the ckpt.</summary>
public sealed record GenConfig(long TextBosId = 128000, long TextEosId = 128001, long AudioEosId = 8193, long EmptyId = 0)
{
    public static GenConfig FromFile(string path)
    {
        var j = JsonDocument.Parse(File.ReadAllText(path)).RootElement;
        long L(string k, long d) => j.TryGetProperty(k, out var v) ? v.GetInt64() : d;
        return new GenConfig(L("text_bos_id", 128000), L("text_eos_id", 128001), L("audio_eos_id", 8193), L("empty_id", 0));
    }
}

public sealed record PromptInputs(Tensor Tokens, Tensor TokensMask, Tensor MuqEmbed, int[] MuqIdx, Tensor Pos);

/// <summary>
/// Port of HeartMuLaGenPipeline.preprocess + _forward. The postprocess (HeartCodec → wav) is step 3 of the plan.
/// </summary>
public sealed class MusicGenerationPipeline
{

    private readonly HeartMuLaModel mula;
    private readonly ITextTokenizer tokenizer;
    private readonly GenConfig cfg;
    private readonly Device device;
    private readonly ScalarType dtype;

    public const double DefaultTemperature = 1.0;
    public const int DefaultTopK = 50;

    private int ParallelNumber => mula.Config.AudioNumCodebooks + 1; // codebooks + 1 text column
    private int MuqDim => mula.Config.MuqDim;

    public MusicGenerationPipeline(HeartMuLaModel mula, ITextTokenizer tokenizer, GenConfig cfg, Device device, ScalarType dtype)
    {
        this.mula = mula;
        this.tokenizer = tokenizer;
        this.cfg = cfg;
        this.device = device;
        this.dtype = dtype;
    }

    private List<long> EncodeWithBosEos(string text)
    {
        var ids = tokenizer.Encode(text).ToList();
        if (ids.Count == 0 || ids[0] != cfg.TextBosId) ids.Insert(0, cfg.TextBosId);
        if (ids[^1] != cfg.TextEosId) ids.Add(cfg.TextEosId);
        return ids;
    }

    public PromptInputs Preprocess(string tags, string lyrics, double cfgScale)
    {
        // Like heartlib: Python's str.lower() (not ToLowerInvariant) and ordinal startswith/endswith (StartsWith(string) depends on the
        // culture and, with ICU, ignores characters such as the zero-width space before "<tag>").
        tags = PythonText.Lower(tags);
        if (!tags.StartsWith("<tag>", StringComparison.Ordinal)) tags = "<tag>" + tags;
        if (!tags.EndsWith("</tag>", StringComparison.Ordinal)) tags += "</tag>";
        var tagsIds = EncodeWithBosEos(tags);

        // ref_audio is not supported in the oss version: the MuQ embedding is a zero vector
        // injected at position len(tags_ids).
        int muqIdx = tagsIds.Count;

        var lyricsIds = EncodeWithBosEos(PythonText.Lower(lyrics));

        int promptLen = tagsIds.Count + 1 + lyricsIds.Count;
        var tokens = torch.zeros(new long[] { promptLen, ParallelNumber }, dtype: ScalarType.Int64);
        tokens[..tagsIds.Count, -1] = torch.tensor(tagsIds.ToArray());
        tokens[(tagsIds.Count + 1).., -1] = torch.tensor(lyricsIds.ToArray());

        var tokensMask = torch.zeros_like(tokens, dtype: ScalarType.Bool);
        tokensMask[.., -1] = torch.tensor(true);

        int bs = cfgScale != 1.0 ? 2 : 1;
        Tensor CfgCat(Tensor t)
        {
            t = t.unsqueeze(0);
            return cfgScale != 1.0 ? torch.cat(new[] { t, t }, 0) : t;
        }

        return new PromptInputs(
            Tokens: CfgCat(tokens),
            TokensMask: CfgCat(tokensMask),
            MuqEmbed: CfgCat(torch.zeros(MuqDim, dtype: dtype)),
            MuqIdx: Enumerable.Repeat(muqIdx, bs).ToArray(),
            Pos: CfgCat(torch.arange(promptLen, dtype: ScalarType.Int64)));
    }

    /// <summary>Returns the audio frames [num_codebooks, T] (long) to hand to HeartCodec.</summary>
    public Tensor Generate(PromptInputs inp, int maxAudioLengthMs = 120_000, double temperature = DefaultTemperature, int topk = DefaultTopK, double cfgScale = 1.5,
                           Action<int, int>? progress = null)
    {
        using var noGrad = torch.no_grad();

        var promptTokens = inp.Tokens.to(device);
        var promptMask = inp.TokensMask.to(device);
        var muq = inp.MuqEmbed.to(device);
        var promptPos = inp.Pos.to(device);

        int bs = cfgScale != 1.0 ? 2 : 1;
        mula.SetupCaches(bs);

        var frames = new List<Tensor>();
        var sw = Stopwatch.StartNew();

        var curr = mula.GenerateFrame(promptTokens, promptMask, promptPos, temperature, topk, cfgScale, muq, inp.MuqIdx);
        frames.Add(curr[0..1].clone());

        int maxFrames = maxAudioLengthMs / 80;
        for (int i = 0; i < maxFrames; i++)
        {
            using var step = NewDisposeScope();
            var (padded, paddedMask) = PadAudioToken(curr, cfg.EmptyId);
            var pos = promptPos[.., ^1..] + (i + 1);
            curr = mula.GenerateFrame(padded, paddedMask, pos, temperature, topk, cfgScale);

            if (torch.any(curr[0..1] >= cfg.AudioEosId).item<bool>())
                break;

            frames.Add(curr[0..1].clone().MoveToOuterDisposeScope());
            curr = curr.MoveToOuterDisposeScope();
            progress?.Invoke(i + 1, maxFrames);
        }

        Console.WriteLine($"  {frames.Count} frames ({frames.Count * 0.08:F1}s of audio) in {sw.Elapsed.TotalSeconds:F1}s -> RTF {sw.Elapsed.TotalSeconds / (frames.Count * 0.08):F2}");

        // stack -> [T, 1, K] -> permute(1,2,0) -> [1, K, T] -> squeeze -> [K, T]
        return torch.stack(frames).permute(1, 2, 0).squeeze(0);
    }

    /// <summary>
    /// Turns the tokens of a frame [B, K] into the input of the backbone's next step: [B, 1, K+1] with the
    /// text column set to <paramref name="emptyId"/> and mask false on it.
    /// </summary>
    public static (Tensor tokens, Tensor mask) PadAudioToken(Tensor token, long emptyId)
    {
        var padded = torch.ones(new long[] { token.shape[0], token.shape[1] + 1 }, dtype: ScalarType.Int64, device: token.device) * emptyId;
        padded[.., ..^1] = token;
        padded = padded.unsqueeze(1);
        var mask = torch.ones_like(padded, dtype: ScalarType.Bool);
        mask[.., .., -1] = torch.tensor(false, device: token.device);
        return (padded, mask);
    }
}
