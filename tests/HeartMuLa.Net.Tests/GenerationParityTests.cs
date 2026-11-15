using MusicGen.Core.Testing;
using System.Diagnostics;
using HeartMuLa.Model;
using HeartMuLa.Pipeline;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// Full generation loop (MusicGenerationPipeline.Generate) with HeartMuLa-oss-3B in bf16 against heartlib: seed 1234,
/// cfg 1.5, topk 50, temperature 1.0 and the reference prompt (126 frames in frames_py.safetensors).
/// It is a stochastic process: a one-ulp bf16 difference can change a token and the error propagates, so only the
/// beginning (the first frames) is required and from there on it reports where and by how much it diverges.
/// </summary>
[Collection(RealModelCollection.Name)]
public sealed class GenerationParityTests
{
    private const int Seed = 1234;
    private const double Cfg = 1.5;
    private const int TopK = 50;
    private const int RequiredIdenticalFrames = 2;

    private readonly RealModelFixture _fx;
    private readonly ITestOutputHelper _out;

    public GenerationParityTests(RealModelFixture fx, ITestOutputHelper output) { _fx = fx; _out = output; }

    private (MusicGenerationPipeline pipe, PromptInputs prompt) Pipeline()
    {
        var gen = GenConfig.FromFile(Path.Combine(Checkpoint.Dir!, "gen_config.json"));
        var pipe = new MusicGenerationPipeline(_fx.Model!, new JsonIdsTokenizer(Path.Combine(Fixtures.Dir, "prompt_ids.json")), gen, _fx.Device, ScalarType.BFloat16);
        var prompt = pipe.Preprocess(
            File.ReadAllText(Path.Combine(Fixtures.Dir, "inputs", "tags.txt")),
            File.ReadAllText(Path.Combine(Fixtures.Dir, "inputs", "lyrics.txt")), Cfg);
        return (pipe, prompt);
    }

    private Tensor Generate(MusicGenerationPipeline pipe, PromptInputs prompt, int seed, int frames)
    {
        Rng.Seed(seed);
        return pipe.Generate(prompt, maxAudioLengthMs: (frames - 1) * 80, temperature: 1.0, topk: TopK, cfgScale: Cfg);
    }

    [RequiresCpuCheckpoint]
    public void FullLoop_OnCpu_ReproducesTheFirstFramesOfPython_AndReportsWhereItDiverges()
    {
        var (pipe, prompt) = Pipeline();
        var reference = Fixtures.Load("reference.safetensors");
        var expected = Fixtures.Load("frames_py.safetensors")["frames"];          // [8, 126]
        long total = expected.shape[1];

        // The prompt built with the pipeline is the same one Python used.
        Assert.True(prompt.Tokens.Equals(reference["tokens"]), "prompt tokens differ from reference.safetensors");
        Assert.True(prompt.TokensMask.Equals(reference["tokens_mask"]), "prompt mask differs");
        Assert.True(prompt.Pos.Equals(reference["pos"]), "prompt positions differ");
        Assert.Equal((int)reference["muq_idx"].item<long>(), prompt.MuqIdx[0]);

        var sw = Stopwatch.StartNew();
        var frames = Generate(pipe, prompt, Seed, (int)total);
        sw.Stop();

        Assert.Equal(expected.shape, frames.shape);
        var equal = frames.eq(expected);                                            // [8, T] bool
        long equalTokens = equal.sum().item<long>();
        var fullFrame = equal.all(0).data<bool>().ToArray();
        int identicalFrames = Array.FindIndex(fullFrame, ok => !ok);
        if (identicalFrames < 0) identicalFrames = fullFrame.Length;
        string firstDiff = "none";
        if (identicalFrames < total)
        {
            int cb = Array.FindIndex(equal[.., identicalFrames].data<bool>().ToArray(), ok => !ok);
            firstDiff = $"frame {identicalFrames}, codebook {cb} (Python {expected[cb, identicalFrames].item<long>()}, C# {frames[cb, identicalFrames].item<long>()})";
        }
        double audioSeconds = total * 0.08;
        _out.WriteLine($"{total} frames in {sw.Elapsed.TotalSeconds:F0} s (RTF {sw.Elapsed.TotalSeconds / audioSeconds:F1})");
        _out.WriteLine($"first difference: {firstDiff}");
        _out.WriteLine($"identical frames from the start: {identicalFrames}; fully equal frames in total: {fullFrame.Count(ok => ok)} of {total}");
        _out.WriteLine($"equal tokens: {equalTokens} of {equal.numel()} ({100.0 * equalTokens / equal.numel():F2} %)");

        Assert.True(identicalFrames >= RequiredIdenticalFrames,
            $"only {identicalFrames} identical frames at the start; {RequiredIdenticalFrames} required. First difference: {firstDiff}");
    }

    [RequiresCheckpoint]
    public void RealModel_Invariants_ShapeTypeRangeAndSameSeedSameOutput()
    {
        const int n = 16;
        var (pipe, prompt) = Pipeline();

        var sw = Stopwatch.StartNew();
        var a = Generate(pipe, prompt, Seed, n);
        sw.Stop();
        var b = Generate(pipe, prompt, Seed, n);

        _out.WriteLine($"{_fx.Device}: {n} frames in {sw.Elapsed.TotalSeconds:F1} s (RTF {sw.Elapsed.TotalSeconds / (n * 0.08):F1})");
        Assert.Equal(new long[] { 8, n }, a.shape);
        Assert.Equal(ScalarType.Int64, a.dtype);
        Assert.True(a.min().item<long>() >= 0 && a.max().item<long>() < 8197, "token outside [0, 8197)");
        var same = a.eq(b);
        long differing = same.numel() - same.sum().item<long>();
        string firstDiff = "none";
        if (differing > 0)
        {
            int t = Array.FindIndex(same.all(0).data<bool>().ToArray(), ok => !ok);
            int cb = Array.FindIndex(same[.., t].data<bool>().ToArray(), ok => !ok);
            firstDiff = $"frame {t}, codebook {cb}";
        }
        _out.WriteLine($"same seed, two generations: {differing} of {same.numel()} tokens differ; first difference: {firstDiff}");
        Assert.True(differing == 0, $"the same seed should give the same output ({differing} tokens differ; first difference: {firstDiff})");
    }

    /// <summary>
    /// Diagnosis of the first divergence (frame 3, codebook 5). With the backbone under teacher forcing up to frame 3 and the
    /// Python tokens, C# has the same inputs as Python at that point; if it picks another token with the same noise, the cause is
    /// that its probabilities differ. The question is by how much: the score that decides the argmax is p/q and, if the difference
    /// between the C# token and the Python token in log(p/q) fits within the noise two logits can have (2 x the C3 tolerance), it is
    /// a bf16 numeric difference and not a bug.
    /// </summary>
    [RequiresCpuCheckpoint]
    public void FirstDivergence_Frame3Codebook5_IsExplainedByNumericNoise()
    {
        const int frame = 3, codebook = 5, pythonToken = 7232, csharpToken = 3908;
        var model = _fx.ModelWithFreshCaches();
        var r = Fixtures.Load("reference.safetensors");
        var frames = Fixtures.Load("frames_py.safetensors")["frames"];               // [8, 126]
        using var _ = no_grad();
        Assert.Equal(pythonToken, (int)frames[codebook, frame].item<long>());

        // Backbone under teacher forcing: call 0 = prompt, call t = Python frame t-1. What matters is the last_h of call 3.
        var promptPos = r["pos"];
        Tensor lastH = null!;
        for (int t = 0; t <= frame; t++)
        {
            Tensor h, logits;
            if (t == 0)
                (h, logits) = model.BackboneLogits(r["tokens"], r["tokens_mask"], promptPos, Cfg,
                    r["muq_embed"].to(ScalarType.BFloat16), new[] { (int)r["muq_idx"].item<long>(), (int)r["muq_idx"].item<long>() });
            else
            {
                var prev = frames[.., (t - 1)..t].reshape(1, 8).repeat(2, 1);
                var (padded, mask) = MusicGenerationPipeline.PadAudioToken(prev, emptyId: 0);
                (h, logits) = model.BackboneLogits(padded, mask, promptPos[.., ^1..] + t, Cfg);
            }
            logits.Dispose();
            lastH = h;
        }

        // Decoder with the Python tokens of frame 3 (the codebooks before 5 are the same ones C# used when it diverged).
        var tokens = frames[.., frame..(frame + 1)].reshape(1, 8).repeat(2, 1);
        var ciLogits = model.DecoderLogits(lastH, tokens);                             // [2, 7, V]
        var guided = Sampling.Guide(ciLogits[.., codebook - 1, ..], Cfg);               // [1, V]
        var probs = Sampling.TopKProbabilities(guided, TopK, 1.0);                      // [1, V] bf16

        // The noise of that draw: the 30th of the sequence with seed 1234 (frames 0-2 = 24 draws, plus 5 from frame 3).
        long vocab = probs.shape[1];
        manual_seed(Seed);
        for (int i = 0; i < 3 * 8 + codebook; i++) Rng.Advance(vocab);
        var q = empty(new long[] { 1, vocab }, dtype: ScalarType.BFloat16).exponential_(1);
        var scores = probs / q;
        long chosen = argmax(scores, dim: -1).item<long>();

        double P(long tok) => probs[0, tok].to(ScalarType.Float64).item<double>();
        double Q(long tok) => q[0, tok].to(ScalarType.Float64).item<double>();
        var rank = argsort(probs[0], descending: true).data<long>().ToArray();
        int RankOf(long tok) => Array.IndexOf(rank, tok) + 1;
        double Score(long tok) => Math.Log(P(tok) / Q(tok));
        double delta = Score(csharpToken) - Score(pythonToken);

        _out.WriteLine($"argmax(p/q) with the Python inputs: {chosen} (C# chose {csharpToken} in the full generation)");
        foreach (var (name, tok) in new[] { ("Python", (long)pythonToken), ("C#    ", (long)csharpToken) })
            _out.WriteLine($"{name} token {tok}: rank {RankOf(tok)} of {TopK}, p={P(tok):E3}, q={Q(tok):E3}, log(p/q)={Score(tok):F4}");
        _out.WriteLine($"log(p/q) difference C# - Python = {delta:F4} (limit {2 * Parity.MaxAbsLogits})");
        _out.WriteLine("top-5 by probability: " + string.Join(", ", rank.Take(5).Select(tok => $"{tok} (p={P(tok):E2})")));

        Assert.Equal(csharpToken, chosen);                                   // the diagnosis reproduces the full generation
        Assert.True(P(pythonToken) > 0, "the Python token is not in the C# top-k: sign of a real bug");
        Assert.True(delta <= 2 * Parity.MaxAbsLogits, $"log(p/q) difference {delta:F4} > {2 * Parity.MaxAbsLogits}: not explainable by numeric noise");
    }
}
