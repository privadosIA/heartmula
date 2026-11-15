using HeartMuLa.Model;
using HeartMuLa.Pipeline;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>One id per word (between 3 and 17, within the text vocabulary of the small model).</summary>
internal sealed class ToyTokenizer : ITextTokenizer
{
    public IReadOnlyList<long> Encode(string text) =>
        text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(w => 3L + w.Length % 15).ToList();
}

/// <summary>
/// Invariants of the generation loop (MusicGenerationPipeline.Generate) with the small model: shape, type, range,
/// reproducibility, CFG rows, stopping at EOS and maximum length. Needs no checkpoint.
/// The prompt is 7 positions long and the small backbone has 16, so at most 8 more frames are generated.
/// </summary>
public class GenerationLoopTests
{
    private const double Cfg = 1.5;
    private const int TopK = 5;           // the small audio vocabulary has 10 tokens
    private static readonly int Vocab = TinyModel.Config.AudioVocabSize;
    private static readonly int Codebooks = TinyModel.Config.AudioNumCodebooks;

    private readonly ITestOutputHelper _out;
    public GenerationLoopTests(ITestOutputHelper output) => _out = output;

    private static Tensor Generate(int seed, long audioEosId, int maxMs)
    {
        var model = TinyModel.Create(1);
        var gen = new GenConfig(TextBosId: 1, TextEosId: 2, AudioEosId: audioEosId, EmptyId: 0);
        var pipe = new MusicGenerationPipeline(model, new ToyTokenizer(), gen, CPU, ScalarType.Float32);
        var inputs = pipe.Preprocess("a", "b", Cfg);
        manual_seed(seed);
        return pipe.Generate(inputs, maxAudioLengthMs: maxMs, temperature: 1.0, topk: TopK, cfgScale: Cfg);
    }

    private static long[] Column(Tensor frames, int t) => Enumerable.Range(0, (int)frames.shape[0]).Select(k => frames[k, t].item<long>()).ToArray();

    [Fact]
    public void Output_HasShapeTypeAndRange()
    {
        const int n = 4;
        var frames = Generate(seed: 7, audioEosId: Vocab, maxMs: n * 80);   // audio_eos_id outside the vocabulary: there is never an EOS

        Assert.Equal(new long[] { Codebooks, n + 1 }, frames.shape);
        Assert.Equal(ScalarType.Int64, frames.dtype);
        Assert.True(frames.min().item<long>() >= 0 && frames.max().item<long>() < Vocab, "token outside [0, audio_vocab_size)");
    }

    [Fact]
    public void SameSeed_SameOutput_AnotherSeed_DifferentOutput()
    {
        var a = Generate(seed: 7, audioEosId: Vocab, maxMs: 6 * 80);
        var b = Generate(seed: 7, audioEosId: Vocab, maxMs: 6 * 80);
        var c = Generate(seed: 8, audioEosId: Vocab, maxMs: 6 * 80);

        Assert.True(a.Equals(b), "the same seed should give the same output");
        Assert.False(a.Equals(c), "another seed should give a different output");
    }

    [Fact]
    public void CfgRows_ReceiveTheSameTokensInEveryFrame()
    {
        var model = TinyModel.Create(2);
        model.SetupCaches(2);
        using var _ = no_grad();
        var audio = (arange(2 * 4 * Codebooks, dtype: ScalarType.Int64) * 7 % Vocab).reshape(2, 4, Codebooks);
        var text = (arange(2 * 4, dtype: ScalarType.Int64) * 5 % 20).reshape(2, 4, 1);
        var tokens = cat(new[] { audio, text }, dim: -1);
        var mask = ones(new long[] { 2, 4, Codebooks + 1 }, dtype: ScalarType.Bool);
        var pos = arange(4, dtype: ScalarType.Int64).unsqueeze(0).repeat(2, 1);

        manual_seed(3);
        var curr = model.GenerateFrame(tokens, mask, pos, 1.0, TopK, Cfg);            // prompt frame
        for (int i = 0; i < 4; i++)
        {
            Assert.True(curr[0].Equals(curr[1]), $"frame {i}: the two CFG rows differ");
            var (padded, paddedMask) = MusicGenerationPipeline.PadAudioToken(curr, emptyId: 0);
            curr = model.GenerateFrame(padded, paddedMask, pos[.., ^1..] + (i + 1), 1.0, TopK, Cfg);
        }
        Assert.True(curr[0].Equals(curr[1]), "last frame: the two CFG rows differ");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    [InlineData(5)]
    public void WithoutEos_GeneratesPromptFramePlusMaxMsOver80(int extraFrames)
    {
        var frames = Generate(seed: 11, audioEosId: Vocab, maxMs: extraFrames * 80);

        Assert.Equal(extraFrames + 1, frames.shape[1]);
    }

    [Fact]
    public void Eos_StopsTheLoop_AndTheEosFrameIsNotInTheOutput()
    {
        const int n = 8;
        var free = Generate(seed: 5, audioEosId: Vocab, maxMs: n * 80);   // reference without EOS with the same seed
        Assert.Equal(n + 1, free.shape[1]);

        // An EOS threshold such that the first loop frame that reaches it (never the prompt one) is frame 2 or later.
        long threshold = -1, stopAt = -1;
        for (long thr = Vocab - 1; thr >= 1 && threshold < 0; thr--)
            for (int t = 1; t <= n; t++)
                if (Column(free, t).Any(v => v >= thr)) { if (t >= 2) { threshold = thr; stopAt = t; } break; }
        Assert.True(threshold >= 0, "no EOS threshold found that cuts generation midway; change the test seed");

        var stopped = Generate(seed: 5, audioEosId: threshold, maxMs: n * 80);

        _out.WriteLine($"audio_eos_id={threshold}: the reference without EOS has {n + 1} frames, with EOS it stops at frame {stopAt}");
        Assert.Equal(stopAt, stopped.shape[1]);                                  // frames 0..stopAt-1; the EOS frame is not emitted
        Assert.True(stopped.Equals(free[.., ..(int)stopAt]), "the frames before EOS should be identical to those of the generation without EOS");
        Assert.True(Enumerable.Range(0, (int)stopped.shape[1]).Skip(1).All(t => Column(stopped, t).All(v => v < threshold)),
            "no output frame (except the prompt one) may contain a token >= audio_eos_id");
    }

    [Fact]
    public void Eos_AtTheFirstLoopFrame_LeavesOnlyThePromptFrame()
    {
        var frames = Generate(seed: 5, audioEosId: 0, maxMs: 6 * 80);   // any token is >= 0

        Assert.Equal(new long[] { Codebooks, 1 }, frames.shape);
    }
}
