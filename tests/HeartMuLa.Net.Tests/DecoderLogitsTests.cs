using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// DecoderLogits and GenerateFrame share the codebook loop; with topk=1 sampling is deterministic (argmax),
/// so GenerateFrame must match the greedy chain built with DecoderLogits. Needs no checkpoint.
/// </summary>
public class DecoderLogitsTests
{
    private static Tensor Guided(Tensor logits, double cfg) => logits[1..] + (logits[..1] - logits[1..]) * cfg;

    [Fact]
    public void GenerateFrame_WithTopK1_EqualsGreedyChainOfDecoderLogits()
    {
        const double cfg = 1.5;
        var model = TinyModel.Create(1);
        model.SetupCaches(2);
        using var _ = no_grad();

        var audio = (arange(2 * 4 * 3, dtype: ScalarType.Int64) * 7 % 10).reshape(2, 4, 3);
        var text = (arange(2 * 4, dtype: ScalarType.Int64) * 5 % 20).reshape(2, 4, 1);
        var tokens = cat(new[] { audio, text }, dim: -1);
        var mask = ones(new long[] { 2, 4, 4 }, dtype: ScalarType.Bool);
        var pos = arange(4, dtype: ScalarType.Int64).unsqueeze(0).repeat(2, 1);

        var sampled = model.GenerateFrame(tokens, mask, pos, temperature: 1.0, topk: 1, cfgScale: cfg);   // [2, K]

        model.ResetCaches();
        var (lastH, c0) = model.BackboneLogits(tokens, mask, pos, cfg);
        int k = (int)sampled.shape[1];
        var chain = new long[k];
        chain[0] = argmax(Guided(c0, cfg), dim: -1).item<long>();
        for (int i = 1; i < k; i++)
        {
            // Codebooks with index >= i do not influence logit i: they are filled with zeros.
            var forced = tensor(chain).reshape(1, k).repeat(2, 1);
            var logits = model.DecoderLogits(lastH, forced);                                               // [2, K-1, V]
            Assert.Equal(new long[] { 2, k - 1, TinyModel.Config.AudioVocabSize }, logits.shape);
            chain[i] = argmax(Guided(logits[.., i - 1, ..], cfg), dim: -1).item<long>();
        }

        var expected = tensor(chain).reshape(1, k).repeat(2, 1);
        Assert.True(sampled.Equals(expected), $"GenerateFrame {string.Join(",", sampled[0].data<long>().ToArray())} != greedy {string.Join(",", chain)}");
    }

    [Fact]
    public void DecoderLogits_DependsOnlyOnPreviousCodebooks()
    {
        var model = TinyModel.Create(2);
        model.SetupCaches(2);
        using var _ = no_grad();
        var lastH = randn(new long[] { 2, 16 }, generator: new Generator(3));

        Tensor Logits(params long[] row) => model.DecoderLogits(lastH, tensor(row.Concat(row).ToArray()).reshape(2, 3));

        var a = Logits(3, 4, 5);
        var lastTokenChanged = Logits(3, 4, 9);   // the last codebook is never consumed: no logit changes
        var codebook1Changed = Logits(3, 7, 5);   // affects the codebook 2 logit, not the codebook 1 one

        Assert.True(a.Equals(lastTokenChanged), "the last codebook token should not influence any logit");
        Assert.True(a[.., 0, ..].Equals(codebook1Changed[.., 0, ..]), "the codebook 1 logit should not depend on the codebook 1 token");
        Assert.False(a[.., 1, ..].Equals(codebook1Changed[.., 1, ..]), "the codebook 2 logit should depend on the codebook 1 token");
    }
}
