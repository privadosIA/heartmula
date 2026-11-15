using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>Small vocoder (same upsampling factors and kernels as the real one, reduced channels) with deterministic weights.</summary>
internal static class TinyVocoder
{
    public static readonly VocoderConfig Config = new(LatentDim: 8, InitChannel: 2);

    public static ScalarVocoder Create(int seed)
    {
        var model = new ScalarVocoder(Config);
        var g = new Generator((ulong)seed);
        using var _ = no_grad();
        foreach (var (name, p) in model.named_parameters())
        {
            if (name.EndsWith("activation1.weight") || name.EndsWith("activation2.weight") || name.EndsWith("activation.weight"))
                p.copy_(full(p.shape, 0.25));                        // PReLU: PyTorch's default value
            else
                p.copy_(randn(p.shape, generator: g) * 0.2);
        }
        return model;
    }

    /// <summary>Expected parameter names, derived from heartlib's architecture (checkpoint keys with the weight_norm folded).</summary>
    public static List<string> ExpectedNames()
    {
        var names = new List<string> { "decoder.0.weight", "decoder.0.bias" };
        for (int b = 1; b <= 5; b++)
        {
            names.Add($"decoder.{b}.up_conv.layer.weight");
            names.Add($"decoder.{b}.up_conv.layer.bias");
            for (int n = 0; n < 5; n++)
            {
                foreach (var conv in new[] { "conv1", "conv2" })
                {
                    names.Add($"decoder.{b}.convs.{n}.{conv}.weight");
                    names.Add($"decoder.{b}.convs.{n}.{conv}.bias");
                }
                names.Add($"decoder.{b}.convs.{n}.activation1.weight");
                names.Add($"decoder.{b}.convs.{n}.activation2.weight");
            }
        }
        names.AddRange(new[] { "decoder.6.conv.weight", "decoder.6.conv.bias", "decoder.6.activation.weight", "decoder.7.weight", "decoder.7.bias" });
        return names;
    }
}

public class ScalarVocoderTests
{
    private static Tensor Latents(int frames, int seed) =>
        randn(new long[] { 2, TinyVocoder.Config.LatentDim, frames }, generator: new Generator((ulong)seed));

    [Fact]
    public void SamplesPerFrame_Is1920_ForTheRealConfiguration() =>
        Assert.Equal(5 * 4 * 4 * 4 * 3 * 2, VocoderConfig.Default.SamplesPerFrame);

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(6)]
    public void Output_Has1920SamplesPerFrame_PerChannel(int frames)
    {
        var model = TinyVocoder.Create(1);
        using var _ = no_grad();

        using var y = model.Decode(Latents(frames, 2));

        Assert.Equal(new long[] { 2, frames * 1920L }, y.shape);
        Assert.True(y.isfinite().all().item<bool>(), "the output has non-finite values");
    }

    [Fact]
    public void Output_OnlyDependsOnLaterInput_ThroughTheTwoFrameLookAheadOfTheFirstConv()
    {
        const int frames = 8, t = 5;
        var model = TinyVocoder.Create(3);
        using var _ = no_grad();
        var x = Latents(frames, 4);
        var x2 = x.clone();
        x2[.., .., t..] = x2[.., .., t..] + 5;                           // alter the input from frame t

        using var a = model.Decode(x);
        using var b = model.Decode(x2);

        var untouched = (t - 2) * 1920L;
        Assert.True(a[.., ..(int)untouched].Equals(b[.., ..(int)untouched]), "samples before (t − 2) · 1920 changed");
        var firstWindow = (a[.., (int)untouched..(int)(untouched + 1920)] - b[.., (int)untouched..(int)(untouched + 1920)]).abs().max().item<float>();
        Assert.True(firstWindow > 0, "the input convolution should look 2 frames ahead: frame t − 2 should change");
    }

    [Fact]
    public void Quantize9_RoundsToNinths_WithTiesToEven()
    {
        var values = new[] { 0f, 0.5f / 9, 1.5f / 9, 2.5f / 9, -0.5f / 9, -1.5f / 9, 0.3f, -0.3f, 0.9999f, 1f, 7.3f };
        using var q = ScalarVocoder.Quantize9(tensor(values));

        var expected = values.Select(v => MathF.Round(v * 9f, MidpointRounding.ToEven) / 9f).ToArray();
        Assert.Equal(expected, q.data<float>().ToArray());
        Assert.All(q.data<float>().ToArray(), v => Assert.False(float.IsNaN(v)));
    }

    [Fact]
    public void ParameterNames_AreTheCheckpointKeysWithTheWeightNormFolded()
    {
        var names = new ScalarVocoder(TinyVocoder.Config).named_parameters().Select(p => p.name).Order(StringComparer.Ordinal).ToList();
        var expected = TinyVocoder.ExpectedNames().Order(StringComparer.Ordinal).ToList();

        Assert.Equal(expected, names);
        Assert.Equal(167, names.Count);                                  // 224 checkpoint keys − 57 folded pairs
    }

    [Fact]
    public void ParameterShapes_FollowTheCheckpointLayout()
    {
        var p = new ScalarVocoder(VocoderConfig.Default with { LatentDim = 128 }).named_parameters().ToDictionary(x => x.name, x => x.parameter.shape);

        Assert.Equal(new long[] { 2048, 128, 5 }, p["decoder.0.weight"]);
        Assert.Equal(new long[] { 2048, 1024, 10 }, p["decoder.1.up_conv.layer.weight"]);   // transpuesta [in, out, k]
        Assert.Equal(new long[] { 1024, 1024, 7 }, p["decoder.1.convs.0.conv1.weight"]);
        Assert.Equal(new long[] { 1024, 1024, 1 }, p["decoder.1.convs.0.conv2.weight"]);
        Assert.Equal(new long[] { 128, 64, 6 }, p["decoder.5.up_conv.layer.weight"]);
        Assert.Equal(new long[] { 64, 64, 7 }, p["decoder.6.conv.weight"]);
        Assert.Equal(new long[] { 1, 64, 7 }, p["decoder.7.weight"]);
        Assert.Equal(new long[] { 1 }, p["decoder.1.convs.0.activation1.weight"]);
    }

    private static void Set(TorchSharp.torch.nn.Module m, string name, Tensor value)
    {
        using var _ = no_grad();
        m.named_parameters().Single(x => x.name == name).parameter.copy_(value);
    }

    [Fact]
    public void PostProcessor_RepeatsEachSampleInPlace_NotTheWholeSequence()
    {
        const int c = 3, t = 5;
        var post = new PostProcessor(c, numSamples: 2, kernel: 7, causal: true);
        var identity = zeros(c, c, 7);
        for (int i = 0; i < c; i++) identity[i, i, 6] = 1f;               // causal: the last tap is the current sample
        Set(post, "conv.weight", identity); Set(post, "conv.bias", zeros(c)); Set(post, "activation.weight", ones(1));
        var x = randn(new long[] { 2, c, t }, generator: new Generator(9));
        using var _ = no_grad();

        using var y = post.forward(x);

        Assert.Equal(new long[] { 2, c, t * 2 }, y.shape);
        for (int i = 0; i < t; i++)
        {
            Assert.True(y[.., .., 2 * i].Equals(x[.., .., i]), $"sample {2 * i}: it should be input {i}");
            Assert.True(y[.., .., 2 * i + 1].Equals(x[.., .., i]), $"sample {2 * i + 1}: it should be input {i}");
        }
    }

    [Fact]
    public void ResidualUnit_AddsItsInput_AndAppliesBothActivations()
    {
        const int c = 4, t = 9;
        var unit = new ResidualUnit(c, dilation: 3, resKernel: 7, causal: true);
        foreach (var (name, p) in unit.named_parameters())
        {
            using var _ = no_grad();
            p.copy_(name.EndsWith("weight") && name.StartsWith("activation") ? full(p.shape, 0.5) : zeros(p.shape));
        }
        var x = randn(new long[] { 1, c, t }, generator: new Generator(10));
        using var __ = no_grad();

        using var zeroConvs = unit.forward(x);
        Assert.True(zeroConvs.Equals(x), "with the convolutions at zero the residual unit should return its input");

        // conv2 contributes a constant bias −1: PReLU(−1) = −0.5; the result is input − 0.5.
        Set(unit, "conv2.bias", full(c, -1.0));
        using var biased = unit.forward(x);
        Assert.True((biased - (x - 0.5)).abs().max().item<float>() < 1e-6f);
    }
}
