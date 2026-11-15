using MusicGen.Core.Layers;
using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace HeartMuLa.Tests;

/// <summary>Helpers shared by the flow-matching tests: deterministic parameters and naive double-precision reference maths.</summary>
internal static class DitTestKit
{
    /// <summary>Fills every parameter with deterministic normal values (the modules allocate some with uninitialised memory).</summary>
    public static void Fill(Module m, int seed, double scale = 0.2)
    {
        var g = new Generator((ulong)seed);
        using var _ = no_grad();
        foreach (var (name, p) in m.named_parameters())
            p.copy_(randn(p.shape, generator: g) * scale);
    }

    public static double[] Doubles(Tensor t) => t.detach().cpu().to(ScalarType.Float64).contiguous().data<double>().ToArray();

    public static Tensor Param(Module m, string name) => m.named_parameters().Single(p => p.name == name).parameter;

    /// <summary>y[o] = Σ_i x[i] · w[o, i]  (nn.Linear without bias), w flattened [out, in].</summary>
    public static double[] Linear(double[] x, double[] w, int outDim, int inDim)
    {
        var y = new double[outDim];
        for (int o = 0; o < outDim; o++)
        {
            double acc = 0;
            for (int i = 0; i < inDim; i++) acc += x[i] * w[o * inDim + i];
            y[o] = acc;
        }
        return y;
    }

    /// <summary>Rotates adjacent pairs of the first <paramref name="ropeDim"/> entries of <paramref name="v"/> for position <paramref name="pos"/>.</summary>
    public static double[] Rope(double[] v, int pos, int ropeDim)
    {
        var o = (double[])v.Clone();
        for (int i = 0; i < ropeDim / 2; i++)
        {
            double angle = pos * (1.0 / Math.Pow(10000.0, 2.0 * i / ropeDim));
            double c = Math.Cos(angle), s = Math.Sin(angle);
            o[2 * i] = v[2 * i] * c - v[2 * i + 1] * s;
            o[2 * i + 1] = v[2 * i] * s + v[2 * i + 1] * c;
        }
        return o;
    }

    public static double MaxAbsDiff(double[] a, double[] b) => a.Zip(b, (x, y) => Math.Abs(x - y)).Max();
}

public class DitBlocksTests
{
    [Fact]
    public void RmsNorm_MatchesTheNaiveFormula_WithoutCastingToFp32()
    {
        const int dim = 12, rows = 5;
        var norm = new FlowRmsNorm(dim);
        DitTestKit.Fill(norm, 1, 1.0);
        var x = randn(new long[] { 2, rows, dim }, generator: new Generator(2)) * 3;

        using var y = norm.forward(x);

        var w = DitTestKit.Doubles(DitTestKit.Param(norm, "weight"));
        var xd = DitTestKit.Doubles(x);
        var expected = new double[xd.Length];
        for (int r = 0; r < 2 * rows; r++)
        {
            double mean = Enumerable.Range(0, dim).Average(i => xd[r * dim + i] * xd[r * dim + i]);
            for (int i = 0; i < dim; i++) expected[r * dim + i] = w[i] * xd[r * dim + i] / Math.Sqrt(mean + 1e-6);
        }
        Assert.True(DitTestKit.MaxAbsDiff(expected, DitTestKit.Doubles(y)) < 1e-5);
        Assert.Equal(ScalarType.Float32, y.dtype);
        Assert.Equal(new[] { "weight" }, norm.named_parameters().Select(p => p.name));
    }

    [Theory]
    [InlineData(8, 8)]       // full rotation
    [InlineData(8, 4)]       // only the first 4 channels, the rest untouched
    [InlineData(128, 128)]   // stage-2 head dimension
    public void Rope_RotatesEachAdjacentPairByPositionTimesInverseFrequency(int dim, int ropeDim)
    {
        const int t = 7;
        var x = randn(new long[] { 1, 2, t, dim }, generator: new Generator(3));
        var rope = new RotaryEmbedding(RopeLayout.AdjacentPairs, RopeFrequencies.Plain(10000), RopeTable.InputDtypePerCall, dim, ropeDim: ropeDim);

        using var y = rope.Apply(x, null, seqDim: 2);

        var xd = DitTestKit.Doubles(x); var yd = DitTestKit.Doubles(y);
        for (int h = 0; h < 2; h++)
            for (int p = 0; p < t; p++)
            {
                var row = xd.Skip((h * t + p) * dim).Take(dim).ToArray();
                var expected = DitTestKit.Rope(row, p, ropeDim);
                Assert.True(DitTestKit.MaxAbsDiff(expected, yd.Skip((h * t + p) * dim).Take(dim).ToArray()) < 2e-5, $"head {h}, position {p}");
            }
    }

    [Fact]
    public void Rope_LeavesPositionZeroUnchanged_AndPreservesNorms()
    {
        var x = randn(new long[] { 1, 1, 5, 16 }, generator: new Generator(4));
        var rope = new RotaryEmbedding(RopeLayout.AdjacentPairs, RopeFrequencies.Plain(10000), RopeTable.InputDtypePerCall, 16);

        using var y = rope.Apply(x, null, seqDim: 2);

        Assert.True(y[.., .., 0].Equals(x[.., .., 0]), "position 0 must not rotate");
        Assert.True((y.pow(2).sum(-1) - x.pow(2).sum(-1)).abs().max().item<float>() < 1e-5f, "a rotation must preserve the norm of each vector");
    }

    [Fact]
    public void Attention_MatchesANaiveSoftmaxAttentionWithRope()
    {
        const int dim = 8, heads = 2, headDim = 4, t = 6;
        var attn = new FlowAttention(dim, heads, headDim, HeartCodecRope.Create(headDim));
        DitTestKit.Fill(attn, 5);
        var x = randn(new long[] { 1, t, dim }, generator: new Generator(6));

        using var y = attn.forward(x);

        double[] W(string n) => DitTestKit.Doubles(DitTestKit.Param(attn, n));
        var wq = W("q_proj.weight"); var wk = W("k_proj.weight"); var wv = W("v_proj.weight"); var wo = W("o_proj.weight");
        var xd = DitTestKit.Doubles(x);
        int inner = heads * headDim;
        var q = new double[t][]; var k = new double[t][]; var v = new double[t][];
        for (int p = 0; p < t; p++)
        {
            var row = xd.Skip(p * dim).Take(dim).ToArray();
            q[p] = DitTestKit.Linear(row, wq, inner, dim); k[p] = DitTestKit.Linear(row, wk, inner, dim); v[p] = DitTestKit.Linear(row, wv, inner, dim);
        }
        var expected = new double[t * dim];
        for (int p = 0; p < t; p++)
        {
            var concat = new double[inner];
            for (int h = 0; h < heads; h++)
            {
                double[] Head(double[] vec) => vec.Skip(h * headDim).Take(headDim).ToArray();
                var qh = DitTestKit.Rope(Head(q[p]), p, headDim);
                var scores = Enumerable.Range(0, t).Select(s => Enumerable.Range(0, headDim).Sum(d => qh[d] * DitTestKit.Rope(Head(k[s]), s, headDim)[d]) / Math.Sqrt(headDim)).ToArray();
                var max = scores.Max(); var exp = scores.Select(s => Math.Exp(s - max)).ToArray(); var sum = exp.Sum();
                for (int d = 0; d < headDim; d++) concat[h * headDim + d] = Enumerable.Range(0, t).Sum(s => exp[s] / sum * Head(v[s])[d]);
            }
            var o = DitTestKit.Linear(concat, wo, dim, inner);
            Array.Copy(o, 0, expected, p * dim, dim);
        }
        Assert.True(DitTestKit.MaxAbsDiff(expected, DitTestKit.Doubles(y)) < 1e-5, "differs from the naive attention");
        Assert.Equal(new long[] { 1, t, dim }, y.shape);
        Assert.Equal(new[] { "k_proj.weight", "o_proj.weight", "q_proj.weight", "v_proj.weight" }, attn.named_parameters().Select(p => p.name).Order());
    }

    [Fact]
    public void Attention_IsNonCausal_ALaterTokenChangesEarlierOutputs()
    {
        var attn = new FlowAttention(8, 2, 4, HeartCodecRope.Create(4));
        DitTestKit.Fill(attn, 7);
        var x = randn(new long[] { 1, 6, 8 }, generator: new Generator(8));
        var x2 = x.clone();
        x2[.., 5] = x2[.., 5] + 3;

        using var a = attn.forward(x);
        using var b = attn.forward(x2);

        Assert.False(a[.., 0].Equals(b[.., 0]), "attention is not causal: the last token must influence the first output");
    }

    [Theory]
    [InlineData(1536, 4096)]
    [InlineData(3072, 8192)]
    [InlineData(512, 1536)]    // 2·2048/3 = 1365 → next multiple of 256
    [InlineData(8, 256)]       // tiny widths still round up to 256
    public void Mlp_HiddenSizeFollowsTheReferenceRule(int dim, int expected) => Assert.Equal(expected, FlowMlp.HiddenSize(dim));

    [Fact]
    public void Mlp_MatchesTheNaiveSwiGlu()
    {
        const int dim = 8, t = 4;
        var mlp = new FlowMlp(dim);
        DitTestKit.Fill(mlp, 9, 0.3);
        var x = randn(new long[] { 1, t, dim }, generator: new Generator(10));
        int hidden = FlowMlp.HiddenSize(dim);

        using var y = mlp.forward(x);

        var wg = DitTestKit.Doubles(DitTestKit.Param(mlp, "gate.weight")); var wu = DitTestKit.Doubles(DitTestKit.Param(mlp, "up.weight")); var wd = DitTestKit.Doubles(DitTestKit.Param(mlp, "down.weight"));
        var xd = DitTestKit.Doubles(x);
        var expected = new double[t * dim];
        for (int p = 0; p < t; p++)
        {
            var row = xd.Skip(p * dim).Take(dim).ToArray();
            var g = DitTestKit.Linear(row, wg, hidden, dim); var u = DitTestKit.Linear(row, wu, hidden, dim);
            var act = g.Zip(u, (gi, ui) => gi / (1 + Math.Exp(-gi)) * ui).ToArray();
            Array.Copy(DitTestKit.Linear(act, wd, dim, hidden), 0, expected, p * dim, dim);
        }
        Assert.True(DitTestKit.MaxAbsDiff(expected, DitTestKit.Doubles(y)) < 1e-5, "differs from the naive SwiGLU");
        Assert.Equal(new[] { "down.weight", "gate.weight", "up.weight" }, mlp.named_parameters().Select(p => p.name).Order());
    }
}
