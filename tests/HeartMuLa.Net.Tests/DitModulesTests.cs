using MusicGen.Core.Layers;
using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

public class DitModulesTests
{
    [Fact]
    public void ProjectLayer_MatchesANaiveKernel3ConvolutionThenLinear()
    {
        const int cin = 3, cout = 4, t = 6, k = 3;
        var layer = new FlowProjectLayer(cin, cout, k);
        DitTestKit.Fill(layer, 1, 0.4);
        var x = randn(new long[] { 1, t, cin }, generator: new Generator(2));

        using var y = layer.forward(x);

        var w1 = DitTestKit.Doubles(DitTestKit.Param(layer, "ffn_1.weight"));   // [cout, cin, k]
        var b1 = DitTestKit.Doubles(DitTestKit.Param(layer, "ffn_1.bias"));
        var w2 = DitTestKit.Doubles(DitTestKit.Param(layer, "ffn_2.weight"));   // [cout, cout]
        var b2 = DitTestKit.Doubles(DitTestKit.Param(layer, "ffn_2.bias"));
        var xd = DitTestKit.Doubles(x);
        var expected = new double[t * cout];
        for (int p = 0; p < t; p++)
        {
            var conv = new double[cout];
            for (int o = 0; o < cout; o++)
            {
                double acc = b1[o];
                for (int c = 0; c < cin; c++)
                    for (int j = 0; j < k; j++)
                    {
                        int src = p - 1 + j;                                    // padding k/2 = 1 on both sides
                        if (src >= 0 && src < t) acc += w1[(o * cin + c) * k + j] * xd[src * cin + c];
                    }
                conv[o] = acc * Math.Pow(k, -0.5);
            }
            var lin = DitTestKit.Linear(conv, w2, cout, cout);
            for (int o = 0; o < cout; o++) expected[p * cout + o] = lin[o] + b2[o];
        }
        Assert.True(DitTestKit.MaxAbsDiff(expected, DitTestKit.Doubles(y)) < 1e-5, "differs from the naive projection");
        Assert.Equal(new long[] { 1, t, cout }, y.shape);
        Assert.Equal(new[] { "ffn_1.bias", "ffn_1.weight", "ffn_2.bias", "ffn_2.weight" }, layer.named_parameters().Select(p => p.name).Order());
        Assert.Equal(new long[] { cout, cin, k }, DitTestKit.Param(layer, "ffn_1.weight").shape);
    }

    [Fact]
    public void TimestepFeatures_AreCosAndSinOfTimeTimes1000TimesTheFrequencies()
    {
        var t = tensor(new[] { 0f, 0.3f, 1f });

        using var f = FlowTimestepEmbedder.Features(t);

        Assert.Equal(new long[] { 3, 512 }, f.shape);
        var fd = DitTestKit.Doubles(f);
        for (int b = 0; b < 3; b++)
            for (int i = 0; i < 256; i++)
            {
                double arg = new[] { 0.0, 0.3, 1.0 }[b] * 1000 * Math.Exp(-Math.Log(10000) * i / 256);
                Assert.True(Math.Abs(Math.Cos(arg) - fd[b * 512 + i]) < 2e-3, $"cos feature {i} of timestep {b}");        // fp32 arguments up to 1000 rad: ~1e-4..1e-3
                Assert.True(Math.Abs(Math.Sin(arg) - fd[b * 512 + 256 + i]) < 2e-3, $"sin feature {i} of timestep {b}");
            }
        Assert.All(Enumerable.Range(0, 256), i => Assert.Equal(1.0, fd[i], 6));                                                // t = 0: cos = 1
        Assert.All(Enumerable.Range(0, 256), i => Assert.Equal(0.0, fd[256 + i], 6));                                          //        sin = 0
    }

    [Fact]
    public void AdaLayerNormSingle_ProjectsSiluOfTheEmbeddingTo6Dim_AndReturnsTheEmbedding()
    {
        const int dim = 6;
        var ada = new FlowAdaLayerNormSingle(dim);
        DitTestKit.Fill(ada, 3, 0.3);
        var t = tensor(new[] { 0.25f, 0.75f });

        var (modulation, embedded) = ada.Forward(t);

        Assert.Equal(new long[] { 2, 6 * dim }, modulation.shape);
        Assert.Equal(new long[] { 2, dim }, embedded.shape);
        // Naive: embedded = Linear2(SiLU(Linear1(features))); modulation = Linear(SiLU(embedded)) + bias.
        double[] P(string n) => DitTestKit.Doubles(DitTestKit.Param(ada, n));
        var feats = DitTestKit.Doubles(FlowTimestepEmbedder.Features(t));
        var w1 = P("emb.timestep_embedder.linear_1.weight"); var b1 = P("emb.timestep_embedder.linear_1.bias");
        var w2 = P("emb.timestep_embedder.linear_2.weight"); var b2 = P("emb.timestep_embedder.linear_2.bias");
        var wl = P("linear.weight"); var bl = P("linear.bias");
        for (int b = 0; b < 2; b++)
        {
            var h = DitTestKit.Linear(feats.Skip(b * 512).Take(512).ToArray(), w1, dim, 512).Select((v, i) => v + b1[i]).Select(v => v / (1 + Math.Exp(-v))).ToArray();
            var e = DitTestKit.Linear(h, w2, dim, dim).Select((v, i) => v + b2[i]).ToArray();
            var m = DitTestKit.Linear(e.Select(v => v / (1 + Math.Exp(-v))).ToArray(), wl, 6 * dim, dim).Select((v, i) => v + bl[i]).ToArray();
            Assert.True(DitTestKit.MaxAbsDiff(e, DitTestKit.Doubles(embedded[b])) < 1e-5, $"embedding of timestep {b}");
            Assert.True(DitTestKit.MaxAbsDiff(m, DitTestKit.Doubles(modulation[b])) < 1e-5, $"modulation of timestep {b}");
        }
        Assert.Equal(new[] { "emb.timestep_embedder.linear_1.bias", "emb.timestep_embedder.linear_1.weight", "emb.timestep_embedder.linear_2.bias", "emb.timestep_embedder.linear_2.weight", "linear.bias", "linear.weight" },
            ada.named_parameters().Select(p => p.name).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void OutputNorm_IsAffinelessLayerNormModulatedWithShiftFirstThenScale()
    {
        const int dim = 8, t = 3;
        var s = randn(new long[] { 2, t, dim }, generator: new Generator(4)) * 2 + 1;
        var table = randn(new long[] { 2, dim }, generator: new Generator(5));
        var embedded = randn(new long[] { 2, dim }, generator: new Generator(6));

        using var y = FlowModulation.OutputNorm(s, table, embedded);

        var sd = DitTestKit.Doubles(s); var td = DitTestKit.Doubles(table); var ed = DitTestKit.Doubles(embedded);
        var expected = new double[2 * t * dim];
        for (int b = 0; b < 2; b++)
            for (int p = 0; p < t; p++)
            {
                var row = sd.Skip((b * t + p) * dim).Take(dim).ToArray();
                double mean = row.Average(), variance = row.Average(v => (v - mean) * (v - mean));
                for (int d = 0; d < dim; d++)
                {
                    double shift = td[0 * dim + d] + ed[b * dim + d];          // chunk 0 of (table[None] + embedded[:, None]) along dim 1
                    double scale = td[1 * dim + d] + ed[b * dim + d];          // chunk 1
                    expected[(b * t + p) * dim + d] = (row[d] - mean) / Math.Sqrt(variance + 1e-6) * (1 + scale) + shift;
                }
            }
        Assert.True(DitTestKit.MaxAbsDiff(expected, DitTestKit.Doubles(y)) < 1e-5, "differs from the naive output normalization");
    }

    private static FlowTransformerBlock Block(int seed)
    {
        var block = new FlowTransformerBlock(dim: 8, heads: 2, headDim: 4);
        DitTestKit.Fill(block, seed, 0.3);
        using var _ = no_grad();
        DitTestKit.Param(block, "scale_shift_table").zero_();
        return block;
    }

    private static Tensor OnlyChunk(int chunk, int batch = 1, int dim = 8, double value = 1.0)
    {
        var m = zeros(batch, 6 * dim);
        m[.., (chunk * dim)..((chunk + 1) * dim)] = full(new long[] { batch, dim }, value);
        return m;
    }

    [Fact]
    public void Block_WithZeroGates_IsTheIdentity()
    {
        var block = Block(7);
        var x = randn(new long[] { 1, 5, 8 }, generator: new Generator(8));
        using var _ = no_grad();

        using var y = block.Forward(x, zeros(1, 48));

        Assert.True(y.Equals(x));
    }

    [Fact]
    public void Block_ModulationChunks_AreInTheReferenceOrder()
    {
        // Chunk order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
        var block = Block(9);
        var x = randn(new long[] { 1, 5, 8 }, generator: new Generator(10));
        var norm1 = new FlowRmsNorm(8); var norm2 = new FlowRmsNorm(8);
        using var _ = no_grad();
        foreach (var (name, p) in block.named_parameters())
        {
            if (name.StartsWith("attn_norm.")) DitTestKit.Param(norm1, "weight").copy_(p);
            if (name.StartsWith("mlp_norm.")) DitTestKit.Param(norm2, "weight").copy_(p);
        }
        var attn = block.GetType().GetField("attn", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(block) as FlowAttention;
        var mlp = block.GetType().GetField("mlp", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(block) as FlowMlp;

        // Only gate_msa = 1: x + attn(norm(x)).
        using (var y = block.Forward(x, OnlyChunk(2)))
            Assert.True((y - (x + attn!.forward(norm1.forward(x)))).abs().max().item<float>() < 1e-5f, "gate_msa is chunk 2");
        // Only gate_mlp = 1: x + mlp(norm(x)).
        using (var y = block.Forward(x, OnlyChunk(5)))
            Assert.True((y - (x + mlp!.forward(norm2.forward(x)))).abs().max().item<float>() < 1e-5f, "gate_mlp is chunk 5");
        // gate_msa = 1 and scale_msa = 1: the attention input is norm(x)·2.
        var both = OnlyChunk(2); both[.., 8..16] = ones(1, 8);
        using (var y = block.Forward(x, both))
            Assert.True((y - (x + attn!.forward(norm1.forward(x) * 2))).abs().max().item<float>() < 1e-5f, "scale_msa is chunk 1 and multiplies by (1 + scale)");
        // gate_msa = 1 and shift_msa = 0.5: the attention input is norm(x) + 0.5.
        var withShift = OnlyChunk(2); withShift[.., 0..8] = full(new long[] { 1, 8 }, 0.5);
        using (var y = block.Forward(x, withShift))
            Assert.True((y - (x + attn!.forward(norm1.forward(x) + 0.5))).abs().max().item<float>() < 1e-5f, "shift_msa is chunk 0");
        // gate_mlp = 1 with scale_mlp = 1 and shift_mlp = 0.5 (chunks 4 and 3), after a zero attention branch.
        var mlpMod = OnlyChunk(5); mlpMod[.., 32..40] = ones(1, 8); mlpMod[.., 24..32] = full(new long[] { 1, 8 }, 0.5);
        using (var y = block.Forward(x, mlpMod))
            Assert.True((y - (x + mlp!.forward(norm2.forward(x) * 2 + 0.5))).abs().max().item<float>() < 1e-5f, "shift_mlp is chunk 3 and scale_mlp chunk 4");
    }

    [Fact]
    public void Block_AddsItsScaleShiftTableToTheModulation()
    {
        var block = Block(11);
        var x = randn(new long[] { 1, 4, 8 }, generator: new Generator(12));
        using var _ = no_grad();
        using var viaModulation = block.Forward(x, OnlyChunk(2));
        DitTestKit.Param(block, "scale_shift_table")[2] = ones(8);          // the table row 2 (gate_msa) = 1 ...
        using var viaTable = block.Forward(x, zeros(1, 48));                 // ... with a zero modulation must give the same result

        Assert.True((viaModulation - viaTable).abs().max().item<float>() < 1e-6f);
    }

    [Fact]
    public void Block_HasTheCheckpointParameterNames()
    {
        var names = new FlowTransformerBlock(8, 2, 4).named_parameters().Select(p => p.name).Order(StringComparer.Ordinal).ToList();

        Assert.Equal(new[]
        {
            "attn.k_proj.weight", "attn.o_proj.weight", "attn.q_proj.weight", "attn.v_proj.weight", "attn_norm.weight",
            "mlp.down.weight", "mlp.gate.weight", "mlp.up.weight", "mlp_norm.weight", "scale_shift_table",
        }, names);
    }
}
