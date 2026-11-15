using MusicGen.Core.Layers;
using HeartMuLa.Codec;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// The assembled DiT estimator. The parameter names and shapes it must have are written here from the architecture of the reference (not read from the
/// implementation), and the same description is compared with a small model and, with the real checkpoint, with the shapes in its headers.
/// </summary>
public class DitEstimatorTests
{
    private static readonly FlowEstimatorConfig Small = new(Heads: 2, HeadDim: 4, InChannels: 12, OutChannels: 5, Layers: 2, Layers2: 1);
    // The real number of blocks (24 + 6) with tiny dimensions: same names as the real estimator without allocating 6 GB.
    private static readonly FlowEstimatorConfig RealStructure = new(Heads: 1, HeadDim: 2, InChannels: 4, OutChannels: 2, Layers: 24, Layers2: 6);

    /// <summary>Parameter name → shape for any configuration, from the architecture of heartlib's LlamaTransformer.</summary>
    internal static Dictionary<string, long[]> ExpectedShapes(FlowEstimatorConfig c)
    {
        var s = new Dictionary<string, long[]>();
        void Project(string name, int cin, int cout, int k)
        {
            s[$"{name}.ffn_1.weight"] = new long[] { cout, cin, k };
            s[$"{name}.ffn_1.bias"] = new long[] { cout };
            s[$"{name}.ffn_2.weight"] = new long[] { cout, cout };
            s[$"{name}.ffn_2.bias"] = new long[] { cout };
        }
        void Ada(string name, int dim)
        {
            s[$"{name}.emb.timestep_embedder.linear_1.weight"] = new long[] { dim, 512 };
            s[$"{name}.emb.timestep_embedder.linear_1.bias"] = new long[] { dim };
            s[$"{name}.emb.timestep_embedder.linear_2.weight"] = new long[] { dim, dim };
            s[$"{name}.emb.timestep_embedder.linear_2.bias"] = new long[] { dim };
            s[$"{name}.linear.weight"] = new long[] { 6 * dim, dim };
            s[$"{name}.linear.bias"] = new long[] { 6 * dim };
        }
        void Blocks(string name, int count, int dim)
        {
            int hidden = FlowMlp.HiddenSize(dim);
            for (int i = 0; i < count; i++)
            {
                var p = $"{name}.{i}";
                foreach (var proj in new[] { "q_proj", "k_proj", "v_proj", "o_proj" }) s[$"{p}.attn.{proj}.weight"] = new long[] { dim, dim };
                s[$"{p}.attn_norm.weight"] = new long[] { dim };
                s[$"{p}.mlp.gate.weight"] = new long[] { hidden, dim };
                s[$"{p}.mlp.up.weight"] = new long[] { hidden, dim };
                s[$"{p}.mlp.down.weight"] = new long[] { dim, hidden };
                s[$"{p}.mlp_norm.weight"] = new long[] { dim };
                s[$"{p}.scale_shift_table"] = new long[] { 6, dim };
            }
        }
        Project("proj_in", c.InChannels, c.Inner, 3);
        Ada("adaln_single", c.Inner);
        Blocks("transformer_blocks", c.Layers, c.Inner);
        s["scale_shift_table"] = new long[] { 2, c.Inner };
        Project("connection_proj", c.InChannels + c.Inner, c.Inner2, 3);
        Ada("adaln_single_2", c.Inner2);
        Blocks("transformer_blocks_2", c.Layers2, c.Inner2);
        s["scale_shift_table_2"] = new long[] { 2, c.Inner2 };
        Project("proj_out", c.Inner2, c.OutChannels, 3);
        return s;
    }

    private static void AssertSameShapes(Dictionary<string, long[]> expected, IEnumerable<(string name, long[] shape)> actual)
    {
        var a = actual.ToDictionary(x => x.name, x => x.shape);
        Assert.Equal(expected.Keys.Order(StringComparer.Ordinal), a.Keys.Order(StringComparer.Ordinal));
        foreach (var (name, shape) in expected) Assert.True(shape.SequenceEqual(a[name]), $"{name}: expected [{string.Join(", ", shape)}], got [{string.Join(", ", a[name])}]");
    }

    [Fact]
    public void ParameterNamesAndShapes_MatchTheReferenceArchitecture_ForASmallModel() =>
        AssertSameShapes(ExpectedShapes(Small), new LlamaTransformer(Small).named_parameters().Select(p => (p.name, p.parameter.shape)));

    [Fact]
    public void TheRealStructure_Has326Parameters()
    {
        var names = new LlamaTransformer(RealStructure).named_parameters().Select(p => p.name).ToList();

        Assert.Equal(326, names.Count);
        Assert.Equal(326, ExpectedShapes(FlowEstimatorConfig.Default).Count);
        Assert.Equal(ExpectedShapes(FlowEstimatorConfig.Default).Keys.Order(StringComparer.Ordinal), names.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void Stage2_HasTwiceTheWidthAndTwiceTheHeadDimension()
    {
        var c = FlowEstimatorConfig.Default;

        Assert.Equal((1536, 3072), (c.Inner, c.Inner2));
        Assert.Equal(24 * 64, c.Inner);
        Assert.Equal(24 * 128, c.Inner2);
    }

    [RequiresCheckpoint]
    public void RealConfiguration_HasExactlyTheShapesOfTheCheckpointHeaders()
    {
        var index = CheckpointIndex.Open(Path.Combine(Checkpoint.Dir!, "HeartCodec-oss"));
        const string prefix = "flow_matching.estimator.";
        var fromFile = index.Tensors.Where(kv => kv.Key.StartsWith(prefix, StringComparison.Ordinal)).Select(kv => (kv.Key[prefix.Length..], kv.Value.Shape)).ToList();

        Assert.Equal(326, fromFile.Count);
        AssertSameShapes(ExpectedShapes(FlowEstimatorConfig.Default), fromFile);
    }

    private static LlamaTransformer SmallModel()
    {
        var m = new LlamaTransformer(Small);
        DitTestKit.Fill(m, 21, 0.15);
        return m;
    }

    [Fact]
    public void Forward_ReturnsOutChannels_IsDeterministicAndFinite()
    {
        var model = SmallModel();
        var x = randn(new long[] { 2, 9, 12 }, generator: new Generator(1));
        var t = tensor(new[] { 0.3f, 0.3f });
        using var _ = no_grad();

        using var a = model.Forward(x, t);
        using var b = model.Forward(x, t);

        Assert.Equal(new long[] { 2, 9, 5 }, a.shape);
        Assert.True(a.Equals(b), "the same input must give the same output");
        Assert.True(a.isfinite().all().item<bool>());
    }

    [Fact]
    public void Forward_DependsOnTheTimestepAndOnTheInput()
    {
        var model = SmallModel();
        var x = randn(new long[] { 1, 6, 12 }, generator: new Generator(2));
        using var _ = no_grad();

        using var t0 = model.Forward(x, tensor(new[] { 0.1f }));
        using var t1 = model.Forward(x, tensor(new[] { 0.9f }));
        using var other = model.Forward(x + 0.5, tensor(new[] { 0.1f }));

        Assert.False(t0.Equals(t1), "the timestep must change the output");
        Assert.False(t0.Equals(other), "the input must change the output");
    }

    [Fact]
    public void Forward_TreatsEachBatchElementIndependently_ButMixesTimeSteps()
    {
        var model = SmallModel();
        var x = randn(new long[] { 2, 6, 12 }, generator: new Generator(3));
        var t = tensor(new[] { 0.2f, 0.7f });
        using var _ = no_grad();

        using var both = model.Forward(x, t);
        using var first = model.Forward(x[0..1], t[0..1]);
        using var second = model.Forward(x[1..2], t[1..2]);

        Assert.True((both[0..1] - first).abs().max().item<float>() < 1e-5f, "batch element 0 changed when batched");
        Assert.True((both[1..2] - second).abs().max().item<float>() < 1e-5f, "batch element 1 changed when batched");

        var x2 = x.clone();
        x2[0, 5] = x2[0, 5] + 2;                                              // alter the last time step of element 0
        using var changed = model.Forward(x2, t);
        Assert.False(both[0, 0].Equals(changed[0, 0]), "attention and the kernel-3 convolutions are not causal: a later frame must influence an earlier output");
        Assert.True(both[1].Equals(changed[1]), "element 1 must not depend on element 0");
    }
}
