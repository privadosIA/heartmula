using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>RVQ lookup, conditioning and masks of the flow-matching module, against naive loops over the parameters.</summary>
public class FlowMatchingTests
{
    private static readonly FlowEstimatorConfig SmallEstimator = new(Heads: 2, HeadDim: 4, InChannels: 12, OutChannels: 5, Layers: 2, Layers2: 1);
    private static readonly FlowMatchingConfig Small = new(Dim: 6, CodebookSize: 10, CodebookDim: 3, NumQuantizers: 2, Estimator: SmallEstimator);

    [Fact]
    public void Parameter_names_are_the_339_inference_keys_of_the_checkpoint()
    {
        var estimator = new FlowEstimatorConfig(Heads: 1, HeadDim: 2, InChannels: 4, OutChannels: 2, Layers: 24, Layers2: 6);
        using var model = new FlowMatching(new FlowMatchingConfig(Dim: 4, CodebookSize: 5, CodebookDim: 2, NumQuantizers: 8, Estimator: estimator));
        var names = model.named_parameters().Select(p => p.name).ToHashSet();

        Assert.Equal(339, names.Count);
        Assert.Contains("cond_feature_emb.weight", names);
        Assert.Contains("cond_feature_emb.bias", names);
        Assert.Contains("zero_cond_embedding1", names);
        Assert.Contains("vq_embed.project_out.weight", names);
        Assert.Contains("vq_embed.project_out.bias", names);
        for (int i = 0; i < 8; i++) Assert.Contains($"vq_embed.layers.{i}._codebook.embed", names);
        Assert.Equal(326, names.Count(n => n.StartsWith("estimator.")));
    }

    [Fact]
    public void Codebook_tables_have_the_checkpoint_shape()
    {
        using var model = new FlowMatching(Small);
        Assert.Equal(new long[] { 1, 10, 3 }, DitTestKit.Param(model, "vq_embed.layers.1._codebook.embed").shape);
        Assert.Equal(new long[] { 6, 3 }, DitTestKit.Param(model, "vq_embed.project_out.weight").shape);
    }

    [Fact]
    public void Condition_matches_a_naive_computation()
    {
        using var model = new FlowMatching(Small);
        DitTestKit.Fill(model, 11);
        const int B = 2, T = 4, Q = 2, Dim = 6, CbDim = 3;
        var codeValues = new long[B * Q * T];
        for (int i = 0; i < codeValues.Length; i++) codeValues[i] = (i * 7 + 3) % 10;
        using var codes = tensor(codeValues, new long[] { B, Q, T });
        // Some frames outside the latent so the zero embedding is exercised.
        using var masks = FlowMatching.LatentMasks(B, 2 * T, latentLength: 6, incontextLength: 0, otherSeg: false, device: CPU);
        using var mu = model.Condition(codes, masks);
        Assert.Equal(new long[] { B, 2 * T, Dim }, mu.shape);

        var embed = Enumerable.Range(0, Q).Select(q => DitTestKit.Doubles(DitTestKit.Param(model, $"vq_embed.layers.{q}._codebook.embed"))).ToArray();
        var wOut = DitTestKit.Doubles(DitTestKit.Param(model, "vq_embed.project_out.weight"));
        var bOut = DitTestKit.Doubles(DitTestKit.Param(model, "vq_embed.project_out.bias"));
        var wCond = DitTestKit.Doubles(DitTestKit.Param(model, "cond_feature_emb.weight"));
        var bCond = DitTestKit.Doubles(DitTestKit.Param(model, "cond_feature_emb.bias"));
        var zero = DitTestKit.Doubles(DitTestKit.Param(model, "zero_cond_embedding1"));

        var expected = new double[B * 2 * T * Dim];
        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++)
            {
                var sum = new double[CbDim];
                for (int q = 0; q < Q; q++)
                {
                    long code = codeValues[(b * Q + q) * T + t];
                    for (int d = 0; d < CbDim; d++) sum[d] += embed[q][code * CbDim + d];
                }
                var projected = DitTestKit.Linear(sum, wOut, Dim, CbDim).Select((v, i) => v + bOut[i]).ToArray();
                var cond = DitTestKit.Linear(projected, wCond, Dim, Dim).Select((v, i) => v + bCond[i]).ToArray();
                for (int rep = 0; rep < 2; rep++)
                {
                    int frame = 2 * t + rep;
                    for (int d = 0; d < Dim; d++)
                        expected[(b * 2 * T + frame) * Dim + d] = frame < 6 ? cond[d] : zero[d];
                }
            }

        Assert.True(DitTestKit.MaxAbsDiff(DitTestKit.Doubles(mu), expected) < 1e-5);
    }

    [Fact]
    public void Masks_are_2_inside_the_latent_and_1_for_the_in_context_frames()
    {
        using var plain = FlowMatching.LatentMasks(1, 8, latentLength: 6, incontextLength: 3, otherSeg: false, device: CPU);
        Assert.Equal(new long[] { 2, 2, 2, 2, 2, 2, 0, 0 }, plain[0].data<long>().ToArray());
        using var other = FlowMatching.LatentMasks(1, 8, latentLength: 6, incontextLength: 3, otherSeg: true, device: CPU);
        Assert.Equal(new long[] { 1, 1, 1, 2, 2, 2, 0, 0 }, other[0].data<long>().ToArray());
    }

    [Fact]
    public void Out_of_range_code_streams_are_rejected()
    {
        using var model = new FlowMatching(Small);
        using var codes = zeros(new long[] { 1, 3, 2 }, ScalarType.Int64);
        using var masks = FlowMatching.LatentMasks(1, 4, 4, 0, false, CPU);
        Assert.Throws<ArgumentException>(() => model.Condition(codes, masks));
    }
}
