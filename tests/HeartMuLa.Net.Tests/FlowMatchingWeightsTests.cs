using System.Text.Json;
using HeartMuLa.Codec;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace HeartMuLa.Tests;

/// <summary>Strict loading of <c>flow_matching.*</c> with small synthetic checkpoints that have the structure of HeartCodec-oss.</summary>
public sealed class FlowMatchingWeightsTests : IDisposable
{
    private static readonly FlowEstimatorConfig SmallEstimator = new(Heads: 2, HeadDim: 4, InChannels: 12, OutChannels: 5, Layers: 2, Layers2: 1);
    private static readonly FlowMatchingConfig Small = new(Dim: 6, CodebookSize: 10, CodebookDim: 3, NumQuantizers: 2, Estimator: SmallEstimator);

    private readonly string _dir = Directory.CreateTempSubdirectory("flow-weights-").FullName;
    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private sealed record Written(Dictionary<string, Tensor> Expected, int Trained, int Foreign);

    /// <summary>Checkpoint with every model tensor, the RVQ training state, and tensors of other modules.</summary>
    private Written Write(int seed, Action<Dictionary<string, Tensor>>? tweak = null)
    {
        using var source = new FlowMatching(Small);
        DitTestKit.Fill(source, seed);
        var tensors = new Dictionary<string, Tensor>();
        var expected = new Dictionary<string, Tensor>();
        foreach (var (name, p) in source.named_parameters())
        {
            tensors[$"flow_matching.{name}"] = p.detach().clone();
            expected[name] = p.detach().clone();
        }
        var g = new Generator((ulong)seed + 7);
        int trained = 0;
        for (int i = 0; i < Small.NumQuantizers; i++)
        {
            tensors[$"flow_matching.vq_embed.layers.{i}._codebook.embed_avg"] = randn(new long[] { 1, 10, 3 }, generator: g);
            tensors[$"flow_matching.vq_embed.layers.{i}._codebook.cluster_size"] = randn(new long[] { 1, 10 }, generator: g);
            tensors[$"flow_matching.vq_embed.layers.{i}._codebook.initted"] = ones(new long[] { 1 });
            trained += 3;
        }
        tensors["flow_matching.vq_embed.project_in.weight"] = randn(new long[] { 3, 6 }, generator: g);
        tensors["flow_matching.vq_embed.project_in.bias"] = randn(new long[] { 3 }, generator: g);
        trained += 2;
        tensors["scalar_model.decoder.0.bias"] = randn(new long[] { 4 }, generator: g);
        tensors["scalar_model.encoder.0.bias"] = randn(new long[] { 4 }, generator: g);
        tweak?.Invoke(tensors);

        var keys = tensors.Keys.Order(StringComparer.Ordinal).ToList();
        var half = keys.Count / 2;
        var weightMap = new Dictionary<string, string>();
        foreach (var (file, ks) in new[] { ("model-00001-of-00002.safetensors", keys.Take(half)), ("model-00002-of-00002.safetensors", keys.Skip(half)) })
        {
            var list = ks.ToList();
            SafeTensorsWriter.Save(Path.Combine(_dir, file), list.ToDictionary(k => k, k => tensors[k]));
            foreach (var k in list) weightMap[k] = file;
        }
        File.WriteAllText(Path.Combine(_dir, "model.safetensors.index.json"), JsonSerializer.Serialize(new { weight_map = weightMap }));
        return new Written(expected, trained, 2);
    }

    private static Dictionary<string, Tensor> Snapshot(Module m) => m.named_parameters().ToDictionary(p => p.name, p => p.parameter.detach().clone());

    [Fact]
    public void Complete_checkpoint_loads_every_model_tensor_and_skips_the_training_state()
    {
        var written = Write(1);
        using var model = new FlowMatching(Small);
        DitTestKit.Fill(model, 2);                                       // different from the checkpoint

        var report = FlowMatchingWeights.Load(model, _dir, CPU);

        Assert.Equal(written.Expected.Count, report.TensorsRead);
        Assert.Equal(written.Expected.Count, report.ModelParameters);
        Assert.Equal(written.Trained, report.IgnoredTrainingState);
        Assert.Equal(written.Foreign, report.IgnoredOutsideFlowMatching);
        foreach (var (name, p) in model.named_parameters())
            Assert.Equal(0.0, (p - written.Expected[name]).abs().max().item<float>());
    }

    [Fact]
    public void Missing_key_fails_before_touching_the_model_listing_it()
    {
        Write(1, t => t.Remove("flow_matching.cond_feature_emb.bias"));
        using var model = new FlowMatching(Small);
        DitTestKit.Fill(model, 2);
        var before = Snapshot(model);

        var ex = Assert.Throws<InvalidDataException>(() => FlowMatchingWeights.Load(model, _dir, CPU));

        Assert.Contains("cond_feature_emb.bias", ex.Message);
        foreach (var (name, p) in model.named_parameters()) Assert.Equal(0.0, (p - before[name]).abs().max().item<float>());
    }

    [Fact]
    public void Unexpected_key_that_is_not_in_the_allow_list_is_reported()
    {
        Write(1, t =>
        {
            t["flow_matching.vq_embed.layers.0._codebook.extra"] = zeros(new long[] { 1 });
            t["flow_matching.vq_embed.project_in.extra"] = zeros(new long[] { 1 });
        });
        using var model = new FlowMatching(Small);
        var ex = Assert.Throws<InvalidDataException>(() => FlowMatchingWeights.Load(model, _dir, CPU));
        Assert.Contains("vq_embed.layers.0._codebook.extra", ex.Message);
        Assert.Contains("vq_embed.project_in.extra", ex.Message);
    }

    [Fact]
    public void Wrong_shape_is_reported_with_both_shapes()
    {
        Write(1, t => t["flow_matching.zero_cond_embedding1"] = zeros(new long[] { 7 }));
        using var model = new FlowMatching(Small);
        var ex = Assert.Throws<InvalidDataException>(() => FlowMatchingWeights.Load(model, _dir, CPU));
        Assert.Contains("zero_cond_embedding1", ex.Message);
        Assert.Contains("[6]", ex.Message);
        Assert.Contains("[7]", ex.Message);
    }

    [Fact]
    public void Every_problem_is_listed_in_one_error()
    {
        Write(1, t =>
        {
            t.Remove("flow_matching.cond_feature_emb.bias");
            t["flow_matching.zero_cond_embedding1"] = zeros(new long[] { 7 });
            t["flow_matching.surprise"] = zeros(new long[] { 1 });
        });
        using var model = new FlowMatching(Small);
        var ex = Assert.Throws<InvalidDataException>(() => FlowMatchingWeights.Load(model, _dir, CPU));
        Assert.Contains("cond_feature_emb.bias", ex.Message);
        Assert.Contains("zero_cond_embedding1", ex.Message);
        Assert.Contains("surprise", ex.Message);
    }

    [Theory]
    [InlineData("vq_embed.layers.3._codebook.embed_avg", true)]
    [InlineData("vq_embed.layers.0._codebook.cluster_size", true)]
    [InlineData("vq_embed.layers.7._codebook.initted", true)]
    [InlineData("vq_embed.project_in.weight", true)]
    [InlineData("vq_embed.project_in.bias", true)]
    [InlineData("vq_embed.layers.0._codebook.embed", false)]
    [InlineData("vq_embed.project_out.weight", false)]
    [InlineData("estimator.proj_in.ffn_1.weight", false)]
    public void Allow_list_covers_exactly_the_training_state(string key, bool ignorable) =>
        Assert.Equal(ignorable, FlowMatchingWeights.IsTrainingState(key));
}
