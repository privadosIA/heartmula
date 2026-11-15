using HeartMuLa.Model;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

public sealed class LoadWeightsTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("load-weights-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static HeartMuLaModel TinyModel(int seed) => Tests.TinyModel.Create(seed);

    private static Dictionary<string, Tensor> StateOf(HeartMuLaModel m) =>
        m.named_parameters().ToDictionary(x => x.name, x => x.parameter.detach().clone());

    /// <summary>Writes a sharded checkpoint (two shards) with the state of <paramref name="source"/>, after applying <paramref name="tweak"/>.</summary>
    private CheckpointIndex WriteCheckpoint(HeartMuLaModel source, Action<Dictionary<string, Tensor>>? tweak = null)
    {
        var state = StateOf(source);
        tweak?.Invoke(state);
        var keys = state.Keys.Order(StringComparer.Ordinal).ToList();
        var half = keys.Count / 2;
        var shards = new[] { ("s1.safetensors", keys.Take(half)), ("s2.safetensors", keys.Skip(half)) };
        var weightMap = new Dictionary<string, string>();
        foreach (var (file, ks) in shards)
        {
            var list = ks.ToList();
            SafeTensorsWriter.Save(Path.Combine(_dir, file), list.ToDictionary(k => k, k => state[k]));
            foreach (var k in list) weightMap[k] = file;
        }
        File.WriteAllText(Path.Combine(_dir, "model.safetensors.index.json"),
            System.Text.Json.JsonSerializer.Serialize(new { weight_map = weightMap }));
        return CheckpointIndex.Open(_dir);
    }

    [Fact]
    public void Load_CopiesEveryTensorExactly()
    {
        var source = TinyModel(1);
        var target = TinyModel(2);
        var before = StateOf(target);
        var expected = StateOf(source);
        Assert.False(before["projection.weight"].Equals(expected["projection.weight"]), "the models should start out different");

        target.LoadWeights(WriteCheckpoint(source), ScalarType.Float32, CPU);

        foreach (var (name, p) in target.named_parameters())
            Assert.True(p.Equals(expected[name]), $"{name} does not match after loading");
    }

    [Fact]
    public void Load_ConvertsToRequestedDtype()
    {
        var source = TinyModel(1);
        var target = TinyModel(2);
        var expected = StateOf(source);

        target.LoadWeights(WriteCheckpoint(source), ScalarType.BFloat16, CPU);

        foreach (var (name, p) in target.named_parameters())
        {
            Assert.Equal(ScalarType.BFloat16, p.dtype);
            Assert.True(p.Equals(expected[name].to(ScalarType.BFloat16)), $"{name} is not the bf16 rounding of the file value");
        }
    }

    [Fact]
    public void Load_MissingKey_FailsListingItAndTouchesNothing()
    {
        var source = TinyModel(1);
        var target = TinyModel(2);
        var before = StateOf(target);
        var checkpoint = WriteCheckpoint(source, s => { s.Remove("projection.weight"); s.Remove("muq_linear.bias"); });

        var ex = Assert.Throws<InvalidDataException>(() => target.LoadWeights(checkpoint, ScalarType.Float32, CPU));

        Assert.Contains("2 model keys are missing from the checkpoint", ex.Message);
        Assert.Contains("projection.weight", ex.Message);
        Assert.Contains("muq_linear.bias", ex.Message);
        foreach (var (name, p) in target.named_parameters())
            Assert.True(p.Equals(before[name]), $"{name} changed even though validation failed");
    }

    [Fact]
    public void Load_UnexpectedKey_FailsListingIt()
    {
        var source = TinyModel(1);
        var target = TinyModel(2);
        var checkpoint = WriteCheckpoint(source, s => s["zzz.unexpected"] = ones(3));

        var ex = Assert.Throws<InvalidDataException>(() => target.LoadWeights(checkpoint, ScalarType.Float32, CPU));

        Assert.Contains("1 checkpoint keys are not in the model", ex.Message);
        Assert.Contains("zzz.unexpected", ex.Message);
    }

    [Fact]
    public void Load_ShapeMismatch_FailsNamingKeyAndBothShapes()
    {
        var source = TinyModel(1);
        var target = TinyModel(2);
        var checkpoint = WriteCheckpoint(source, s => s["codebook0_head.weight"] = zeros(7, 3));

        var ex = Assert.Throws<InvalidDataException>(() => target.LoadWeights(checkpoint, ScalarType.Float32, CPU));

        Assert.Contains("codebook0_head.weight", ex.Message);
        Assert.Contains("the model has [10, 16]", ex.Message);
        Assert.Contains("the checkpoint [7, 3]", ex.Message);
    }

    [Fact]
    public void Load_AllProblemsAreReportedInOneError()
    {
        var source = TinyModel(1);
        var target = TinyModel(2);
        var checkpoint = WriteCheckpoint(source, s =>
        {
            s.Remove("projection.weight");
            s["zzz.unexpected"] = ones(3);
            s["codebook0_head.weight"] = zeros(7, 3);
        });

        var ex = Assert.Throws<InvalidDataException>(() => target.LoadWeights(checkpoint, ScalarType.Float32, CPU));

        Assert.Contains("missing from the checkpoint", ex.Message);
        Assert.Contains("not in the model", ex.Message);
        Assert.Contains("shape mismatches", ex.Message);
    }
}
