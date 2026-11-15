using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;
using HeartMuLa.Codec;
using MusicGen.Core.Weights;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// Strict loading of the vocoder's decoder. With small synthetic checkpoints that replicate the structure of HeartCodec-oss (weight_norm pairs
/// only in the layers that have them, and tensors of the encoder and the flow matching that are not the decoder's), and with the real checkpoint.
/// </summary>
public sealed class CodecWeightsTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("codec-weights-").FullName;
    private readonly ITestOutputHelper _out;

    public CodecWeightsTests(ITestOutputHelper output) => _out = output;
    public void Dispose() => Directory.Delete(_dir, recursive: true);

    // Layers with weight_norm in HeartCodec: the input, the upsampling and the two convolutions of each residual unit, and the final one. The post-processor
    // convolution and the PReLUs do not have it.
    private static readonly Regex WeightNormed = new(@"^decoder\.(0|7)\.weight$|^decoder\.[1-5]\.up_conv\.layer\.weight$|^decoder\.[1-5]\.convs\.\d\.conv[12]\.weight$");

    private sealed record Written(Dictionary<string, Tensor> Expected, int RawDecoderTensors, int Foreign);

    /// <summary>Writes a synthetic HeartCodec checkpoint in two shards with its index. <paramref name="tweak"/> modifies the tensor dictionary before writing it.</summary>
    private Written WriteCodecCheckpoint(int seed, Action<Dictionary<string, Tensor>>? tweak = null)
    {
        var source = TinyVocoder.Create(seed);
        var g = new Generator((ulong)seed + 100);
        var tensors = new Dictionary<string, Tensor>();
        var expected = new Dictionary<string, Tensor>();
        int rawDecoder = 0;
        using var _ = no_grad();
        foreach (var (name, p) in source.named_parameters())
        {
            if (WeightNormed.IsMatch(name))
            {
                var layer = name[..^".weight".Length];
                var v = randn(p.shape, generator: g);
                var gg = rand(new long[] { p.shape[0], 1, 1 }, generator: g) + 0.5;
                tensors[$"scalar_model.{layer}.parametrizations.weight.original0"] = gg;
                tensors[$"scalar_model.{layer}.parametrizations.weight.original1"] = v;
                rawDecoder += 2;
                // Expected value computed by hand in double precision, without going through WeightNorm.
                var w = v.to(ScalarType.Float64) * (gg.to(ScalarType.Float64) / v.to(ScalarType.Float64).pow(2).sum(new long[] { 1, 2 }, keepdim: true).sqrt());
                expected[name] = w.to(ScalarType.Float32);
            }
            else
            {
                var t = randn(p.shape, generator: g);
                tensors[$"scalar_model.{name}"] = t;
                expected[name] = t;
                rawDecoder++;
            }
        }
        // Tensors outside the decoder: they are ignored without reading them or counting them as extra.
        tensors["scalar_model.encoder.0.bias"] = randn(new long[] { 4 }, generator: g);
        tensors["scalar_model.encoder.0.parametrizations.weight.original0"] = randn(new long[] { 4, 1, 1 }, generator: g);
        tensors["flow_matching.estimator.proj_in.weight"] = randn(new long[] { 3, 3 }, generator: g);
        int foreign = 3;

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
        return new Written(expected, rawDecoder, foreign);
    }

    private static Dictionary<string, Tensor> Snapshot(ScalarVocoder m) =>
        m.named_parameters().ToDictionary(p => p.name, p => p.parameter.detach().clone());

    [Fact]
    public void CompleteCheckpoint_LoadsEveryTensor_AndFoldsTheWeightNormPairs()
    {
        var written = WriteCodecCheckpoint(seed: 1);
        var model = TinyVocoder.Create(2);                                // weights different from the checkpoint's

        var report = CodecWeights.LoadVocoder(model, _dir, CPU);

        Assert.Equal(224, report.TensorsRead);                            // 167 final weights: 57 pairs (114 tensors) + 110 plain ones
        Assert.Equal(57, report.PairsFolded);
        Assert.Equal(written.Foreign, report.IgnoredOutsideDecoder);
        Assert.Equal(167, report.ModelParameters);
        Assert.Equal(written.RawDecoderTensors, report.TensorsRead);
        foreach (var (name, p) in model.named_parameters())
            Assert.True((p - written.Expected[name]).abs().max().item<float>() < 1e-5f, $"{name} does not match the expected value");
    }

    [Fact]
    public void MissingKey_FailsBeforeTouchingTheModel_ListingIt()
    {
        WriteCodecCheckpoint(1, t => t.Remove("scalar_model.decoder.6.conv.bias"));
        var model = TinyVocoder.Create(2);
        var before = Snapshot(model);

        var ex = Assert.Throws<InvalidDataException>(() => CodecWeights.LoadVocoder(model, _dir, CPU));

        Assert.Contains("1 model keys are missing from the checkpoint", ex.Message);
        Assert.Contains("decoder.6.conv.bias", ex.Message);
        foreach (var (name, p) in model.named_parameters()) Assert.True(p.Equals(before[name]), $"{name} changed even though validation failed");
    }

    [Fact]
    public void UnexpectedDecoderKey_Fails_ListingIt()
    {
        WriteCodecCheckpoint(1, t => t["scalar_model.decoder.9.weight"] = ones(3));
        var ex = Assert.Throws<InvalidDataException>(() => CodecWeights.LoadVocoder(TinyVocoder.Create(2), _dir, CPU));

        Assert.Contains("1 decoder keys in the checkpoint are not in the model", ex.Message);
        Assert.Contains("decoder.9.weight", ex.Message);
    }

    [Fact]
    public void WrongShape_Fails_NamingTheKeyAndBothShapes()
    {
        WriteCodecCheckpoint(1, t => t["scalar_model.decoder.6.conv.weight"] = zeros(7, 3, 5));
        var ex = Assert.Throws<InvalidDataException>(() => CodecWeights.LoadVocoder(TinyVocoder.Create(2), _dir, CPU));

        Assert.Contains("decoder.6.conv.weight", ex.Message);
        Assert.Contains("the checkpoint [7, 3, 5]", ex.Message);
        Assert.Contains("the model has [2, 2, 7]", ex.Message);
    }

    [Theory]
    [InlineData("scalar_model.decoder.0.parametrizations.weight.original1", "decoder.0", "it has original0 but not original1")]
    [InlineData("scalar_model.decoder.7.parametrizations.weight.original0", "decoder.7", "it has original1 but not original0")]
    public void IncompleteWeightNormPair_Fails_NamingTheLayer(string removedKey, string layer, string detail)
    {
        WriteCodecCheckpoint(1, t => t.Remove(removedKey));
        var ex = Assert.Throws<InvalidDataException>(() => CodecWeights.LoadVocoder(TinyVocoder.Create(2), _dir, CPU));

        Assert.Contains($"incomplete weight_norm pair in {layer}", ex.Message);
        Assert.Contains(detail, ex.Message);
    }

    [Fact]
    public void EveryProblemIsReportedInOneError()
    {
        WriteCodecCheckpoint(1, t =>
        {
            t.Remove("scalar_model.decoder.6.conv.bias");
            t["scalar_model.decoder.9.weight"] = ones(3);
            t["scalar_model.decoder.6.conv.weight"] = zeros(7, 3, 5);
            t.Remove("scalar_model.decoder.0.parametrizations.weight.original1");
        });
        var ex = Assert.Throws<InvalidDataException>(() => CodecWeights.LoadVocoder(TinyVocoder.Create(2), _dir, CPU));

        Assert.Contains("missing from the checkpoint", ex.Message);
        Assert.Contains("not in the model", ex.Message);
        Assert.Contains("shape mismatches", ex.Message);
        Assert.Contains("incomplete weight_norm pair in decoder.0", ex.Message);
    }

    [Fact]
    public void ForeignTensors_AreNeitherReadNorCountedAsUnexpected()
    {
        // Many foreign tensors, even with keys similar to the decoder's but outside the exact prefix.
        WriteCodecCheckpoint(1, t =>
        {
            for (int i = 0; i < 20; i++) t[$"scalar_model.encoder.{i}.weight"] = randn(new long[] { 5, 5 });
            t["flow_matching.vq_embed.layers.0.codebook"] = randn(new long[] { 4, 4 });
        });
        var report = CodecWeights.LoadVocoder(TinyVocoder.Create(2), _dir, CPU);

        Assert.Equal(224, report.TensorsRead);                            // does not grow with the foreign ones
        Assert.Equal(3 + 20 + 1, report.IgnoredOutsideDecoder);
    }

    // ------------------------------------------------------------------ with the real checkpoint

    private static long ReadCharsOfThisProcess()
    {
        const string path = "/proc/self/io";
        if (!File.Exists(path)) return -1;
        var line = File.ReadLines(path).First(l => l.StartsWith("rchar:"));
        return long.Parse(line.Split(':')[1].Trim());
    }

    [RequiresCheckpoint]
    public void RealCheckpoint_LoadsAll224DecoderTensors_Folding57Pairs_WithoutReadingTheRest()
    {
        var model = new ScalarVocoder();
        var dir = Path.Combine(Checkpoint.Dir!, "HeartCodec-oss");
        var index = CheckpointIndex.Open(dir);                            // headers only
        long rchar0 = ReadCharsOfThisProcess();

        var sw = Stopwatch.StartNew();
        var report = CodecWeights.LoadVocoder(model, index, CPU);
        sw.Stop();
        long readMb = ReadCharsOfThisProcess() < 0 ? -1 : (ReadCharsOfThisProcess() - rchar0) / 1_000_000;

        _out.WriteLine($"load: {sw.Elapsed.TotalSeconds:F1} s; file tensors read {report.TensorsRead}, pairs folded {report.PairsFolded}, foreign ignored {report.IgnoredOutsideDecoder}, model parameters {report.ModelParameters}; bytes read ≈ {readMb} MB (the whole checkpoint is 6,200 MB)");
        Assert.Equal(818, index.Tensors.Count);
        Assert.Equal(224, report.TensorsRead);
        Assert.Equal(57, report.PairsFolded);
        Assert.Equal(818 - 224, report.IgnoredOutsideDecoder);
        Assert.Equal(167, report.ModelParameters);
        foreach (var (name, p) in model.named_parameters())
            Assert.True(p.isfinite().all().item<bool>(), $"{name} has non-finite values");
        if (readMb >= 0) Assert.True(readMb < 800, $"≈ {readMb} MB were read: it should read only the decoder (≈ 335 MB), not the whole checkpoint");
    }
}
