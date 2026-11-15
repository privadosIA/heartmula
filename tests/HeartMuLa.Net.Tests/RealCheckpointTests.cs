using MusicGen.Core.Testing;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using HeartMuLa.Model;
using MusicGen.Core.Weights;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// Raw safetensors reader, deliberately independent of <see cref="CheckpointIndex"/>:
/// it parses the index and the headers on its own so that the tests do not compare against themselves.
/// </summary>
internal sealed class RawCheckpoint
{
    public sealed record Entry(string File, string Dtype, long[] Shape, long AbsoluteStart, long Length);

    public Dictionary<string, Entry> Tensors { get; } = new();
    public HashSet<string> IndexKeys { get; } = new();
    public int Shards { get; }

    public RawCheckpoint(string dir)
    {
        var weightMap = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "model.safetensors.index.json")))
            .RootElement.GetProperty("weight_map");
        foreach (var p in weightMap.EnumerateObject()) IndexKeys.Add(p.Name);
        var files = weightMap.EnumerateObject().Select(p => p.Value.GetString()!).Distinct().ToList();
        Shards = files.Count;

        foreach (var file in files)
        {
            var path = Path.Combine(dir, file);
            using var fs = File.OpenRead(path);
            var len = new byte[8];
            fs.ReadExactly(len);
            long headerLen = BitConverter.ToInt64(len);
            var header = new byte[headerLen];
            fs.ReadExactly(header);
            foreach (var p in JsonDocument.Parse(header).RootElement.EnumerateObject())
            {
                if (p.Name == "__metadata__") continue;
                var offsets = p.Value.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetInt64()).ToArray();
                Tensors[p.Name] = new Entry(
                    path,
                    p.Value.GetProperty("dtype").GetString()!,
                    p.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray(),
                    8 + headerLen + offsets[0],
                    offsets[1] - offsets[0]);
            }
        }
    }

    public byte[] ReadBytes(string key)
    {
        var e = Tensors[key];
        using var fs = File.OpenRead(e.File);
        fs.Seek(e.AbsoluteStart, SeekOrigin.Begin);
        var bytes = new byte[e.Length];
        fs.ReadExactly(bytes);
        return bytes;
    }
}

[Collection(RealModelCollection.Name)]
public sealed class RealCheckpointTests
{
    private const long MaxSampledTensorBytes = 64L << 20; // the giant tensors (embeddings) are covered by shapes, not by bytes

    private readonly RealModelFixture _fx;
    private readonly ITestOutputHelper _out;

    public RealCheckpointTests(RealModelFixture fx, ITestOutputHelper output) { _fx = fx; _out = output; }

    /// <summary>Deterministic sample: every 8th key in alphabetical order among those of reasonable size, plus those with a different texture.</summary>
    private static List<string> Sample(RawCheckpoint raw)
    {
        var keys = raw.Tensors.Where(kv => kv.Value.Length <= MaxSampledTensorBytes).Select(kv => kv.Key).Order(StringComparer.Ordinal).ToList();
        var sample = keys.Where((_, i) => i % 8 == 0).ToHashSet();
        foreach (var must in new[] { "audio_head", "muq_linear.bias", "backbone.norm.scale", "backbone.layers.0.attn.q_proj.weight", "projection.weight" })
            if (raw.Tensors.ContainsKey(must)) sample.Add(must);
        return sample.Order(StringComparer.Ordinal).ToList();
    }

    [RequiresCheckpoint]
    public void Keys_MatchTheIndex_AndTheModelHasNoMissingNorExtra()
    {
        var raw = new RawCheckpoint(_fx.Dir!);
        var index = CheckpointIndex.Open(_fx.Dir!);

        Assert.Equal(289, raw.IndexKeys.Count);
        Assert.Equal(4, raw.Shards);
        Assert.True(raw.IndexKeys.SetEquals(index.Tensors.Keys), "the CheckpointIndex keys differ from the weight_map");
        Assert.True(raw.IndexKeys.SetEquals(raw.Tensors.Keys), "the shard headers differ from the weight_map");

        var modelKeys = _fx.Model!.named_parameters().Select(x => x.name).Concat(_fx.Model.named_buffers().Select(x => x.name)).ToHashSet();
        Assert.Empty(modelKeys.Except(raw.IndexKeys));  // extra in the model
        Assert.Empty(raw.IndexKeys.Except(modelKeys));  // missing from the model
    }

    [RequiresCheckpoint]
    public void Shapes_MatchTheHeaders_AndParametersAreBf16()
    {
        var raw = new RawCheckpoint(_fx.Dir!);

        foreach (var (name, p) in _fx.Model!.named_parameters())
        {
            Assert.Equal(raw.Tensors[name].Shape, p.shape);
            Assert.Equal(ScalarType.BFloat16, p.dtype);
        }
    }

    [RequiresCheckpoint]
    public void F32Tensors_ReadWithoutConversion_AreByteIdenticalToTheFile()
    {
        var raw = new RawCheckpoint(_fx.Dir!);
        var index = CheckpointIndex.Open(_fx.Dir!);
        var sample = Sample(raw);
        Assert.True(sample.Count >= 20, $"sample too small: {sample.Count}");

        foreach (var key in sample)
        {
            Assert.Equal("F32", raw.Tensors[key].Dtype);
            using var t = index.Read(key);
            Assert.Equal(ScalarType.Float32, t.dtype);
            Assert.Equal(raw.Tensors[key].Shape, t.shape);
            Assert.True(raw.ReadBytes(key).AsSpan().SequenceEqual(t.bytes), $"{key}: the bytes differ from those of the file");
        }
    }

    [RequiresCheckpoint]
    public void Bf16Parameters_EqualTheNearestEvenOracle_OnASample()
    {
        var raw = new RawCheckpoint(_fx.Dir!);
        var parameters = _fx.Model!.named_parameters().ToDictionary(x => x.name, x => x.parameter);
        var sample = Sample(raw);

        long values = 0;
        foreach (var key in sample)
        {
            var f32 = MemoryMarshal.Cast<byte, float>(raw.ReadBytes(key));
            var loaded = MemoryMarshal.Cast<byte, ushort>(parameters[key].cpu().bytes);
            Assert.Equal(f32.Length, loaded.Length);
            for (int i = 0; i < f32.Length; i++)
                if (loaded[i] != Bf16.RoundNearestEven(f32[i]))
                    Assert.Fail($"{key}[{i}]: F32 0x{BitConverter.SingleToUInt32Bits(f32[i]):X8} → loaded 0x{loaded[i]:X4}, oracle 0x{Bf16.RoundNearestEven(f32[i]):X4}");
            values += f32.Length;
        }
        _out.WriteLine($"{sample.Count} tensors / {values:N0} values equal to the oracle");
    }

    [RequiresCpuCheckpoint]
    public void PeakMemory_LoadingBf16OnCpu_StaysWithin1_5xOfTheWeights()
    {
        double gib(long b) => b / 1024.0 / 1024 / 1024;
        var limit = (long)(1.5 * _fx.WeightsBytes);
        _out.WriteLine($"bf16 load: {_fx.LoadTime.TotalSeconds:F0} s, weights {gib(_fx.WeightsBytes):F2} GiB, " +
                       $"peak {gib(_fx.PeakWorkingSetBytes):F2} GiB ({(double)_fx.PeakWorkingSetBytes / _fx.WeightsBytes:F2}x), limit {gib(limit):F2} GiB");

        Assert.True(_fx.PeakWorkingSetBytes <= limit,
            $"peak {gib(_fx.PeakWorkingSetBytes):F2} GiB > limit {gib(limit):F2} GiB (1.5 x {gib(_fx.WeightsBytes):F2} GiB)");
    }
}
