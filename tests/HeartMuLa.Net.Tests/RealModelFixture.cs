using System.Diagnostics;
using HeartMuLa.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// A single HeartMuLa-oss-3B loaded in bf16 for all the tests that use the real model (loading, parity...).
/// Being a collection fixture, xUnit runs those classes one after another so there are never two live models
/// (or two memory peaks) at the same time. The device comes from HEARTMULA_DEVICE (cpu by default).
/// The model (~7 GiB) only exists if HEARTMULA_CKPT is set.
/// </summary>
public sealed class RealModelFixture : IDisposable
{
    public string? Dir { get; }
    public Device Device { get; }
    public HeartMuLaModel? Model { get; }

    /// <summary>Peak resident memory of the process right after loading, before any check.</summary>
    public long PeakWorkingSetBytes { get; }
    public long WeightsBytes { get; }
    public TimeSpan LoadTime { get; }

    public RealModelFixture()
    {
        Device = new Device(Checkpoint.DeviceName);
        if (Checkpoint.Dir is not { } root) return;
        Dir = Path.Combine(root, "HeartMuLa-oss-3B");
        var config = HeartMuLaConfig.FromFile(Path.Combine(Dir, "config.json"));

        var sw = Stopwatch.StartNew();
        Model = HeartMuLaModel.Create(config, ScalarType.BFloat16);
        Model.LoadWeights(Dir, ScalarType.BFloat16, Device);
        sw.Stop();

        PeakWorkingSetBytes = Process.GetCurrentProcess().PeakWorkingSet64;
        WeightsBytes = Model.parameters().Sum(p => p.numel() * p.element_size());
        LoadTime = sw.Elapsed;
    }

    private bool _cachesReady;

    /// <summary>The model with the KV caches (batch 2, CFG) ready and empty: created the first time and reset afterwards.</summary>
    public HeartMuLaModel ModelWithFreshCaches()
    {
        var model = Model ?? throw new InvalidOperationException("no checkpoint");
        if (_cachesReady) model.ResetCaches();
        else { model.SetupCaches(2); _cachesReady = true; }
        return model;
    }

    public void Dispose() => Model?.Dispose();
}

[CollectionDefinition(Name)]
public sealed class RealModelCollection : ICollectionFixture<RealModelFixture>
{
    public const string Name = "RealModel";
}
