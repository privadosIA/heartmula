using System.Diagnostics;
using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// HeartCodec-oss's flow matching (5.97 GB checkpoint, 339 tensors loaded) and its vocoder (335 MB) read once for the tests that use the real checkpoint, on the device of
/// HEARTMULA_DEVICE (CPU by default). It does not touch the 7 GiB HeartMuLa model. It only exists when HEARTMULA_CKPT is set.
/// </summary>
public sealed class FlowMatchingFixture : IDisposable
{
    public Device Device { get; }
    public FlowMatching? Model { get; }
    public ScalarVocoder? Vocoder { get; }
    public FlowMatchingLoadReport? Report { get; }
    public TimeSpan LoadTime { get; }
    public long PeakRssAfterLoadMb { get; }

    public FlowMatchingFixture()
    {
        Device = new Device(Checkpoint.DeviceName);
        if (Checkpoint.Dir is not { } root) return;
        var sw = Stopwatch.StartNew();
        Model = new FlowMatching();
        Report = FlowMatchingWeights.Load(Model, Path.Combine(root, "HeartCodec-oss"), Device);
        Model.eval();
        Vocoder = new ScalarVocoder();
        CodecWeights.LoadVocoder(Vocoder, Path.Combine(root, "HeartCodec-oss"), Device);
        Vocoder.eval();
        LoadTime = sw.Elapsed;
        PeakRssAfterLoadMb = Process.GetCurrentProcess().PeakWorkingSet64 / 1048576;
    }

    public void Dispose() { Model?.Dispose(); Vocoder?.Dispose(); }
}

[CollectionDefinition(Name)]
public sealed class FlowMatchingCollection : ICollectionFixture<FlowMatchingFixture>
{
    public const string Name = "CodecFlowMatching";
}
