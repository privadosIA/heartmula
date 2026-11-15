using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// HeartCodec-oss's vocoder decoder (335 MB, fp32) loaded once for the tests with the real checkpoint, on the device of
/// HEARTMULA_DEVICE (CPU by default). It does not touch HeartMuLa's 7 GiB model. It only exists if HEARTMULA_CKPT is defined.
/// </summary>
public sealed class CodecVocoderFixture : IDisposable
{
    public Device Device { get; }
    public ScalarVocoder? Vocoder { get; }
    public TimeSpan LoadTime { get; }

    public CodecVocoderFixture()
    {
        Device = new Device(Checkpoint.DeviceName);
        if (Checkpoint.Dir is not { } root) return;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Vocoder = new ScalarVocoder();
        CodecWeights.LoadVocoder(Vocoder, Path.Combine(root, "HeartCodec-oss"), Device);
        Vocoder.eval();
        LoadTime = sw.Elapsed;
    }

    public void Dispose() => Vocoder?.Dispose();
}

[CollectionDefinition(Name)]
public sealed class CodecVocoderCollection : ICollectionFixture<CodecVocoderFixture>
{
    public const string Name = "CodecVocoder";
}
