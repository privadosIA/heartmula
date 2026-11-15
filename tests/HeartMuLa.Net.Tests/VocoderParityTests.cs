using System.Diagnostics;
using HeartMuLa.Codec;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// The vocoder in fp32 against Python's reference audio. decode_in_seg0 [2,128,744] is the exact input of `scalar_model.decode` in the golden
/// run (Apple M4, CPU, fp32, tools/dump_reference.py codec); with a single segment detokenize returns the output trimmed to
/// target_len = 126/12.5·48000 = 483 840 samples, which is codec_wav.safetensors.
/// </summary>
[Collection(CodecVocoderCollection.Name)]
public sealed class VocoderParityTests
{
    // CPU: measured max_abs 1.0e-6, cos_sim 1.0000000 and 121 dB (float32 rounding noise); thresholds with ≈ 4× margin.
    // CUDA with TF32 (cuDNN's default, and what heartlib's Python would use): measured max_abs 4.1e-4 (4070) and 3.7e-4 (5060 Ti), 66 dB; thresholds
    // 1e-3 and 60 dB (the spec's initial ones, with ≈ 2.4× margin in max_abs). With TF32 off the GPU gives the same result as the CPU (max_abs 8.2e-7, 122 dB).
    private const int TargetLen = 483_840;
    private static (double MinCosSim, double MaxAbs, double MinSnrDb) Limits(Device d) =>
        d.type == DeviceType.CPU ? (0.999999, 4e-6, 100) : (0.9999, 1e-3, 60);

    private readonly CodecVocoderFixture _fx;
    private readonly ITestOutputHelper _out;

    public VocoderParityTests(CodecVocoderFixture fx, ITestOutputHelper output) { _fx = fx; _out = output; }

    [RequiresCheckpoint]
    public void ReferenceLatents_GiveThePythonWav_OnTheConfiguredDevice()
    {
        var latents = Fixtures.Load("codec_latents.safetensors")["decode_in_seg0"];
        var expected = Fixtures.Load("codec_wav.safetensors")["wav"];
        Assert.Equal(new long[] { 2, 128, 744 }, latents.shape);
        Assert.Equal(new long[] { 2, TargetLen }, expected.shape);
        using var _ = no_grad();

        var sw = Stopwatch.StartNew();
        using var y = _fx.Vocoder!.Decode(latents.to(_fx.Device));          // [2, 744·1920]
        if (_fx.Device.type == DeviceType.CUDA) cuda.synchronize();
        sw.Stop();
        var firstTime = sw.Elapsed;
        var peakMb = Process.GetCurrentProcess().PeakWorkingSet64 / 1048576;
        Assert.Equal(new long[] { 2, 744L * 1920 }, y.shape);
        var got = y[.., ..TargetLen];

        // Second decoding only to measure the warm time (on GPU the first one includes CUDA and cuDNN initialization).
        sw.Restart();
        using (var again = _fx.Vocoder!.Decode(latents.to(_fx.Device))) { if (_fx.Device.type == DeviceType.CUDA) cuda.synchronize(); }
        var warm = sw.Elapsed;

        var r = AudioParity.Compare(got, expected);
        var perChannel = new[] { AudioParity.Compare(got[0..1], expected[0..1]), AudioParity.Compare(got[1..2], expected[1..2]) };
        _out.WriteLine($"device {_fx.Device}: decoding (first) {firstTime.TotalSeconds:F1} s and warm {warm.TotalSeconds:F1} s, process peak RSS {peakMb} MB, vocoder load {_fx.LoadTime.TotalSeconds:F1} s");
        _out.WriteLine($"overall: {r}");
        _out.WriteLine($"channel 0: {perChannel[0]}");
        _out.WriteLine($"channel 1: {perChannel[1]}");
        _out.WriteLine($"peak |output| = {got.abs().max().item<float>():F4}, peak |reference| = {expected.abs().max().item<float>():F4}");

        var (minCos, maxAbs, minSnr) = Limits(_fx.Device);
        // On GPU cuDNN convolutions use TF32 by default (Ampere and later): it is repeated with TF32 off to see how much of the error that is.
        if (_fx.Device.type == DeviceType.CUDA)
        {
            var previous = backends.cudnn.allow_tf32;
            backends.cudnn.allow_tf32 = false;
            try
            {
                using var exact = _fx.Vocoder!.Decode(latents.to(_fx.Device));
                var rExact = AudioParity.Compare(exact[.., ..TargetLen], expected);
                _out.WriteLine($"TF32 off: {rExact}");
                _out.WriteLine($"TF32 on (default): max_abs {r.MaxAbs:E3} → off {rExact.MaxAbs:E3} ({r.MaxAbs / rExact.MaxAbs:F0}× less); SNR {r.SnrDb:F1} → {rExact.SnrDb:F1} dB");
                var (exactCos, exactAbs, exactSnr) = Limits(new Device(DeviceType.CPU));
                Assert.True(rExact.CosSim >= exactCos && rExact.MaxAbs <= exactAbs && rExact.SnrDb >= exactSnr,
                    $"with TF32 off the GPU should equal the CPU (cos_sim ≥ {exactCos}, max_abs ≤ {exactAbs:E0}, SNR ≥ {exactSnr} dB): {rExact}");
            }
            finally { backends.cudnn.allow_tf32 = previous; }
        }

        Assert.True(r.CosSim >= minCos, $"cos_sim {r.CosSim:F7} < {minCos}");
        Assert.True(r.MaxAbs <= maxAbs, $"max_abs {r.MaxAbs:E3} > {maxAbs:E0}");
        Assert.True(r.SnrDb >= minSnr, $"SNR {r.SnrDb:F1} dB < {minSnr} dB");
    }
}
