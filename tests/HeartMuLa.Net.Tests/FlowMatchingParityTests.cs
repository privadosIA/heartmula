using System.Diagnostics;
using HeartMuLa.Codec;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace HeartMuLa.Tests;

/// <summary>
/// The real flow matching against the Python reference (golden run: Apple M4, CPU, fp32, tools/dump_reference.py codec). codes_seg0 [1,8,372] plus the
/// recorded noise randn_1_inference_codes [1,744,256] (the tensor that actually drives flow matching in detokenize) must give latents_seg0.
/// </summary>
[Collection(FlowMatchingCollection.Name)]
public sealed class FlowMatchingParityTests
{
    // CPU (this machine, fp32): measured cos_sim 1.0000012, max_abs 4.9e-6 and 118.7 dB over 30 layers x 10 Euler steps; limits with about 4x margin.
    // CUDA with the library defaults (cuDNN TF32 on, which is what Python's heartlib would use): measured max_abs 5.5e-3 and 68.5 dB on both GPUs; limits
    // 2e-2 and 60 dB. With cuDNN TF32 off the GPUs match the CPU (max_abs 6.5e-6 to 1.2e-5, 114 to 117 dB): see TfOffLatentLimits.
    private static (double MinCosSim, double MaxAbs, double MinSnrDb) Limits(Device d) =>
        d.type == DeviceType.CPU ? (0.999999, 2e-5, 100) : (0.99999, 2e-2, 60);
    private static readonly (double MinCosSim, double MaxAbs, double MinSnrDb) TfOffLatentLimits = (0.999999, 5e-5, 100);
    private static readonly (double MinCosSim, double MaxAbs, double MinSnrDb) TfOffWavLimits = (0.999999, 4e-6, 100);

    private readonly FlowMatchingFixture _fx;
    private readonly ITestOutputHelper _out;

    public FlowMatchingParityTests(FlowMatchingFixture fx, ITestOutputHelper output) { _fx = fx; _out = output; }

    [RequiresCheckpoint]
    public void Real_checkpoint_loads_339_tensors_and_skips_26_of_training_state()
    {
        var r = _fx.Report!;
        _out.WriteLine($"{r} in {_fx.LoadTime.TotalSeconds:F1} s on {_fx.Device}, peak RSS after load {_fx.PeakRssAfterLoadMb} MB");
        Assert.Equal(339, r.TensorsRead);
        Assert.Equal(339, r.ModelParameters);
        Assert.Equal(26, r.IgnoredTrainingState);
        Assert.Equal(453, r.IgnoredOutsideFlowMatching);
    }

    [RequiresCheckpoint]
    public void Reference_codes_and_noise_give_the_python_latents()
    {
        var f = Fixtures.Load("codec_latents.safetensors");
        var codes = f["codes_seg0"].to(ScalarType.Int64).to(_fx.Device);
        var noise = f["randn_1_inference_codes"].to(_fx.Device);
        var expected = f["latents_seg0"];
        Assert.Equal(new long[] { 1, 8, 372 }, codes.shape);
        Assert.Equal(new long[] { 1, 744, 256 }, expected.shape);
        using var _ = no_grad();

        // Probe: one step only, to see that it runs and how long a step takes before spending the full run.
        var sw = Stopwatch.StartNew();
        using (var probe = _fx.Model!.InferenceCodes(codes, 744, 0, 1.25, 1, noise))
        {
            if (_fx.Device.type == DeviceType.CUDA) cuda.synchronize();
            Assert.Equal(expected.shape, probe.shape);
            _out.WriteLine($"1 step: {sw.Elapsed.TotalSeconds:F1} s");
        }

        sw.Restart();
        using var got = _fx.Model!.InferenceCodes(codes, 744, 0, 1.25, 10, noise).cpu();
        sw.Stop();
        var peakMb = Process.GetCurrentProcess().PeakWorkingSet64 / 1048576;
        var cos = functional.cosine_similarity(got.reshape(-1), expected.reshape(-1), 0).item<float>();
        var diff = (got - expected).abs();
        var maxAbs = diff.max().item<float>();
        var snr = 10 * Math.Log10(expected.to(ScalarType.Float64).pow(2).sum().item<double>() / (got - expected).to(ScalarType.Float64).pow(2).sum().item<double>());
        _out.WriteLine($"{_fx.Device}: 10 steps {sw.Elapsed.TotalSeconds:F1} s, peak RSS {peakMb} MB");
        _out.WriteLine($"cos_sim {cos:F7}, max_abs {maxAbs:E3}, SNR {snr:F1} dB, |expected| rms {expected.pow(2).mean().sqrt().item<float>():F4}");

        var (minCos, maxAbsLimit, minSnr) = Limits(_fx.Device);
        Assert.True(cos >= minCos, $"cos_sim {cos:F7} < {minCos}");
        Assert.True(maxAbs <= maxAbsLimit, $"max_abs {maxAbs:E3} > {maxAbsLimit:E0}");
        Assert.True(snr >= minSnr, $"SNR {snr:F1} dB < {minSnr} dB");
    }

    [RequiresCheckpoint]
    public void Reference_codes_and_noise_give_the_python_wav_end_to_end()
    {
        const int TargetLen = 483_840;
        var f = Fixtures.Load("codec_latents.safetensors");
        var codes = f["codes_seg0"].to(ScalarType.Int64).to(_fx.Device);
        var noise = f["randn_1_inference_codes"].to(_fx.Device);
        var expected = Fixtures.Load("codec_wav.safetensors")["wav"];

        var sw = Stopwatch.StartNew();
        using var wav = HeartCodecDecoder.DecodeSegment(_fx.Model!, _fx.Vocoder!, codes, noise: noise);
        if (_fx.Device.type == DeviceType.CUDA) cuda.synchronize();
        sw.Stop();
        Assert.Equal(new long[] { 2, 744L * 1920 }, wav.shape);
        var r = AudioParity.Compare(wav[.., ..TargetLen].cpu(), expected);
        _out.WriteLine($"{_fx.Device}: codes to wav {sw.Elapsed.TotalSeconds:F1} s, peak RSS {Process.GetCurrentProcess().PeakWorkingSet64 / 1048576} MB");
        _out.WriteLine($"end to end: {r}");

        var (minCos, maxAbs, minSnr) = EndToEndLimits(_fx.Device);
        Assert.True(r.CosSim >= minCos, $"cos_sim {r.CosSim:F7} < {minCos}");
        Assert.True(r.MaxAbs <= maxAbs, $"max_abs {r.MaxAbs:E3} > {maxAbs:E0}");
        Assert.True(r.SnrDb >= minSnr, $"SNR {r.SnrDb:F1} dB < {minSnr} dB");
    }

    // CPU measured end to end: cos_sim 1.0000000, max_abs 1.0e-6, 121.3 dB (the vocoder limits, about 4x margin).
    // CUDA defaults (cuDNN TF32 on): the flow-matching error of 5.5e-3 goes through the vocoder and grows to max_abs 3.6e-2 to 3.7e-2, cos_sim 0.99988 to 0.99992,
    // 36 to 38 dB on both GPUs; limits max_abs 0.15, cos_sim 0.999, 30 dB. With cuDNN TF32 off: max_abs 8.2e-7 and 122 dB, as the CPU.
    private static (double MinCosSim, double MaxAbs, double MinSnrDb) EndToEndLimits(Device d) =>
        d.type == DeviceType.CPU ? (0.999999, 4e-6, 100) : (0.999, 0.15, 30);

    /// <summary>On CUDA: which TF32 setting costs accuracy, for the latents and end to end. Does nothing on CPU.</summary>
    [RequiresCheckpoint]
    public void Gpu_precision_modes_are_measured_on_the_configured_cuda_device()
    {
        if (_fx.Device.type != DeviceType.CUDA) return;
        var f = Fixtures.Load("codec_latents.safetensors");
        var codes = f["codes_seg0"].to(ScalarType.Int64).to(_fx.Device);
        var noise = f["randn_1_inference_codes"].to(_fx.Device);
        var expected = f["latents_seg0"];
        var expectedWav = Fixtures.Load("codec_wav.safetensors")["wav"];
        using var _ = no_grad();

        var cudnn = backends.cudnn.allow_tf32;
        var matmul = backends.cuda.matmul.allow_tf32;
        try
        {
            foreach (var (name, cudnnTf32, matmulTf32) in new[] { ("defaults", cudnn, matmul), ("cuDNN TF32 off", false, matmul), ("cuBLAS TF32 off", cudnn, false), ("both off", false, false) })
            {
                backends.cudnn.allow_tf32 = cudnnTf32;
                backends.cuda.matmul.allow_tf32 = matmulTf32;
                var sw = Stopwatch.StartNew();
                using var latents = _fx.Model!.InferenceCodes(codes, 744, 0, 1.25, 10, noise);
                cuda.synchronize();
                var time = sw.Elapsed.TotalSeconds;
                using var got = latents.cpu();
                var cos = functional.cosine_similarity(got.reshape(-1), expected.reshape(-1), 0).item<float>();
                var maxAbs = (got - expected).abs().max().item<float>();
                var snr = 10 * Math.Log10(expected.to(ScalarType.Float64).pow(2).sum().item<double>() / (got - expected).to(ScalarType.Float64).pow(2).sum().item<double>());
                using var wav = HeartCodecDecoder.DecodeSegment(_fx.Model!, _fx.Vocoder!, codes, noise: noise);
                var r = AudioParity.Compare(wav[.., ..483_840].cpu(), expectedWav);
                _out.WriteLine($"{_fx.Device} {name} (cudnn={cudnnTf32}, matmul={matmulTf32}): latents {time:F1} s cos {cos:F7} max_abs {maxAbs:E3} SNR {snr:F1} dB | wav {r}");
                if (!cudnnTf32)
                {
                    Assert.True(cos >= TfOffLatentLimits.MinCosSim && maxAbs <= TfOffLatentLimits.MaxAbs && snr >= TfOffLatentLimits.MinSnrDb,
                        $"{name}: with cuDNN TF32 off the latents should match the CPU's: cos {cos:F7}, max_abs {maxAbs:E3}, SNR {snr:F1} dB");
                    Assert.True(r.CosSim >= TfOffWavLimits.MinCosSim && r.MaxAbs <= TfOffWavLimits.MaxAbs && r.SnrDb >= TfOffWavLimits.MinSnrDb,
                        $"{name}: with cuDNN TF32 off the wav should match the CPU's: {r}");
                }
            }
        }
        finally { backends.cudnn.allow_tf32 = cudnn; backends.cuda.matmul.allow_tf32 = matmul; }
    }
}
