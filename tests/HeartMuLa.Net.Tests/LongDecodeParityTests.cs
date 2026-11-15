using System.Diagnostics;
using HeartMuLa.Codec;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// The windowed decode against the Python golden run of 400 frames (2 windows, tools/dump_reference.py codec_long, x86 CPU, fp32). Window 1 is the first
/// real check of the in-context path: the C7 solver was only tested with stubs for it.
/// </summary>
[Collection(FlowMatchingCollection.Name)]
public sealed class LongDecodeParityTests
{
    // CPU measured: window 0 max_abs 1.2e-6 / 124.9 dB, window 1 from our own window 0 5.8e-6 / 123.0 dB, window 1 from the reference window 0 5.5e-6 / 124.0 dB
    // (in-context frames exactly equal). Limits with about 4x margin in max_abs.
    // GPU with cuDNN TF32 off (the default of the decode), measured: 4070 window 0 1.6e-5 / 113.2 dB, window 1 4.5e-5 / 108.3 dB; 5060 Ti 1.3e-5 / 115.6 dB, 3.3e-5 / 111.2 dB.
    // Limits max_abs 1.5e-4 and 100 dB (about 3x and 8 dB of margin).
    private double LatentCos => 0.999999;
    private double LatentMaxAbs => _fx.Device.type == DeviceType.CPU ? 2.5e-5 : 1.5e-4;
    private double LatentSnr => _fx.Device.type == DeviceType.CPU ? 110 : 100;

    private readonly FlowMatchingFixture _fx;
    private readonly ITestOutputHelper _out;

    public LongDecodeParityTests(FlowMatchingFixture fx, ITestOutputHelper output) { _fx = fx; _out = output; }

    private static AudioParity.Result Compare(Tensor got, Tensor expected) => AudioParity.Compare(got.reshape(1, -1).cpu(), expected.reshape(1, -1).cpu());

    [RequiresCheckpoint]
    public void Window_1_with_the_reference_window_0_tail_matches_the_python_latents()
    {
        var f = Fixtures.Load("codec_long_latents.safetensors");
        var dev = _fx.Device;
        using var _ = no_grad();
        var tail = f["latents_seg0"][.., ^104..];
        var trueLatents = cat(new[] { tail, f["randn_2_detokenize"] }, 1).to(dev);

        var sw = Stopwatch.StartNew();
        var previousTf32 = backends.cudnn.allow_tf32;
        backends.cudnn.allow_tf32 = false;                                           // what the decode does by default
        using var got = _fx.Model!.InferenceCodes(f["codes_seg1"].to(ScalarType.Int64).to(dev), 744, 104, 1.25, 10, f["randn_3_inference_codes"].to(dev), trueLatents, otherSeg: true).cpu();
        sw.Stop();
        backends.cudnn.allow_tf32 = previousTf32;
        var r = Compare(got, f["latents_seg1"]);
        var head = Compare(got[.., ..104], f["latents_seg1"][.., ..104]);
        var rest = Compare(got[.., 104..], f["latents_seg1"][.., 104..]);
        _out.WriteLine($"{dev}: window 1 {sw.Elapsed.TotalSeconds:F1} s");
        _out.WriteLine($"all frames: {r}");
        _out.WriteLine($"in-context frames (0..103): {head}");
        _out.WriteLine($"generated frames (104..743): {rest}");

        Assert.Equal(0.0, (got[.., ..104] - tail).abs().max().item<float>());           // the in-context frames come back exactly
        Assert.True(r.CosSim >= LatentCos && r.MaxAbs <= LatentMaxAbs && r.SnrDb >= LatentSnr, $"{r}");
    }

    [RequiresCheckpoint]
    public void Whole_chain_from_the_codes_gives_the_python_latents_and_wav()
    {
        var f = Fixtures.Load("codec_long_latents.safetensors");
        var expectedWav = Fixtures.Load("codec_long_wav.safetensors")["wav"];
        var dev = _fx.Device;
        var codes = f["frames"].to(ScalarType.Int64).unsqueeze(0).to(dev);
        var latents = new List<Tensor>();
        Tensor Flow(Tensor seg, long ctx, Tensor noise, Tensor? trueLatents)
        {
            var r = _fx.Model!.InferenceCodes(seg, 744, ctx, 1.25, 10, noise, trueLatents, otherSeg: true);
            latents.Add(r.cpu().DetachFromDisposeScope());
            return r;
        }
        HeartCodecDecoder.WindowNoise Noise(int w) => w == 0
            ? new(null, f["randn_1_inference_codes"].to(dev))
            : new(f["randn_2_detokenize"].to(dev), f["randn_3_inference_codes"].to(dev));

        var sw = Stopwatch.StartNew();
        using var wav = HeartCodecDecoder.DecodeWith(Flow, l => _fx.Vocoder!.Decode(FlowMatching.ReorderForVocoder(l)), codes, 744, 256, Noise);
        sw.Stop();
        var peakMb = Process.GetCurrentProcess().PeakWorkingSet64 / 1048576;

        var l0 = Compare(latents[0], f["latents_seg0"]);
        var l1 = Compare(latents[1], f["latents_seg1"]);
        var r = AudioParity.Compare(wav, expectedWav);
        var overlapStart = DetokenizePlan.HopSamples;                                    // the cross-fade region of the trimmed audio
        var region = AudioParity.Compare(wav[.., overlapStart..(overlapStart + DetokenizePlan.OverlapSamples)], expectedWav[.., overlapStart..(overlapStart + DetokenizePlan.OverlapSamples)]);
        _out.WriteLine($"{dev}: 2 windows, codes to wav {sw.Elapsed.TotalSeconds:F1} s, peak RSS {peakMb} MB");
        _out.WriteLine($"latents window 0: {l0}");
        _out.WriteLine($"latents window 1 (in-context from our own window 0): {l1}");
        _out.WriteLine($"wav: {r}");
        _out.WriteLine($"cross-fade region: {region}");

        // The vocoder starts with round(9·x)/9: a latent that lands on the other side of a rounding boundary changes one quantization step.
        var flipFrames = new List<(int Window, int Frame, long Count)>();
        for (int w = 0; w < 2; w++)
        {
            var ours = FlowMatching.ReorderForVocoder(latents[w]);
            var theirs = f[$"decode_in_seg{w}"];
            var flipped = (ScalarVocoder.Quantize9(ours) != ScalarVocoder.Quantize9(theirs)).any(1).any(0).contiguous();    // [T] frames with at least one flipped value
            var flags = flipped.data<bool>().ToArray();
            long total = (ScalarVocoder.Quantize9(ours) != ScalarVocoder.Quantize9(theirs)).sum().item<long>();
            _out.WriteLine($"window {w}: {total} of {ours.numel()} quantized latent values differ from Python's, in frames [{string.Join(", ", Enumerable.Range(0, flags.Length).Where(i => flags[i]))}]");
            for (int i = 0; i < flags.Length; i++) if (flags[i]) flipFrames.Add((w, i, 1));
        }
        var err = (wav - expectedWav).abs().contiguous();
        var flat = err.reshape(-1).data<float>().ToArray();
        long n = wav.shape[1];
        int big = flat.Count(e => e > 1e-5);
        _out.WriteLine($"samples with error > 1e-5: {big} of {flat.Length}; > 1e-4: {flat.Count(e => e > 1e-4)}");
        var spans = new List<string>(); long prev = -1000000, startSpan = 0;
        for (long i = 0; i < flat.Length; i++)
        {
            if (flat[i] <= 1e-5) continue;
            long t = i % n;
            if (t - prev > 4800) { if (prev >= 0) spans.Add($"[{startSpan / 48000.0:F2}..{prev / 48000.0:F2}s]"); startSpan = t; }
            prev = t;
        }
        spans.Add($"[{startSpan / 48000.0:F2}..{prev / 48000.0:F2}s]");
        _out.WriteLine($"spans with error > 1e-5: {string.Join(" ", spans)}; flipped frames start at sample 1920 x frame (+ {DetokenizePlan.HopSamples} for window 1)");

        Assert.Equal(new long[] { 2, 1_536_000 }, wav.shape);
        Assert.True(l0.CosSim >= LatentCos && l0.MaxAbs <= LatentMaxAbs && l0.SnrDb >= LatentSnr, $"window 0: {l0}");
        Assert.True(l1.CosSim >= LatentCos && l1.MaxAbs <= LatentMaxAbs && l1.SnrDb >= LatentSnr, $"window 1: {l1}");

        // The vocoder is discontinuous: on this clip one of 380 928 quantized latent values lands on the other side of a rounding boundary (window 0, frame 358)
        // and changes the audio locally (measured: cos_sim 0.9999999, max_abs 4.1e-3, 65.9 dB, all of it in 14.24 s to 15.07 s). So the whole-chain audio is checked
        // for what it can promise: few flips, global agreement, and error confined to the neighbourhood of a flipped frame (which was measured to last 0.83 s).
        Assert.True(flipFrames.Count <= 3, $"{flipFrames.Count} flipped frames");           // measured: 1 on the CPU and 1 on each GPU
        Assert.True(r.CosSim >= 0.99999 && r.SnrDb >= 50, $"wav: {r}");
        var stray = Enumerable.Range(0, flat.Length).Where(i => flat[i] > 1e-5 && !flipFrames.Any(ff =>
        {
            long flipSample = (ff.Window == 0 ? 0 : DetokenizePlan.HopSamples) + ff.Frame * 1920L;
            long t = i % n;
            return t >= flipSample - 4800 && t <= flipSample + 72_000;                    // 0.1 s before to 1.5 s after (measured: 0.83 s)
        })).ToList();
        Assert.True(stray.Count == 0, $"{stray.Count} samples with error above 1e-5 away from any flipped frame");
    }

    [RequiresCheckpoint]
    public void A_single_window_decode_equals_DecodeSegment_trimmed()
    {
        var f = Fixtures.Load("codec_latents.safetensors");
        var dev = _fx.Device;
        var codes = f["codes_seg0"].to(ScalarType.Int64).to(dev)[.., .., ..126];           // the 126 reference frames (the segment is those repeated)
        var noise = f["randn_1_inference_codes"].to(dev);
        using var _ = no_grad();
        using var viaDecode = HeartCodecDecoder.Decode(_fx.Model!, _fx.Vocoder!, codes, noise: _ => new(null, noise));
        var previousTf32 = backends.cudnn.allow_tf32;
        backends.cudnn.allow_tf32 = false;                                           // DecodeSegment (C7) leaves the library default; Decode turns TF32 off
        using var segment = HeartCodecDecoder.DecodeSegment(_fx.Model!, _fx.Vocoder!, f["codes_seg0"].to(ScalarType.Int64).to(dev), noise: noise);
        backends.cudnn.allow_tf32 = previousTf32;
        var r = AudioParity.Compare(viaDecode, segment[.., ..483_840].cpu());
        _out.WriteLine($"Decode vs DecodeSegment (126 frames): {r}");
        Assert.Equal(new long[] { 2, 483_840 }, viaDecode.shape);
        Assert.True(r.MaxAbs <= 1e-7, $"{r}");                                             // same computation: bit-level equality expected
    }

    /// <summary>The reference's own vocoder inputs of both windows, assembled by our cross-fade and trim, give its audio (quantization cannot flip: the inputs are identical).</summary>
    [RequiresCheckpoint]
    public void Reference_window_inputs_through_the_vocoder_and_the_assembler_give_the_python_wav()
    {
        var f = Fixtures.Load("codec_long_latents.safetensors");
        var expectedWav = Fixtures.Load("codec_long_wav.safetensors")["wav"];
        using var _ = no_grad();
        var plan = DetokenizePlan.For(400);
        using var assembler = plan.CreateAssembler();
        var previousTf32 = backends.cudnn.allow_tf32;
        backends.cudnn.allow_tf32 = false;                                           // what the decode does by default; with TF32 the vocoder alone is at 66 dB on a GPU
        try
        {
            for (int w = 0; w < 2; w++)
            {
                using var y = _fx.Vocoder!.Decode(f[$"decode_in_seg{w}"].to(_fx.Device));
                assembler.Add(y);
            }
        }
        finally { backends.cudnn.allow_tf32 = previousTf32; }
        using var wav = assembler.Finish();
        var r = AudioParity.Compare(wav, expectedWav);
        var overlap = AudioParity.Compare(wav[.., DetokenizePlan.HopSamples..(DetokenizePlan.HopSamples + DetokenizePlan.OverlapSamples)],
                                          expectedWav[.., DetokenizePlan.HopSamples..(DetokenizePlan.HopSamples + DetokenizePlan.OverlapSamples)]);
        _out.WriteLine($"{_fx.Device}: reference inputs -> vocoder -> assembler: {r}");
        _out.WriteLine($"cross-fade region: {overlap}");
        Assert.Equal(new long[] { 2, 1_536_000 }, wav.shape);
        Assert.True(r.CosSim >= 0.999999 && r.MaxAbs <= 4e-6 && r.SnrDb >= 100, $"{r}");
    }

    /// <summary>On CUDA with cuDNN TF32 allowed (heartlib's default): how much the in-context chain degrades. Does nothing on CPU.</summary>
    [RequiresCheckpoint]
    public void Gpu_with_tf32_allowed_is_measured()
    {
        if (_fx.Device.type != DeviceType.CUDA) return;
        var f = Fixtures.Load("codec_long_latents.safetensors");
        var expectedWav = Fixtures.Load("codec_long_wav.safetensors")["wav"];
        var dev = _fx.Device;
        var codes = f["frames"].to(ScalarType.Int64).unsqueeze(0).to(dev);
        var latents = new List<Tensor>();
        Tensor Flow(Tensor seg, long ctx, Tensor noise, Tensor? trueLatents)
        {
            var r = _fx.Model!.InferenceCodes(seg, 744, ctx, 1.25, 10, noise, trueLatents, otherSeg: true);
            latents.Add(r.cpu().DetachFromDisposeScope());
            return r;
        }
        HeartCodecDecoder.WindowNoise Noise(int w) => w == 0
            ? new(null, f["randn_1_inference_codes"].to(dev))
            : new(f["randn_2_detokenize"].to(dev), f["randn_3_inference_codes"].to(dev));
        var sw = Stopwatch.StartNew();
        using var wav = HeartCodecDecoder.DecodeWith(Flow, l => _fx.Vocoder!.Decode(FlowMatching.ReorderForVocoder(l)), codes, 744, 256, Noise, allowCudnnTf32: true);
        cuda.synchronize();
        var l0 = Compare(latents[0], f["latents_seg0"]);
        var l1 = Compare(latents[1], f["latents_seg1"]);
        var r = AudioParity.Compare(wav, expectedWav);
        _out.WriteLine($"{dev} TF32 allowed: {sw.Elapsed.TotalSeconds:F1} s");
        _out.WriteLine($"  latents window 0 {l0}");
        _out.WriteLine($"  latents window 1 {l1}");
        _out.WriteLine($"  wav {r}");
        Assert.True(l0.CosSim >= 0.99999 && l0.SnrDb >= 60 && l1.CosSim >= 0.99999 && l1.SnrDb >= 60 && r.CosSim >= 0.999 && r.SnrDb >= 30, $"{l0} | {l1} | {r}");
    }
}
