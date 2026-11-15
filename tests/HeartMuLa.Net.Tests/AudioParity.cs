using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>fp32 audio parity metrics (codec-vocoder spec): cosine, maximum error and signal-to-error ratio.</summary>
public static class AudioParity
{
    public readonly record struct Result(double CosSim, double MaxAbs, double SnrDb, double RmsError, long WorstSample, int WorstChannel)
    {
        public override string ToString() =>
            $"cos_sim={CosSim:F7} max_abs={MaxAbs:E3} (channel {WorstChannel}, sample {WorstSample}, {WorstSample / 48000.0:F3} s) rms_err={RmsError:E3} SNR={SnrDb:F1} dB";
    }

    /// <summary>Compares two audios <c>[channels, samples]</c>, on CPU and in double precision.</summary>
    public static Result Compare(Tensor got, Tensor expected)
    {
        using var _ = no_grad();
        var g = got.detach().cpu().to(ScalarType.Float64);
        var e = expected.detach().cpu().to(ScalarType.Float64);
        if (!g.shape.AsSpan().SequenceEqual(e.shape)) throw new InvalidOperationException($"shapes differ: [{string.Join(",", g.shape)}] vs [{string.Join(",", e.shape)}]");
        var diff = g - e;
        var cos = ((g * e).sum() / (g.norm() * e.norm())).item<double>();
        var abs = diff.abs();
        var maxAbs = abs.max().item<double>();
        var flat = abs.reshape(-1).argmax().item<long>();
        long samples = g.shape[1];
        var rmsErr = diff.pow(2).mean().sqrt().item<double>();
        var rmsRef = e.pow(2).mean().sqrt().item<double>();
        return new Result(cos, maxAbs, 20 * Math.Log10(rmsRef / rmsErr), rmsErr, flat % samples, (int)(flat / samples));
    }
}
