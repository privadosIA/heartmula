using MusicGen.Core.Testing;
using HeartMuLa.Model;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// C# sampling (CFG + top-k + temperature + exponential noise) must pick the same tokens as heartlib when it receives
/// the same logits and the same seed. The Python logits come from frames_logits.safetensors and decoder_logits.safetensors and
/// the expected tokens from frames_py.safetensors (seed 1234, cfg 1.5, topk 50, temperature 1.0). Needs no checkpoint.
///
/// A frame consumes the global RNG 8 times (codebook 0 and codebooks 1..7), each with an `exponential_` over a bf16 [1, V].
/// Where there are no Python logits, a filler draw with that same shape and dtype is made just to advance the RNG.
/// </summary>
public class SamplingParityTests
{
    private const int Seed = 1234;
    private const double Cfg = 1.5;
    private const int TopK = 50;
    private const double Temperature = 1.0;
    private const int Codebooks = 8;

    private readonly ITestOutputHelper _out;
    public SamplingParityTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Codebook0_AllFramesWithPythonLogits_PickPythonTokens()
    {
        var logits = Fixtures.Load("frames_logits.safetensors")["c0_logits"];   // [calls, 2, V] float32 (bf16 values)
        var frames = Fixtures.Load("frames_py.safetensors")["frames"];          // [8, T]
        long calls = logits.shape[0], vocab = logits.shape[2];
        Assert.Equal(126, calls);
        Assert.Equal(calls, frames.shape[1]);

        manual_seed(Seed);
        var mismatches = new List<string>();
        for (int t = 0; t < calls; t++)
        {
            using var picked = Sampling.SampleGuided(logits[t].to(ScalarType.BFloat16), Cfg, TopK, Temperature);   // [2, 1]
            long row0 = picked[0, 0].item<long>(), row1 = picked[1, 0].item<long>();
            long expected = frames[0, t].item<long>();
            if (row0 != row1) mismatches.Add($"frame {t}: the two CFG rows differ ({row0} vs {row1})");
            if (row0 != expected) mismatches.Add($"frame {t}: C# {row0}, Python {expected}");
            for (int k = 1; k < Codebooks; k++) Rng.Advance(vocab);               // draws of codebooks 1..7
        }

        _out.WriteLine($"codebook 0: {calls - mismatches.Count(m => m.Contains("Python"))} of {calls} tokens equal to frames_py");
        Assert.True(mismatches.Count == 0, "different tokens:\n  " + string.Join("\n  ", mismatches));
    }

    [Fact]
    public void Codebooks1To7_InDecoderFixtureFrames_PickPythonTokens()
    {
        var c0Logits = Fixtures.Load("frames_logits.safetensors")["c0_logits"];   // [calls, 2, V]
        var dec = Fixtures.Load("decoder_logits.safetensors");
        var frameIdx = dec["frame_idx"].data<long>().ToArray();
        var ciLogits = dec["ci_logits"];                                           // [n, 2, 7, V]
        var frames = Fixtures.Load("frames_py.safetensors")["frames"];
        long calls = c0Logits.shape[0], vocab = c0Logits.shape[2];
        Assert.Equal(new long[] { 0, 1, 2, 63, 124, 125 }, frameIdx);

        manual_seed(Seed);
        int compared = 0;
        var mismatches = new List<string>();
        for (int t = 0; t < calls; t++)
        {
            using (Sampling.SampleGuided(c0Logits[t].to(ScalarType.BFloat16), Cfg, TopK, Temperature)) { }   // codebook 0: same draw as in production
            int j = Array.IndexOf(frameIdx, (long)t);
            for (int k = 1; k < Codebooks; k++)
            {
                if (j < 0) { Rng.Advance(vocab); continue; }
                using var picked = Sampling.SampleGuided(ciLogits[j, .., k - 1, ..].to(ScalarType.BFloat16), Cfg, TopK, Temperature);
                long got = picked[0, 0].item<long>(), expected = frames[k, t].item<long>();
                compared++;
                if (got != expected) mismatches.Add($"frame {t} codebook {k}: C# {got}, Python {expected}");
                if (picked[1, 0].item<long>() != got) mismatches.Add($"frame {t} codebook {k}: the two rows differ");
            }
        }

        _out.WriteLine($"codebooks 1-7: {compared - mismatches.Count(m => m.Contains("Python"))} of {compared} tokens equal to frames_py");
        Assert.Equal(42, compared);
        Assert.True(mismatches.Count == 0, "different tokens:\n  " + string.Join("\n  ", mismatches));
    }

    [Fact]
    public void AnotherSeed_GivesADifferentCodebook0Sequence()
    {
        var logits = Fixtures.Load("frames_logits.safetensors")["c0_logits"];
        var frames = Fixtures.Load("frames_py.safetensors")["frames"];
        long calls = logits.shape[0], vocab = logits.shape[2];

        manual_seed(Seed + 1);
        int equal = 0;
        for (int t = 0; t < calls; t++)
        {
            using var picked = Sampling.SampleGuided(logits[t].to(ScalarType.BFloat16), Cfg, TopK, Temperature);
            if (picked[0, 0].item<long>() == frames[0, t].item<long>()) equal++;
            for (int k = 1; k < Codebooks; k++) Rng.Advance(vocab);
        }

        _out.WriteLine($"seed {Seed + 1}: {equal} of {calls} codebook 0 tokens match frames_py (seed {Seed}: 126 of 126)");
        Assert.True(equal < calls, "with another seed the sequence should be different");
        Assert.True(equal < calls / 2, $"too many tokens match with another seed ({equal}); the test does not distinguish seeds");
    }
}
