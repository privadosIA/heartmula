using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>The window planning against a literal transcription of the reference loops, written independently of <see cref="DetokenizePlan"/>.</summary>
public class DetokenizePlanTests
{
    /// <summary>Reference arithmetic as the Python does it, step by step (while loops and all).</summary>
    private static (int Padded, int[] Starts, long Target) Reference(int n)
    {
        int minSamples = (int)(29.76 * 12.5);
        int hop = minSamples / 93 * 80;
        int ovlp = minSamples - hop;
        long target = (long)(n / 12.5 * 48000);
        int len = n;
        if (len < minSamples) { while (len < minSamples) len *= 2; len = minSamples; }   // cat([codes, codes]) then cut
        if ((len - ovlp) % hop > 0)
        {
            int lenCodes = (int)Math.Ceiling((len - ovlp) / (double)hop) * hop + ovlp;
            while (len < lenCodes) len *= 2;                                              // repeated, then cut
            len = lenCodes;
        }
        var starts = new List<int>();
        for (int s = 0; s < len - hop + 1; s += hop) starts.Add(s);
        return (len, starts.ToArray(), target);
    }

    [Fact]
    public void Constants_are_the_reference_ones()
    {
        Assert.Equal(372, DetokenizePlan.WindowFrames);
        Assert.Equal(320, DetokenizePlan.HopFrames);
        Assert.Equal(52, DetokenizePlan.OverlapFrames);
        Assert.Equal(104, DetokenizePlan.OverlapLatentFrames);
        Assert.Equal(1_428_480, DetokenizePlan.WindowSamples);
        Assert.Equal(1_228_800, DetokenizePlan.HopSamples);
        Assert.Equal(199_680, DetokenizePlan.OverlapSamples);
    }

    // Values computed with the reference formulas in Python (math.ceil, //, int()).
    [Theory]
    [InlineData(1, 372, 1, 3840)]
    [InlineData(126, 372, 1, 483_840)]
    [InlineData(371, 372, 1, 1_424_640)]
    [InlineData(372, 372, 1, 1_428_480)]
    [InlineData(373, 692, 2, 1_432_320)]
    [InlineData(400, 692, 2, 1_536_000)]
    [InlineData(691, 692, 2, 2_653_440)]
    [InlineData(692, 692, 2, 2_657_280)]
    [InlineData(693, 1012, 3, 2_661_120)]
    [InlineData(1000, 1012, 3, 3_840_000)]
    public void Known_lengths(int n, int padded, int windows, long target)
    {
        var plan = DetokenizePlan.For(n);
        Assert.Equal(padded, plan.PaddedFrames);
        Assert.Equal(windows, plan.Windows);
        Assert.Equal(target, plan.TargetSamples);
        Assert.Equal(Enumerable.Range(0, windows).Select(w => w * 320), plan.Starts);
    }

    [Fact]
    public void Matches_the_literal_reference_loops_for_every_length_up_to_3000()
    {
        for (int n = 1; n <= 3000; n++)
        {
            var (padded, starts, target) = Reference(n);
            var plan = DetokenizePlan.For(n);
            Assert.True(plan.PaddedFrames == padded && plan.TargetSamples == target && plan.Starts.SequenceEqual(starts), $"N = {n}");
            // Every window fits inside the padded codes.
            Assert.All(plan.Starts, st => Assert.True(st + DetokenizePlan.WindowFrames <= plan.PaddedFrames, $"N = {n}, start {st}"));
        }
    }

    [Fact]
    public void Zero_frames_are_rejected() => Assert.Throws<ArgumentOutOfRangeException>(() => DetokenizePlan.For(0));

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(126)]
    [InlineData(373)]
    [InlineData(400)]
    public void Repeat_keeps_the_original_codes_first_and_repeats_them_in_order(int n)
    {
        var plan = DetokenizePlan.For(n);
        var values = Enumerable.Range(0, 2 * n).Select(i => (long)(1000 + i)).ToArray();     // [1, 2, n]: stream q holds 1000 + q·n + t
        using var codes = tensor(values, new long[] { 1, 2, n });
        using var repeated = plan.Repeat(codes);

        Assert.Equal(new long[] { 1, 2, plan.PaddedFrames }, repeated.shape);
        var got = repeated.data<long>().ToArray();
        for (int q = 0; q < 2; q++)
            for (int t = 0; t < plan.PaddedFrames; t++)
                Assert.Equal(values[q * n + t % n], got[q * plan.PaddedFrames + t]);
        for (int t = 0; t < n; t++) Assert.Equal(values[t], got[t]);                          // the first N frames are the originals
    }

    [Fact]
    public void Repeat_of_a_window_sized_input_returns_it_unchanged()
    {
        var plan = DetokenizePlan.For(372);
        using var codes = zeros(new long[] { 1, 8, 372 }, ScalarType.Int64);
        Assert.Same(codes, plan.Repeat(codes));
    }

    [Fact]
    public void Repeat_rejects_codes_of_another_length()
    {
        var plan = DetokenizePlan.For(400);
        using var codes = zeros(new long[] { 1, 8, 399 }, ScalarType.Int64);
        Assert.Throws<ArgumentException>(() => plan.Repeat(codes));
    }
}
