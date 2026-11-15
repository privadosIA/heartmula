using MusicGen.Core.Audio;
using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>The cross-faded assembly against a naive double-precision loop, with the real window and overlap sizes.</summary>
public class WindowAssemblerTests
{
    private const int W = 1_428_480, Ovlp = 199_680, Hop = 1_228_800;

    private static float Sample(int window, int channel, int t) => (float)Math.Sin(0.0007 * t + window * 1.3 + channel * 0.5);

    private static Tensor MakeWindow(int window)
    {
        var data = new float[2 * W];
        for (int c = 0; c < 2; c++)
            for (int t = 0; t < W; t++) data[c * W + t] = Sample(window, c, t);
        return tensor(data, new long[] { 2, W });
    }

    [Fact]
    public void Ramp_is_numpy_linspace()
    {
        var r = WindowAssembler.Ramp(5);
        Assert.Equal(new[] { 0.0, 0.25, 0.5, 0.75, 1.0 }, r);
        var big = WindowAssembler.Ramp(Ovlp);
        Assert.Equal(0.0, big[0]);
        Assert.Equal(1.0, big[^1]);
        Assert.Equal(0.5, big[Ovlp / 2], 4);
        Assert.True(Enumerable.Range(1, Ovlp - 1).All(i => big[i] > big[i - 1]));
    }

    [Fact]
    public void Three_windows_are_joined_with_a_linear_cross_fade_and_trimmed()
    {
        var plan = DetokenizePlan.For(693);                                   // 3 windows, 2 661 120 target samples
        Assert.Equal(3, plan.Windows);
        using var assembler = plan.CreateAssembler();
        for (int k = 0; k < 3; k++) { using var w = MakeWindow(k); assembler.Add(w); }
        using var result = assembler.Finish();

        Assert.Equal(new long[] { 2, plan.TargetSamples }, result.shape);
        var got = result.data<float>().ToArray();
        var ramp = WindowAssembler.Ramp(Ovlp);
        double worst = 0;
        long len = result.shape[1];
        for (int c = 0; c < 2; c++)
            for (long t = 0; t < len; t++)
            {
                int k = (int)Math.Min(2, t / Hop);                            // window whose own samples cover t (the last one takes the tail)
                // Naive expectation: window k starts at k·Hop; the first Ovlp samples of window k > 0 mix the previous window's tail (fade out) and this one (fade in).
                double expected;
                long local = t - (long)k * Hop;
                if (k > 0 && local < Ovlp)
                    expected = Sample(k - 1, c, (int)(local + Hop)) * (1 - ramp[local]) + (double)Sample(k, c, (int)local) * ramp[local];
                else expected = Sample(k, c, (int)local);
                worst = Math.Max(worst, Math.Abs(got[c * len + t] - expected));
            }
        Assert.True(worst < 1e-6, $"max difference {worst:E2}");
    }

    [Fact]
    public void Samples_outside_the_overlaps_are_bit_exact_and_the_overlap_edges_are_the_pure_signals()
    {
        var plan = DetokenizePlan.For(400);                                   // 2 windows
        using var assembler = plan.CreateAssembler();
        using (var w = MakeWindow(0)) assembler.Add(w);
        using (var w = MakeWindow(1)) assembler.Add(w);
        using var result = assembler.Finish();
        var got = result.data<float>().ToArray();
        long len = result.shape[1];

        Assert.Equal(Sample(0, 1, 12345), got[len + 12345]);                  // untouched head of the first window
        Assert.Equal(Sample(1, 0, Ovlp + 777), got[Hop + Ovlp + 777]);        // untouched part of the second window, after the overlap
        Assert.Equal(Sample(0, 0, Hop), got[Hop]);                            // ramp 0: only the old window
        double lastMixed = Sample(1, 0, Ovlp - 1);                            // ramp 1: only the new window
        Assert.Equal(lastMixed, got[Hop + Ovlp - 1], 6);
    }

    [Fact]
    public void Output_length_is_the_sum_of_hops_before_trimming()
    {
        var plan = DetokenizePlan.For(693);
        using var assembler = plan.CreateAssembler();
        for (int k = 0; k < 3; k++) { using var w = MakeWindow(k); assembler.Add(w); }
        using var result = assembler.Finish();
        long untrimmed = W + 2L * Hop;
        Assert.True(plan.TargetSamples <= untrimmed);
        Assert.Equal(plan.TargetSamples, result.shape[1]);
    }

    [Fact]
    public void Misuse_is_rejected()
    {
        var plan = DetokenizePlan.For(400);
        using var assembler = plan.CreateAssembler();
        using var wrong = zeros(new long[] { 2, 100 });
        Assert.Throws<ArgumentException>(() => assembler.Add(wrong));
        Assert.Throws<InvalidOperationException>(() => assembler.Finish());   // no window added yet
        using (var w = MakeWindow(0)) assembler.Add(w);
        using (var w = MakeWindow(1)) assembler.Add(w);
        using var extra = zeros(new long[] { 2, W });
        Assert.Throws<InvalidOperationException>(() => assembler.Add(extra));
    }
}

public class DetokenizePlanAssemblerTests
{
    [Fact]
    public void EveryPlan_FitsTheAssemblerItBuilds()
    {
        // The assembler validates target <= window + (windows - 1) * hop; the plans of every length up to 4 minutes must satisfy it.
        for (int frames = 1; frames <= 3000; frames++)
        {
            var plan = DetokenizePlan.For(frames);
            long total = DetokenizePlan.WindowSamples + (long)(plan.Windows - 1) * DetokenizePlan.HopSamples;
            Assert.True(plan.TargetSamples <= total, $"frames {frames}: target {plan.TargetSamples} > total {total}");
        }
    }
}
