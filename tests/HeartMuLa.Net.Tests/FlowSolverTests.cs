using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>The Euler/CFG solver with stub estimators whose output is known in closed form.</summary>
public class FlowSolverTests
{
    private const int C = 5, MuDim = 6;

    /// <summary>out = 0.1·x + 0.3·mu[..., :C] + 0.2·incontext; records every call.</summary>
    private sealed class Stub : IFlowEstimator
    {
        public readonly List<double[]> Timesteps = new();
        public readonly List<Tensor> Inputs = new();

        public Tensor Forward(Tensor input, Tensor timestep)
        {
            Timesteps.Add(DitTestKit.Doubles(timestep));
            Inputs.Add(input.clone().DetachFromDisposeScope());
            var x = input[.., .., ..C];
            var incontext = input[.., .., C..(2 * C)];
            var mu = input[.., .., (2 * C)..(2 * C + C)];
            return 0.1 * x + 0.3 * mu + 0.2 * incontext;
        }
    }

    private static Tensor Rand(int seed, params long[] shape) => randn(shape, generator: new Generator((ulong)seed), dtype: ScalarType.Float32);

    [Fact]
    public void Matches_a_naive_euler_loop_with_guidance_when_there_is_no_in_context()
    {
        const int B = 1, T = 4, Steps = 10;
        const double g = 1.25;
        var stub = new Stub();
        using var noise = Rand(1, B, T, C);
        using var mu = Rand(2, B, T, MuDim);
        using var zeros3 = zeros(B, T, C);
        using var result = FlowMatching.Solve(stub, noise, zeros3, 0, mu, g, Steps);

        var x = DitTestKit.Doubles(noise);
        var m = DitTestKit.Doubles(mu);
        double dt = 1.0 / Steps;
        for (int s = 0; s < Steps; s++)
        {
            var next = new double[x.Length];
            for (int f = 0; f < T; f++)
                for (int c = 0; c < C; c++)
                {
                    int i = f * C + c;
                    double uncond = 0.1 * x[i];
                    double cond = 0.1 * x[i] + 0.3 * m[f * MuDim + c];
                    next[i] = x[i] + dt * (uncond + g * (cond - uncond));
                }
            x = next;
        }
        Assert.True(DitTestKit.MaxAbsDiff(DitTestKit.Doubles(result), x) < 1e-5);
    }

    [Fact]
    public void Each_step_runs_a_batch_of_two_with_the_unconditional_half_first_and_the_times_of_the_linspace()
    {
        var stub = new Stub();
        using var noise = Rand(3, 1, 3, C);
        using var mu = Rand(4, 1, 3, MuDim) + 1.0;
        using var zeros3 = zeros(1, 3, C);
        using var _ = FlowMatching.Solve(stub, noise, zeros3, 0, mu, 2.0, 10);

        Assert.Equal(10, stub.Timesteps.Count);
        for (int s = 0; s < 10; s++)
        {
            Assert.Equal(2, stub.Timesteps[s].Length);
            Assert.Equal(stub.Timesteps[s][0], stub.Timesteps[s][1]);
            Assert.Equal(s / 10.0, stub.Timesteps[s][0], 5);
            var input = stub.Inputs[s];
            Assert.Equal(new long[] { 2, 3, 2 * C + MuDim }, input.shape);
            Assert.Equal(0.0, input[0, .., (2 * C)..].abs().max().item<float>());       // unconditional: mu zeroed
            Assert.True(DitTestKit.MaxAbsDiff(DitTestKit.Doubles(input[1, .., (2 * C)..]), DitTestKit.Doubles(mu[0])) < 1e-7);
        }
    }

    [Fact]
    public void Same_noise_gives_identical_results_and_different_noise_does_not()
    {
        using var mu = Rand(5, 1, 3, MuDim);
        using var zeros3 = zeros(1, 3, C);
        using var n1 = Rand(6, 1, 3, C);
        using var n2 = Rand(7, 1, 3, C);
        using var a = FlowMatching.Solve(new Stub(), n1, zeros3, 0, mu, 1.25, 10);
        using var b = FlowMatching.Solve(new Stub(), n1, zeros3, 0, mu, 1.25, 10);
        using var c = FlowMatching.Solve(new Stub(), n2, zeros3, 0, mu, 1.25, 10);
        Assert.Equal(0.0, (a - b).abs().max().item<float>());
        Assert.True((a - c).abs().max().item<float>() > 1e-3);
    }

    [Fact]
    public void Without_in_context_the_first_step_sees_the_noise_untouched()
    {
        var stub = new Stub();
        using var noise = Rand(8, 1, 3, C);
        using var mu = Rand(9, 1, 3, MuDim);
        using var incontext = Rand(10, 1, 3, C);
        using var _ = FlowMatching.Solve(stub, noise, incontext, 0, mu, 1.25, 10);
        Assert.True(DitTestKit.MaxAbsDiff(DitTestKit.Doubles(stub.Inputs[0][0, .., ..C]), DitTestKit.Doubles(noise[0])) == 0);
    }

    [Fact]
    public void In_context_frames_are_blended_at_every_step_and_equal_the_in_context_latents_in_the_result()
    {
        const int K = 2;
        var stub = new Stub();
        using var noise = Rand(11, 1, 4, C);
        using var mu = Rand(12, 1, 4, MuDim);
        using var incontext = Rand(13, 1, 4, C);
        using var result = FlowMatching.Solve(stub, noise, incontext, K, mu, 1.25, 10);

        for (int s = 0; s < 10; s++)
        {
            double t = s / 10.0;
            var seen = DitTestKit.Doubles(stub.Inputs[s][0, ..K, ..C]);
            var n = DitTestKit.Doubles(noise[0, ..K]);
            var ic = DitTestKit.Doubles(incontext[0, ..K]);
            var expected = n.Select((v, i) => (1 - (1 - 1e-6) * t) * v + t * ic[i]).ToArray();
            Assert.True(DitTestKit.MaxAbsDiff(seen, expected) < 1e-5, $"step {s}");
        }
        Assert.Equal(0.0, (result[.., ..K] - incontext[.., ..K]).abs().max().item<float>());
    }

    [Fact]
    public void InferenceCodes_keeps_the_in_context_latents_and_masks_the_rest_of_them()
    {
        var estimator = new FlowEstimatorConfig(Heads: 2, HeadDim: 4, InChannels: 12, OutChannels: 5, Layers: 1, Layers2: 1);
        // InChannels = 2 * OutChannels + Dim so the concatenation fits.
        var config = new FlowMatchingConfig(Dim: 2, CodebookSize: 10, CodebookDim: 3, NumQuantizers: 2, Estimator: estimator);
        using var model = new FlowMatching(config);
        DitTestKit.Fill(model, 21, 0.1);
        using var codes = randint(0, 10, new long[] { 1, 2, 4 }, dtype: ScalarType.Int64, generator: new Generator(22));
        using var noise = Rand(23, 1, 8, 5);
        using var trueLatents = Rand(24, 1, 8, 5);
        using var result = model.InferenceCodes(codes, latentLength: 8, incontextLength: 3, guidanceScale: 1.25, numSteps: 2,
            noise: noise, trueLatents: trueLatents, otherSeg: true);

        Assert.Equal(new long[] { 1, 8, 5 }, result.shape);
        Assert.Equal(0.0, (result[.., ..3] - trueLatents[.., ..3]).abs().max().item<float>());
        Assert.True((result[.., 3..] - trueLatents[.., 3..]).abs().max().item<float>() > 1e-4);
    }
}
