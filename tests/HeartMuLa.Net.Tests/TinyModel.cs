using HeartMuLa.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>Toy HeartMuLa model (no checkpoint) for logic tests.</summary>
public static class TinyModel
{
    public static readonly HeartMuLaConfig Config = new()
    {
        TextVocabSize = 20,
        AudioVocabSize = 10,
        AudioNumCodebooks = 3,
        MuqDim = 6,
    };
    public static readonly LlamaArgs Backbone = new(20, 2, 4, 2, 16, 16, 32, 1e-5, 10_000, 32);
    public static readonly LlamaArgs Decoder = new(20, 1, 2, 1, 8, 8, 16, 1e-5, 10_000, 32);

    /// <summary>
    /// Small model with all parameters filled deterministically: the constructor creates some
    /// with <c>torch.empty</c> (uninitialized memory, sometimes NaN, and NaN != NaN would break the comparisons).
    /// Local generator so as not to depend on global state under parallel execution.
    /// </summary>
    public static HeartMuLaModel Create(int seed)
    {
        var model = new HeartMuLaModel(Config, Backbone, Decoder);
        var g = new Generator((ulong)seed);
        using var _ = no_grad();
        foreach (var p in model.parameters())
            p.copy_(randn(p.shape, generator: g));
        return model;
    }
}
