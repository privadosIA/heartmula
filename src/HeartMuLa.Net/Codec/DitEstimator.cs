using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace HeartMuLa.Codec;

/// <summary>The estimator seen by the solver: features and flow time in, velocity out. Lets tests inject stubs.</summary>
public interface IFlowEstimator
{
    /// <param name="input"><c>[B, T, in_channels]</c></param>
    /// <param name="timestep"><c>[B]</c> flow time</param>
    /// <returns><c>[B, T, out_channels]</c></returns>
    Tensor Forward(Tensor input, Tensor timestep);
}

/// <summary>
/// Configuration of the estimator (<c>config.json</c> of HeartCodec-oss): stage 1 has width <c>Heads · HeadDim</c> (1536 = 24 × 64) and stage 2 has
/// twice that (3072 = 24 × 128: same number of heads, twice the head dimension).
/// </summary>
public sealed record FlowEstimatorConfig(int Heads = 24, int HeadDim = 64, int InChannels = 1024, int OutChannels = 256, int Layers = 24, int Layers2 = 6)
{
    /// <summary>Width of stage 1: <c>Heads · HeadDim</c>.</summary>
    public int Inner => Heads * HeadDim;
    /// <summary>Width of stage 2: twice <see cref="Inner"/>.</summary>
    public int Inner2 => Inner * 2;
    /// <summary>The HeartCodec-oss configuration.</summary>
    public static FlowEstimatorConfig Default { get; } = new();
}

/// <summary>
/// <c>LlamaTransformer</c> of heartlib's flow matching with AdaLN-single: <c>proj_in</c>, a first stage of <c>Layers</c> blocks, an output normalization
/// modulated by the embedded timestep, concatenation with the input, <c>connection_proj</c>, a second stage of <c>Layers2</c> wider blocks, its own output
/// normalization and <c>proj_out</c>. Parameter names are the checkpoint keys under <c>flow_matching.estimator.</c>.
/// </summary>
public sealed class LlamaTransformer : Module, IFlowEstimator
{
    private readonly FlowProjectLayer proj_in;
    private readonly ModuleList<FlowTransformerBlock> transformer_blocks;
    private readonly ModuleList<FlowTransformerBlock> transformer_blocks_2;
    private readonly FlowProjectLayer connection_proj;
    private readonly FlowProjectLayer proj_out;
    private readonly Parameter scale_shift_table;
    private readonly Parameter scale_shift_table_2;
    private readonly FlowAdaLayerNormSingle adaln_single;
    private readonly FlowAdaLayerNormSingle adaln_single_2;

    /// <summary>Configuration the estimator was built with.</summary>
    public FlowEstimatorConfig Config { get; }

    /// <summary>Builds the estimator from <paramref name="config"/> (the HeartCodec-oss default when null).</summary>
    public LlamaTransformer(FlowEstimatorConfig? config = null) : base(nameof(LlamaTransformer))
    {
        Config = config ?? FlowEstimatorConfig.Default;
        var c = Config;
        proj_in = new FlowProjectLayer(c.InChannels, c.Inner, 3);
        transformer_blocks = new ModuleList<FlowTransformerBlock>();
        for (int i = 0; i < c.Layers; i++) transformer_blocks.Add(new FlowTransformerBlock(c.Inner, c.Heads, c.HeadDim));
        transformer_blocks_2 = new ModuleList<FlowTransformerBlock>();
        for (int i = 0; i < c.Layers2; i++) transformer_blocks_2.Add(new FlowTransformerBlock(c.Inner2, c.Heads, c.HeadDim * 2));
        connection_proj = new FlowProjectLayer(c.InChannels + c.Inner, c.Inner2, 3);
        proj_out = new FlowProjectLayer(c.Inner2, c.OutChannels, 3);
        scale_shift_table = new Parameter(zeros(2, c.Inner), requires_grad: false);
        scale_shift_table_2 = new Parameter(zeros(2, c.Inner2), requires_grad: false);
        adaln_single = new FlowAdaLayerNormSingle(c.Inner);
        adaln_single_2 = new FlowAdaLayerNormSingle(c.Inner2);
        RegisterComponents();
    }

    /// <summary>Predicts the velocity <c>[B, T, out_channels]</c> from the features <c>[B, T, in_channels]</c> and the flow time <c>[B]</c>.</summary>
    public Tensor Forward(Tensor hiddenStates, Tensor timestep)
    {
        using var scope = NewDisposeScope();
        var s = proj_in.forward(hiddenStates);
        var (modulation, embedded) = adaln_single.Forward(timestep);
        foreach (var block in transformer_blocks)
        {
            var next = block.Forward(s, modulation);
            s.Dispose();
            s = next;
        }
        s = FlowModulation.OutputNorm(s, scale_shift_table, embedded);

        var x = connection_proj.forward(cat(new[] { hiddenStates, s }, dim: -1));
        var (modulation2, embedded2) = adaln_single_2.Forward(timestep);
        foreach (var block in transformer_blocks_2)
        {
            var next = block.Forward(x, modulation2);
            x.Dispose();
            x = next;
        }
        x = FlowModulation.OutputNorm(x, scale_shift_table_2, embedded2);
        return proj_out.forward(x).MoveToOuterDisposeScope();
    }
}
