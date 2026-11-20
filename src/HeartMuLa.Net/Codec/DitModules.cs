using MusicGen.Core.Layers;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace HeartMuLa.Codec;

/// <summary>
/// <c>ProjectLayer</c>: a convolution over time (<c>ffn_1</c>, kernel <c>k</c>, padding <c>k/2</c>) applied to the transposed input, scaled by
/// <c>k^−0.5</c>, followed by a <c>Linear</c> (<c>ffn_2</c>). Input and output are <c>[B, T, C]</c>.
/// </summary>
public sealed class FlowProjectLayer : Module<Tensor, Tensor>
{
    private readonly Conv1d ffn_1;
    private readonly Linear ffn_2;
    private readonly int kernel;

    /// <summary>Builds the layer: a convolution with the given <paramref name="kernel"/> from <paramref name="inChannels"/> to <paramref name="outChannels"/>, then a Linear of <paramref name="outChannels"/>.</summary>
    public FlowProjectLayer(int inChannels, int outChannels, int kernel) : base(nameof(FlowProjectLayer))
    {
        this.kernel = kernel;
        ffn_1 = Conv1d(inChannels, outChannels, kernel, padding: kernel / 2);
        ffn_2 = Linear(outChannels, outChannels);
        RegisterComponents();
    }

    /// <summary>Applies the convolution, the <c>k^−0.5</c> scaling and the Linear to <c>[B, T, C]</c>.</summary>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var h = ffn_1.forward(x.transpose(1, 2)).transpose(1, 2) * Math.Pow(kernel, -0.5);
        return ffn_2.forward(h).MoveToOuterDisposeScope();
    }
}

/// <summary><c>Linear → SiLU → Linear</c>.</summary>
public sealed class FlowTimestepEmbedding : Module<Tensor, Tensor>
{
    private readonly Linear linear_1;
    private readonly Linear linear_2;

    /// <summary>Builds the two Linear layers: <paramref name="inChannels"/> → <paramref name="dim"/> → <paramref name="dim"/>.</summary>
    public FlowTimestepEmbedding(int inChannels, int dim) : base(nameof(FlowTimestepEmbedding))
    {
        linear_1 = Linear(inChannels, dim);
        linear_2 = Linear(dim, dim);
        RegisterComponents();
    }

    /// <summary>Maps features <c>[B, inChannels]</c> to <c>[B, dim]</c>.</summary>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        return linear_2.forward(F.silu(linear_1.forward(x))).MoveToOuterDisposeScope();
    }
}

/// <summary>Sinusoidal embedding of the flow time <c>t</c> (512 features: cos then sin of <c>t·1000·exp(−ln 10000·i/256)</c>) and its MLP.</summary>
public sealed class FlowTimestepEmbedder : Module<Tensor, Tensor>
{
    /// <summary>Number of sinusoidal features of a timestep.</summary>
    public const int FlowTSize = 512;
    private readonly FlowTimestepEmbedding timestep_embedder;

    /// <summary>Builds the embedder whose MLP outputs <paramref name="dim"/> features.</summary>
    public FlowTimestepEmbedder(int dim) : base(nameof(FlowTimestepEmbedder))
    {
        timestep_embedder = new FlowTimestepEmbedding(FlowTSize, dim);
        RegisterComponents();
    }

    /// <summary>The 512 sinusoidal features <c>[B, 512]</c> of timesteps <c>[B]</c>.</summary>
    public static Tensor Features(Tensor timesteps, double maxPeriod = 10000, double scale = 1000)
    {
        using var scope = NewDisposeScope();
        int half = FlowTSize / 2;
        var freqs = exp(-Math.Log(maxPeriod) * arange(0, half, dtype: timesteps.dtype, device: timesteps.device) / half);
        var args = timesteps.unsqueeze(1) * freqs.unsqueeze(0) * scale;
        return cat(new[] { cos(args), sin(args) }, dim: -1).MoveToOuterDisposeScope();
    }

    /// <summary>Embeds the timesteps <c>[B]</c> into <c>[B, dim]</c>.</summary>
    public override Tensor forward(Tensor timesteps)
    {
        using var scope = NewDisposeScope();
        return timestep_embedder.forward(Features(timesteps)).MoveToOuterDisposeScope();
    }
}

/// <summary>AdaLN-single: embeds the timestep and projects <c>SiLU(embedding)</c> to the <c>6·dim</c> modulation shared by a stage's blocks.</summary>
public sealed class FlowAdaLayerNormSingle : Module
{
    private readonly FlowTimestepEmbedder emb;
    private readonly Linear linear;

    /// <summary>Builds the module for hidden size <paramref name="dim"/>.</summary>
    public FlowAdaLayerNormSingle(int dim) : base(nameof(FlowAdaLayerNormSingle))
    {
        emb = new FlowTimestepEmbedder(dim);
        linear = Linear(dim, 6 * dim);
        RegisterComponents();
    }

    /// <returns>(modulation <c>[B, 6·dim]</c>, embedded timestep <c>[B, dim]</c>)</returns>
    public (Tensor modulation, Tensor embedded) Forward(Tensor timesteps)
    {
        var embedded = emb.forward(timesteps);
        return (linear.forward(F.silu(embedded)), embedded);
    }
}

/// <summary>Modulation arithmetic shared by the blocks and the stage outputs.</summary>
public static class FlowModulation
{
    /// <summary>
    /// <c>norm(s) · (1 + scale) + shift</c> with <c>shift, scale = (table[None] + embedded[:, None]).chunk(2, dim = 1)</c> (shift first) and a LayerNorm without
    /// affine parameters, eps 1e-6. <paramref name="s"/> is <c>[B, T, D]</c>, <paramref name="table"/> <c>[2, D]</c>, <paramref name="embedded"/> <c>[B, D]</c>.
    /// </summary>
    public static Tensor OutputNorm(Tensor s, Tensor table, Tensor embedded)
    {
        using var scope = NewDisposeScope();
        var chunks = (table.unsqueeze(0) + embedded.unsqueeze(1)).chunk(2, dim: 1);
        var shift = chunks[0];
        var scale = chunks[1];
        var normed = F.layer_norm(s, new long[] { s.shape[^1] }, null, null, 1e-6);
        return (normed * (1 + scale) + shift).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// DiT block with AdaLN-single: <c>x += gate_msa · attn(RMSNorm(x)·(1 + scale_msa) + shift_msa)</c> then
/// <c>x += gate_mlp · mlp(RMSNorm(x)·(1 + scale_mlp) + shift_mlp)</c>. The six vectors are
/// <c>shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp</c>, in that order, from <c>scale_shift_table [6, dim] + modulation [B, 6, dim]</c>.
/// </summary>
public sealed class FlowTransformerBlock : Module
{
    private readonly FlowRmsNorm attn_norm;
    private readonly FlowAttention attn;
    private readonly FlowRmsNorm mlp_norm;
    private readonly FlowMlp mlp;
    private readonly Parameter scale_shift_table;

    /// <summary>Builds a block of width <paramref name="dim"/> with <paramref name="heads"/> attention heads of size <paramref name="headDim"/>.</summary>
    public FlowTransformerBlock(int dim, int heads, int headDim) : base(nameof(FlowTransformerBlock))
    {
        attn_norm = new FlowRmsNorm(dim);
        attn = new FlowAttention(dim, heads, headDim, HeartCodecRope.Create(headDim));
        mlp_norm = new FlowRmsNorm(dim);
        mlp = new FlowMlp(dim);
        scale_shift_table = new Parameter(zeros(6, dim), requires_grad: false);
        RegisterComponents();
    }

    /// <param name="x"><c>[B, T, dim]</c></param>
    /// <param name="modulation"><c>[B, 6·dim]</c> from <see cref="FlowAdaLayerNormSingle"/></param>
    public Tensor Forward(Tensor x, Tensor modulation)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0];
        var c = (scale_shift_table.unsqueeze(0) + modulation.reshape(b, 6, -1)).chunk(6, dim: 1);   // six [B, 1, dim]
        var shiftMsa = c[0]; var scaleMsa = c[1]; var gateMsa = c[2];
        var shiftMlp = c[3]; var scaleMlp = c[4]; var gateMlp = c[5];

        var h = attn_norm.forward(x) * (1 + scaleMsa) + shiftMsa;
        x = x + gateMsa * attn.forward(h);
        h = mlp_norm.forward(x) * (1 + scaleMlp) + shiftMlp;
        x = x + gateMlp * mlp.forward(h);
        return x.MoveToOuterDisposeScope();
    }
}
