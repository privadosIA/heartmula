using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace HeartMuLa.Codec;

/// <summary>Configuration of the flow-matching decoder (<c>config.json</c> of HeartCodec-oss).</summary>
public sealed record FlowMatchingConfig(int Dim = 512, int CodebookSize = 8192, int CodebookDim = 32, int NumQuantizers = 8, FlowEstimatorConfig? Estimator = null)
{
    public FlowEstimatorConfig EstimatorConfig => Estimator ?? FlowEstimatorConfig.Default;

    /// <summary>Frames per second of the latents: the codes run at 12.5 Hz and are upsampled ×2.</summary>
    public const int LatentFramesPerCodeFrame = 2;

    public static FlowMatchingConfig Default { get; } = new();
}

/// <summary>The codebook of one RVQ layer: only the <c>embed</c> table <c>[1, size, dim]</c> (the EMA statistics are training state and are not loaded).</summary>
public sealed class VqCodebook : Module
{
    private readonly Parameter embed;

    public VqCodebook(int size, int dim) : base(nameof(VqCodebook))
    {
        embed = new Parameter(zeros(1, size, dim), requires_grad: false);
        RegisterComponents();
    }

    /// <summary><c>codes [B, T]</c> → vectors <c>[B, T, dim]</c>.</summary>
    public Tensor Lookup(Tensor codes)
    {
        using var scope = NewDisposeScope();
        var table = embed[0];                                            // [size, dim]
        return table.index_select(0, codes.reshape(-1)).reshape(codes.shape[0], codes.shape[1], -1).MoveToOuterDisposeScope();
    }
}

/// <summary>One RVQ layer. Its codebook must be registered as <c>_codebook</c> (leading underscore, as in the checkpoint keys).</summary>
public sealed class VqLayer : Module
{
    private readonly VqCodebook codebook;

    public VqLayer(int size, int dim) : base(nameof(VqLayer))
    {
        codebook = new VqCodebook(size, dim);
        register_module("_codebook", codebook);                          // explicit: RegisterComponents would derive the name from the field
    }

    public Tensor Lookup(Tensor codes) => codebook.Lookup(codes);
}

/// <summary>
/// Residual VQ as used for decoding: <c>get_output_from_indices</c> sums the vector each layer selects for its code and applies <c>project_out</c>
/// (<c>codebook_dim → dim</c>). <c>project_in</c> and the EMA tensors are only needed for training and do not exist here.
/// </summary>
public sealed class ResidualVq : Module
{
    private readonly ModuleList<VqLayer> layers;
    private readonly Linear project_out;

    public int NumQuantizers { get; }

    public ResidualVq(int dim, int codebookSize, int codebookDim, int numQuantizers) : base(nameof(ResidualVq))
    {
        NumQuantizers = numQuantizers;
        layers = new ModuleList<VqLayer>();
        for (int i = 0; i < numQuantizers; i++) layers.Add(new VqLayer(codebookSize, codebookDim));
        project_out = Linear(codebookDim, dim);
        RegisterComponents();
    }

    /// <param name="codes"><c>[B, T, Q]</c> int64</param>
    /// <returns><c>[B, T, dim]</c></returns>
    public Tensor GetOutputFromIndices(Tensor codes)
    {
        using var scope = NewDisposeScope();
        if (codes.shape[2] != NumQuantizers) throw new ArgumentException($"expected {NumQuantizers} code streams, got {codes.shape[2]}");
        Tensor? sum = null;
        int q = 0;
        foreach (var layer in layers)
        {
            var v = layer.Lookup(codes[.., .., q++]);
            sum = sum is null ? v : sum + v;
        }
        return project_out.forward(sum!).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// HeartCodec's flow matching: RVQ code lookup and conditioning, and the DiT estimator. Parameter names are the checkpoint keys under
/// <c>flow_matching.</c> (339 of the 365 tensors; the other 26 are RVQ training state).
/// </summary>
public sealed class FlowMatching : Module
{
    private readonly ResidualVq vq_embed;
    private readonly Linear cond_feature_emb;
    private readonly Parameter zero_cond_embedding1;
    private readonly LlamaTransformer estimator;

    public FlowMatchingConfig Config { get; }
    public int LatentDim => Config.EstimatorConfig.OutChannels;
    public IFlowEstimator Estimator => estimator;

    public FlowMatching(FlowMatchingConfig? config = null) : base(nameof(FlowMatching))
    {
        Config = config ?? FlowMatchingConfig.Default;
        var c = Config;
        vq_embed = new ResidualVq(c.Dim, c.CodebookSize, c.CodebookDim, c.NumQuantizers);
        cond_feature_emb = Linear(c.Dim, c.Dim);
        zero_cond_embedding1 = new Parameter(zeros(c.Dim), requires_grad: false);
        estimator = new LlamaTransformer(c.EstimatorConfig);
        RegisterComponents();
    }

    /// <summary>
    /// Flow-matching mask: 0 outside the latent, 2 for the frames to generate and, for the <c>other_seg</c> scenario, 1 for the first
    /// <paramref name="incontextLength"/> frames (in-context latents taken from the previous segment).
    /// </summary>
    public static Tensor LatentMasks(long batch, long frames, long latentLength, long incontextLength, bool otherSeg, Device device)
    {
        var masks = zeros(new long[] { batch, frames }, dtype: ScalarType.Int64, device: device);
        masks[.., ..(int)latentLength] = full(new long[] { batch, latentLength }, 2L, dtype: ScalarType.Int64, device: device);
        if (otherSeg && incontextLength > 0)
            masks[.., ..(int)incontextLength] = ones(new long[] { batch, incontextLength }, dtype: ScalarType.Int64, device: device);
        return masks;
    }

    /// <summary>
    /// Conditioning <c>mu [B, 2T, dim]</c> of a segment: quantized codes → <c>cond_feature_emb</c> → nearest ×2 in time, with the frames whose mask is
    /// below 0.5 (outside the latent) replaced by <c>zero_cond_embedding1</c>.
    /// </summary>
    /// <param name="codes"><c>[B, Q, T]</c> int64</param>
    /// <param name="masks"><c>[B, 2T]</c> from <see cref="LatentMasks"/></param>
    public Tensor Condition(Tensor codes, Tensor masks)
    {
        using var scope = NewDisposeScope();
        var quantized = vq_embed.GetOutputFromIndices(codes.transpose(1, 2));              // [B, T, dim]
        quantized = cond_feature_emb.forward(quantized);
        quantized = functional.interpolate(quantized.permute(0, 2, 1), scale_factor: new double[] { FlowMatchingConfig.LatentFramesPerCodeFrame }, mode: InterpolationMode.Nearest)
            .permute(0, 2, 1);                                                             // [B, 2T, dim]
        var conditioned = (masks > 0.5).unsqueeze(-1);
        var unconditioned = (masks < 0.5).unsqueeze(-1);
        return (conditioned * quantized + unconditioned * zero_cond_embedding1.unsqueeze(0)).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Latents <c>[B, frames, 256]</c> of one segment (reference <c>inference_codes</c>): conditioning from the codes and an Euler solve with
    /// classifier-free guidance. With <paramref name="incontextLength"/> &gt; 0 the first frames are taken from <paramref name="trueLatents"/> instead of generated.
    /// </summary>
    /// <param name="codes"><c>[B, Q, T]</c> int64</param>
    /// <param name="noise"><c>[B, 2T, 256]</c>; <c>torch.randn</c> when null</param>
    /// <param name="trueLatents"><c>[B, 2T, 256]</c> in-context latents; zeros when null</param>
    public Tensor InferenceCodes(Tensor codes, long latentLength, long incontextLength, double guidanceScale, int numSteps,
        Tensor? noise = null, Tensor? trueLatents = null, bool otherSeg = false)
    {
        using var scope = NewDisposeScope();
        long batch = codes.shape[0];
        long frames = codes.shape[2] * FlowMatchingConfig.LatentFramesPerCodeFrame;
        var device = codes.device;
        var masks = LatentMasks(batch, frames, latentLength, incontextLength, otherSeg, device);
        var mu = Condition(codes, masks);
        var x = noise is null ? randn(new long[] { batch, frames, LatentDim }, device: device) : noise.to(mu.dtype, device);
        var incontextX = trueLatents is null ? zeros_like(x) : trueLatents.to(mu.dtype, device);
        incontextX = incontextX * ((masks > 0.5) * (masks < 1.5)).unsqueeze(-1);
        long incontext = (masks == 1).sum(1)[0].item<long>();
        return Solve(x, incontextX, incontext, mu, guidanceScale, numSteps).MoveToOuterDisposeScope();
    }

    /// <summary>Euler solver of the reference (<c>solve_euler</c>): <c>t_span = linspace(0, 1, numSteps + 1)</c>, batch-2 guidance (unconditional first).</summary>
    public Tensor Solve(Tensor x0, Tensor incontextX, long incontextLength, Tensor mu, double guidanceScale, int numSteps)
        => Solve(estimator, x0, incontextX, incontextLength, mu, guidanceScale, numSteps);

    public static Tensor Solve(IFlowEstimator estimator, Tensor x0, Tensor incontextX, long incontextLength, Tensor mu, double guidanceScale, int numSteps)
    {
        using var scope = NewDisposeScope();
        var tSpan = linspace(0, 1, numSteps + 1, dtype: x0.dtype, device: x0.device);
        var t = tSpan[0].clone();
        var dt = tSpan[1] - tSpan[0];
        var noise = x0.clone();
        var x = x0.clone();
        var zeroMu = zeros_like(mu);
        for (int step = 1; step <= numSteps; step++)
        {
            using var stepScope = NewDisposeScope();
            if (incontextLength > 0)
            {
                var blended = (1 - (1 - 1e-6) * t) * noise[.., ..(int)incontextLength] + t * incontextX[.., ..(int)incontextLength];
                x[.., ..(int)incontextLength] = blended;
            }
            var input = cat(new[] { cat(new[] { x, x }, 0), cat(new[] { incontextX, incontextX }, 0), cat(new[] { zeroMu, mu }, 0) }, 2);
            var timestep = t.reshape(1).repeat(2);
            var dphi = estimator.Forward(input, timestep);
            var halves = dphi.chunk(2, 0);
            var step_ = halves[0] + guidanceScale * (halves[1] - halves[0]);
            var next = x + dt * step_;
            x.copy_(next);
            t = t + dt;
            if (step < numSteps) dt = tSpan[step + 1] - t;
            t.MoveToOuterDisposeScope();
            dt.MoveToOuterDisposeScope();
        }
        if (incontextLength > 0) x[.., ..(int)incontextLength] = incontextX[.., ..(int)incontextLength];
        return x.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Latents <c>[B, T, 256]</c> → vocoder input <c>[2B, 128, T]</c>: each frame holds two channels' 128 values side by side
    /// (<c>reshape(B, T, 2, 128)</c>), which become separate batch items (item 2b + c is channel c of batch b).
    /// </summary>
    public static Tensor ReorderForVocoder(Tensor latents)
    {
        using var scope = NewDisposeScope();
        long b = latents.shape[0], t = latents.shape[1];
        long half = latents.shape[2] / 2;
        return latents.reshape(b, t, 2, half).permute(0, 2, 1, 3).reshape(2 * b, t, half).transpose(1, 2).contiguous().MoveToOuterDisposeScope();
    }
}
