using MusicGen.Core.Layers;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace HeartMuLa.Model;

/// <summary>
/// Hyperparameters of a torchtune-style `llama3_2` TransformerDecoder.
/// The four heartlib "flavors" are in <see cref="Flavors"/>.
/// </summary>
public sealed record LlamaArgs(
    int VocabSize,
    int NumLayers,
    int NumHeads,
    int NumKvHeads,
    int EmbedDim,
    int MaxSeqLen,
    int IntermediateDim,
    double NormEps,
    double RopeBase,
    double ScaleFactor)
{
    /// <summary>Dimension of each attention head: <c>EmbedDim / NumHeads</c>.</summary>
    public int HeadDim => EmbedDim / NumHeads;
}

/// <summary>The Llama configurations used by HeartMuLa.</summary>
public static class Flavors
{
    // Copied 1:1 from modeling_heartmula.py
    /// <summary>The <c>llama-3B</c> flavor: 28 layers, 24 heads (8 KV heads), embedding dimension 3072.</summary>
    public static readonly LlamaArgs Llama3B = new(128_256, 28, 24, 8, 3072, 8192, 8192, 1e-5, 500_000, 32);
    /// <summary>The <c>llama-300M</c> flavor: 3 layers, 8 heads (4 KV heads), embedding dimension 3072.</summary>
    public static readonly LlamaArgs Llama300M = new(128_256, 3, 8, 4, 3072, 2048, 8192, 1e-5, 500_000, 32);
    /// <summary>The <c>llama-7B</c> flavor: 32 layers, 32 heads (8 KV heads), embedding dimension 4096.</summary>
    public static readonly LlamaArgs Llama7B = new(128_256, 32, 32, 8, 4096, 8192, 14336, 1e-5, 500_000, 32);
    /// <summary>The <c>llama-400M</c> flavor: 4 layers, 8 heads (4 KV heads), embedding dimension 3072.</summary>
    public static readonly LlamaArgs Llama400M = new(128_256, 4, 8, 4, 3072, 2048, 8192, 1e-5, 500_000, 32);

    /// <summary>Returns the flavor with the given name (<c>llama-3B</c>, <c>llama-300M</c>, <c>llama-7B</c> or <c>llama-400M</c>); throws <see cref="ArgumentException"/> otherwise.</summary>
    public static LlamaArgs Get(string name) => name switch
    {
        "llama-3B" => Llama3B,
        "llama-300M" => Llama300M,
        "llama-7B" => Llama7B,
        "llama-400M" => Llama400M,
        _ => throw new ArgumentException($"unknown flavor: {name}")
    };
}

/// <summary>torchtune's MultiHeadAttention with GQA and a positional KV cache.</summary>
public sealed class Attention : Module<Tensor, Tensor, Tensor?, Tensor>
{
    private readonly Linear q_proj, k_proj, v_proj, output_proj;
    private readonly int numHeads, numKvHeads, headDim;
    private readonly RotaryEmbedding rope;

    // KV cache: [b, n_kv, max_seq_len, head_dim]. It is not part of the state_dict.
    private Tensor? kCache, vCache;

    /// <summary>Builds the attention layer for the hyperparameters <paramref name="a"/>; the KV cache is allocated later by <see cref="SetupCaches"/>.</summary>
    public Attention(LlamaArgs a) : base(nameof(Attention))
    {
        numHeads = a.NumHeads;
        numKvHeads = a.NumKvHeads;
        headDim = a.HeadDim;
        q_proj = Linear(a.EmbedDim, a.NumHeads * a.HeadDim, hasBias: false);
        k_proj = Linear(a.EmbedDim, a.NumKvHeads * a.HeadDim, hasBias: false);
        v_proj = Linear(a.EmbedDim, a.NumKvHeads * a.HeadDim, hasBias: false);
        output_proj = Linear(a.NumHeads * a.HeadDim, a.EmbedDim, hasBias: false);
        // torchtune's Llama3ScaledRoPE: Llama 3 frequency scaling, a float32 table for MaxSeqLen, rotation over ADJACENT pairs (not HuggingFace's rotate_half).
        rope = new RotaryEmbedding(RopeLayout.AdjacentPairs, RopeFrequencies.Llama3Scaled(a.RopeBase, a.ScaleFactor), RopeTable.Float32Cached, a.HeadDim, a.MaxSeqLen);
        RegisterComponents();
    }

    /// <summary>True when the KV caches are allocated.</summary>
    public bool CachesEnabled => kCache is not null;

    /// <summary>Allocates the KV caches <c>[batchSize, kv_heads, maxSeqLen, head_dim]</c> in <paramref name="dtype"/> on <paramref name="device"/>, replacing any previous ones.</summary>
    public void SetupCaches(int batchSize, int maxSeqLen, ScalarType dtype, Device device)
    {
        ResetCaches();
        kCache?.Dispose(); vCache?.Dispose();
        kCache = torch.zeros(new long[] { batchSize, numKvHeads, maxSeqLen, headDim }, dtype: dtype, device: device);
        vCache = torch.zeros(new long[] { batchSize, numKvHeads, maxSeqLen, headDim }, dtype: dtype, device: device);
    }

    /// <summary>Zeroes the KV caches in place (does nothing if they are not allocated).</summary>
    public void ResetCaches()
    {
        kCache?.zero_();
        vCache?.zero_();
    }

    /// <summary>Disposes the caches (not part of the state_dict, so disposing the module does not free them).</summary>
    public void FreeCaches()
    {
        kCache?.Dispose(); vCache?.Dispose();
        kCache = vCache = null;
    }

    /// <param name="x">[b, s, embed_dim]</param>
    /// <param name="inputPos">[b, s] long</param>
    /// <param name="mask">[b, s, max_seq_len] bool (true = attend). May be null only without a cache.</param>
    public override Tensor forward(Tensor x, Tensor inputPos, Tensor? mask)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0], s = x.shape[1];

        var q = q_proj.forward(x).view(b, s, numHeads, headDim);
        var k = k_proj.forward(x).view(b, s, numKvHeads, headDim);
        var v = v_proj.forward(x).view(b, s, numKvHeads, headDim);

        q = rope.Apply(q, inputPos, seqDim: 1);
        k = rope.Apply(k, inputPos, seqDim: 1);

        q = q.transpose(1, 2); // [b, n_h, s, hd]
        k = k.transpose(1, 2); // [b, n_kv, s, hd]
        v = v.transpose(1, 2);

        if (kCache is not null && vCache is not null)
        {
            // Positions are the same across the whole batch (CFG duplicates the same prompt), so we use row 0.
            var pos = inputPos[0].reshape(-1);
            kCache.index_copy_(2, pos, k.to(kCache.dtype));
            vCache.index_copy_(2, pos, v.to(vCache.dtype));
            k = kCache;
            v = vCache;
        }

        // Expand KV to n_heads (GQA) as torchtune does: [b, n_kv, 1, L, hd] -> [b, n_kv, q_per_kv, L, hd] -> [b, n_h, L, hd]
        int qPerKv = numHeads / numKvHeads;
        if (qPerKv > 1)
        {
            long L = k.shape[2];
            k = k.unsqueeze(2).expand(b, numKvHeads, qPerKv, L, headDim).reshape(b, numHeads, L, headDim);
            v = v.unsqueeze(2).expand(b, numKvHeads, qPerKv, L, headDim).reshape(b, numHeads, L, headDim);
        }

        Tensor? attnMask = mask?.unsqueeze(1); // [b, 1, s, L]
        var o = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attnMask, 0.0, false);
        o = o.transpose(1, 2).reshape(b, s, -1);
        return output_proj.forward(o).MoveToOuterDisposeScope();
    }
}

/// <summary>torchtune's FeedForward: w2(silu(w1(x)) * w3(x)).</summary>
public sealed class FeedForward : Module<Tensor, Tensor>
{
    private readonly Linear w1, w2, w3;

    /// <summary>Builds the layer from <paramref name="dim"/> to <paramref name="hidden"/> and back.</summary>
    public FeedForward(int dim, int hidden) : base(nameof(FeedForward))
    {
        w1 = Linear(dim, hidden, hasBias: false);
        w2 = Linear(hidden, dim, hasBias: false);
        w3 = Linear(dim, hidden, hasBias: false);
        RegisterComponents();
    }

    /// <summary>Applies <c>w2(silu(w1(x)) * w3(x))</c> to <c>[b, s, dim]</c>.</summary>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        return w2.forward(F.silu(w1.forward(x)) * w3.forward(x)).MoveToOuterDisposeScope();
    }
}

/// <summary>A decoder layer: pre-norm self-attention and pre-norm feed-forward, each with a residual connection.</summary>
public sealed class TransformerLayer : Module<Tensor, Tensor, Tensor?, Tensor>
{
    private readonly Attention attn;
    private readonly FeedForward mlp;
    private readonly RmsNorm sa_norm, mlp_norm;

    /// <summary>Builds the layer for the hyperparameters <paramref name="a"/>.</summary>
    public TransformerLayer(LlamaArgs a) : base(nameof(TransformerLayer))
    {
        attn = new Attention(a);
        mlp = new FeedForward(a.EmbedDim, a.IntermediateDim);
        sa_norm = new RmsNorm(a.EmbedDim, a.NormEps);
        mlp_norm = new RmsNorm(a.EmbedDim, a.NormEps);
        RegisterComponents();
    }

    /// <summary>The layer's attention module.</summary>
    public Attention Attn => attn;

    /// <summary>Applies attention and feed-forward to <c>[b, s, embed_dim]</c> at positions <paramref name="inputPos"/> under <paramref name="mask"/>.</summary>
    public override Tensor forward(Tensor x, Tensor inputPos, Tensor? mask)
    {
        using var scope = NewDisposeScope();
        var h = attn.forward(sa_norm.forward(x), inputPos, mask) + x;
        var o = mlp.forward(mlp_norm.forward(h)) + h;
        return o.MoveToOuterDisposeScope();
    }
}

/// <summary>
/// torchtune's TransformerDecoder WITHOUT tok_embeddings or output (heartlib replaces them with Identity):
/// it takes the embedding [b, s, embed_dim] directly and returns [b, s, embed_dim] normalized.
/// </summary>
public sealed class TransformerDecoder : Module<Tensor, Tensor, Tensor?, Tensor>
{
    private readonly ModuleList<TransformerLayer> layers;
    private readonly RmsNorm norm;
    /// <summary>Hyperparameters the decoder was built with.</summary>
    public LlamaArgs Args { get; }
    /// <summary>Maximum sequence length (the override if one was given, otherwise that of <see cref="Args"/>).</summary>
    public int MaxSeqLen { get; }

    /// <summary>Builds the decoder for <paramref name="a"/>, with <paramref name="maxSeqLenOverride"/> in place of <c>a.MaxSeqLen</c> when given.</summary>
    public TransformerDecoder(LlamaArgs a, int? maxSeqLenOverride = null) : base(nameof(TransformerDecoder))
    {
        Args = a;
        MaxSeqLen = maxSeqLenOverride ?? a.MaxSeqLen;
        layers = new ModuleList<TransformerLayer>();
        for (int i = 0; i < a.NumLayers; i++) layers.Add(new TransformerLayer(a));
        norm = new RmsNorm(a.EmbedDim, a.NormEps);
        RegisterComponents();
    }

    /// <summary>True when the KV caches are allocated.</summary>
    public bool CachesEnabled => layers.Count > 0 && layers[0].Attn.CachesEnabled;

    /// <summary>Allocates the KV caches of every layer for batches up to <paramref name="batchSize"/> (<paramref name="maxSeqLen"/> positions, <see cref="MaxSeqLen"/> when null).</summary>
    public void SetupCaches(int batchSize, ScalarType dtype, Device device, int? maxSeqLen = null)
    {
        foreach (var l in layers) l.Attn.SetupCaches(batchSize, maxSeqLen ?? MaxSeqLen, dtype, device);
    }

    /// <summary>Zeroes the KV caches of every layer.</summary>
    public void ResetCaches()
    {
        foreach (var l in layers) l.Attn.ResetCaches();
    }

    /// <summary>Disposes the KV caches of every layer.</summary>
    public void FreeCaches()
    {
        foreach (var l in layers) l.Attn.FreeCaches();
    }

    /// <summary>Runs the embedding <c>[b, s, embed_dim]</c> through every layer and the final normalization, at positions <paramref name="inputPos"/> under <paramref name="mask"/>.</summary>
    public override Tensor forward(Tensor h, Tensor inputPos, Tensor? mask)
    {
        using var scope = NewDisposeScope();
        foreach (var layer in layers)
            h = layer.forward(h, inputPos, mask);
        return norm.forward(h).MoveToOuterDisposeScope();
    }
}
