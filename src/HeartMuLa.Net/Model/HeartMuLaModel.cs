using System.Text.Json;
using MusicGen.Core.Weights;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace HeartMuLa.Model;

/// <summary>Equivalent of HeartMuLaConfig (config.json inside HeartMuLa-oss-3B/).</summary>
public sealed class HeartMuLaConfig
{
    /// <summary>Name of the backbone's Llama flavor (see <see cref="Flavors"/>).</summary>
    public string BackboneFlavor { get; init; } = "llama-3B";
    /// <summary>Name of the codebook decoder's Llama flavor (see <see cref="Flavors"/>).</summary>
    public string DecoderFlavor { get; init; } = "llama-300M";
    /// <summary>Size of the text vocabulary.</summary>
    public int TextVocabSize { get; init; } = 128256;
    /// <summary>Size of each audio codebook's vocabulary (the audio codes plus special ids).</summary>
    public int AudioVocabSize { get; init; } = 8197;
    /// <summary>Number of audio codebooks per frame.</summary>
    public int AudioNumCodebooks { get; init; } = 8;
    /// <summary>Dimension of the MuQ reference-audio embedding.</summary>
    public int MuqDim { get; init; } = 512;

    /// <summary>Reads the configuration from the <c>config.json</c> at <paramref name="path"/>; missing keys keep their defaults.</summary>
    public static HeartMuLaConfig FromFile(string path)
    {
        var j = JsonDocument.Parse(File.ReadAllText(path)).RootElement;
        string S(string k, string d) => j.TryGetProperty(k, out var v) ? v.GetString() ?? d : d;
        int I(string k, int d) => j.TryGetProperty(k, out var v) ? v.GetInt32() : d;
        return new HeartMuLaConfig
        {
            BackboneFlavor = S("backbone_flavor", "llama-3B"),
            DecoderFlavor = S("decoder_flavor", "llama-300M"),
            TextVocabSize = I("text_vocab_size", 128256),
            AudioVocabSize = I("audio_vocab_size", 8197),
            AudioNumCodebooks = I("audio_num_codebooks", 8),
            MuqDim = I("muq_dim", 512),
        };
    }
}

/// <summary>
/// Port of HeartMuLa(PreTrainedModel) from modeling_heartmula.py.
/// The field names match the checkpoint keys, so
/// <c>load_state_dict</c> fits without renaming anything:
///   backbone.layers.N.attn.q_proj.weight, backbone.norm.scale, decoder.*, text_embeddings.weight,
///   audio_embeddings.weight, unconditional_text_embedding.weight, projection.weight,
///   codebook0_head.weight, audio_head, muq_linear.{weight,bias}
/// </summary>
public sealed class HeartMuLaModel : Module
{
    /// <summary>Configuration the model was built with.</summary>
    public HeartMuLaConfig Config { get; }

    private readonly TransformerDecoder backbone;
    private readonly TransformerDecoder decoder;
    private readonly Embedding text_embeddings;
    private readonly Embedding audio_embeddings;
    private readonly Embedding unconditional_text_embedding;
    private readonly Linear projection;
    private readonly Linear codebook0_head;
    private readonly Parameter audio_head;
    private readonly Linear muq_linear;

    private Tensor? backboneCausalMask; // [max_seq, max_seq] bool
    private Tensor? decoderCausalMask;  // [num_codebooks, num_codebooks] bool

    /// <summary>Builds the model with the flavors named in <paramref name="config"/>.</summary>
    public HeartMuLaModel(HeartMuLaConfig config)
        : this(config, Flavors.Get(config.BackboneFlavor), Flavors.Get(config.DecoderFlavor)) { }

    /// <summary>Builds the model with explicit hyperparameters for the backbone (<paramref name="bArgs"/>) and the codebook decoder (<paramref name="dArgs"/>).</summary>
    public HeartMuLaModel(HeartMuLaConfig config, LlamaArgs bArgs, LlamaArgs dArgs) : base(nameof(HeartMuLaModel))
    {
        Config = config;

        backbone = new TransformerDecoder(bArgs);
        // The decoder only needs as many positions as codebooks (setup_caches(decoder_max_seq_len=num_codebooks))
        decoder = new TransformerDecoder(dArgs, maxSeqLenOverride: config.AudioNumCodebooks);

        text_embeddings = Embedding(config.TextVocabSize, bArgs.EmbedDim);
        audio_embeddings = Embedding(config.AudioVocabSize * config.AudioNumCodebooks, bArgs.EmbedDim);
        unconditional_text_embedding = Embedding(1, bArgs.EmbedDim);
        projection = Linear(bArgs.EmbedDim, dArgs.EmbedDim, hasBias: false);
        codebook0_head = Linear(bArgs.EmbedDim, config.AudioVocabSize, hasBias: false);
        audio_head = new Parameter(torch.empty(config.AudioNumCodebooks - 1, dArgs.EmbedDim, config.AudioVocabSize));
        muq_linear = Linear(config.MuqDim, bArgs.EmbedDim);
        RegisterComponents();
    }

    /// <summary>
    /// Builds the model directly in <paramref name="dtype"/>. Without this, the ~3.9 B parameters are
    /// first allocated in fp32 (~14.7 GiB) and only converted afterwards; with libtorch's default dtype
    /// set during construction the peak drops to ~8 GiB in bf16.
    /// </summary>
    public static HeartMuLaModel Create(HeartMuLaConfig config, ScalarType dtype)
    {
        var previous = torch.get_default_dtype();
        torch.set_default_dtype(dtype);
        try { return new HeartMuLaModel(config); }
        finally { torch.set_default_dtype(previous); }
    }

    /// <summary>The backbone transformer, which processes the frame sequence.</summary>
    public TransformerDecoder Backbone => backbone;
    /// <summary>The codebook decoder transformer, which predicts codebooks 1 to K−1 of a frame.</summary>
    public TransformerDecoder Decoder => decoder;

    // ---------------------------------------------------------------- loading

    /// <summary>
    /// Strict load of the directory's safetensors: the model's keys and shapes must match
    /// the checkpoint's exactly (checked with headers only, before reading any data).
    /// </summary>
    public void LoadWeights(string checkpointDir, ScalarType dtype, Device device)
    {
        Console.WriteLine($"Loading HeartMuLa weights from {checkpointDir} ...");
        LoadWeights(CheckpointIndex.Open(checkpointDir), dtype, device);
        Console.WriteLine("  OK");
    }

    /// <summary>
    /// Copies in place, tensor by tensor and under no_grad, converting each one to <paramref name="dtype"/>:
    /// memory only holds the model and one tensor in flight. Throws <see cref="InvalidDataException"/> with
    /// all the discrepancies (missing, extra, shapes) without having read or modified anything.
    /// </summary>
    public void LoadWeights(CheckpointIndex checkpoint, ScalarType dtype, Device device)
    {
        // Parameters and buffers; the KV caches and masks are not registered, so they are not part of the state.
        StrictLoader.Load(this, checkpoint, device, new StrictLoadOptions("The checkpoint does not match the model:")
        {
            ReadDtype = dtype,
            ModelDtype = dtype,
            IncludeBuffers = true,
            UnexpectedLabel = "checkpoint keys",
        });
    }

    // ---------------------------------------------------------------- caches

    /// <summary>Allocates the KV caches of the backbone and the decoder for batches up to <paramref name="maxBatchSize"/> and builds the causal masks, using the dtype and device of the parameters.</summary>
    public void SetupCaches(int maxBatchSize)
    {
        var p = parameters().First();
        var dtype = p.dtype;
        var device = p.device;

        backbone.SetupCaches(maxBatchSize, dtype, device);
        decoder.SetupCaches(maxBatchSize, dtype, device, maxSeqLen: Config.AudioNumCodebooks);

        backboneCausalMask?.Dispose();
        decoderCausalMask?.Dispose();
        backboneCausalMask = torch.tril(torch.ones(backbone.MaxSeqLen, backbone.MaxSeqLen, dtype: ScalarType.Bool, device: device));
        decoderCausalMask = torch.tril(torch.ones(Config.AudioNumCodebooks, Config.AudioNumCodebooks, dtype: ScalarType.Bool, device: device));
    }

    /// <summary>True when the KV caches of both transformers are allocated.</summary>
    public bool CachesEnabled => backbone.CachesEnabled && decoder.CachesEnabled;

    /// <summary>Frees the KV caches and the causal masks (they live outside the parameters, so disposing the model alone would leave them on the device).</summary>
    public void FreeCaches()
    {
        backbone.FreeCaches();
        decoder.FreeCaches();
        backboneCausalMask?.Dispose(); backboneCausalMask = null;
        decoderCausalMask?.Dispose(); decoderCausalMask = null;
    }

    /// <summary>Zeroes the KV caches of the backbone and the decoder.</summary>
    public void ResetCaches()
    {
        backbone.ResetCaches();
        decoder.ResetCaches();
    }

    private static Tensor IndexCausalMask(Tensor mask, Tensor inputPos)
    {
        // mask[input_pos, :] -> [b, s, max_seq]
        long b = inputPos.shape[0], s = inputPos.shape[1];
        return mask.index_select(0, inputPos.reshape(-1)).view(b, s, mask.shape[1]);
    }

    // ---------------------------------------------------------------- embeddings

    private Tensor EmbedAudio(int codebook, Tensor tokens)
        => audio_embeddings.forward(tokens + codebook * Config.AudioVocabSize);

    /// <param name="tokens">[B, S, num_codebooks+1] long. Last column = text, the rest = audio.</param>
    /// <param name="uncondMask">[B] bool, true on the CFG "unconditional" rows; null if there is no CFG.</param>
    /// <returns>[B, S, num_codebooks+1, embed_dim]</returns>
    private Tensor EmbedTokens(Tensor tokens, Tensor? uncondMask)
    {
        long B = tokens.shape[0], S = tokens.shape[1];
        var textEmbeds = text_embeddings.forward(tokens[.., .., -1]); // [B, S, D]

        if (uncondMask is not null)
        {
            var uncondTextEmbed = unconditional_text_embedding.forward(torch.zeros(1, dtype: ScalarType.Int64, device: tokens.device));
            var maskExpanded = uncondMask.view(B, 1, 1).expand_as(textEmbeds);
            textEmbeds = torch.where(maskExpanded, uncondTextEmbed, textEmbeds);
        }
        textEmbeds = textEmbeds.unsqueeze(-2); // [B, S, 1, D]

        var offsets = Config.AudioVocabSize * torch.arange(Config.AudioNumCodebooks, device: tokens.device);
        var audioTokens = tokens[.., .., ..^1] + offsets; // [B, S, K]
        var audioEmbeds = audio_embeddings.forward(audioTokens.view(-1)).reshape(B, S, Config.AudioNumCodebooks, -1);

        return torch.cat(new[] { audioEmbeds, textEmbeds }, dim: -2);
    }

    // ---------------------------------------------------------------- generation

    /// <summary>
    /// First half of generate_frame: embeddings → backbone → codebook 0 logits.
    /// Split out so it can be compared with the golden run without sampling (sampling is random).
    /// </summary>
    /// <param name="tokens">[B, S, K+1] long</param>
    /// <param name="tokensMask">[B, S, K+1] bool</param>
    /// <param name="inputPos">[B, S] long</param>
    /// <param name="cfgScale">Classifier-free guidance scale; the unconditional text embedding is used on the second half of the batch when above 1</param>
    /// <param name="continuousSegments">[B, muq_dim] or null (MuQ embedding of the reference audio; always zeros in oss)</param>
    /// <param name="starts">per-batch index where continuousSegments is injected</param>
    /// <returns>(last_h [B, D], c0_logits [B, audio_vocab])</returns>
    public (Tensor lastH, Tensor c0Logits) BackboneLogits(
        Tensor tokens, Tensor tokensMask, Tensor inputPos, double cfgScale,
        Tensor? continuousSegments = null, int[]? starts = null)
    {
        using var scope = NewDisposeScope();
        long b = tokens.shape[0];
        if (!backbone.CachesEnabled) throw new InvalidOperationException("backbone caches are not enabled");
        if (backboneCausalMask is null) throw new InvalidOperationException("call SetupCaches first");

        var currBackboneMask = IndexCausalMask(backboneCausalMask, inputPos);

        long actualB = b / 2;
        Tensor? uncondMask = null;
        if (cfgScale > 1.0 && b > 1)
        {
            uncondMask = torch.cat(new[]
            {
                torch.zeros(actualB, dtype: ScalarType.Bool, device: tokens.device),
                torch.ones(actualB, dtype: ScalarType.Bool, device: tokens.device)
            });
        }

        var embeds = EmbedTokens(tokens, uncondMask);                 // [B, S, K+1, D]
        var maskedEmbeds = embeds * tokensMask.unsqueeze(-1);
        var h = maskedEmbeds.sum(new long[] { 2 }, type: embeds.dtype); // [B, S, D]

        if (continuousSegments is not null && starts is not null)
        {
            var seg = muq_linear.forward(continuousSegments.to(h.dtype)); // [B, D]
            if (uncondMask is not null)
            {
                var uncondEmbed = unconditional_text_embedding.forward(torch.zeros(1, dtype: ScalarType.Int64, device: tokens.device));
                var maskExpanded = uncondMask.view(b, 1).expand_as(seg);
                seg = torch.where(maskExpanded, uncondEmbed, seg);
            }
            for (long i = 0; i < b; i++)
                h[i, starts[i]] = seg[i];
        }

        h = backbone.forward(h, inputPos, currBackboneMask);
        var lastH = h[.., -1, ..];                          // [B, D]
        var c0Logits = codebook0_head.forward(lastH);       // [B, audio_vocab]
        return (lastH.MoveToOuterDisposeScope(), c0Logits.MoveToOuterDisposeScope());
    }

    /// <summary>
    /// Port of generate_frame: returns [B, num_codebooks] with the tokens of the next frame (80 ms).
    /// </summary>
    /// <param name="tokens">[B, S, K+1] long</param>
    /// <param name="tokensMask">[B, S, K+1] bool</param>
    /// <param name="inputPos">[B, S] long</param>
    /// <param name="temperature">Sampling temperature</param>
    /// <param name="topk">Number of candidates kept by the top-k filter</param>
    /// <param name="cfgScale">Classifier-free guidance scale</param>
    /// <param name="continuousSegments">[B, muq_dim] or null (MuQ embedding of the reference audio; always zeros in oss)</param>
    /// <param name="starts">per-batch index where continuousSegments is injected</param>
    public Tensor GenerateFrame(
        Tensor tokens, Tensor tokensMask, Tensor inputPos,
        double temperature, int topk, double cfgScale,
        Tensor? continuousSegments = null, int[]? starts = null)
    {
        using var scope = NewDisposeScope();
        if (decoderCausalMask is null) throw new InvalidOperationException("call SetupCaches first");

        var (lastH, c0Logits) = BackboneLogits(tokens, tokensMask, inputPos, cfgScale, continuousSegments, starts);

        var c0Sample = Sampling.SampleGuided(c0Logits, cfgScale, topk, temperature);
        var (sample, _) = RunDecoder(lastH, c0Sample, (i, ciLogits) => Sampling.SampleGuided(ciLogits, cfgScale, topk, temperature));

        return sample.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Raw logits (before CFG) of the decoder's codebooks 1..K-1 when it receives the backbone's hidden state and the
    /// already-known tokens of the previous codebooks (teacher forcing), without sampling. It is the same loop
    /// <see cref="GenerateFrame"/> uses; only how each token is chosen changes. The logit of codebook i only depends on
    /// tokens 0..i-1, so those with a higher index can be any valid value.
    /// </summary>
    /// <param name="lastH">[B, D] backbone hidden state</param>
    /// <param name="tokens">[B, K] long: codebook 0 and the following ones</param>
    /// <returns>[B, K-1, audio_vocab]</returns>
    public Tensor DecoderLogits(Tensor lastH, Tensor tokens)
    {
        using var scope = NewDisposeScope();
        if (decoderCausalMask is null) throw new InvalidOperationException("call SetupCaches first");

        var (_, logits) = RunDecoder(lastH, tokens[.., 0..1], (i, _) => tokens[.., i..(i + 1)]);
        return torch.stack(logits, dim: 1).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Loop over codebooks 1..K-1 of generate_frame. <paramref name="chooseToken"/> receives the codebook index and its raw
    /// logits [B, V] and returns the token [B, 1] that goes into the next step (sampled or forced).
    /// </summary>
    private (Tensor sample, List<Tensor> logits) RunDecoder(Tensor lastH, Tensor c0Sample, Func<int, Tensor, Tensor> chooseToken)
    {
        var c0Embed = EmbedAudio(0, c0Sample);              // [B, 1, D]

        decoder.ResetCaches();
        var currH = torch.cat(new[] { lastH.unsqueeze(1), c0Embed }, dim: 1); // [B, 2, D]
        var currSample = c0Sample.clone();
        var currPos = torch.arange(0, currH.shape[1], device: currH.device).unsqueeze(0).repeat(currH.shape[0], 1);
        currH = currH.to(lastH.dtype);

        var logits = new List<Tensor>();
        for (int i = 1; i < Config.AudioNumCodebooks; i++)
        {
            var currDecoderMask = IndexCausalMask(decoderCausalMask!, currPos);
            var decoderH = decoder.forward(projection.forward(currH), currPos, currDecoderMask);
            var ciLogits = torch.mm(decoderH[.., -1, ..], audio_head[i - 1]);
            logits.Add(ciLogits);

            var ciSample = chooseToken(i, ciLogits);
            var ciEmbed = EmbedAudio(i, ciSample);
            currH = ciEmbed;
            currSample = torch.cat(new[] { currSample, ciSample }, dim: 1);
            currPos = currPos[.., ^1..] + 1;
        }

        return (currSample, logits);
    }
}

/// <summary>Sampling of the next audio tokens from logits: classifier-free guidance, temperature and top-k.</summary>
public static class Sampling
{
    /// <summary>
    /// Sampling for one step of generate_frame over the raw logits <paramref name="logits"/> [B, V]: with CFG (even batch
    /// &gt; 1, <paramref name="cfgScale"/> &gt; 1) it guides the conditional row (first half) with the unconditional one (second) and repeats the
    /// chosen token in both rows; without CFG it samples each row. Returns [B, 1] long.
    /// </summary>
    public static Tensor SampleGuided(Tensor logits, double cfgScale, int topk, double temperature)
    {
        if (!UsesCfg(logits, cfgScale))
            return SampleTopK(logits, topk, temperature);

        using var scope = NewDisposeScope();
        var guided = Guide(logits, cfgScale);
        return SampleTopK(guided, topk, temperature).repeat(2, 1).MoveToOuterDisposeScope();
    }

    private static bool UsesCfg(Tensor logits, double cfgScale) => cfgScale > 1.0 && logits.shape[0] > 1 && logits.shape[0] % 2 == 0;

    /// <summary>CFG-guided logits: <c>uncond + (cond - uncond) * cfgScale</c> with cond = first half of the batch and uncond = second. [B, V] → [B/2, V].</summary>
    public static Tensor Guide(Tensor logits, double cfgScale)
    {
        long half = logits.shape[0] / 2;
        var cond = logits[..(int)half];
        var uncond = logits[(int)half..];
        return uncond + (cond - uncond) * cfgScale;
    }

    /// <summary>Probabilities after temperature, top-k filter, log_softmax and softmax (the deterministic part of sample_topk). [B, V].</summary>
    public static Tensor TopKProbabilities(Tensor logits, int topk, double temperature)
    {
        using var scope = NewDisposeScope();
        logits = logits / temperature;

        var (vals, _) = torch.topk(logits, topk);
        var threshold = vals[.., ^1..];                       // [B, 1]
        var filtered = logits.masked_fill(logits < threshold, double.NegativeInfinity);
        var probs = F.softmax(F.log_softmax(filtered, dim: -1), dim: -1);
        return probs.MoveToOuterDisposeScope();
    }

    /// <summary>Port of sample_topk + _multinomial_sample_one_no_sync (Gumbel/exponential trick).</summary>
    public static Tensor SampleTopK(Tensor logits, int topk, double temperature)
    {
        using var scope = NewDisposeScope();
        var probs = TopKProbabilities(logits, topk, temperature);

        var q = torch.empty_like(probs).exponential_(1);
        // The original returns int32; we use int64 so Embedding/cat do not cause trouble.
        var sample = torch.argmax(probs / q, dim: -1, keepdim: true).to(ScalarType.Int64);
        return sample.MoveToOuterDisposeScope();
    }
}
