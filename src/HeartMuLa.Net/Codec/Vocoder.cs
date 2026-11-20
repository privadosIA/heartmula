using MusicGen.Core.Layers;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace HeartMuLa.Codec;

/// <summary>
/// Vocoder configuration (<c>scalar_model.decoder</c> of HeartCodec-oss, config.json): 128 latent channels, 64 initial channels,
/// upsampling 5·4·4·4·3 with kernels 10·8·8·8·6, input kernel 5 (non-causal), residual kernel 7 and a final ×2 duplication: 1920 samples per frame.
/// </summary>
public sealed record VocoderConfig(
    int LatentDim = 128,
    int InitChannel = 64,
    int[]? UpFactors = null,
    int[]? UpKernels = null,
    int DelayKernel = 5,
    int ResKernel = 7,
    int DefaultKernel = 7,
    int NumSamples = 2,
    int NumBands = 1,
    bool Causal = true)
{
    /// <summary>Upsampling factor of each decoder block (default 5, 4, 4, 4, 3).</summary>
    public int[] Factors => UpFactors ?? new[] { 5, 4, 4, 4, 3 };
    /// <summary>Kernel size of each upsampling convolution (default 10, 8, 8, 8, 6).</summary>
    public int[] Kernels => UpKernels ?? new[] { 10, 8, 8, 8, 6 };

    /// <summary>Audio samples per latent frame: the product of the upsampling factors times the final duplication.</summary>
    public int SamplesPerFrame => Factors.Aggregate(1, (a, b) => a * b) * NumSamples;

    /// <summary>The HeartCodec-oss configuration.</summary>
    public static VocoderConfig Default { get; } = new();
}

/// <summary>conv1 (dilated k) + PReLU, conv2 (k=1) + PReLU, and a residual sum.</summary>
public sealed class ResidualUnit : Module<Tensor, Tensor>
{
    private readonly Conv1dLayer conv1;
    private readonly Conv1dLayer conv2;
    private readonly PReLULayer activation1;
    private readonly PReLULayer activation2;

    /// <summary>Builds the unit for <paramref name="channels"/> channels: first convolution with the given <paramref name="dilation"/> and <paramref name="resKernel"/>, second convolution of kernel 1.</summary>
    public ResidualUnit(int channels, int dilation, int resKernel, bool causal) : base(nameof(ResidualUnit))
    {
        conv1 = new Conv1dLayer(channels, channels, resKernel, dilation, causal);
        conv2 = new Conv1dLayer(channels, channels, 1, 1, causal);
        activation1 = new PReLULayer();
        activation2 = new PReLULayer();
        RegisterComponents();
    }

    /// <summary>Applies both convolutions with their activations and adds the input.</summary>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var h = activation1.forward(conv1.forward(x));
        h = activation2.forward(conv2.forward(h));
        return (h + x).MoveToOuterDisposeScope();
    }
}

/// <summary>Upsampling: causal <c>ConvTranspose1d</c> without activation (the field is called <c>layer</c> as in the checkpoint).</summary>
public sealed class UpsampleLayer : Module<Tensor, Tensor>
{
    private readonly ConvTranspose1dLayer layer;

    /// <summary>Builds the transposed convolution from <paramref name="inChannels"/> to <paramref name="outChannels"/> with the given <paramref name="kernel"/> and <paramref name="stride"/>.</summary>
    public UpsampleLayer(int inChannels, int outChannels, int kernel, int stride) : base(nameof(UpsampleLayer))
    {
        layer = new ConvTranspose1dLayer(inChannels, outChannels, kernel, stride);
        RegisterComponents();
    }

    /// <summary>Upsamples <c>[B, C, T]</c> in time by the stride.</summary>
    public override Tensor forward(Tensor x) => layer.forward(x);
}

/// <summary>Upsampling block (<c>up_conv</c>) followed by five residual units with dilations 1, 3, 5, 7 and 9.</summary>
public sealed class ResDecoderBlock : Module<Tensor, Tensor>
{
    private static readonly int[] Dilations = { 1, 3, 5, 7, 9 };
    private readonly UpsampleLayer up_conv;
    private readonly ModuleList<ResidualUnit> convs;

    /// <summary>Builds a block that upsamples by <paramref name="stride"/> from <paramref name="inChannels"/> to <paramref name="outChannels"/>, followed by residual units of kernel <paramref name="resKernel"/>.</summary>
    public ResDecoderBlock(int inChannels, int outChannels, int stride, int upKernel, int resKernel, bool causal) : base(nameof(ResDecoderBlock))
    {
        up_conv = new UpsampleLayer(inChannels, outChannels, upKernel, stride);
        convs = new ModuleList<ResidualUnit>();
        foreach (var d in Dilations) convs.Add(new ResidualUnit(outChannels, d, resKernel, causal));
        RegisterComponents();
    }

    /// <summary>Upsamples <c>[B, C, T]</c> and applies the residual units.</summary>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var h = up_conv.forward(x);
        foreach (var unit in convs)
        {
            var next = unit.forward(h);
            h.Dispose();
            h = next;
        }
        return h.MoveToOuterDisposeScope();
    }
}

/// <summary>Repeats each sample <c>numSamples</c> times in time and applies a convolution (plain weight, no weight_norm) and PReLU.</summary>
public sealed class PostProcessor : Module<Tensor, Tensor>
{
    private readonly int numSamples;
    private readonly Conv1dLayer conv;
    private readonly PReLULayer activation;

    /// <summary>Builds the post-processor for <paramref name="channels"/> channels, repeating each sample <paramref name="numSamples"/> times and convolving with the given <paramref name="kernel"/>.</summary>
    public PostProcessor(int channels, int numSamples, int kernel, bool causal) : base(nameof(PostProcessor))
    {
        this.numSamples = numSamples;
        conv = new Conv1dLayer(channels, channels, kernel, 1, causal);
        activation = new PReLULayer();
        RegisterComponents();
    }

    /// <summary>Repeats each sample in time, then applies the convolution and PReLU: <c>[B, C, T]</c> → <c>[B, C, T·numSamples]</c>.</summary>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var rep = x.repeat_interleave(numSamples, dim: 2);            // [B, C, T] → [B, C, T·numSamples]
        return activation.forward(conv.forward(rep)).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// HeartCodec's <c>ScalarModel.decode</c>: 9-level quantization and the 8 layers of <c>decoder</c>. Input <c>[B, latent, T]</c>, output
/// <c>[B, T·1920]</c> (each batch element, e.g. each stereo channel, is decoded as mono). Parameter names replicate the checkpoint's keys
/// after folding the weight_norm: the file's key <c>scalar_model.decoder.N…</c> is <c>decoder.N…</c> here.
/// </summary>
public sealed class ScalarVocoder : Module<Tensor, Tensor>
{
    private readonly ModuleList<Module<Tensor, Tensor>> decoder;

    /// <summary>Configuration the vocoder was built with.</summary>
    public VocoderConfig Config { get; }

    /// <summary>Builds the vocoder from <paramref name="config"/> (the HeartCodec-oss default when null).</summary>
    public ScalarVocoder(VocoderConfig? config = null) : base(nameof(ScalarVocoder))
    {
        Config = config ?? VocoderConfig.Default;
        var c = Config;
        int n = c.Factors.Length;
        decoder = new ModuleList<Module<Tensor, Tensor>>();

        // 0: symmetric input (non-causal): in the original code this convolution does not receive `causal`.
        decoder.Add(new Conv1dLayer(c.LatentDim, c.InitChannel * (1 << n), c.DelayKernel, 1, causal: false));
        // 1..n: upsampling blocks, channels init·2^n → init·2^(n−1) → … → init.
        for (int i = 0; i < n; i++)
            decoder.Add(new ResDecoderBlock(c.InitChannel << (n - i), c.InitChannel << (n - i - 1), c.Factors[i], c.Kernels[i], c.ResKernel, c.Causal));
        // n+1: temporal duplication + convolution + PReLU, and n+2: final convolution to `num_bands` channels.
        if (c.NumSamples > 1) decoder.Add(new PostProcessor(c.InitChannel, c.NumSamples, c.DefaultKernel, c.Causal));
        decoder.Add(new Conv1dLayer(c.InitChannel, c.NumBands, c.DefaultKernel, 1, c.Causal));
        RegisterComponents();
    }

    /// <summary><c>round(9·x)/9</c> (<c>round_func9</c>): <c>torch.round</c> rounds to the nearest even on ties.</summary>
    public static Tensor Quantize9(Tensor x) => round(x * 9) / 9;

    /// <summary>Quantizes and decodes the latents <c>[B, latent, T]</c> to audio <c>[B, T·1920]</c>.</summary>
    public Tensor Decode(Tensor latents)
    {
        using var scope = NewDisposeScope();
        var h = Quantize9(latents);
        foreach (var layer in decoder)
        {
            var next = layer.forward(h);
            h.Dispose();                       // the activations are huge (up to ~0.7 GB): they are freed as soon as they are no longer needed
            h = next;
        }
        return h.squeeze(1).MoveToOuterDisposeScope();
    }

    /// <summary>Same as <see cref="Decode"/>.</summary>
    public override Tensor forward(Tensor x) => Decode(x);
}
