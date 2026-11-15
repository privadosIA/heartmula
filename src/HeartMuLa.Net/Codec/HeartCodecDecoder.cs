using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Codec;

/// <summary>Codes to audio for one segment: flow matching (codes → latents), the stereo reordering and the vocoder.</summary>
public static class HeartCodecDecoder
{
    /// <param name="codes"><c>[1, 8, T]</c> int64 (T ≤ 372 frames of 12.5 Hz)</param>
    /// <param name="noise"><c>[1, 2T, 256]</c>; <c>torch.randn</c> when null</param>
    /// <returns><c>[2, 2T · 1920]</c> samples at 48 kHz (not trimmed)</returns>
    public static Tensor DecodeSegment(FlowMatching flow, ScalarVocoder vocoder, Tensor codes, double guidanceScale = 1.25, int numSteps = 10, Tensor? noise = null)
    {
        using var scope = NewDisposeScope();
        using var _ = no_grad();
        long latentLength = codes.shape[2] * FlowMatchingConfig.LatentFramesPerCodeFrame;
        var latents = flow.InferenceCodes(codes, latentLength, 0, guidanceScale, numSteps, noise);
        return vocoder.Decode(FlowMatching.ReorderForVocoder(latents)).MoveToOuterDisposeScope();
    }

    /// <summary>Noise of one window: the padding that completes the in-context latents to a full window (not for window 0) and the flow-matching noise.</summary>
    public readonly record struct WindowNoise(Tensor? Padding, Tensor Flow);

    /// <summary>
    /// Codes of any length → stereo audio <c>[2, TargetSamples]</c> (CPU, float32), as heartlib's <c>detokenize</c>: overlapping windows, in-context latents
    /// from the tail of the previous window, linear cross-fade and trim. cuDNN TF32 is off during the call unless <paramref name="allowCudnnTf32"/>.
    /// </summary>
    /// <param name="codes"><c>[1, 8, N]</c> int64</param>
    /// <param name="noise">noise of each window; default <c>randn</c> on the codes' device, in the order the reference draws it (per window: padding, then flow)</param>
    public static Tensor Decode(FlowMatching flow, ScalarVocoder vocoder, Tensor codes, double guidanceScale = 1.25, int numSteps = 10,
        Func<int, WindowNoise>? noise = null, bool allowCudnnTf32 = false, Action<int, int>? onWindow = null)
    {
        long frames = DetokenizePlan.WindowFrames * FlowMatchingConfig.LatentFramesPerCodeFrame;
        long channels = flow.LatentDim;
        Tensor Flow(Tensor segment, long incontext, Tensor flowNoise, Tensor? trueLatents) =>
            flow.InferenceCodes(segment, frames, incontext, guidanceScale, numSteps, flowNoise, trueLatents, otherSeg: true);
        return DecodeWith(Flow, latents => vocoder.Decode(FlowMatching.ReorderForVocoder(latents)), codes, frames, channels, noise, allowCudnnTf32, onWindow);
    }

    /// <summary>The window loop with the two models as functions, so it can be tested without them.</summary>
    public static Tensor DecodeWith(Func<Tensor, long, Tensor, Tensor?, Tensor> flowMatching, Func<Tensor, Tensor> vocode, Tensor codes,
        long latentFrames, long latentChannels, Func<int, WindowNoise>? noise = null, bool allowCudnnTf32 = false, Action<int, int>? onWindow = null)
    {
        if (codes.dim() != 3 || codes.shape[0] != 1) throw new ArgumentException("codes must be [1, quantizers, frames]");
        var plan = DetokenizePlan.For((int)codes.shape[2]);
        var device = codes.device;
        var previousTf32 = backends.cudnn.allow_tf32;
        backends.cudnn.allow_tf32 = allowCudnnTf32;
        using var _ = no_grad();
        var padded = plan.Repeat(codes);                              // the very same tensor when no repetition is needed: not ours to dispose
        using var assembler = plan.CreateAssembler();
        Tensor? tail = null;                                           // last OverlapLatentFrames latent frames of the previous window
        try
        {
            for (int w = 0; w < plan.Windows; w++)
            {
                using var scope = NewDisposeScope();
                onWindow?.Invoke(w, plan.Windows);
                var segment = padded[.., .., plan.Starts[w]..(plan.Starts[w] + DetokenizePlan.WindowFrames)];
                var n = noise?.Invoke(w) ?? default;
                Tensor latents;
                if (w == 0)
                {
                    var flowNoise = n.Flow ?? randn(new long[] { 1, latentFrames, latentChannels }, device: device);
                    latents = flowMatching(segment, 0, flowNoise, null);
                }
                else
                {
                    int ctx = DetokenizePlan.OverlapLatentFrames;
                    var padding = n.Padding ?? randn(new long[] { 1, latentFrames - ctx, latentChannels }, device: device);
                    var flowNoise = n.Flow ?? randn(new long[] { 1, latentFrames, latentChannels }, device: device);
                    var trueLatents = cat(new[] { tail!.to(device), padding.to(tail.dtype, device) }, 1);
                    latents = flowMatching(segment, ctx, flowNoise, trueLatents);
                }
                tail?.Dispose();
                tail = latents[.., ^DetokenizePlan.OverlapLatentFrames..].clone().DetachFromDisposeScope();
                using var audio = vocode(latents);
                assembler.Add(audio[.., ..DetokenizePlan.WindowSamples]);
            }
            return assembler.Finish();
        }
        finally
        {
            tail?.Dispose();
            if (!ReferenceEquals(padded, codes)) padded.Dispose();
            backends.cudnn.allow_tf32 = previousTf32;
        }
    }
}
