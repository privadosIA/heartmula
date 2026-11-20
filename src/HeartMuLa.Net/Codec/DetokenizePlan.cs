using MusicGen.Core.Audio;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Codec;

/// <summary>
/// Windows of the reference <c>detokenize</c> (heartlib <c>heartcodec/models/decoder.py</c>) for <c>N</c> code frames: windows of 372 frames with a hop of
/// 320 (overlap 52), codes repeated up to the padded length, and the number of samples the final audio is cut to.
/// </summary>
public sealed record DetokenizePlan(int Frames, int PaddedFrames, int[] Starts, long TargetSamples)
{
    /// <summary>Duration of one decoding window in seconds.</summary>
    public const double Duration = 29.76;
    /// <summary>Code frames of one window: <c>int(29.76 · 12.5)</c>.</summary>
    public static readonly int WindowFrames = (int)(Duration * 12.5);
    /// <summary>Frames between window starts: <c>372 // 93 · 80</c>.</summary>
    public static readonly int HopFrames = WindowFrames / 93 * 80;
    /// <summary>Code frames shared by two consecutive windows (window minus hop).</summary>
    public static readonly int OverlapFrames = WindowFrames - HopFrames;
    /// <summary>Latent frames of the overlap (the latents run at twice the code rate).</summary>
    public static readonly int OverlapLatentFrames = OverlapFrames * 2;
    /// <summary>Sample rate of the decoded audio in Hz.</summary>
    public const int SampleRate = 48000;
    /// <summary>Samples a window decodes to: <c>int(29.76 · 48000)</c>.</summary>
    public static readonly int WindowSamples = (int)(Duration * SampleRate);
    /// <summary>Samples between window starts: <c>WindowSamples // 93 · 80</c>.</summary>
    public static readonly int HopSamples = WindowSamples / 93 * 80;
    /// <summary>Samples shared by two consecutive windows (window minus hop).</summary>
    public static readonly int OverlapSamples = WindowSamples - HopSamples;

    /// <summary>Number of windows of the plan.</summary>
    public int Windows => Starts.Length;

    /// <summary>The assembler that joins this plan's windows (stereo by default) and cuts the result to <see cref="TargetSamples"/>.</summary>
    public WindowAssembler CreateAssembler(long channels = 2) => new(WindowSamples, HopSamples, OverlapSamples, Windows, TargetSamples, channels);

    /// <summary>Builds the plan for <paramref name="frames"/> code frames; throws <see cref="ArgumentOutOfRangeException"/> when there are fewer than one.</summary>
    public static DetokenizePlan For(int frames)
    {
        if (frames < 1) throw new ArgumentOutOfRangeException(nameof(frames), "at least one code frame is needed");
        long target = (long)(frames / 12.5 * SampleRate);
        int length = Math.Max(frames, WindowFrames);                  // shorter inputs are repeated up to one window
        if ((length - OverlapFrames) % HopFrames > 0)
            length = (int)Math.Ceiling((length - OverlapFrames) / (double)HopFrames) * HopFrames + OverlapFrames;
        var starts = new List<int>();
        for (int s = 0; s <= length - HopFrames; s += HopFrames) starts.Add(s);
        return new DetokenizePlan(frames, length, starts.ToArray(), target);
    }

    /// <summary>
    /// <c>codes [B, Q, N]</c> → <c>[B, Q, PaddedFrames]</c>: the reference repeats the codes (<c>cat([codes, codes])</c>) and cuts, so the result is the codes
    /// followed by themselves again, in order.
    /// </summary>
    public Tensor Repeat(Tensor codes)
    {
        if (codes.shape[2] != Frames) throw new ArgumentException($"the plan is for {Frames} frames, got {codes.shape[2]}");
        if (PaddedFrames == Frames) return codes;
        using var scope = NewDisposeScope();
        var index = arange(PaddedFrames, dtype: ScalarType.Int64, device: codes.device) % Frames;
        return codes.index_select(2, index).MoveToOuterDisposeScope();
    }
}
