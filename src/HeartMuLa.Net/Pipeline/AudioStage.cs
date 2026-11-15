using MusicGen.Core.Audio;
using HeartMuLa.Codec;
using HeartMuLa.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Pipeline;

/// <summary>Options of <c>generate</c> that turn the frames into a WAV: <c>--wav</c>, <c>--wav-format</c>, <c>--codec-device</c> and <c>--codec-tf32</c>.</summary>
public sealed record AudioOptions(string WavPath, WavFormat Format, string? CodecDevice, bool CodecTf32)
{
    public static readonly string[] Formats = { "pcm16", "float32" };

    /// <summary>
    /// Reads the options through <paramref name="get"/> (null when absent). Returns null when no WAV is requested. Invalid values and options that need
    /// <c>--wav</c> throw <see cref="ArgumentException"/>, so the caller can fail before loading any model.
    /// </summary>
    public static AudioOptions? Parse(Func<string, string?> get)
    {
        var wav = get("wav");
        var format = get("wav-format");
        var device = get("codec-device");
        var tf32 = get("codec-tf32");
        if (wav is null)
        {
            var orphan = new[] { ("wav-format", format), ("codec-device", device), ("codec-tf32", tf32) }.FirstOrDefault(o => o.Item2 is not null);
            if (orphan.Item1 is not null) throw new ArgumentException($"--{orphan.Item1} only makes sense with --wav <path>");
            return null;
        }
        if (wav is "true" or "") throw new ArgumentException("--wav needs a file path");
        var fmt = (format ?? "pcm16") switch
        {
            "pcm16" => WavFormat.Pcm16,
            "float32" => WavFormat.Float32,
            var other => throw new ArgumentException($"--wav-format must be one of {string.Join(" or ", Formats)}, got '{other}'"),
        };
        if (device is "true" or "") throw new ArgumentException("--codec-device needs a device such as cpu or cuda:1");
        if (device is not null)
        {
            try { _ = new Device(device); }
            catch (Exception e) when (e is ArgumentException or InvalidOperationException or FormatException)
            { throw new ArgumentException($"--codec-device '{device}' is not a valid device: {e.Message}"); }
        }
        return new AudioOptions(wav, fmt, device, tf32 is not null);
    }
}

public static class AudioStage
{
    public const string CodecFolder = "HeartCodec-oss";

    /// <summary>
    /// Releases a HeartMuLa model and everything that keeps device memory alive before the codec is read: KV caches and masks, the parameters and the
    /// prompt tensors. TorchSharp 0.107 has no <c>cuda.empty_cache()</c>: the freed blocks stay in libtorch's caching allocator, which hands them to the
    /// next allocations and returns its cached blocks to the driver by itself when an allocation would otherwise fail.
    /// </summary>
    public static void Unload(HeartMuLaModel model, params IDisposable?[] others)
    {
        model.FreeCaches();
        model.Dispose();
        foreach (var o in others) o?.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    /// <summary>
    /// Releases what one generation allocated (KV caches, causal masks and the given prompt/frame tensors) but keeps the model's parameters, so the same
    /// model can generate again: <see cref="HeartMuLaModel.SetupCaches"/> rebuilds the caches on the next <c>Generate</c>. Use it instead of
    /// <see cref="Unload"/> when the codec runs on another device or both fit together.
    /// </summary>
    public static void ReleaseGeneration(HeartMuLaModel model, params IDisposable?[] others)
    {
        model.FreeCaches();
        foreach (var o in others) o?.Dispose();
    }

    /// <summary>Loads the codec on <paramref name="device"/>, decodes <paramref name="frames"/> (<c>[8, T]</c>) and writes the WAV; the codec is released afterwards.</summary>
    public static long DecodeToWav(Tensor frames, string checkpointDir, Device device, AudioOptions options, Action<string>? log = null, Action<int, int>? progress = null)
    {
        var codecDir = Path.Combine(checkpointDir, CodecFolder);
        log?.Invoke($"loading HeartCodec on {device} (fp32)");
        using var flow = new FlowMatching();
        FlowMatchingWeights.Load(flow, codecDir, device);
        flow.eval();
        using var vocoder = new ScalarVocoder();
        CodecWeights.LoadVocoder(vocoder, codecDir, device);
        vocoder.eval();

        using var codes = frames.to(ScalarType.Int64).unsqueeze(0).to(device);
        using var wav = HeartCodecDecoder.Decode(flow, vocoder, codes, allowCudnnTf32: options.CodecTf32, onWindow: progress);
        WavWriter.Write(options.WavPath, wav, DetokenizePlan.SampleRate, options.Format);
        return wav.shape[1];
    }
}
