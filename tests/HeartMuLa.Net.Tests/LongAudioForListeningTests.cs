using MusicGen.Core.Testing;
using MusicGen.Core.Audio;
using HeartMuLa.Codec;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// Leaves in HEARTMULA_AUDIO_DIR the 32 s two-window decode of the C# codec next to the Python reference of the same codes (the cross-fade at 25.6 s to 29.8 s is
/// the thing to listen for). It never writes when the variable is not set.
/// </summary>
[Collection(FlowMatchingCollection.Name)]
public sealed class LongAudioForListeningTests
{
    private readonly FlowMatchingFixture _fx;
    private readonly ITestOutputHelper _out;

    public LongAudioForListeningTests(FlowMatchingFixture fx, ITestOutputHelper output) { _fx = fx; _out = output; }

    [RequiresAudioDir]
    public void Writes_the_long_decode_next_to_the_python_reference()
    {
        var dir = Checkpoint.AudioDir!;
        Directory.CreateDirectory(dir);
        var f = Fixtures.Load("codec_long_latents.safetensors");
        var reference = Fixtures.Load("codec_long_wav.safetensors")["wav"];
        var dev = _fx.Device;
        var codes = f["frames"].to(ScalarType.Int64).unsqueeze(0).to(dev);
        HeartCodecDecoder.WindowNoise Noise(int w) => w == 0
            ? new(null, f["randn_1_inference_codes"].to(dev))
            : new(f["randn_2_detokenize"].to(dev), f["randn_3_inference_codes"].to(dev));

        using var wav = HeartCodecDecoder.Decode(_fx.Model!, _fx.Vocoder!, codes, noise: Noise);
        var oursPath = Path.Combine(dir, "c8_reference_long_csharp.wav");
        var pythonPath = Path.Combine(dir, "c8_reference_long_python.wav");
        WavWriter.Write(oursPath, wav.contiguous(), 48000, WavFormat.Pcm16);
        WavWriter.Write(pythonPath, reference, 48000, WavFormat.Pcm16);

        var a = WavFile.Read(oursPath);
        var b = WavFile.Read(pythonPath);
        var sa = a.Shorts(); var sb = b.Shorts();
        int differing = Enumerable.Range(0, sa.Length).Count(i => sa[i] != sb[i]);
        int worst = Enumerable.Range(0, sa.Length).Max(i => Math.Abs(sa[i] - sb[i]));
        _out.WriteLine($"{oursPath} ({new FileInfo(oursPath).Length} bytes)");
        _out.WriteLine($"{pythonPath} ({new FileInfo(pythonPath).Length} bytes)");
        _out.WriteLine($"{a.Channels} channels, {a.SampleRate} Hz, {a.Bits} bits, {sa.Length / a.Channels} samples ({sa.Length / a.Channels / 48000.0:F2} s); PCM16 samples that differ: {differing} of {sa.Length}, largest difference {worst} step(s)");

        Assert.Equal(a.FileLength, b.FileLength);
        Assert.Equal((a.Tag, a.Channels, a.SampleRate, a.Bits), (b.Tag, b.Channels, b.SampleRate, b.Bits));
        Assert.Equal(1_536_000 * 2 * 2, a.Data.Length);
        // One quantized latent value flips versus Python at 14.3 s (see design.md), which moves the audio by up to 4e-3 (about 130 PCM16 steps) for 0.8 s.
        Assert.True(worst <= 150, $"the two versions differ by {worst} PCM16 steps");
    }
}
