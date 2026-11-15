using MusicGen.Core.Testing;
using MusicGen.Core.Audio;
using HeartMuLa.Codec;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// Leaves in HEARTMULA_AUDIO_DIR the vocoder's audio and Python's reference audio to compare them by ear. It never writes if the variable is not defined.
/// </summary>
[Collection(CodecVocoderCollection.Name)]
public sealed class AudioForListeningTests
{
    private const int TargetLen = 483_840;
    private readonly CodecVocoderFixture _fx;
    private readonly ITestOutputHelper _out;

    public AudioForListeningTests(CodecVocoderFixture fx, ITestOutputHelper output) { _fx = fx; _out = output; }

    [RequiresAudioDir]
    public void WritesTheVocoderAudioAndThePythonReference_InTheSameFormat()
    {
        var dir = Checkpoint.AudioDir!;
        Directory.CreateDirectory(dir);
        var latents = Fixtures.Load("codec_latents.safetensors")["decode_in_seg0"];
        var reference = Fixtures.Load("codec_wav.safetensors")["wav"];
        using var _ = no_grad();

        using var y = _fx.Vocoder!.Decode(latents.to(_fx.Device));
        var ours = y[.., ..TargetLen].cpu().contiguous();
        var oursPath = Path.Combine(dir, "c6_vocoder_from_reference_latents.wav");
        var pythonPath = Path.Combine(dir, "c6_reference_python.wav");
        WavWriter.Write(oursPath, ours, 48000, WavFormat.Pcm16);
        WavWriter.Write(pythonPath, reference, 48000, WavFormat.Pcm16);

        var a = WavFile.Read(oursPath);
        var b = WavFile.Read(pythonPath);
        var sa = a.Shorts(); var sb = b.Shorts();
        int differing = Enumerable.Range(0, sa.Length).Count(i => sa[i] != sb[i]);
        int worst = Enumerable.Range(0, sa.Length).Max(i => Math.Abs(sa[i] - sb[i]));
        _out.WriteLine($"{oursPath} ({new FileInfo(oursPath).Length} bytes)");
        _out.WriteLine($"{pythonPath} ({new FileInfo(pythonPath).Length} bytes)");
        _out.WriteLine($"{a.Channels} channels, {a.SampleRate} Hz, {a.Bits} bits, {sa.Length / a.Channels} samples ({sa.Length / a.Channels / 48000.0:F2} s); PCM16 samples that differ between the two: {differing} of {sa.Length}, largest difference {worst} step(s)");

        Assert.Equal(a.FileLength, b.FileLength);                          // same size and format: the comparison by ear is fair
        Assert.Equal((a.Tag, a.Channels, a.SampleRate, a.Bits), (b.Tag, b.Channels, b.SampleRate, b.Bits));
        Assert.Equal(2, a.Channels); Assert.Equal(48000u, a.SampleRate); Assert.Equal(16, a.Bits);
        Assert.Equal(TargetLen * 2 * 2, a.Data.Length);
        Assert.True(worst <= 1, $"the two versions differ by {worst} PCM16 steps: at most 1 (float32 rounding error) is expected");
    }
}
