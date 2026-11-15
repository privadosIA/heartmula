using MusicGen.Core.Testing;
using MusicGen.Core.Audio;
using HeartMuLa.Codec;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>
/// Leaves in HEARTMULA_AUDIO_DIR the first audio produced from tokens by the C# codec (codes → flow matching → vocoder), in the same format as the Python
/// reference of C6, to compare them by ear. It never writes when the variable is not set.
/// </summary>
[Collection(FlowMatchingCollection.Name)]
public sealed class AudioFromTokensTests
{
    private const int TargetLen = 483_840;
    private readonly FlowMatchingFixture _fx;
    private readonly ITestOutputHelper _out;

    public AudioFromTokensTests(FlowMatchingFixture fx, ITestOutputHelper output) { _fx = fx; _out = output; }

    [RequiresAudioDir]
    public void Writes_the_wav_decoded_from_the_reference_codes_next_to_the_python_one()
    {
        var dir = Checkpoint.AudioDir!;
        Directory.CreateDirectory(dir);
        var f = Fixtures.Load("codec_latents.safetensors");
        var codes = f["codes_seg0"].to(ScalarType.Int64).to(_fx.Device);
        var noise = f["randn_1_inference_codes"].to(_fx.Device);
        var reference = Fixtures.Load("codec_wav.safetensors")["wav"];

        using var wav = HeartCodecDecoder.DecodeSegment(_fx.Model!, _fx.Vocoder!, codes, noise: noise);
        var oursPath = Path.Combine(dir, "c7_codes_to_wav_csharp.wav");
        var pythonPath = Path.Combine(dir, "c6_reference_python.wav");
        WavWriter.Write(oursPath, wav[.., ..TargetLen].cpu().contiguous(), 48000, WavFormat.Pcm16);
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
        Assert.Equal(TargetLen * 2 * 2, a.Data.Length);
        Assert.True(worst <= 1, $"the two versions differ by {worst} PCM16 steps: at most 1 (float32 rounding) is expected");
    }
}
