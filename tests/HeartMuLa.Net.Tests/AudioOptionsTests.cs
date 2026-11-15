using MusicGen.Core.Testing;
using MusicGen.Core.Audio;
using HeartMuLa.Codec;
using HeartMuLa.Pipeline;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>The audio options of <c>generate</c> and the unloading of the model before the codec.</summary>
public class AudioOptionsTests
{
    private static AudioOptions? Parse(params (string Key, string Value)[] args)
    {
        var map = args.ToDictionary(a => a.Key, a => a.Value);
        return AudioOptions.Parse(k => map.GetValueOrDefault(k));
    }

    [Fact]
    public void Without_wav_nothing_is_requested() => Assert.Null(Parse());

    [Fact]
    public void Wav_alone_gives_pcm16_on_the_models_device_without_tf32()
    {
        var o = Parse(("wav", "song.wav"))!;
        Assert.Equal("song.wav", o.WavPath);
        Assert.Equal(WavFormat.Pcm16, o.Format);
        Assert.Null(o.CodecDevice);
        Assert.False(o.CodecTf32);
    }

    [Fact]
    public void All_options_are_read()
    {
        var o = Parse(("wav", "a.wav"), ("wav-format", "float32"), ("codec-device", "cpu"), ("codec-tf32", "true"))!;
        Assert.Equal(WavFormat.Float32, o.Format);
        Assert.Equal("cpu", o.CodecDevice);
        Assert.True(o.CodecTf32);
    }

    [Fact]
    public void Invalid_format_names_the_accepted_values()
    {
        var e = Assert.Throws<ArgumentException>(() => Parse(("wav", "a.wav"), ("wav-format", "mp3")));
        Assert.Contains("pcm16", e.Message);
        Assert.Contains("float32", e.Message);
        Assert.Contains("mp3", e.Message);
    }

    [Theory]
    [InlineData("wav-format", "float32")]
    [InlineData("codec-device", "cpu")]
    [InlineData("codec-tf32", "true")]
    public void Codec_options_without_wav_are_rejected(string key, string value)
    {
        var e = Assert.Throws<ArgumentException>(() => Parse((key, value)));
        Assert.Contains($"--{key}", e.Message);
        Assert.Contains("--wav", e.Message);
    }

    [Fact]
    public void Wav_without_a_path_is_rejected() => Assert.Throws<ArgumentException>(() => Parse(("wav", "true")));

    [Fact]
    public void An_unknown_codec_device_is_rejected() => Assert.Throws<ArgumentException>(() => Parse(("wav", "a.wav"), ("codec-device", "quantum:3")));

    [Fact]
    public void Unload_frees_the_caches_the_parameters_and_the_given_tensors()
    {
        var model = TinyModel.Create(3);
        model.SetupCaches(2);
        Assert.True(model.CachesEnabled);
        var prompt = zeros(new long[] { 4 });
        var param = model.parameters().First();

        AudioStage.Unload(model, prompt);

        Assert.False(model.CachesEnabled);
        Assert.Throws<InvalidOperationException>(() => prompt.shape);                    // disposed: its handle is gone
        Assert.Throws<InvalidOperationException>(() => param.shape);
    }

    [Fact]
    public void ReleaseGeneration_frees_caches_and_tensors_but_keeps_the_model_usable()
    {
        var model = TinyModel.Create(3);
        model.SetupCaches(2);
        var prompt = zeros(new long[] { 4 });
        var param = model.parameters().First();

        AudioStage.ReleaseGeneration(model, prompt);

        Assert.False(model.CachesEnabled);
        Assert.Throws<InvalidOperationException>(() => prompt.shape);
        Assert.Equal(param.shape, model.parameters().First().shape);                     // parameters still alive
        model.SetupCaches(2);                                                           // and the next generation can rebuild the caches
        Assert.True(model.CachesEnabled);
    }

    [RequiresCudaFact]
    public void Unload_leaves_room_for_a_model_sized_allocation_on_the_same_gpu()
    {
        var device = new Device(DeviceType.CUDA, 0);
        var model = TinyModel.Create(4);
        model.to(device);
        model.SetupCaches(2);
        AudioStage.Unload(model);
        // The model is gone and a fresh one of the same size can take its place on the device.
        using var again = TinyModel.Create(5);
        again.to(device);
        Assert.Equal(DeviceType.CUDA, again.parameters().First().device_type);
    }
}
