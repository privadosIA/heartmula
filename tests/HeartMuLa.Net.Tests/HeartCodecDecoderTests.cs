using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>The window loop with stub models: what each window receives, the noise order, precision settings and the assembled result.</summary>
public class HeartCodecDecoderTests
{
    private const int Frames = 744, Channels = 256;

    private sealed class Call
    {
        public Tensor Segment = null!;
        public long Incontext;
        public Tensor Noise = null!;
        public Tensor? TrueLatents;
        public Tensor Output = null!;
        public bool Tf32;
    }

    /// <summary>Flow stub: output = noise + 0.001 · mean of the codes of the segment; records everything it receives.</summary>
    private static (Func<Tensor, long, Tensor, Tensor?, Tensor> Flow, List<Call> Calls) StubFlow()
    {
        var calls = new List<Call>();
        Tensor Flow(Tensor segment, long incontext, Tensor noise, Tensor? trueLatents)
        {
            var output = noise + 0.001 * segment.to(ScalarType.Float32).mean();
            calls.Add(new Call
            {
                Segment = segment.clone().DetachFromDisposeScope(),
                Incontext = incontext,
                Noise = noise.clone().DetachFromDisposeScope(),
                TrueLatents = trueLatents?.clone().DetachFromDisposeScope(),
                Output = output.clone().DetachFromDisposeScope(),
                Tf32 = backends.cudnn.allow_tf32,
            });
            return output;
        }
        return (Flow, calls);
    }

    /// <summary>Vocoder stub: [2, 128, 744] → [2, 744·1920] repeating the channel mean of each frame.</summary>
    private static Tensor StubVocoder(Tensor latents) => latents.reshape(1, Frames, 2, 128).permute(0, 2, 1, 3).reshape(2, Frames, 128).mean(new long[] { 2 }).repeat_interleave(1920, 1);

    private static Tensor Codes(int n)
    {
        var values = new long[8 * n];
        for (int q = 0; q < 8; q++)
            for (int t = 0; t < n; t++) values[q * n + t] = 100 * q + t;
        return tensor(values, new long[] { 1, 8, n });
    }

    private static HeartCodecDecoder.WindowNoise NoiseOf(int w) =>
        new(w == 0 ? null : randn(new long[] { 1, 640, Channels }, generator: new Generator((ulong)(50 + w))),
            randn(new long[] { 1, Frames, Channels }, generator: new Generator((ulong)(10 + w))));

    [Fact]
    public void Second_window_receives_the_tail_of_the_first_as_in_context_followed_by_its_padding_noise()
    {
        var (flow, calls) = StubFlow();
        using var codes = Codes(400);
        using var _ = HeartCodecDecoder.DecodeWith(flow, StubVocoder, codes, Frames, Channels, NoiseOf);

        Assert.Equal(2, calls.Count);
        Assert.Equal(0, calls[0].Incontext);
        Assert.Null(calls[0].TrueLatents);
        Assert.Equal(104, calls[1].Incontext);
        using var pad = NoiseOf(1).Padding!;
        using var tail = calls[0].Output[.., ^104..];
        Assert.Equal(new long[] { 1, 744, 256 }, calls[1].TrueLatents!.shape);
        Assert.Equal(0.0, (calls[1].TrueLatents![.., ..104] - tail).abs().max().item<float>());
        Assert.Equal(0.0, (calls[1].TrueLatents![.., 104..] - pad).abs().max().item<float>());
        // Each window gets its own noise.
        using var n0 = NoiseOf(0).Flow; using var n1 = NoiseOf(1).Flow;
        Assert.Equal(0.0, (calls[0].Noise - n0).abs().max().item<float>());
        Assert.Equal(0.0, (calls[1].Noise - n1).abs().max().item<float>());
    }

    [Fact]
    public void Windows_get_the_code_segments_of_the_padded_codes()
    {
        var (flow, calls) = StubFlow();
        using var codes = Codes(400);
        using var _ = HeartCodecDecoder.DecodeWith(flow, StubVocoder, codes, Frames, Channels, NoiseOf);

        // Padded codes are the 400 frames followed by the first 292 again (692 frames); window 1 starts at 320.
        for (int w = 0; w < 2; w++)
        {
            Assert.Equal(new long[] { 1, 8, 372 }, calls[w].Segment.shape);
            var got = calls[w].Segment.data<long>().ToArray();
            for (int q = 0; q < 8; q++)
                for (int t = 0; t < 372; t++)
                    Assert.Equal(100 * q + (320 * w + t) % 400, got[q * 372 + t]);
        }
    }

    [Fact]
    public void Result_is_the_cross_faded_stub_audio_trimmed_to_the_target()
    {
        var (flow, calls) = StubFlow();
        using var codes = Codes(400);
        using var result = HeartCodecDecoder.DecodeWith(flow, StubVocoder, codes, Frames, Channels, NoiseOf);
        Assert.Equal(new long[] { 2, 1_536_000 }, result.shape);

        using var a0 = StubVocoder(calls[0].Output); using var a1 = StubVocoder(calls[1].Output);
        using var assembler = DetokenizePlan.For(400).CreateAssembler();
        assembler.Add(a0); assembler.Add(a1);
        using var expected = assembler.Finish();
        Assert.Equal(0.0, (result - expected).abs().max().item<float>());
    }

    [Fact]
    public void A_single_window_is_the_vocoder_output_trimmed()
    {
        var (flow, calls) = StubFlow();
        using var codes = Codes(126);
        using var result = HeartCodecDecoder.DecodeWith(flow, StubVocoder, codes, Frames, Channels, NoiseOf);
        Assert.Single(calls);
        Assert.Equal(new long[] { 2, 483_840 }, result.shape);
        using var audio = StubVocoder(calls[0].Output);
        Assert.Equal(0.0, (result - audio[.., ..483_840]).abs().max().item<float>());
        Assert.Equal(new long[] { 1, 8, 126 }, codes.shape);                   // the caller's codes are untouched and still alive
        Assert.Equal(125L, codes[0, 0, 125].item<long>());
    }

    [Fact]
    public void Default_noise_is_drawn_from_the_seeded_stream_in_reference_order()
    {
        var (flow, calls) = StubFlow();
        using var codes = Codes(400);
        manual_seed(5);
        using var first = HeartCodecDecoder.DecodeWith(flow, StubVocoder, codes, Frames, Channels);
        var (flow2, calls2) = StubFlow();
        manual_seed(5);
        using var second = HeartCodecDecoder.DecodeWith(flow2, StubVocoder, codes, Frames, Channels);
        Assert.Equal(0.0, (first - second).abs().max().item<float>());                 // same seed → identical audio

        // Order: flow noise of window 0, then the padding of window 1, then its flow noise.
        manual_seed(5);
        using var flow0 = randn(1, Frames, Channels);
        using var pad1 = randn(1, 640, Channels);
        using var flow1 = randn(1, Frames, Channels);
        Assert.Equal(0.0, (calls[0].Noise - flow0).abs().max().item<float>());
        Assert.Equal(0.0, (calls[1].TrueLatents![.., 104..] - pad1).abs().max().item<float>());
        Assert.Equal(0.0, (calls[1].Noise - flow1).abs().max().item<float>());

        manual_seed(6);
        var (flow3, calls3) = StubFlow();
        using var other = HeartCodecDecoder.DecodeWith(flow3, StubVocoder, codes, Frames, Channels);
        Assert.True((other - first).abs().max().item<float>() > 1e-3);                 // another seed → another audio
    }

    [Fact]
    public void Cudnn_tf32_is_off_during_the_decode_and_restored_afterwards()
    {
        var previous = backends.cudnn.allow_tf32;
        try
        {
            backends.cudnn.allow_tf32 = true;
            var (flow, calls) = StubFlow();
            using var codes = Codes(126);
            using (HeartCodecDecoder.DecodeWith(flow, StubVocoder, codes, Frames, Channels, NoiseOf)) { }
            Assert.False(calls[0].Tf32);
            Assert.True(backends.cudnn.allow_tf32);

            backends.cudnn.allow_tf32 = false;
            var (flow2, calls2) = StubFlow();
            using (HeartCodecDecoder.DecodeWith(flow2, StubVocoder, codes, Frames, Channels, NoiseOf, allowCudnnTf32: true)) { }
            Assert.True(calls2[0].Tf32);
            Assert.False(backends.cudnn.allow_tf32);
        }
        finally { backends.cudnn.allow_tf32 = previous; }
    }

    [Fact]
    public void Setting_is_restored_even_when_a_model_throws()
    {
        var previous = backends.cudnn.allow_tf32;
        try
        {
            backends.cudnn.allow_tf32 = true;
            using var codes = Codes(126);
            Assert.Throws<InvalidOperationException>(() => HeartCodecDecoder.DecodeWith((_, _, _, _) => throw new InvalidOperationException("boom"), StubVocoder, codes, Frames, Channels, NoiseOf));
            Assert.True(backends.cudnn.allow_tf32);
        }
        finally { backends.cudnn.allow_tf32 = previous; }
    }

    [Fact]
    public void Codes_of_the_wrong_shape_are_rejected()
    {
        using var codes = zeros(new long[] { 2, 8, 10 }, ScalarType.Int64);
        Assert.Throws<ArgumentException>(() => HeartCodecDecoder.DecodeWith((_, _, _, _) => throw new InvalidOperationException(), StubVocoder, codes, Frames, Channels));
    }
}
