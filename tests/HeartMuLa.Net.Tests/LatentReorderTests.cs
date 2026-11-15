using HeartMuLa.Codec;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

public class LatentReorderTests
{
    [Fact]
    public void Layout_on_synthetic_data()
    {
        const int B = 2, T = 3, H = 4;
        var values = new float[B * T * 2 * H];
        for (int i = 0; i < values.Length; i++) values[i] = i;
        using var latents = tensor(values, new long[] { B, T, 2 * H });
        using var r = FlowMatching.ReorderForVocoder(latents);
        Assert.Equal(new long[] { 2 * B, H, T }, r.shape);
        var got = r.data<float>().ToArray();
        for (int b = 0; b < B; b++)
            for (int ch = 0; ch < 2; ch++)
                for (int h = 0; h < H; h++)
                    for (int t = 0; t < T; t++)
                        Assert.Equal(values[(b * T + t) * 2 * H + ch * H + h], got[(((b * 2) + ch) * H + h) * T + t]);
    }

    [Fact]
    public void Reference_latents_reorder_exactly_into_the_recorded_vocoder_input()
    {
        var fixture = Fixtures.Load("codec_latents.safetensors");
        var latents = fixture["latents_seg0"];
        var expected = fixture["decode_in_seg0"];
        using var r = FlowMatching.ReorderForVocoder(latents);
        Assert.Equal(expected.shape, r.shape);
        Assert.Equal(0.0, (r - expected).abs().max().item<float>());
    }
}
