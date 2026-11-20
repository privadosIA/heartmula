using MusicGen.Core.Layers;

namespace HeartMuLa.Codec;

/// <summary>
/// The rotary embedding of HeartCodec's flow-matching DiT (transformer.py in heartlib): ADJACENT pairs, base 10000, no Llama-3 scaling, cos/sin
/// computed on every call in the input dtype, over the full head dimension. It is NOT the HeartMuLa Llama stack's (Llama3Scaled, float32 table).
/// </summary>
public static class HeartCodecRope
{
    /// <summary>Creates the rotary embedding for attention heads of dimension <paramref name="headDim"/>.</summary>
    public static RotaryEmbedding Create(int headDim) =>
        new(RopeLayout.AdjacentPairs, RopeFrequencies.Plain(10000), RopeTable.InputDtypePerCall, headDim);
}
