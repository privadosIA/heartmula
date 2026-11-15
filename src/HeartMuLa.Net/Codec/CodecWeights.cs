using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Codec;

/// <summary>Summary of a vocoder load: tensors read from the file, weight_norm pairs folded and tensors outside the decoder ignored.</summary>
public sealed record VocoderLoadReport(int TensorsRead, int PairsFolded, int IgnoredOutsideDecoder, int ModelParameters);

/// <summary>
/// Strict loading of the vocoder's decoder (<c>scalar_model.decoder.*</c>) from the HeartCodec checkpoint. It only looks at headers to validate and only
/// reads the decoder's tensors (the <c>scalar_model</c> encoder and the <c>flow_matching</c> are neither read nor counted as extra). Each pair
/// <c>parametrizations.weight.original0/1</c> is folded into a plain <c>weight</c> (<see cref="WeightNorm"/>), so that the file's key without the
/// <c>scalar_model.</c> prefix and with the pair replaced by <c>.weight</c> is exactly the model's.
/// </summary>
public static class CodecWeights
{
    public const string DecoderPrefix = "scalar_model.decoder.";
    private const string StripPrefix = "scalar_model.";

    public static VocoderLoadReport LoadVocoder(ScalarVocoder model, string codecDir, Device device) =>
        LoadVocoder(model, CheckpointIndex.Open(codecDir), device);

    public static VocoderLoadReport LoadVocoder(ScalarVocoder model, CheckpointIndex checkpoint, Device device)
    {
        var r = StrictLoader.Load(model, checkpoint, device, new StrictLoadOptions("The HeartCodec checkpoint does not match the vocoder:")
        {
            Prefix = DecoderPrefix,
            StripPrefix = StripPrefix,
            FoldWeightNorm = true,
            ReadDtype = ScalarType.Float32,
            UnexpectedLabel = "decoder keys in the checkpoint",
        });
        return new VocoderLoadReport(r.TensorsRead, r.PairsFolded, r.OutsidePrefix, r.ModelTensors);
    }
}
