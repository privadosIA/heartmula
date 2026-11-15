using System.Text.RegularExpressions;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Codec;

/// <summary>Summary of a flow-matching load: tensors read, RVQ training-state tensors skipped by the allow-list, tensors outside <c>flow_matching.</c> not looked at.</summary>
public sealed record FlowMatchingLoadReport(int TensorsRead, int IgnoredTrainingState, int IgnoredOutsideFlowMatching, int ModelParameters, long BytesRead);

/// <summary>
/// Strict loader of <c>flow_matching.*</c> from the HeartCodec checkpoint. Only headers are used to validate (one error lists every discrepancy before any
/// data is read) and tensors are then read one by one. The RVQ tensors that only exist for training (EMA codebook statistics and <c>project_in</c>) are
/// skipped through an explicit allow-list, so anything else the model does not have is still reported as unexpected. <c>scalar_model.*</c> is not read here.
/// </summary>
public static class FlowMatchingWeights
{
    public const string Prefix = "flow_matching.";

    private static readonly Regex TrainingState = new(
        @"^vq_embed\.(layers\.\d+\._codebook\.(embed_avg|cluster_size|initted)|project_in\.(weight|bias))$", RegexOptions.CultureInvariant | RegexOptions.Compiled);

    public static bool IsTrainingState(string modelKey) => TrainingState.IsMatch(modelKey);

    public static FlowMatchingLoadReport Load(FlowMatching model, string codecDir, Device device) => Load(model, CheckpointIndex.Open(codecDir), device);

    public static FlowMatchingLoadReport Load(FlowMatching model, CheckpointIndex checkpoint, Device device)
    {
        var r = StrictLoader.Load(model, checkpoint, device, new StrictLoadOptions("The HeartCodec checkpoint does not match the flow-matching model:")
        {
            Prefix = Prefix,
            Skip = IsTrainingState,
            ReadDtype = ScalarType.Float32,
            UnexpectedLabel = "flow_matching keys in the checkpoint",
        });
        return new FlowMatchingLoadReport(r.TensorsRead, r.SkippedByAllowList, r.OutsidePrefix, r.ModelTensors, r.BytesRead);
    }
}
