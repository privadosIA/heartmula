using HeartMuLa.Server.Jobs;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Server.Worker;

/// <summary>Turns the generated frames <c>[8, T]</c> into the song's WAV; where that happens is what the isolation modes decide.</summary>
public interface IFrameDecoder
{
    /// <summary>Writes the WAV at <paramref name="wavPath"/> and returns its length in milliseconds.</summary>
    int Decode(Tensor framesCpu, SongRequest request, string wavPath, Action<JobStage, int, int> progress, CancellationToken ct);
}

/// <summary><c>codec</c> isolation: the frames go to a file, a worker process decodes them with HeartCodec and exits, and the codec's memory goes back to the system.</summary>
public sealed class WorkerFrameDecoder : IFrameDecoder
{
    private readonly WorkerSettings settings;
    private readonly WorkerCommand command;

    public WorkerFrameDecoder(WorkerSettings settings, WorkerCommand? command = null)
    {
        this.settings = settings;
        this.command = command ?? WorkerRunner.CurrentExecutable;
    }

    /// <summary>The frames file of a song: next to its WAV, named so that the startup sweep of <c>*.tmp</c> files removes a leftover.</summary>
    public static string FramesPath(string wavPath) =>
        (wavPath.EndsWith(".wav.tmp", StringComparison.Ordinal) ? wavPath[..^".wav.tmp".Length] : wavPath) + ".frames.tmp";

    public int Decode(Tensor framesCpu, SongRequest request, string wavPath, Action<JobStage, int, int> progress, CancellationToken ct)
    {
        var framesPath = FramesPath(wavPath);
        try
        {
            SafeTensorsWriter.Save(framesPath, new Dictionary<string, Tensor> { ["frames"] = framesCpu }, new Dictionary<string, string>());
            var worker = new WorkerRequest(WorkerProtocol.CodecMode, request.Tags, request.Lyrics, request.DurationMs, request.Seed, request.CfgScale,
                                           request.Temperature, request.TopK, framesPath, settings);
            WorkerRunner.Run(command, worker, wavPath, progress, ct);
            return WavInfo.DurationMs(wavPath);
        }
        finally
        {
            try { File.Delete(framesPath); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }
}
