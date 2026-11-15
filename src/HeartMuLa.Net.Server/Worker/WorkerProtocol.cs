using System.Text.Json;
using HeartMuLa.Server.Jobs;

namespace HeartMuLa.Server.Worker;

/// <summary>The settings a worker needs, already resolved by the service (a child does not inherit settings given on the command line).</summary>
public sealed record WorkerSettings(string Checkpoint, string Version, string ModelDevice, string CodecDevice, bool CodecTf32);

/// <summary>
/// What the service tells a worker, as one JSON line. <c>Mode</c> is <c>song</c> (load the model, generate, decode) or <c>codec</c> (only decode the
/// frames in <c>FramesPath</c>).
/// </summary>
public sealed record WorkerRequest(string Mode, string Tags, string Lyrics, int DurationMs, long Seed, double CfgScale, double Temperature, int TopK,
                                   string? FramesPath, WorkerSettings Settings);

/// <summary>
/// The conversation between the service and a worker process, as JSON lines: the request on the worker's standard input, one progress line per step on
/// its standard output. Both sides use this class, so they cannot drift apart.
/// </summary>
public static class WorkerProtocol
{
    /// <summary>The argument that makes the service executable act as a worker; the mode and the WAV path follow it.</summary>
    public const string Argument = "--worker";
    public const string SongMode = "song", CodecMode = "codec";

    /// <summary>Exit codes of a worker: the song is written; the song could not be made (the cause is the last line of standard error); cancelled.</summary>
    public const int Done = 0, Failed = 1, Cancelled = 3;

    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    private sealed record ProgressLine(JobStage Stage, int Current, int Total);

    public static string EncodeRequest(WorkerRequest request) => JsonSerializer.Serialize(request, Json);

    public static WorkerRequest DecodeRequest(string line) =>
        JsonSerializer.Deserialize<WorkerRequest>(line, Json) ?? throw new JsonException("the request line is empty");

    public static string EncodeProgress(JobStage stage, int current, int total) => JsonSerializer.Serialize(new ProgressLine(stage, current, total), Json);

    /// <summary>Reads a progress line; false for anything else (a stray line from a native library, say), which the caller ignores.</summary>
    public static bool TryDecodeProgress(string line, out (JobStage Stage, int Current, int Total) progress)
    {
        progress = default;
        if (!line.StartsWith('{')) return false;
        try
        {
            using var doc = JsonDocument.Parse(line);
            if (!doc.RootElement.TryGetProperty("stage", out _)) return false;               // JSON, but not ours
            var p = doc.RootElement.Deserialize<ProgressLine>(Json);
            if (p is null) return false;
            progress = (p.Stage, p.Current, p.Total);
            return true;
        }
        catch (JsonException) { return false; }
    }
}
