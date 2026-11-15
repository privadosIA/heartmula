using HeartMuLa.Server.Jobs;

namespace HeartMuLa.Server.Worker;

public enum ModelStatus { Loading, Ready, Failed }

public sealed record GeneratedSong(int AudioDurationMs);

public sealed record DeviceInfo(string Model, string Codec);

/// <summary>
/// The part of the service that touches the model. <see cref="Load"/> and <see cref="Generate"/> are blocking and always called from the worker's own
/// thread, one at a time; the rest may be read from any thread.
/// </summary>
public interface ISongGenerator
{
    ModelStatus Status { get; }
    string? LoadError { get; }
    DeviceInfo Devices { get; }

    /// <summary>True when the model is loaded for each song by a worker process (nothing is resident), so health says <c>on-demand</c> instead of <c>ready</c>.</summary>
    bool LoadsPerSong => false;

    /// <summary>Checkpoint files the model needs that are not there (empty when all are).</summary>
    IReadOnlyList<string> Missing { get; }

    /// <summary>Reads the weights onto the model device. Blocks until done. On failure sets <see cref="Status"/> to Failed (it does not throw).</summary>
    void Load();

    /// <summary>
    /// Generates the song and writes the WAV to <paramref name="wavPath"/>. <paramref name="progress"/> is called after every frame / codec window; the
    /// implementation must observe <paramref name="ct"/> there and throw <see cref="OperationCanceledException"/> to stop.
    /// </summary>
    GeneratedSong Generate(SongRequest request, string wavPath, Action<JobStage, int, int> progress, CancellationToken ct);
}
