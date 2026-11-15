using System.Collections.Concurrent;
using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Worker;

namespace HeartMuLa.Server.Tests;

/// <summary>Stands in for the model: writes a tiny valid WAV, reports progress and lets a test pause, fail or cancel at a given step.</summary>
public sealed class FakeSongGenerator : ISongGenerator
{
    public int Steps { get; set; } = 4;
    public ModelStatus Status { get; private set; } = ModelStatus.Loading;
    public string? LoadError { get; private set; }
    public DeviceInfo Devices { get; set; } = new("fake:0", "fake:1");
    public IReadOnlyList<string> Missing { get; set; } = Array.Empty<string>();
    public int LoadCalls;
    public bool FailLoad { get; set; }
    /// <summary>When set, <see cref="Load"/> waits for it (keeps the model in the loading state).</summary>
    public ManualResetEventSlim? LoadGate { get; set; }

    /// <summary>Called before each step with the request and the step number (1-based). May block; may throw.</summary>
    public Action<SongRequest, int>? OnStep { get; set; }
    public ConcurrentQueue<string> Started { get; } = new();

    public void Load()
    {
        Interlocked.Increment(ref LoadCalls);
        LoadGate?.Wait(10_000);
        if (FailLoad) { LoadError = "cannot read the weights"; Status = ModelStatus.Failed; }
        else Status = ModelStatus.Ready;
    }

    public GeneratedSong Generate(SongRequest request, string wavPath, Action<JobStage, int, int> progress, CancellationToken ct)
    {
        Started.Enqueue(request.Tags);
        for (int i = 1; i <= Steps; i++)
        {
            OnStep?.Invoke(request, i);
            progress(JobStage.Generating, i, Steps);
        }
        progress(JobStage.Decoding, 1, 1);
        int ms = Math.Min(request.DurationMs, 400);
        File.WriteAllBytes(wavPath, WavBytes(ms));
        return new GeneratedSong(ms);
    }

    /// <summary>Stereo 48 kHz PCM16 silence of <paramref name="ms"/> milliseconds in a valid WAV container.</summary>
    public static byte[] WavBytes(int ms)
    {
        int frames = 48 * ms, dataLen = frames * 4;
        using var m = new MemoryStream();
        using var w = new BinaryWriter(m);
        w.Write("RIFF"u8); w.Write(36 + dataLen); w.Write("WAVE"u8);
        w.Write("fmt "u8); w.Write(16); w.Write((short)1); w.Write((short)2); w.Write(48000); w.Write(48000 * 4); w.Write((short)4); w.Write((short)16);
        w.Write("data"u8); w.Write(dataLen); w.Write(new byte[dataLen]);
        return m.ToArray();
    }
}
