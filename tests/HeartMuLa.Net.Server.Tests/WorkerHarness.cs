using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;
using HeartMuLa.Server.Storage;
using HeartMuLa.Server.Webhook;
using HeartMuLa.Server.Worker;
using Microsoft.Extensions.Logging.Abstractions;

namespace HeartMuLa.Server.Tests;

public sealed class RecordingSink : IJobEndedSink
{
    public List<(string Id, JobState State)> Ended { get; } = new();
    public void JobEnded(Job job) { lock (Ended) Ended.Add((job.Id, job.State)); }
}

/// <summary>A worker over a fake generator and a temp directory, running for real on its own thread.</summary>
public sealed class WorkerHarness : IAsyncDisposable
{
    public string Dir { get; } = Path.Combine(Path.GetTempPath(), "heartmula-worker-" + Guid.NewGuid().ToString("N"));
    public FakeSongGenerator Generator { get; } = new();
    public RecordingSink Sink { get; } = new();
    public ServiceOptions Options { get; }
    public SongService Songs { get; }
    public GenerationWorker Worker { get; }

    public WorkerHarness(int maxQueued = 16)
    {
        Options = new ServiceOptions { Storage = new StorageOptions { Directory = Dir }, Jobs = new JobsOptions { MaxQueued = maxQueued } };
        var results = new ResultStore(Options.Storage, TimeProvider.System);
        Songs = new SongService(new JobStore(), results, Sink, Options, TimeProvider.System);
        Worker = new GenerationWorker(Songs, Generator, NullLogger<GenerationWorker>.Instance);
    }

    public Task StartAsync() => Worker.StartAsync(CancellationToken.None);

    public Job Submit(string tags = "piano", int durationMs = 10_000, string? callback = null) =>
        Songs.Submit(new SongRequest(tags, "la", durationMs, 1, 1.5, callback)) ?? throw new InvalidOperationException("queue full");

    public static async Task Eventually(Func<bool> condition, int timeoutMs = 5000)
    {
        var until = DateTime.UtcNow.AddMilliseconds(timeoutMs);
        while (!condition())
        {
            if (DateTime.UtcNow > until) throw new TimeoutException("condition not met in time");
            await Task.Delay(5);
        }
    }

    /// <summary>Like the synchronous one, for conditions that make a request: blocking a thread on it (<c>.Result</c>) starves the pool on a machine with few cores.</summary>
    public static async Task Eventually(Func<Task<bool>> condition, int timeoutMs = 5000)
    {
        var until = DateTime.UtcNow.AddMilliseconds(timeoutMs);
        while (!await condition())
        {
            if (DateTime.UtcNow > until) throw new TimeoutException("condition not met in time");
            await Task.Delay(5);
        }
    }

    public async ValueTask DisposeAsync()
    {
        try { await Worker.StopAsync(new CancellationTokenSource(5000).Token); } catch { }
        Worker.Dispose();
        if (Directory.Exists(Dir)) Directory.Delete(Dir, recursive: true);
    }
}
