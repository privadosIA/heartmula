using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Webhook;

namespace HeartMuLa.Server.Worker;

/// <summary>
/// Runs the queued jobs one at a time on its own thread: loads the model once, then for each job generates, stores the result and tells the webhook sink.
/// A job's failure never stops the loop.
/// </summary>
public sealed class GenerationWorker : BackgroundService
{
    private readonly SongService songs;
    private readonly ISongGenerator generator;
    private readonly ILogger<GenerationWorker> log;
    private volatile Job? current;

    public GenerationWorker(SongService songs, ISongGenerator generator, ILogger<GenerationWorker> log)
    {
        this.songs = songs; this.generator = generator; this.log = log;
    }

    protected override Task ExecuteAsync(CancellationToken stoppingToken) =>
        Task.Factory.StartNew(() => Run(stoppingToken), CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

    private void Run(CancellationToken stopping)
    {
        // Shutdown cancels the running job; the loop then ends because the token is cancelled.
        using var onStop = stopping.Register(() => current?.Cancellation.Cancel());

        try { generator.Load(); }
        catch (Exception e) { log.LogError(e, "Loading the model failed"); }

        var reader = songs.Reader;
        try
        {
            while (reader.WaitToReadAsync(stopping).AsTask().GetAwaiter().GetResult())
            {
                while (reader.TryRead(out var job))
                {
                    Process(job, stopping);
                    if (stopping.IsCancellationRequested) return;
                }
            }
        }
        catch (OperationCanceledException) { /* shutting down */ }
    }

    private void Process(Job job, CancellationToken stopping)
    {
        var now = songs.Time;
        if (!job.TryStart(now.GetUtcNow())) return;                 // cancelled while it waited

        current = job;
        if (stopping.IsCancellationRequested) job.Cancellation.Cancel();
        var ct = job.Cancellation.Token;
        var results = songs.Results;
        try
        {
            results.Save(job.View());
            if (generator.Status != ModelStatus.Ready)
                throw new InvalidOperationException($"the model is not loaded: {generator.LoadError ?? generator.Status.ToString()}");

            var song = generator.Generate(job.Request, results.WavTempPath(job.Id), (stage, cur, total) =>
            {
                ct.ThrowIfCancellationRequested();
                job.ReportProgress(stage, cur, total);
            }, ct);
            ct.ThrowIfCancellationRequested();

            if (songs.Jobs.Get(job.Id) is not null) results.CommitWav(job.Id);        // a forgotten job gets no file
            job.Succeed(song.AudioDurationMs, now.GetUtcNow());
        }
        catch (OperationCanceledException)
        {
            DeleteTemp(job);
            job.MarkCancelled(now.GetUtcNow());
        }
        catch (Exception e)
        {
            log.LogError(e, "Job {Id} failed", job.Id);
            DeleteTemp(job);
            job.Fail(e.Message, now.GetUtcNow());
        }
        finally
        {
            current = null;
        }

        try { songs.Publish(job); }                                 // stores the final state and tells the webhook; a job forgotten while it ran leaves nothing
        catch (Exception e) { log.LogError(e, "Could not store the final state of job {Id}", job.Id); }
    }

    private void DeleteTemp(Job job)
    {
        try { File.Delete(songs.Results.WavTempPath(job.Id)); }
        catch (IOException) { }
    }
}
