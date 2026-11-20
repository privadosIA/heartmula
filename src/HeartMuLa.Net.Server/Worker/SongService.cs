using System.Threading.Channels;
using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;
using HeartMuLa.Server.Storage;
using HeartMuLa.Server.Webhook;

namespace HeartMuLa.Server.Worker;

/// <summary>What <see cref="SongService.CancelOrRemove"/> did with a job.</summary>
public enum CancelOutcome
{
    /// <summary>There is no job with that id.</summary>
    NotFound,
    /// <summary>The job is running; it has been signalled and the worker will end it.</summary>
    Cancelling,
    /// <summary>The job was queued and is now cancelled.</summary>
    Cancelled,
    /// <summary>The job had finished; it and its stored result were removed.</summary>
    Removed,
}

/// <summary>What the endpoints use: submit, look up, cancel / remove. Owns the queue the worker reads.</summary>
public sealed class SongService
{
    private readonly Channel<Job> queue = Channel.CreateUnbounded<Job>(new UnboundedChannelOptions { SingleReader = true });
    private readonly JobStore store;
    private readonly ResultStore results;
    private readonly IJobEndedSink ended;
    private readonly ServiceOptions options;
    private readonly TimeProvider time;
    private readonly object gate = new();   // makes "store the finished job" and "forget the job" exclude each other

    /// <summary>Creates the service over the job store, the result store, the webhook sink, the settings and the clock.</summary>
    public SongService(JobStore store, ResultStore results, IJobEndedSink ended, ServiceOptions options, TimeProvider time)
    {
        this.store = store; this.results = results; this.ended = ended; this.options = options; this.time = time;
    }

    /// <summary>The reading side of the queue, for the generation worker.</summary>
    public ChannelReader<Job> Reader => queue.Reader;
    /// <summary>The store of jobs.</summary>
    public JobStore Jobs => store;
    /// <summary>The store of results (sidecar files and WAVs) on disk.</summary>
    public ResultStore Results => results;
    /// <summary>The clock used for job timestamps.</summary>
    public TimeProvider Time => time;

    /// <summary>Creates the job and queues it. Null when the queue is full.</summary>
    public Job? Submit(SongRequest request)
    {
        var job = new Job(JobStore.NewId(), request, time.GetUtcNow());
        if (!store.TryAdd(job, options.Jobs.MaxQueued)) return null;
        results.Save(job.View());
        queue.Writer.TryWrite(job);
        return job;
    }

    /// <summary>
    /// DELETE semantics: a queued job is cancelled at once; a running one is signalled and the worker ends it; a finished one has its stored result removed.
    /// </summary>
    public CancelOutcome CancelOrRemove(string id)
    {
        var job = store.Get(id);
        if (job is null) return CancelOutcome.NotFound;

        var before = job.State;
        if (Job.IsFinal(before))
        {
            store.Remove(id);
            results.Delete(id);
            return CancelOutcome.Removed;
        }

        job.RequestCancel(time.GetUtcNow());
        if (before == JobState.Queued)
        {
            Publish(job);
            return CancelOutcome.Cancelled;
        }
        return CancelOutcome.Cancelling;
    }

    /// <summary>
    /// Forgets a job (the orchestrator's DELETE): cancels it if it is queued or running, removes it from the store and deletes its files. A running
    /// generation that ends later finds the job gone and leaves nothing behind. False when there is no such job.
    /// </summary>
    public bool Forget(string id)
    {
        var job = store.Get(id);
        if (job is null) return false;
        lock (gate) store.Remove(id);           // first: the worker checks the store before it saves anything
        job.RequestCancel(time.GetUtcNow());
        lock (gate) results.Delete(id);
        return true;
    }

    /// <summary>
    /// The worker's last step for a job that reached a final state: store its sidecar and tell the webhook sink, unless the job was forgotten meanwhile, in
    /// which case everything it left is deleted. Atomic with <see cref="Forget"/>, so a forget that races with the end of the job never leaves a file behind.
    /// </summary>
    public void Publish(Job job)
    {
        lock (gate)
        {
            if (store.Get(job.Id) is null)
            {
                results.Delete(job.Id);
                return;
            }
            results.Save(job.View());
        }
        ended.JobEnded(job);
    }

    /// <summary>Completes the queue: no more jobs can be queued and the reader finishes once it has read the ones left.</summary>
    public void Complete() => queue.Writer.TryComplete();
}
