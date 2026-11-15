using System.Collections.Concurrent;

namespace HeartMuLa.Server.Jobs;

/// <summary>In-memory index of the jobs. The queue limit is enforced here, counting jobs that are still <see cref="JobState.Queued"/>.</summary>
public sealed class JobStore
{
    private readonly ConcurrentDictionary<string, Job> jobs = new();
    private readonly object admission = new();

    public static string NewId() => Guid.NewGuid().ToString("N");

    /// <summary>Ids are 32 lowercase hex characters: anything else is not a job and never reaches the file system.</summary>
    public static bool IsValidId(string? id) => id is { Length: 32 } && id.All(c => c is (>= '0' and <= '9') or (>= 'a' and <= 'f'));

    public int QueuedCount => jobs.Values.Count(j => j.State == JobState.Queued);

    public Job? Running => jobs.Values.FirstOrDefault(j => j.State == JobState.Running);

    /// <summary>Adds a new queued job unless <paramref name="maxQueued"/> jobs are already waiting.</summary>
    public bool TryAdd(Job job, int maxQueued)
    {
        lock (admission)
        {
            if (QueuedCount >= maxQueued) return false;
            return jobs.TryAdd(job.Id, job);
        }
    }

    /// <summary>Adds a job restored from disk (no limit).</summary>
    public void Restore(Job job) => jobs[job.Id] = job;

    public Job? Get(string id) => IsValidId(id) && jobs.TryGetValue(id, out var j) ? j : null;

    public bool Remove(string id) => jobs.TryRemove(id, out _);

    public IReadOnlyList<Job> All() => jobs.Values.ToList();
}
