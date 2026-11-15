using System.Text.Json;
using System.Text.Json.Serialization;
using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;

namespace HeartMuLa.Server.Storage;

/// <summary>
/// Results on disk: <c>{id}.wav</c> (written as <c>{id}.wav.tmp</c> and renamed when complete, so a crash never leaves a truncated song that looks finished)
/// and <c>{id}.json</c> with the job's picture (request, state, webhook delivery).
/// </summary>
public sealed class ResultStore
{
    internal static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private readonly string directory;
    private readonly TimeSpan retention;
    private readonly TimeProvider time;

    public ResultStore(StorageOptions options, TimeProvider time)
    {
        directory = Path.GetFullPath(options.Directory);
        retention = TimeSpan.FromHours(options.RetentionHours);
        this.time = time;
        Directory.CreateDirectory(directory);
    }

    public string DirectoryPath => directory;
    public string WavPath(string id) => Path.Combine(directory, id + ".wav");
    public string WavTempPath(string id) => Path.Combine(directory, id + ".wav.tmp");
    private string SidecarPath(string id) => Path.Combine(directory, id + ".json");

    /// <summary>Atomically replaces the sidecar of a job.</summary>
    public void Save(JobView view)
    {
        var tmp = SidecarPath(view.Id) + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(view, Json));
        File.Move(tmp, SidecarPath(view.Id), overwrite: true);
    }

    /// <summary>Makes the finished WAV visible (rename of the temporary file).</summary>
    public void CommitWav(string id) => File.Move(WavTempPath(id), WavPath(id), overwrite: true);

    public bool HasWav(string id) => File.Exists(WavPath(id));

    /// <summary>Deletes everything stored for a job.</summary>
    public void Delete(string id)
    {
        foreach (var p in new[] { WavPath(id), WavTempPath(id), SidecarPath(id), SidecarPath(id) + ".tmp" })
            if (File.Exists(p)) File.Delete(p);
    }

    /// <summary>
    /// Startup scan. Finished jobs come back as they were; jobs that were queued or running become <c>failed</c> ("interrupted by restart") and their sidecar is
    /// rewritten; a succeeded job whose WAV is gone is dropped; leftover temporary files are deleted. Unreadable sidecars are ignored.
    /// </summary>
    public IReadOnlyList<Job> LoadAll()
    {
        var jobs = new List<Job>();
        foreach (var tmp in System.IO.Directory.EnumerateFiles(directory, "*.tmp")) File.Delete(tmp);
        foreach (var path in System.IO.Directory.EnumerateFiles(directory, "*.json"))
        {
            JobView? view;
            try { view = JsonSerializer.Deserialize<JobView>(File.ReadAllText(path), Json); }
            catch (JsonException) { continue; }
            if (view is null || !JobStore.IsValidId(view.Id) || Path.GetFileNameWithoutExtension(path) != view.Id) continue;

            var job = Job.FromView(view);
            if (!Job.IsFinal(view.State))
            {
                job.Interrupt(time.GetUtcNow());
                Save(job.View());
            }
            else if (view.State == JobState.Succeeded && !HasWav(view.Id))
            {
                Delete(view.Id);
                continue;
            }
            jobs.Add(job);
        }
        return jobs;
    }

    /// <summary>Deletes the results of final jobs that ended more than the retention ago and returns their ids.</summary>
    public IReadOnlyList<string> Sweep(IEnumerable<Job> jobs)
    {
        var now = time.GetUtcNow();
        var expired = new List<string>();
        foreach (var job in jobs)
        {
            var v = job.View();
            if (Job.IsFinal(v.State) && v.EndedAt is { } ended && now - ended > retention)
            {
                Delete(v.Id);
                expired.Add(v.Id);
            }
        }
        return expired;
    }
}
