using System.Text.Json;
using System.Text.Json.Serialization;

namespace HeartMuLa.Server.Jobs;

/// <summary>Enums travel as lower-case words ("queued", "running"): the API contract and the sidecar files use them.</summary>
public sealed class LowerCaseEnumConverter<T> : JsonConverter<T> where T : struct, Enum
{
    public override T Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
        Enum.TryParse<T>(reader.GetString(), ignoreCase: true, out var v) ? v : throw new JsonException($"Unknown {typeof(T).Name}: '{reader.GetString()}'");

    public override void Write(Utf8JsonWriter writer, T value, JsonSerializerOptions options) => writer.WriteStringValue(value.ToString().ToLowerInvariant());
}

[JsonConverter(typeof(LowerCaseEnumConverter<JobState>))]
public enum JobState { Queued, Running, Succeeded, Failed, Cancelled }

[JsonConverter(typeof(LowerCaseEnumConverter<JobStage>))]
public enum JobStage { Queued, Loading, Generating, Decoding, Done }

[JsonConverter(typeof(LowerCaseEnumConverter<WebhookState>))]
public enum WebhookState { Pending, Delivered, Failed }

/// <summary>What the client asked for, with every default already applied (the seed included).</summary>
public sealed record SongRequest(string Tags, string Lyrics, int DurationMs, long Seed, double CfgScale, string? CallbackUrl,
                                 double Temperature = 1.0, int TopK = 50);

public sealed record JobProgress(JobStage Stage, int Current, int Total);

public sealed record WebhookStatus(WebhookState State, int Attempts, string? LastError);

/// <summary>Immutable picture of a job: what the API returns and what the sidecar file stores.</summary>
public sealed record JobView(
    string Id,
    JobState State,
    SongRequest Request,
    DateTimeOffset CreatedAt,
    DateTimeOffset? StartedAt,
    DateTimeOffset? EndedAt,
    JobProgress Progress,
    string? Error,
    int? AudioDurationMs,
    WebhookStatus? Webhook);

/// <summary>
/// One song job. Every state change goes through a method that enforces the allowed transitions
/// (queued → running → succeeded | failed | cancelled, and queued → cancelled). Once a job is running only the worker moves it:
/// <see cref="RequestCancel"/> on a running job just signals <see cref="Cancellation"/> and the worker ends it with <see cref="MarkCancelled"/>.
/// </summary>
public sealed class Job
{
    private readonly object gate = new();
    private JobState state = JobState.Queued;
    private JobProgress progress = new(JobStage.Queued, 0, 0);
    private DateTimeOffset? startedAt, endedAt;
    private string? error;
    private int? audioDurationMs;
    private WebhookStatus? webhook;

    public string Id { get; }
    public SongRequest Request { get; }
    public DateTimeOffset CreatedAt { get; }
    public CancellationTokenSource Cancellation { get; } = new();

    public Job(string id, SongRequest request, DateTimeOffset createdAt)
    {
        Id = id;
        Request = request;
        CreatedAt = createdAt;
        if (request.CallbackUrl is not null) webhook = new WebhookStatus(WebhookState.Pending, 0, null);
    }

    /// <summary>Rebuilds a job from its stored picture (startup scan).</summary>
    public static Job FromView(JobView v) => new(v.Id, v.Request, v.CreatedAt)
    {
        state = v.State,
        progress = v.Progress,
        startedAt = v.StartedAt,
        endedAt = v.EndedAt,
        error = v.Error,
        audioDurationMs = v.AudioDurationMs,
        webhook = v.Webhook,
    };

    public static bool IsFinal(JobState s) => s is JobState.Succeeded or JobState.Failed or JobState.Cancelled;

    public JobView View()
    {
        lock (gate) return new JobView(Id, state, Request, CreatedAt, startedAt, endedAt, progress, error, audioDurationMs, webhook);
    }

    public JobState State { get { lock (gate) return state; } }

    /// <summary>Queued → running. False when the job is no longer queued (it was cancelled while waiting).</summary>
    public bool TryStart(DateTimeOffset now)
    {
        lock (gate)
        {
            if (state != JobState.Queued) return false;
            state = JobState.Running;
            startedAt = now;
            progress = new JobProgress(JobStage.Generating, 0, 0);
            return true;
        }
    }

    public void ReportProgress(JobStage stage, int current, int total)
    {
        lock (gate)
        {
            if (state != JobState.Running) throw new InvalidOperationException($"Progress can only be reported while running (job is {state})");
            progress = new JobProgress(stage, current, total);
        }
    }

    public void Succeed(int durationMs, DateTimeOffset now) => Finish(JobState.Succeeded, now, null, durationMs);
    public void Fail(string message, DateTimeOffset now) => Finish(JobState.Failed, now, message, null);

    /// <summary>Ends the job as cancelled: from running (the worker, after unwinding) or from queued.</summary>
    public void MarkCancelled(DateTimeOffset now) => Finish(JobState.Cancelled, now, null, null);

    /// <summary>
    /// Cancels a queued job at once, or signals a running one (the worker ends it). False when the job is already final.
    /// </summary>
    public bool RequestCancel(DateTimeOffset now)
    {
        lock (gate)
        {
            if (IsFinal(state)) return false;
            if (state == JobState.Queued)
            {
                Finalize(JobState.Cancelled, now, null, null);
                return true;
            }
        }
        Cancellation.Cancel();
        return true;
    }

    /// <summary>Marks a job found unfinished at startup as failed.</summary>
    public void Interrupt(DateTimeOffset now) => Finish(JobState.Failed, now, "interrupted by restart", null, allowFromQueued: true);

    public void SetWebhook(WebhookStatus status)
    {
        lock (gate)
        {
            if (Request.CallbackUrl is null) throw new InvalidOperationException("The job has no callbackUrl");
            webhook = status;
        }
    }

    private void Finish(JobState target, DateTimeOffset now, string? message, int? durationMs, bool allowFromQueued = false)
    {
        lock (gate)
        {
            bool fromRunning = state == JobState.Running;
            bool fromQueued = state == JobState.Queued && (target == JobState.Cancelled || allowFromQueued);
            if (!fromRunning && !fromQueued) throw new InvalidOperationException($"A job cannot go from {state} to {target}");
            Finalize(target, now, message, durationMs);
        }
    }

    private void Finalize(JobState target, DateTimeOffset now, string? message, int? durationMs)
    {
        state = target;
        endedAt = now;
        error = message;
        audioDurationMs = durationMs;
        progress = new JobProgress(target == JobState.Succeeded ? JobStage.Done : progress.Stage, progress.Current, progress.Total);
    }
}
