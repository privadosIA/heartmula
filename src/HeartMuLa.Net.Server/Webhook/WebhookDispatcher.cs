using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;
using HeartMuLa.Server.Storage;

namespace HeartMuLa.Server.Webhook;

/// <summary>Body of the notification sent to a job's <c>callbackUrl</c>.</summary>
public sealed record WebhookPayload(string Id, JobState State, long Seed, int? DurationMs, string? Error,
                                    DateTimeOffset CreatedAt, DateTimeOffset? StartedAt, DateTimeOffset? EndedAt, string? AudioUrl);

/// <summary>
/// Sends one signed JSON POST per finished job that has a <c>callbackUrl</c>. It has its own queue and runs every delivery in the background, so a slow or
/// dead receiver never touches the generation worker. A delivery is 1 attempt plus one retry per configured delay; a 2xx answer counts as delivered
/// (redirects are not followed). The outcome is stored with the job; deliveries still pending at startup are resumed.
/// </summary>
public sealed class WebhookDispatcher : BackgroundService, IJobEndedSink
{
    /// <summary>Name of the request header carrying the event name.</summary>
    public const string EventHeader = "X-HeartMuLa-Event";
    /// <summary>Name of the request header carrying the HMAC signature (sent only when <c>Webhook:Secret</c> is set).</summary>
    public const string SignatureHeader = "X-HeartMuLa-Signature";
    /// <summary>Value of the event header: the event sent when a job has ended.</summary>
    public const string EventName = "song.finished";
    private const int MaxConcurrentDeliveries = 8;

    private static readonly JsonSerializerOptions BodyJson = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
    };

    private readonly Channel<Job> queue = Channel.CreateUnbounded<Job>();
    private readonly JobStore store;
    private readonly ResultStore results;
    private readonly WebhookOptions options;
    private readonly HttpClient http;
    private readonly ILogger<WebhookDispatcher> log;

    /// <summary>Creates the dispatcher; redirects are never followed.</summary>
    /// <param name="store">Index of the jobs, used to resume pending deliveries and to skip deleted jobs.</param>
    /// <param name="results">Result store where the delivery outcome is saved.</param>
    /// <param name="options">Service options; the <c>Webhook</c> section is used.</param>
    /// <param name="log">Logger.</param>
    public WebhookDispatcher(JobStore store, ResultStore results, ServiceOptions options, ILogger<WebhookDispatcher> log)
    {
        this.store = store; this.results = results; this.options = options.Webhook; this.log = log;
        // Redirects are never followed: a receiver cannot send the service somewhere else.
        http = new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = false, UseProxy = false }) { Timeout = Timeout.InfiniteTimeSpan };
    }

    /// <summary>Queues a notification when the job has a <c>callbackUrl</c>; returns at once.</summary>
    /// <param name="job">The job that ended.</param>
    public void JobEnded(Job job)
    {
        if (job.Request.CallbackUrl is not null) queue.Writer.TryWrite(job);
    }

    /// <summary>Resumes deliveries left pending by the previous run, then delivers queued notifications in the background, at most 8 at a time, until the service stops.</summary>
    /// <param name="stopping">Signalled when the service is shutting down.</param>
    protected override async Task ExecuteAsync(CancellationToken stopping)
    {
        // Resume what the previous run left pending (jobs restored from disk by Maintenance, which starts first).
        foreach (var job in store.All())
        {
            var v = job.View();
            if (Job.IsFinal(v.State) && v.Webhook is { State: WebhookState.Pending }) queue.Writer.TryWrite(job);
        }

        using var slots = new SemaphoreSlim(MaxConcurrentDeliveries);
        var running = new List<Task>();
        try
        {
            await foreach (var job in queue.Reader.ReadAllAsync(stopping))
            {
                await slots.WaitAsync(stopping);
                running.Add(Task.Run(async () =>
                {
                    try { await DeliverAsync(job, stopping); }
                    catch (OperationCanceledException) { /* shutting down: stays pending, resumed at the next start */ }
                    catch (Exception e) { log.LogError(e, "Webhook delivery for job {Id} crashed", job.Id); }
                    finally { slots.Release(); }
                }, CancellationToken.None));
                running.RemoveAll(t => t.IsCompleted);
            }
        }
        catch (OperationCanceledException) { }
        await Task.WhenAll(running);
    }

    internal async Task DeliverAsync(Job job, CancellationToken stopping)
    {
        var view = job.View();
        var url = view.Request.CallbackUrl!;
        var body = JsonSerializer.SerializeToUtf8Bytes(new WebhookPayload(
            view.Id, view.State, view.Request.Seed, view.AudioDurationMs, view.Error, view.CreatedAt, view.StartedAt, view.EndedAt,
            view.State == JobState.Succeeded ? $"/songs/{view.Id}/audio" : null), BodyJson);

        int attempts = view.Webhook?.Attempts ?? 0;
        string? lastError = view.Webhook?.LastError;
        for (int n = 0; n <= options.EffectiveRetryDelaysSeconds.Length; n++)
        {
            if (n > 0) await Task.Delay(TimeSpan.FromSeconds(options.EffectiveRetryDelaysSeconds[n - 1]), stopping);
            if (store.Get(job.Id) is null) return;                                  // deleted meanwhile: nothing to report

            attempts++;
            lastError = await TryOnce(url, body, stopping);
            bool delivered = lastError is null;
            bool last = n == options.EffectiveRetryDelaysSeconds.Length;

            job.SetWebhook(new WebhookStatus(delivered ? WebhookState.Delivered : last ? WebhookState.Failed : WebhookState.Pending, attempts, lastError));
            if (store.Get(job.Id) is null) return;
            results.Save(job.View());
            if (delivered) return;
            log.LogWarning("Webhook for job {Id} failed (attempt {Attempt}): {Error}", job.Id, attempts, lastError);
        }
    }

    /// <summary>One attempt. Null when the receiver answered 2xx, otherwise the reason.</summary>
    private async Task<string?> TryOnce(string url, byte[] body, CancellationToken stopping)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(stopping);
        timeout.CancelAfter(TimeSpan.FromSeconds(options.TimeoutSeconds));
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, url) { Content = new ByteArrayContent(body) };
            request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            request.Headers.Add(EventHeader, EventName);
            if (!string.IsNullOrEmpty(options.Secret)) request.Headers.Add(SignatureHeader, Sign(options.Secret, body));

            using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
            return response.IsSuccessStatusCode ? null : $"HTTP {(int)response.StatusCode}";
        }
        catch (OperationCanceledException) when (stopping.IsCancellationRequested) { throw; }
        catch (OperationCanceledException) { return $"timeout after {options.TimeoutSeconds} s"; }
        catch (HttpRequestException e) { return e.Message; }
    }

    /// <summary><c>sha256=</c> followed by the hex HMAC-SHA256 of the body.</summary>
    public static string Sign(string secret, byte[] body) =>
        "sha256=" + Convert.ToHexString(HMACSHA256.HashData(Encoding.UTF8.GetBytes(secret), body)).ToLowerInvariant();

    /// <summary>Releases the HTTP client.</summary>
    public override void Dispose()
    {
        http.Dispose();
        base.Dispose();
    }
}
