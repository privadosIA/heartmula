using HeartMuLa.Server.Jobs;

namespace HeartMuLa.Server.Api;

/// <summary>Body of <c>POST /songs</c>. Everything but tags and lyrics is optional.</summary>
/// <param name="Tags">Comma-separated style tags, e.g. "piano,happy".</param>
/// <param name="Lyrics">Lyrics with optional section markers such as [verse].</param>
/// <param name="DurationMs">Maximum length of the song in milliseconds (the model can stop earlier). Default and maximum come from the settings.</param>
/// <param name="Seed">64-bit seed; the same seed with the same inputs and devices gives the same audio. Chosen by the server when omitted.</param>
/// <param name="CfgScale">Classifier-free guidance scale (default 1.5).</param>
/// <param name="Temperature">Sampling temperature, 0.1 to 2.0 (default 1.0).</param>
/// <param name="TopK">Top-k sampling, 1 to 500 (default 50).</param>
/// <param name="CallbackUrl">Absolute http(s) URL notified with a POST when the job ends.</param>
public sealed record CreateSongRequest(string? Tags, string? Lyrics, int? DurationMs, long? Seed, double? CfgScale, string? CallbackUrl, double? Temperature, int? TopK);

/// <summary>What the API returns for a job.</summary>
public sealed record SongResponse(
    string Id,
    JobState State,
    string Tags,
    string Lyrics,
    int DurationMs,
    long Seed,
    double CfgScale,
    double Temperature,
    int TopK,
    string? CallbackUrl,
    DateTimeOffset CreatedAt,
    DateTimeOffset? StartedAt,
    DateTimeOffset? EndedAt,
    JobProgress Progress,
    string? Error,
    int? AudioDurationMs,
    string? AudioUrl,
    WebhookStatus? Webhook)
{
    public static SongResponse From(JobView v) => new(
        v.Id, v.State, v.Request.Tags, v.Request.Lyrics, v.Request.DurationMs, v.Request.Seed, v.Request.CfgScale, v.Request.Temperature, v.Request.TopK, v.Request.CallbackUrl,
        v.CreatedAt, v.StartedAt, v.EndedAt, v.Progress, v.Error, v.AudioDurationMs,
        v.State == JobState.Succeeded ? $"/songs/{v.Id}/audio" : null, v.Webhook);
}

/// <summary>GPU of the model, as the orchestrator's client reads it (an unknown number is null).</summary>
public sealed record GpuInfo([property: System.Text.Json.Serialization.JsonPropertyName("name")] string? Name,
                             [property: System.Text.Json.Serialization.JsonPropertyName("free_mb")] long? FreeMb,
                             [property: System.Text.Json.Serialization.JsonPropertyName("total_mb")] long? TotalMb);

/// <summary>
/// <c>GET /health</c>. Always 200: <c>ready</c> says whether songs can be made. The first six fields are the orchestrator's; the rest are this
/// service's own.
/// </summary>
public sealed record HealthResponse(
    bool Ready, string Version, bool Busy, int Queued,
    [property: System.Text.Json.Serialization.JsonIgnore(Condition = System.Text.Json.Serialization.JsonIgnoreCondition.Never)] GpuInfo? Gpu,
    IReadOnlyList<string> Missing,
    string Status, string Model, string? ModelError, string ModelDevice, string CodecDevice, string? RunningJob);
