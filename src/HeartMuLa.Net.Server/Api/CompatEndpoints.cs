using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Worker;

namespace HeartMuLa.Server.Api;

/// <summary>
/// The orchestrator's HTTP contract (the one its shared music-service client speaks to the HeartMuLa and DiffRhythm2 containers), served over the
/// same <see cref="SongService"/> as the service's own API: <c>POST /generate</c>, <c>GET /jobs/{id}</c>, <c>GET /jobs/{id}/audio</c>,
/// <c>DELETE /jobs/{id}</c>. Validation, limits, defaults, messages and status codes follow the container service; only the JSON differs from
/// <c>/songs</c>. <c>GET /health</c> is shared (see <see cref="SongEndpoints"/>).
/// </summary>
public static class CompatEndpoints
{
    /// <summary>Largest request body, in bytes, that <c>POST /generate</c> accepts (larger ones get 413).</summary>
    public const int MaxBodyBytes = 1_000_000;
    // The methods each route does not serve (they answer "not found", not 405, as the container does).
    private static readonly string[] GenerateOtherMethods = { "GET", "PUT", "PATCH", "DELETE" };
    private static readonly string[] HealthOtherMethods = { "POST", "PUT", "PATCH", "DELETE" };
    private static readonly string[] JobOtherMethods = { "POST", "PUT", "PATCH" };
    private static readonly string[] AudioOtherMethods = { "POST", "PUT", "PATCH", "DELETE" };
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private static readonly Regex HexId = new("^[0-9a-f]+$", RegexOptions.Compiled, RegexTimeout);
    private static readonly Regex CommaSpaces = new(@"\s*,\s*", RegexOptions.Compiled, RegexTimeout);

    /// <summary>Maps the orchestrator-compatible routes (<c>/generate</c>, <c>/jobs/{id}</c>, <c>/jobs/{id}/audio</c>), the not-found answers for other methods and paths, and a trailing-slash normalisation.</summary>
    public static void MapCompatEndpoints(this WebApplication app)
    {
        // The container accepts a trailing slash.
        app.Use((context, next) =>
        {
            var p = context.Request.Path;
            if (p.HasValue && p.Value!.Length > 1 && p.Value.EndsWith('/') && IsCompatPath(p)) context.Request.Path = p.Value.TrimEnd('/');
            return next(context);
        });

        app.MapPost("/generate", Generate).ExcludeFromDescription();
        app.MapGet("/jobs/{id}", GetJob).ExcludeFromDescription();
        app.MapGet("/jobs/{id}/audio", GetAudio).ExcludeFromDescription();
        app.MapDelete("/jobs/{id}", Forget).ExcludeFromDescription();

        // Any other method on these paths, and any other path, is "not found" (not 405) as in the container.
        app.MapMethods("/generate", GenerateOtherMethods, NotFound).ExcludeFromDescription();
        app.MapMethods("/health", HealthOtherMethods, NotFound).ExcludeFromDescription();
        app.MapMethods("/jobs/{id}", JobOtherMethods, (string id) => NotFound()).ExcludeFromDescription();
        app.MapMethods("/jobs/{id}/audio", AudioOtherMethods, (string id) => NotFound()).ExcludeFromDescription();
        app.MapFallback(NotFound).ExcludeFromDescription();
    }

    private static bool IsCompatPath(PathString p) =>
        p.StartsWithSegments("/generate") || p.StartsWithSegments("/jobs") || p.StartsWithSegments("/health");

    private static IResult NotFound() => Results.Json(new { error = "not found" }, statusCode: StatusCodes.Status404NotFound);
    private static IResult Unknown() => Results.Json(new { error = "unknown job" }, statusCode: StatusCodes.Status404NotFound);
    private static IResult Error(int status, string message) => Results.Json(new { error = message }, statusCode: status);

    // ------------------------------------------------------------------ POST /generate

    private static async Task<IResult> Generate(HttpContext http, SongService songs)
    {
        if (http.Request.ContentLength > MaxBodyBytes) return Error(413, "the request is too large");
        var body = await ReadBody(http.Request);
        if (body is null) return Error(413, "the request is too large");

        var (request, problem) = Validate(body);
        if (request is null) return Error(400, problem!);

        var job = songs.Submit(request);
        if (job is null)
        {
            http.Response.Headers.RetryAfter = "30";
            return Error(429, "the queue is full");
        }
        return Results.Json(new { id = job.Id }, statusCode: StatusCodes.Status202Accepted);
    }

    /// <summary>The body, or null when it is larger than the limit (a chunked body has no length to look at beforehand).</summary>
    private static async Task<byte[]?> ReadBody(HttpRequest request)
    {
        using var buffer = new MemoryStream();
        var chunk = new byte[16 * 1024];
        int n;
        while ((n = await request.Body.ReadAsync(chunk)) > 0)
        {
            await buffer.WriteAsync(chunk.AsMemory(0, n));
            if (buffer.Length > MaxBodyBytes) return null;
        }
        return buffer.ToArray();
    }

    private static readonly (string Key, double Lo, double Hi, double Default, string LoText, string HiText, bool Whole)[] Limits =
    {
        ("max_ms", 5_000, 360_000, 240_000, "5000", "360000", true),
        ("temperature", 0.1, 2.0, 1.0, "0.1", "2.0", false),
        ("topk", 1, 500, 50, "1", "500", true),
        ("cfg_scale", 1.0, 10.0, 3.0, "1.0", "10.0", false),
    };

    /// <summary>Turns a body into the song request, or says why not. The messages are the container's.</summary>
    public static (SongRequest? Request, string? Problem) Validate(byte[] body)
    {
        JsonElement root;
        try { root = JsonDocument.Parse(body.Length == 0 ? "{}"u8.ToArray() : body).RootElement; }
        catch (JsonException e) { return (null, e.Message); }
        if (root.ValueKind != JsonValueKind.Object) return (null, "the body must be a JSON object");

        if (!root.TryGetProperty("lyrics", out var lyricsEl) || lyricsEl.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(lyricsEl.GetString()))
            return (null, "lyrics must be a non-empty text");
        var lyrics = lyricsEl.GetString()!;
        if (lyrics.Length > 20_000) return (null, "lyrics are too long (20000 characters at most)");

        string tags = "";
        if (root.TryGetProperty("tags", out var tagsEl))
        {
            if (tagsEl.ValueKind == JsonValueKind.Array) tags = string.Join(",", tagsEl.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.String ? e.GetString() : e.ToString()));
            else if (tagsEl.ValueKind == JsonValueKind.String) tags = tagsEl.GetString()!;
            else return (null, "tags must be a text or a list of words");
        }
        tags = CommaSpaces.Replace(tags.Trim().ToLowerInvariant(), ",").Trim(',');
        if (tags.Length == 0) return (null, "tags must name at least one style");

        var values = new double[Limits.Length];
        for (int i = 0; i < Limits.Length; i++)
        {
            var (key, lo, hi, def, loText, hiText, whole) = Limits[i];
            double value = def;
            if (root.TryGetProperty(key, out var el))
            {
                if (el.ValueKind != JsonValueKind.Number || !el.TryGetDouble(out value)) return (null, $"{key} must be a number");
                if (value < lo || value > hi) return (null, $"{key} must be between {loText} and {hiText}");
                if (whole) value = Math.Truncate(value);
            }
            values[i] = value;
        }

        long seed = 0;
        if (root.TryGetProperty("seed", out var seedEl) && (seedEl.ValueKind != JsonValueKind.Number || !IsWhole(seedEl, out seed) || seed < 0))
            return (null, "seed must be a whole number, 0 or more");

        return (new SongRequest(tags, lyrics, (int)values[0], seed, values[3], null, values[1], (int)values[2]), null);
    }

    private static bool IsWhole(JsonElement number, out long value)
    {
        value = 0;
        var raw = number.GetRawText();
        return !raw.Contains('.') && !raw.Contains('e') && !raw.Contains('E') && long.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
    }

    // ------------------------------------------------------------------ jobs

    /// <summary>The state in the orchestrator's words: queued, running, done, failed.</summary>
    public static string CompatState(JobState s) => s switch
    {
        JobState.Queued => "queued",
        JobState.Running => "running",
        JobState.Succeeded => "done",
        _ => "failed",                                              // failed, and cancelled through the service's own API
    };

    private static IResult GetJob(string id, SongService songs)
    {
        if (!HexId.IsMatch(id)) return NotFound();
        if (songs.Jobs.Get(id) is not { } job) return Unknown();
        var v = job.View();
        var end = v.EndedAt ?? songs.Time.GetUtcNow();
        return Results.Json(new
        {
            id = v.Id,
            state = CompatState(v.State),
            progress = new { done = v.Progress.Current, total = v.Progress.Total },
            error = v.State == JobState.Cancelled ? "cancelled" : v.Error ?? "",
            seconds = v.StartedAt is { } started ? Math.Round((end - started).TotalSeconds, 1) : 0,
        });
    }

    private static IResult GetAudio(string id, SongService songs)
    {
        if (!HexId.IsMatch(id)) return NotFound();
        if (songs.Jobs.Get(id) is not { } job) return Unknown();
        var state = CompatState(job.State);
        if (state != "done") return Error(409, $"the job is {state}");
        if (!songs.Results.HasWav(id)) return Error(410, "the audio is gone");
        return Results.File(songs.Results.WavPath(id), "audio/wav", enableRangeProcessing: true);
    }

    private static IResult Forget(string id, SongService songs)
    {
        if (!HexId.IsMatch(id)) return NotFound();
        return songs.Forget(id) ? Results.Json(new { ok = true }) : Unknown();
    }
}
