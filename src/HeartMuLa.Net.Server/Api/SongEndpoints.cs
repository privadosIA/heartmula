using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;
using HeartMuLa.Server.Worker;
using MusicGen.Core;

namespace HeartMuLa.Server.Api;

public static class SongEndpoints
{
    public static void MapSongEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/songs", CreateSong)
            .WithName("CreateSong").WithTags("Songs")
            .WithSummary("Queue a song")
            .WithDescription("Creates a job and answers at once. Follow it with GET /songs/{id}, or give a callbackUrl to be notified when it ends.")
            .Produces<SongResponse>(StatusCodes.Status202Accepted)
            .ProducesValidationProblem()
            .ProducesProblem(StatusCodes.Status429TooManyRequests)
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        app.MapGet("/songs/{id}", GetSong)
            .WithName("GetSong").WithTags("Songs")
            .WithSummary("State and progress of a job")
            .Produces<SongResponse>()
            .ProducesProblem(StatusCodes.Status404NotFound)
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        app.MapGet("/songs/{id}/audio", GetAudio)
            .WithName("GetSongAudio").WithTags("Songs")
            .WithSummary("Download the finished song as a WAV")
            .Produces<byte[]>(StatusCodes.Status200OK, "audio/wav")
            .ProducesProblem(StatusCodes.Status404NotFound)
            .ProducesProblem(StatusCodes.Status409Conflict)
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        app.MapDelete("/songs/{id}", DeleteSong)
            .WithName("DeleteSong").WithTags("Songs")
            .WithSummary("Cancel a queued or running job, or remove a finished one")
            .Produces<SongResponse>(StatusCodes.Status202Accepted)
            .Produces(StatusCodes.Status204NoContent)
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        app.MapGet("/health", Health)
            .WithName("Health").WithTags("Service")
            .WithSummary("Model state, devices and queue (always 200; see `ready`)")
            .Produces<HealthResponse>();
    }

    private static IResult CreateSong(CreateSongRequest? body, SongService songs, ServiceOptions options, HttpContext http)
    {
        var errors = Validate(body, options);
        if (errors.Count > 0) return Results.ValidationProblem(errors);

        var request = new SongRequest(
            Tags: body!.Tags!.Trim(),
            Lyrics: body.Lyrics!,
            DurationMs: body.DurationMs ?? options.Jobs.DefaultDurationMs,
            Seed: body.Seed ?? Seeding.NewSeed(),
            CfgScale: body.CfgScale ?? options.Jobs.DefaultCfgScale,
            Temperature: body.Temperature ?? 1.0,
            TopK: body.TopK ?? 50,
            CallbackUrl: string.IsNullOrWhiteSpace(body.CallbackUrl) ? null : body.CallbackUrl!.Trim());

        var job = songs.Submit(request);
        if (job is null)
        {
            http.Response.Headers.RetryAfter = "30";
            return Results.Problem(title: "The queue is full", detail: $"{options.Jobs.MaxQueued} jobs are already waiting; try again later.",
                                   statusCode: StatusCodes.Status429TooManyRequests);
        }

        return Results.Accepted($"/songs/{job.Id}", SongResponse.From(job.View()));
    }

    private static Dictionary<string, string[]> Validate(CreateSongRequest? body, ServiceOptions options)
    {
        var errors = new Dictionary<string, string[]>();
        if (body is null) return new() { ["body"] = new[] { "A JSON body is required." } };
        if (string.IsNullOrWhiteSpace(body.Tags)) errors["tags"] = new[] { "tags is required." };
        if (string.IsNullOrWhiteSpace(body.Lyrics)) errors["lyrics"] = new[] { "lyrics is required." };
        if (body.DurationMs is { } d && (d < 1000 || d > options.Jobs.MaxDurationMs))
            errors["durationMs"] = new[] { $"durationMs must be between 1000 and {options.Jobs.MaxDurationMs}." };
        if (body.CfgScale is { } c && !(c > 0)) errors["cfgScale"] = new[] { "cfgScale must be greater than 0." };
        if (body.Temperature is { } t && !(t >= 0.1 && t <= 2.0)) errors["temperature"] = new[] { "temperature must be between 0.1 and 2.0." };
        if (body.TopK is { } k && (k < 1 || k > 500)) errors["topK"] = new[] { "topK must be between 1 and 500." };
        if (!string.IsNullOrWhiteSpace(body.CallbackUrl))
        {
            if (!Uri.TryCreate(body.CallbackUrl.Trim(), UriKind.Absolute, out var uri) || uri.Scheme is not ("http" or "https"))
                errors["callbackUrl"] = new[] { "callbackUrl must be an absolute http or https URL." };
            else if (!options.IsWebhookHostAllowed(uri.Host))
                errors["callbackUrl"] = new[] { $"The host '{uri.Host}' is not allowed by the service settings." };
        }
        return errors;
    }

    private static IResult GetSong(string id, SongService songs) =>
        songs.Jobs.Get(id) is { } job ? Results.Ok(SongResponse.From(job.View())) : NotFound(id);

    private static IResult GetAudio(string id, SongService songs)
    {
        var job = songs.Jobs.Get(id);
        if (job is null) return NotFound(id);
        var state = job.State;
        if (state != JobState.Succeeded || !songs.Results.HasWav(id))
            return Results.Problem(title: "The song is not available", detail: $"The job is {state.ToString().ToLowerInvariant()}.", statusCode: StatusCodes.Status409Conflict);
        return Results.File(songs.Results.WavPath(id), "audio/wav", fileDownloadName: $"{id}.wav", enableRangeProcessing: true);
    }

    private static IResult DeleteSong(string id, SongService songs)
    {
        var outcome = songs.CancelOrRemove(id);
        return outcome switch
        {
            CancelOutcome.Cancelling or CancelOutcome.Cancelled => Results.Accepted($"/songs/{id}", SongResponse.From(songs.Jobs.Get(id)!.View())),
            _ => Results.NoContent(),                                                  // removed, or already gone: DELETE is idempotent
        };
    }

    private static IResult Health(SongService songs, ISongGenerator generator, IGpuProbe gpu, ServiceOptions options)
    {
        var running = songs.Jobs.Running;
        var status = generator.Status;
        var devices = generator.Devices;
        var body = new HealthResponse(
            Ready: status == ModelStatus.Ready,
            Version: options.Service.Version,
            Busy: running is not null,
            Queued: songs.Jobs.QueuedCount,
            Gpu: gpu.Read(devices.Model),
            Missing: generator.Missing,
            Status: status == ModelStatus.Ready ? "ok" : "unavailable",
            Model: status == ModelStatus.Ready && generator.LoadsPerSong ? "on-demand" : status.ToString().ToLowerInvariant(),
            ModelError: generator.LoadError,
            ModelDevice: devices.Model,
            CodecDevice: devices.Codec,
            RunningJob: running?.Id);
        return Results.Ok(body);                                  // always 200: `ready` carries the news, as the orchestrator's client expects
    }

    private static IResult NotFound(string id) =>
        Results.Problem(title: "Song not found", detail: $"There is no job '{id}' (it never existed or its result expired).", statusCode: StatusCodes.Status404NotFound);
}
