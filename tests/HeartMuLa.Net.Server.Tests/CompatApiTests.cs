using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using static HeartMuLa.Server.Tests.WorkerHarness;

namespace HeartMuLa.Server.Tests;

/// <summary>The orchestrator's contract (what its shared music-service client sends and expects), on the fake generator.</summary>
public sealed class CompatApiTests : IDisposable
{
    private readonly ApiFactory factory = new();
    private HttpClient? client;
    private HttpClient C => client ??= factory.CreateClient();
    public void Dispose() { client?.Dispose(); factory.Dispose(); }

    private static readonly object Song = new { lyrics = "[verse]\nla la", tags = "piano,happy" };
    private static JsonElement Parse(string s) => JsonDocument.Parse(s).RootElement.Clone();

    private async Task<(HttpStatusCode Status, JsonElement Body)> Send(HttpMethod method, string path, object? body = null, HttpClient? c = null)
    {
        var r = await (c ?? C).SendAsync(new HttpRequestMessage(method, path) { Content = body is null ? null : JsonContent.Create(body) });
        return (r.StatusCode, Parse(await r.Content.ReadAsStringAsync()));
    }

    private async Task<string> Generate(object? body = null)
    {
        var (status, json) = await Send(HttpMethod.Post, "/generate", body ?? Song);
        Assert.Equal(HttpStatusCode.Accepted, status);
        return json.GetProperty("id").GetString()!;
    }

    private Task<JsonElement> Job(string id) => Send(HttpMethod.Get, $"/jobs/{id}").ContinueWith(t => t.Result.Body);
    private Task WaitState(string id, string state) => Eventually(() => Job(id).Result.GetProperty("state").GetString() == state);

    [Fact]
    public async Task A_song_goes_from_generate_to_jobs_to_audio_to_delete()
    {
        var id = await Generate();
        Assert.Matches("^[0-9a-f]+$", id);
        await WaitState(id, "done");

        var job = await Job(id);
        Assert.Equal(("done", ""), (job.GetProperty("state").GetString(), job.GetProperty("error").GetString()));
        Assert.True(job.GetProperty("progress").GetProperty("total").GetInt32() > 0);
        Assert.True(job.GetProperty("seconds").GetDouble() >= 0);

        var audio = await C.GetAsync($"/jobs/{id}/audio");
        Assert.Equal(HttpStatusCode.OK, audio.StatusCode);
        Assert.Equal("audio/wav", audio.Content.Headers.ContentType!.MediaType);
        var bytes = await audio.Content.ReadAsByteArrayAsync();
        Assert.Equal("RIFF", Encoding.ASCII.GetString(bytes, 0, 4));
        Assert.Equal(bytes.Length, audio.Content.Headers.ContentLength);

        var del = await Send(HttpMethod.Delete, $"/jobs/{id}");
        Assert.Equal((HttpStatusCode.OK, true), (del.Status, del.Body.GetProperty("ok").GetBoolean()));
        var gone = await Send(HttpMethod.Get, $"/jobs/{id}");
        Assert.Equal((HttpStatusCode.NotFound, "unknown job"), (gone.Status, gone.Body.GetProperty("error").GetString()));
    }

    [Fact]
    public async Task Omitted_options_get_the_containers_defaults_and_the_job_is_the_same_one_through_both_apis()
    {
        var id = await Generate();
        var native = Parse(await C.GetStringAsync($"/songs/{id}"));
        Assert.Equal((240_000, 3.0, 1.0, 50, 0L), (native.GetProperty("durationMs").GetInt32(), native.GetProperty("cfgScale").GetDouble(),
            native.GetProperty("temperature").GetDouble(), native.GetProperty("topK").GetInt32(), native.GetProperty("seed").GetInt64()));
        Assert.Equal("piano,happy", native.GetProperty("tags").GetString());
    }

    [Fact]
    public async Task Given_options_are_used_and_fractional_whole_numbers_are_truncated()
    {
        var id = await Generate(new { lyrics = "x", tags = "a", max_ms = 5000.9, temperature = 0.7, topk = 40.7, cfg_scale = 2.5, seed = 123456789012 });
        var n = Parse(await C.GetStringAsync($"/songs/{id}"));
        Assert.Equal((5000, 0.7, 40, 2.5, 123456789012L), (n.GetProperty("durationMs").GetInt32(), n.GetProperty("temperature").GetDouble(),
            n.GetProperty("topK").GetInt32(), n.GetProperty("cfgScale").GetDouble(), n.GetProperty("seed").GetInt64()));
    }

    [Theory]
    [InlineData("{\"lyrics\":\"x\",\"tags\":[\"Piano\",\" Happy \"]}", "piano,happy")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"  Rock , Guitar ,\"}", "rock,guitar")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\",Pop,\"}", "pop")]
    public async Task Tags_are_normalised_like_the_container_does(string body, string expected)
    {
        var r = await C.PostAsync("/generate", new StringContent(body, Encoding.UTF8, "application/json"));
        var id = Parse(await r.Content.ReadAsStringAsync()).GetProperty("id").GetString();
        Assert.Equal(expected, Parse(await C.GetStringAsync($"/songs/{id}")).GetProperty("tags").GetString());
    }

    [Theory]
    [InlineData("{\"tags\":\"a\"}", "lyrics must be a non-empty text")]
    [InlineData("{\"lyrics\":\"   \",\"tags\":\"a\"}", "lyrics must be a non-empty text")]
    [InlineData("{\"lyrics\":5,\"tags\":\"a\"}", "lyrics must be a non-empty text")]
    [InlineData("{\"lyrics\":\"x\"}", "tags must name at least one style")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\" , \"}", "tags must name at least one style")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":7}", "tags must be a text or a list of words")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":null}", "tags must be a text or a list of words")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"max_ms\":4999}", "max_ms must be between 5000 and 360000")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"max_ms\":360001}", "max_ms must be between 5000 and 360000")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"temperature\":0.05}", "temperature must be between 0.1 and 2.0")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"topk\":0}", "topk must be between 1 and 500")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"cfg_scale\":11}", "cfg_scale must be between 1.0 and 10.0")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"cfg_scale\":\"3\"}", "cfg_scale must be a number")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"topk\":true}", "topk must be a number")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"seed\":-1}", "seed must be a whole number, 0 or more")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"seed\":5.0}", "seed must be a whole number, 0 or more")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"seed\":true}", "seed must be a whole number, 0 or more")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"seed\":\"7\"}", "seed must be a whole number, 0 or more")]
    [InlineData("{\"lyrics\":\"x\",\"tags\":\"a\",\"seed\":9223372036854775808}", "seed must be a whole number, 0 or more")]
    [InlineData("[1,2]", "the body must be a JSON object")]
    public async Task An_invalid_field_is_a_400_with_the_containers_message_and_no_job(string body, string message)
    {
        var r = await C.PostAsync("/generate", new StringContent(body, Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.BadRequest, r.StatusCode);
        Assert.Equal(message, Parse(await r.Content.ReadAsStringAsync()).GetProperty("error").GetString());
        Assert.Equal(0, Parse(await C.GetStringAsync("/health")).GetProperty("queued").GetInt32());
    }

    [Fact]
    public async Task Lyrics_over_20000_characters_are_refused()
    {
        var (status, body) = await Send(HttpMethod.Post, "/generate", new { lyrics = new string('a', 20_001), tags = "a" });
        Assert.Equal(HttpStatusCode.BadRequest, status);
        Assert.Equal("lyrics are too long (20000 characters at most)", body.GetProperty("error").GetString());
        Assert.Equal(HttpStatusCode.Accepted, (await Send(HttpMethod.Post, "/generate", new { lyrics = new string('a', 20_000), tags = "a" })).Status);
    }

    [Fact]
    public async Task Not_json_is_a_400_an_empty_body_asks_for_lyrics_and_a_body_over_a_million_bytes_is_a_413()
    {
        var notJson = await C.PostAsync("/generate", new StringContent("not json", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.BadRequest, notJson.StatusCode);
        Assert.True(Parse(await notJson.Content.ReadAsStringAsync()).TryGetProperty("error", out _));

        var empty = await C.PostAsync("/generate", new StringContent("", Encoding.UTF8, "application/json"));
        Assert.Equal("lyrics must be a non-empty text", Parse(await empty.Content.ReadAsStringAsync()).GetProperty("error").GetString());

        var big = await C.PostAsync("/generate", new StringContent("{\"lyrics\":\"" + new string('a', 1_000_000) + "\",\"tags\":\"a\"}", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.RequestEntityTooLarge, big.StatusCode);
        Assert.Equal("the request is too large", Parse(await big.Content.ReadAsStringAsync()).GetProperty("error").GetString());
    }

    [Fact]
    public async Task A_full_queue_is_a_429_with_the_containers_body_and_retry_after()
    {
        using var small = new ApiFactory(("Jobs:MaxQueued", "1"));
        var release = new ManualResetEventSlim();
        small.Generator.OnStep = (_, step) => { if (step == 1) release.Wait(5000); };
        var c = small.CreateClient();
        var first = (await Send(HttpMethod.Post, "/generate", Song, c)).Body.GetProperty("id").GetString();
        await Eventually(() => Send(HttpMethod.Get, $"/jobs/{first}", null, c).Result.Body.GetProperty("state").GetString() == "running");
        Assert.Equal(HttpStatusCode.Accepted, (await Send(HttpMethod.Post, "/generate", Song, c)).Status);

        var r = await c.PostAsJsonAsync("/generate", Song);
        Assert.Equal(HttpStatusCode.TooManyRequests, r.StatusCode);
        Assert.True(r.Headers.RetryAfter is not null);
        Assert.Equal("the queue is full", Parse(await r.Content.ReadAsStringAsync()).GetProperty("error").GetString());
        release.Set();
    }

    [Fact]
    public async Task The_job_view_follows_the_job_queued_running_failed_and_cancelled_through_the_native_api()
    {
        var release = new ManualResetEventSlim();
        factory.Generator.Steps = 5;
        factory.Generator.OnStep = (r, step) =>
        {
            if (r.Tags == "bad") throw new InvalidOperationException("CUDA out of memory");
            if (r.Tags == "first" && step == 3) release.Wait(5000);
        };
        var first = await Generate(new { lyrics = "x", tags = "first" });
        var queued = await Generate(new { lyrics = "x", tags = "second" });
        var bad = await Generate(new { lyrics = "x", tags = "bad" });
        await Eventually(() => Job(first).Result.GetProperty("progress").GetProperty("done").GetInt32() == 2);

        var running = await Job(first);
        Assert.Equal(("running", 5), (running.GetProperty("state").GetString(), running.GetProperty("progress").GetProperty("total").GetInt32()));
        var q = await Job(queued);
        Assert.Equal(("queued", 0.0), (q.GetProperty("state").GetString(), q.GetProperty("seconds").GetDouble()));

        Assert.Equal(HttpStatusCode.Accepted, (await C.DeleteAsync($"/songs/{queued}")).StatusCode);      // cancelled through the native API
        release.Set();
        await WaitState(bad, "failed");
        var failed = await Job(bad);
        Assert.Equal("CUDA out of memory", failed.GetProperty("error").GetString());
        var cancelled = await Job(queued);
        Assert.Equal(("failed", "cancelled"), (cancelled.GetProperty("state").GetString(), cancelled.GetProperty("error").GetString()));
    }

    [Fact]
    public async Task Audio_is_409_with_the_state_before_the_job_is_done_and_410_when_the_file_is_gone()
    {
        var release = new ManualResetEventSlim();
        factory.Generator.OnStep = (r, step) =>
        {
            if (r.Tags == "bad") throw new InvalidOperationException("boom");
            if (step == 1) release.Wait(5000);
        };
        var id = await Generate();
        var early = await C.GetAsync($"/jobs/{id}/audio");
        Assert.Equal(HttpStatusCode.Conflict, early.StatusCode);
        Assert.Matches("^the job is (queued|running)$", Parse(await early.Content.ReadAsStringAsync()).GetProperty("error").GetString()!);

        release.Set();
        await WaitState(id, "done");
        File.Delete(Path.Combine(factory.Dir, id + ".wav"));
        var gone = await C.GetAsync($"/jobs/{id}/audio");
        Assert.Equal(HttpStatusCode.Gone, gone.StatusCode);
        Assert.Equal("the audio is gone", Parse(await gone.Content.ReadAsStringAsync()).GetProperty("error").GetString());

        var bad = await Generate(new { lyrics = "x", tags = "bad" });
        await WaitState(bad, "failed");
        Assert.Equal("the job is failed", Parse(await (await C.GetAsync($"/jobs/{bad}/audio")).Content.ReadAsStringAsync()).GetProperty("error").GetString());
    }

    [Fact]
    public async Task Delete_forgets_a_running_job_which_stops_and_leaves_nothing_behind()
    {
        factory.Generator.Steps = 1_000_000;
        factory.Generator.OnStep = (_, _) => Thread.Sleep(2);
        var id = await Generate();
        await WaitState(id, "running");

        Assert.Equal(HttpStatusCode.OK, (await Send(HttpMethod.Delete, $"/jobs/{id}")).Status);
        Assert.Equal(HttpStatusCode.NotFound, (await Send(HttpMethod.Get, $"/jobs/{id}")).Status);
        Assert.Equal(HttpStatusCode.NotFound, (await C.GetAsync($"/songs/{id}")).StatusCode);          // forgotten through both APIs
        await Eventually(() => Parse(C.GetStringAsync("/health").Result).GetProperty("busy").GetBoolean() == false);   // the generation stopped
        Assert.Empty(Directory.GetFiles(factory.Dir));
    }

    [Theory]
    [InlineData("GET", "/nope")]
    [InlineData("PUT", "/generate")]
    [InlineData("GET", "/generate")]
    [InlineData("POST", "/jobs/abc")]
    [InlineData("POST", "/health")]
    [InlineData("GET", "/jobs/XYZ")]                                  // not lower-case hex: matches no route
    [InlineData("GET", "/jobs/ABC123")]
    public async Task Anything_else_is_not_found_in_the_containers_shape(string method, string path)
    {
        var (status, body) = await Send(new HttpMethod(method), path);
        Assert.Equal((HttpStatusCode.NotFound, "not found"), (status, body.GetProperty("error").GetString()));
    }

    [Fact]
    public async Task An_unknown_hex_job_id_is_unknown_job_for_every_verb()
    {
        foreach (var (method, path) in new[] { ("GET", "/jobs/abc123"), ("GET", "/jobs/abc123/audio"), ("DELETE", "/jobs/abc123") })
        {
            var (status, body) = await Send(new HttpMethod(method), path);
            Assert.Equal((HttpStatusCode.NotFound, "unknown job"), (status, body.GetProperty("error").GetString()));
        }
    }

    [Fact]
    public async Task A_trailing_slash_is_accepted_on_every_route()
    {
        Assert.Equal(HttpStatusCode.OK, (await C.GetAsync("/health/")).StatusCode);
        var (status, body) = await Send(HttpMethod.Post, "/generate/", Song);
        Assert.Equal(HttpStatusCode.Accepted, status);
        var id = body.GetProperty("id").GetString();
        Assert.Equal(HttpStatusCode.OK, (await C.GetAsync($"/jobs/{id}/")).StatusCode);
        await WaitState(id!, "done");
        Assert.Equal(HttpStatusCode.OK, (await C.GetAsync($"/jobs/{id}/audio/")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await C.DeleteAsync($"/jobs/{id}/")).StatusCode);
    }

    [Fact]
    public async Task With_a_key_the_container_routes_answer_401_in_their_own_shape_and_accept_the_bearer_token()
    {
        using var secured = new ApiFactory(("Security:ApiKey", "s3cret"));
        using var anonymous = secured.CreateClient();
        var r = await anonymous.PostAsJsonAsync("/generate", Song);
        Assert.Equal(HttpStatusCode.Unauthorized, r.StatusCode);
        Assert.Equal("a bearer token is required", Parse(await r.Content.ReadAsStringAsync()).GetProperty("error").GetString());
        Assert.Equal(HttpStatusCode.Unauthorized, (await anonymous.GetAsync("/jobs/abc")).StatusCode);

        using var bearer = secured.CreateClient();
        bearer.DefaultRequestHeaders.Add("Authorization", "Bearer s3cret");
        Assert.Equal(HttpStatusCode.Accepted, (await bearer.PostAsJsonAsync("/generate", Song)).StatusCode);
        using var wrong = secured.CreateClient();
        wrong.DefaultRequestHeaders.Add("Authorization", "Bearer nope");
        Assert.Equal(HttpStatusCode.Unauthorized, (await wrong.PostAsJsonAsync("/generate", Song)).StatusCode);
    }
}
