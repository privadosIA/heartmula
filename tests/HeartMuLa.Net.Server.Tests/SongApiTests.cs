using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HeartMuLa.Server.Worker;
using static HeartMuLa.Server.Tests.WorkerHarness;

namespace HeartMuLa.Server.Tests;

public sealed class SongApiTests : IDisposable
{
    private readonly ApiFactory factory = new();
    private HttpClient Client => client ??= factory.CreateClient();
    private HttpClient? client;

    public void Dispose() { client?.Dispose(); factory.Dispose(); }

    private static readonly object Valid = new { tags = "piano,happy", lyrics = "[verse]\nla la" };

    private async Task<JsonElement> Post(object body, HttpStatusCode expected = HttpStatusCode.Accepted)
    {
        var r = await Client.PostAsJsonAsync("/songs", body);
        Assert.Equal(expected, r.StatusCode);
        return JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement.Clone();
    }

    private async Task<JsonElement> Get(string id) =>
        JsonDocument.Parse(await Client.GetStringAsync($"/songs/{id}")).RootElement.Clone();

    private Task WaitState(string id, string state) => Eventually(() => Get(id).Result.GetProperty("state").GetString() == state);

    [Fact]
    public async Task A_valid_request_is_accepted_with_an_id_a_location_and_state_queued()
    {
        factory.Generator.OnStep = (_, _) => Thread.Sleep(20);
        var r = await Client.PostAsJsonAsync("/songs", Valid);
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        var json = JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement;
        var id = json.GetProperty("id").GetString()!;
        Assert.Matches("^[0-9a-f]{32}$", id);
        Assert.Equal($"/songs/{id}", r.Headers.Location!.OriginalString);
        Assert.Equal("queued", json.GetProperty("state").GetString());
    }

    [Fact]
    public async Task Omitted_options_get_the_defaults_and_are_reported_back()
    {
        var json = await Post(Valid);
        Assert.Equal(30_000, json.GetProperty("durationMs").GetInt32());
        Assert.Equal(1.5, json.GetProperty("cfgScale").GetDouble());
        Assert.Equal((1.0, 50), (json.GetProperty("temperature").GetDouble(), json.GetProperty("topK").GetInt32()));
        Assert.True(json.TryGetProperty("seed", out _));
        var explicitSeed = await Post(new { tags = "a", lyrics = "b", seed = 42, durationMs = 5000, cfgScale = 2.0 });
        Assert.Equal((42L, 5000, 2.0), (explicitSeed.GetProperty("seed").GetInt64(), explicitSeed.GetProperty("durationMs").GetInt32(), explicitSeed.GetProperty("cfgScale").GetDouble()));
    }

    [Theory]
    [InlineData("{\"lyrics\":\"x\"}", "tags")]
    [InlineData("{\"tags\":\"  \",\"lyrics\":\"x\"}", "tags")]
    [InlineData("{\"tags\":\"a\"}", "lyrics")]
    [InlineData("{\"tags\":\"a\",\"lyrics\":\"x\",\"durationMs\":500}", "durationMs")]
    [InlineData("{\"tags\":\"a\",\"lyrics\":\"x\",\"durationMs\":999999999}", "durationMs")]
    [InlineData("{\"tags\":\"a\",\"lyrics\":\"x\",\"cfgScale\":0}", "cfgScale")]
    [InlineData("{\"tags\":\"a\",\"lyrics\":\"x\",\"temperature\":0.01}", "temperature")]
    [InlineData("{\"tags\":\"a\",\"lyrics\":\"x\",\"temperature\":2.5}", "temperature")]
    [InlineData("{\"tags\":\"a\",\"lyrics\":\"x\",\"topK\":0}", "topK")]
    [InlineData("{\"tags\":\"a\",\"lyrics\":\"x\",\"topK\":501}", "topK")]
    [InlineData("{\"tags\":\"a\",\"lyrics\":\"x\",\"callbackUrl\":\"ftp://x/y\"}", "callbackUrl")]
    [InlineData("{\"tags\":\"a\",\"lyrics\":\"x\",\"callbackUrl\":\"not a url\"}", "callbackUrl")]
    public async Task An_invalid_body_is_a_400_that_names_the_field_and_creates_no_job(string body, string field)
    {
        var r = await Client.PostAsync("/songs", new StringContent(body, System.Text.Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.BadRequest, r.StatusCode);
        Assert.Equal("application/problem+json", r.Content.Headers.ContentType!.MediaType);
        var errors = JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement.GetProperty("errors");
        Assert.True(errors.TryGetProperty(field, out _), $"no error for {field}");
        var health = JsonDocument.Parse(await Client.GetStringAsync("/health")).RootElement;
        Assert.Equal(0, health.GetProperty("queued").GetInt32());
    }

    [Fact]
    public async Task A_full_queue_is_a_429_with_retry_after_and_creates_no_job()
    {
        using var small = new ApiFactory(("Jobs:MaxQueued", "1"));
        var release = new ManualResetEventSlim();
        small.Generator.OnStep = (_, step) => { if (step == 1) release.Wait(5000); };
        var c = small.CreateClient();

        var first = await c.PostAsJsonAsync("/songs", Valid);
        var firstId = JsonDocument.Parse(await first.Content.ReadAsStringAsync()).RootElement.GetProperty("id").GetString();
        await Eventually(() => c.GetStringAsync($"/songs/{firstId}").Result.Contains("\"running\""));
        Assert.Equal(HttpStatusCode.Accepted, (await c.PostAsJsonAsync("/songs", Valid)).StatusCode);     // waits in the queue
        var rejected = await c.PostAsJsonAsync("/songs", Valid);

        Assert.Equal(HttpStatusCode.TooManyRequests, rejected.StatusCode);
        Assert.True(rejected.Headers.RetryAfter is not null);
        release.Set();
    }

    [Fact]
    public async Task A_running_job_shows_its_progress_and_an_unknown_id_is_404()
    {
        var release = new ManualResetEventSlim();
        factory.Generator.Steps = 5;
        factory.Generator.OnStep = (_, step) => { if (step == 3) release.Wait(5000); };
        var id = (await Post(Valid)).GetProperty("id").GetString()!;
        await Eventually(() => Get(id).Result.GetProperty("progress").GetProperty("current").GetInt32() == 2);

        var json = await Get(id);
        Assert.Equal("running", json.GetProperty("state").GetString());
        Assert.Equal("generating", json.GetProperty("progress").GetProperty("stage").GetString());
        Assert.Equal(5, json.GetProperty("progress").GetProperty("total").GetInt32());
        release.Set();

        Assert.Equal(HttpStatusCode.NotFound, (await Client.GetAsync("/songs/" + new string('0', 32))).StatusCode);
        var bad = await Client.GetAsync("/songs/..%2F..%2Fetc");
        Assert.Equal(HttpStatusCode.NotFound, bad.StatusCode);
        Assert.Equal("application/problem+json", bad.Content.Headers.ContentType!.MediaType);
    }

    [Fact]
    public async Task The_audio_of_a_finished_job_is_a_wav_and_before_that_it_is_409()
    {
        var release = new ManualResetEventSlim();
        factory.Generator.OnStep = (_, step) => { if (step == 1) release.Wait(5000); };
        var id = (await Post(new { tags = "a", lyrics = "b", durationMs = 1000 })).GetProperty("id").GetString()!;

        var early = await Client.GetAsync($"/songs/{id}/audio");
        Assert.Equal(HttpStatusCode.Conflict, early.StatusCode);
        var earlyText = await early.Content.ReadAsStringAsync();
        Assert.True(earlyText.Contains("queued") || earlyText.Contains("running"), earlyText);          // the problem body reports the state

        release.Set();
        await WaitState(id, "succeeded");
        var audio = await Client.GetAsync($"/songs/{id}/audio");
        Assert.Equal(HttpStatusCode.OK, audio.StatusCode);
        Assert.Equal("audio/wav", audio.Content.Headers.ContentType!.MediaType);
        var bytes = await audio.Content.ReadAsByteArrayAsync();
        Assert.Equal("RIFF", System.Text.Encoding.ASCII.GetString(bytes, 0, 4));
        Assert.Equal(2, BitConverter.ToInt16(bytes, 22));                       // channels
        Assert.Equal(48000, BitConverter.ToInt32(bytes, 24));                   // sample rate
        int frames = (bytes.Length - 44) / 4;
        Assert.InRange(Math.Abs(frames * 1000.0 / 48000 - (await Get(id)).GetProperty("audioDurationMs").GetInt32()), 0, 80);
    }

    [Fact]
    public async Task Cancelling_a_running_job_is_202_and_it_ends_cancelled_without_audio()
    {
        factory.Generator.Steps = 1_000_000;
        factory.Generator.OnStep = (_, _) => Thread.Sleep(2);
        var id = (await Post(Valid)).GetProperty("id").GetString()!;
        await WaitState(id, "running");

        var del = await Client.DeleteAsync($"/songs/{id}");
        Assert.Equal(HttpStatusCode.Accepted, del.StatusCode);
        await WaitState(id, "cancelled");
        Assert.Equal(HttpStatusCode.Conflict, (await Client.GetAsync($"/songs/{id}/audio")).StatusCode);
    }

    [Fact]
    public async Task Removing_a_finished_job_is_204_then_everything_about_it_is_404_and_deleting_again_is_still_204()
    {
        var id = (await Post(new { tags = "a", lyrics = "b", durationMs = 1000 })).GetProperty("id").GetString()!;
        await WaitState(id, "succeeded");

        Assert.Equal(HttpStatusCode.NoContent, (await Client.DeleteAsync($"/songs/{id}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await Client.GetAsync($"/songs/{id}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await Client.GetAsync($"/songs/{id}/audio")).StatusCode);
        Assert.Equal(HttpStatusCode.NoContent, (await Client.DeleteAsync($"/songs/{id}")).StatusCode);
    }

    [Fact]
    public async Task A_failed_job_reports_its_error_and_the_audio_is_409()
    {
        factory.Generator.OnStep = (_, _) => throw new InvalidOperationException("CUDA out of memory");
        var id = (await Post(Valid)).GetProperty("id").GetString()!;
        await WaitState(id, "failed");
        Assert.Equal("CUDA out of memory", (await Get(id)).GetProperty("error").GetString());
        Assert.Equal(HttpStatusCode.Conflict, (await Client.GetAsync($"/songs/{id}/audio")).StatusCode);
    }
}
