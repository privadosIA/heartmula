using System.Net;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;
using HeartMuLa.Server.Storage;
using static HeartMuLa.Server.Tests.WorkerHarness;

namespace HeartMuLa.Server.Tests;

public sealed class WebhookTests : IAsyncDisposable
{
    private readonly FakeReceiver receiver = new FakeReceiver().StartAsync().Result;
    private readonly List<ApiFactory> factories = new();

    public async ValueTask DisposeAsync()
    {
        foreach (var f in factories) f.Dispose();
        await receiver.DisposeAsync();
    }

    /// <summary>A service whose webhook retries do not wait (4 attempts in all) unless overridden.</summary>
    private (ApiFactory Factory, HttpClient Client) Service(params (string, string)[] extra)
    {
        var settings = new[] { ("Webhook:RetryDelaysSeconds:0", "0"), ("Webhook:RetryDelaysSeconds:1", "0"), ("Webhook:RetryDelaysSeconds:2", "0") }.Concat(extra).ToArray();
        var f = new ApiFactory(settings);
        factories.Add(f);
        return (f, f.CreateClient());
    }

    private static JsonElement Parse(string s) => JsonDocument.Parse(s).RootElement.Clone();

    private static async Task<string> Post(HttpClient c, object body)
    {
        var r = await c.PostAsJsonAsync("/songs", body);
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        return Parse(await r.Content.ReadAsStringAsync()).GetProperty("id").GetString()!;
    }

    private static async Task<JsonElement> Get(HttpClient c, string id) => Parse(await c.GetStringAsync($"/songs/{id}"));

    private static Task WebhookState(HttpClient c, string id, string state) =>
        Eventually(async () => (await Get(c, id)).TryGetProperty("webhook", out var w) && w.GetProperty("state").GetString() == state);

    [Fact]
    public async Task A_succeeded_job_notifies_the_receiver_once_and_is_marked_delivered()
    {
        var (_, c) = Service();
        var id = await Post(c, new { tags = "a", lyrics = "b", seed = 99, callbackUrl = receiver.Url + "/hook" });
        await WebhookState(c, id, "delivered");

        var got = Assert.Single(receiver.Requests);
        Assert.Equal("/hook", got.Path);
        Assert.Equal("song.finished", got.Headers["X-HeartMuLa-Event"]);
        Assert.StartsWith("application/json", got.Headers["Content-Type"]);
        var body = Parse(got.Body);
        Assert.Equal((id, "succeeded", 99L, $"/songs/{id}/audio"), (body.GetProperty("id").GetString(), body.GetProperty("state").GetString(), body.GetProperty("seed").GetInt64(), body.GetProperty("audioUrl").GetString()));
        Assert.True(body.GetProperty("durationMs").GetInt32() > 0);
        Assert.Equal(1, (await Get(c, id)).GetProperty("webhook").GetProperty("attempts").GetInt32());
    }

    [Fact]
    public async Task Failed_and_cancelled_jobs_are_notified_too()
    {
        var (f, c) = Service();
        var release = new ManualResetEventSlim();
        f.Generator.OnStep = (r, step) =>
        {
            if (r.Tags == "bad") throw new InvalidOperationException("CUDA out of memory");
            if (r.Tags == "first" && step == 1) release.Wait(5000);
        };
        var first = await Post(c, new { tags = "first", lyrics = "b" });                          // holds the worker
        var bad = await Post(c, new { tags = "bad", lyrics = "b", callbackUrl = receiver.Url + "/bad" });
        var gone = await Post(c, new { tags = "gone", lyrics = "b", callbackUrl = receiver.Url + "/gone" });
        Assert.Equal(HttpStatusCode.Accepted, (await c.DeleteAsync($"/songs/{gone}")).StatusCode);   // cancelled while queued
        release.Set();
        await WebhookState(c, bad, "delivered");
        await WebhookState(c, gone, "delivered");

        var byPath = receiver.Requests.ToDictionary(r => r.Path, r => Parse(r.Body));
        Assert.Equal("failed", byPath["/bad"].GetProperty("state").GetString());
        Assert.Equal("CUDA out of memory", byPath["/bad"].GetProperty("error").GetString());
        Assert.Equal("cancelled", byPath["/gone"].GetProperty("state").GetString());
        Assert.False(byPath["/gone"].TryGetProperty("audioUrl", out _));
        Assert.NotEmpty(first);
    }

    [Fact]
    public async Task With_a_secret_the_body_is_signed_with_hmac_sha256_and_without_one_it_is_not()
    {
        var (_, signed) = Service(("Webhook:Secret", "topsecret"));
        var id = await Post(signed, new { tags = "a", lyrics = "b", callbackUrl = receiver.Url });
        await WebhookState(signed, id, "delivered");
        var got = receiver.Requests.Single();
        var expected = "sha256=" + Convert.ToHexString(new HMACSHA256(Encoding.UTF8.GetBytes("topsecret")).ComputeHash(Encoding.UTF8.GetBytes(got.Body))).ToLowerInvariant();
        Assert.Equal(expected, got.Headers["X-HeartMuLa-Signature"]);

        var (_, plain) = Service();
        var id2 = await Post(plain, new { tags = "a", lyrics = "b", callbackUrl = receiver.Url });
        await WebhookState(plain, id2, "delivered");
        Assert.False(receiver.Requests.Last().Headers.ContainsKey("X-HeartMuLa-Signature"));
    }

    [Fact]
    public async Task A_receiver_that_fails_twice_gets_the_event_again_until_it_answers_2xx()
    {
        receiver.Behavior = (ctx, n) => { ctx.Response.StatusCode = n < 3 ? 500 : 200; return Task.CompletedTask; };
        var (_, c) = Service();
        var id = await Post(c, new { tags = "a", lyrics = "b", callbackUrl = receiver.Url });
        await WebhookState(c, id, "delivered");
        Assert.Equal(3, receiver.Requests.Count);
        Assert.Equal(3, (await Get(c, id)).GetProperty("webhook").GetProperty("attempts").GetInt32());
    }

    [Fact]
    public async Task A_receiver_that_never_answers_ok_ends_failed_after_four_attempts_and_the_job_is_untouched()
    {
        receiver.Behavior = (ctx, _) => { ctx.Response.StatusCode = 500; return Task.CompletedTask; };
        var (_, c) = Service();
        var id = await Post(c, new { tags = "a", lyrics = "b", callbackUrl = receiver.Url });
        await WebhookState(c, id, "failed");

        var json = await Get(c, id);
        Assert.Equal(4, json.GetProperty("webhook").GetProperty("attempts").GetInt32());
        Assert.Equal("HTTP 500", json.GetProperty("webhook").GetProperty("lastError").GetString());
        Assert.Equal("succeeded", json.GetProperty("state").GetString());
        Assert.Equal(HttpStatusCode.OK, (await c.GetAsync($"/songs/{id}/audio")).StatusCode);
    }

    [Fact]
    public async Task A_redirect_is_not_followed_and_counts_as_a_failed_attempt()
    {
        receiver.Behavior = (ctx, _) => { ctx.Response.StatusCode = 302; ctx.Response.Headers.Location = "/elsewhere"; return Task.CompletedTask; };
        var (_, c) = Service();
        var id = await Post(c, new { tags = "a", lyrics = "b", callbackUrl = receiver.Url + "/hook" });
        await WebhookState(c, id, "failed");
        Assert.DoesNotContain(receiver.Requests, r => r.Path == "/elsewhere");
        Assert.Equal("HTTP 302", (await Get(c, id)).GetProperty("webhook").GetProperty("lastError").GetString());
    }

    [Fact]
    public async Task A_receiver_that_hangs_never_delays_the_next_job()
    {
        receiver.Behavior = async (ctx, _) => await Task.Delay(Timeout.Infinite, ctx.RequestAborted);
        var (_, c) = Service(("Webhook:TimeoutSeconds", "1"));
        var slow = await Post(c, new { tags = "a", lyrics = "b", callbackUrl = receiver.Url });
        var next = await Post(c, new { tags = "next", lyrics = "b" });

        await Eventually(async () => (await Get(c, next)).GetProperty("state").GetString() == "succeeded");
        Assert.Equal("pending", (await Get(c, slow)).GetProperty("webhook").GetProperty("state").GetString());   // still being retried
        await WebhookState(c, slow, "failed");
        Assert.Contains("timeout", (await Get(c, slow)).GetProperty("webhook").GetProperty("lastError").GetString());
    }

    [Fact]
    public async Task A_host_outside_the_allowed_list_is_rejected_at_post_time_and_one_on_it_is_accepted()
    {
        var (_, c) = Service(("Webhook:AllowedHosts:0", "orchestrator.local"));
        var rejected = await c.PostAsJsonAsync("/songs", new { tags = "a", lyrics = "b", callbackUrl = receiver.Url });
        Assert.Equal(HttpStatusCode.BadRequest, rejected.StatusCode);
        Assert.Contains("callbackUrl", await rejected.Content.ReadAsStringAsync());

        var (_, c2) = Service(("Webhook:AllowedHosts:0", "127.0.0.1"));
        Assert.Equal(HttpStatusCode.Accepted, (await c2.PostAsJsonAsync("/songs", new { tags = "a", lyrics = "b", callbackUrl = receiver.Url })).StatusCode);
    }

    [Fact]
    public async Task Without_a_callback_nothing_is_sent_and_the_job_has_no_webhook_object()
    {
        var (_, c) = Service();
        var id = await Post(c, new { tags = "a", lyrics = "b" });
        await Eventually(() => Get(c, id).Result.GetProperty("state").GetString() == "succeeded");
        await Task.Delay(150);
        Assert.Empty(receiver.Requests);
        Assert.False((await Get(c, id)).TryGetProperty("webhook", out _));
    }

    [Fact]
    public async Task Deleting_a_finished_result_sends_no_event()
    {
        var (_, c) = Service();
        var id = await Post(c, new { tags = "a", lyrics = "b", callbackUrl = receiver.Url });
        await WebhookState(c, id, "delivered");
        Assert.Equal(HttpStatusCode.NoContent, (await c.DeleteAsync($"/songs/{id}")).StatusCode);
        await Task.Delay(150);
        Assert.Single(receiver.Requests);
    }

    [Fact]
    public async Task Deliveries_pending_at_startup_are_resumed_and_unfinished_jobs_are_reported_failed_to_their_receiver()
    {
        // The state a previous run left behind: a finished job whose notification never went out, and a job that was running when the service stopped.
        var dir = Path.Combine(Path.GetTempPath(), "heartmula-resume-" + Guid.NewGuid().ToString("N"));
        var store = new ResultStore(new StorageOptions { Directory = dir }, TimeProvider.System);
        var done = new Job(JobStore.NewId(), new SongRequest("a", "b", 1000, 5, 1.5, receiver.Url + "/done"), DateTimeOffset.UtcNow);
        done.TryStart(DateTimeOffset.UtcNow);
        File.WriteAllBytes(store.WavTempPath(done.Id), FakeSongGenerator.WavBytes(200)); store.CommitWav(done.Id);
        done.Succeed(200, DateTimeOffset.UtcNow);
        store.Save(done.View());
        var interrupted = new Job(JobStore.NewId(), new SongRequest("a", "b", 1000, 6, 1.5, receiver.Url + "/interrupted"), DateTimeOffset.UtcNow);
        interrupted.TryStart(DateTimeOffset.UtcNow);
        store.Save(interrupted.View());

        var factory = new ApiFactory(dir, ("Webhook:RetryDelaysSeconds:0", "0"), ("Webhook:RetryDelaysSeconds:1", "0"), ("Webhook:RetryDelaysSeconds:2", "0"));
        factories.Add(factory);
        var c = factory.CreateClient();
        await WebhookState(c, done.Id, "delivered");
        await WebhookState(c, interrupted.Id, "delivered");

        var byPath = receiver.Requests.ToDictionary(r => r.Path, r => Parse(r.Body));
        Assert.Equal("succeeded", byPath["/done"].GetProperty("state").GetString());
        Assert.Equal(("failed", "interrupted by restart"), (byPath["/interrupted"].GetProperty("state").GetString(), byPath["/interrupted"].GetProperty("error").GetString()));
        Assert.Equal(2, receiver.Requests.Count);                                   // each exactly once
    }
}
