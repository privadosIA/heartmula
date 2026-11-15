using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using static HeartMuLa.Server.Tests.WorkerHarness;

namespace HeartMuLa.Server.Tests;

public class HealthTests
{
    private static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();

    [Fact]
    public async Task Health_is_200_with_ready_false_while_the_model_loads_and_jobs_are_still_queued_then_ready_true()
    {
        using var factory = new ApiFactory();
        var gate = factory.Generator.LoadGate = new ManualResetEventSlim();
        using var client = factory.CreateClient();

        var loading = await client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, loading.StatusCode);                           // the orchestrator's client treats 4xx/5xx as an error
        var l = Json(await loading.Content.ReadAsStringAsync());
        Assert.Equal(("loading", false), (l.GetProperty("model").GetString(), l.GetProperty("ready").GetBoolean()));

        var post = await client.PostAsJsonAsync("/songs", new { tags = "a", lyrics = "b" });
        Assert.Equal(HttpStatusCode.Accepted, post.StatusCode);                       // queued meanwhile
        Assert.Equal(1, Json(await (await client.GetAsync("/health")).Content.ReadAsStringAsync()).GetProperty("queued").GetInt32());

        gate.Set();
        await Eventually(() => Json(client.GetStringAsync("/health").Result).GetProperty("ready").GetBoolean());
        var ready = Json(await client.GetStringAsync("/health"));
        Assert.Equal(("ok", "ready", "fake:0", "fake:1"),
                     (ready.GetProperty("status").GetString(), ready.GetProperty("model").GetString(), ready.GetProperty("modelDevice").GetString(), ready.GetProperty("codecDevice").GetString()));
    }

    [Fact]
    public async Task Health_carries_the_orchestrators_fields_with_its_names_and_types()
    {
        using var factory = new ApiFactory();
        factory.Generator.Missing = new[] { "tokenizer.json" };
        using var client = factory.CreateClient();
        await Eventually(() => Json(client.GetStringAsync("/health").Result).GetProperty("model").GetString() == "ready");
        var h = Json(await client.GetStringAsync("/health"));

        Assert.Equal(JsonValueKind.True, h.GetProperty("ready").ValueKind);
        Assert.Equal("1", h.GetProperty("version").GetString());
        Assert.Equal(JsonValueKind.False, h.GetProperty("busy").ValueKind);
        Assert.Equal(0, h.GetProperty("queued").GetInt32());
        Assert.Equal(JsonValueKind.Null, h.GetProperty("gpu").ValueKind);               // not a CUDA device
        Assert.Equal(new[] { "tokenizer.json" }, h.GetProperty("missing").EnumerateArray().Select(e => e.GetString()));
    }

    [Fact]
    public async Task Health_is_200_with_the_reason_when_the_model_failed_to_load()
    {
        using var factory = new ApiFactory();
        factory.Generator.FailLoad = true;
        using var client = factory.CreateClient();
        await Eventually(() => client.GetAsync("/health").Result.Content.ReadAsStringAsync().Result.Contains("failed"));
        var r = await client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        var h = Json(await r.Content.ReadAsStringAsync());
        Assert.False(h.GetProperty("ready").GetBoolean());
        Assert.Contains("cannot read the weights", h.GetProperty("modelError").GetString());
    }

    [Fact]
    public async Task Health_busy_is_true_while_a_song_is_made()
    {
        using var factory = new ApiFactory();
        var release = new ManualResetEventSlim();
        factory.Generator.OnStep = (_, step) => { if (step == 1) release.Wait(5000); };
        using var client = factory.CreateClient();
        await client.PostAsJsonAsync("/songs", new { tags = "a", lyrics = "b" });
        await Eventually(() => Json(client.GetStringAsync("/health").Result).GetProperty("busy").GetBoolean());
        release.Set();
    }

    [Theory]
    [InlineData("NVIDIA GeForce RTX 4070, 11643, 12282\nNVIDIA GeForce RTX 5060 Ti, 15877, 16311\n", 2, "NVIDIA GeForce RTX 5060 Ti", 15877L, 16311L)]
    [InlineData("Some GPU, [N/A], 8192\n", 1, "Some GPU", null, 8192L)]
    public void The_nvidia_smi_output_is_parsed_one_gpu_per_line(string output, int count, string lastName, long? free, long? total)
    {
        var gpus = HeartMuLa.Server.Worker.NvidiaSmiGpuProbe.Parse(output.Replace("\\n", "\n"));
        Assert.Equal(count, gpus.Count);
        Assert.Equal((lastName, free, total), (gpus[^1].Name, gpus[^1].FreeMb, gpus[^1].TotalMb));
    }

    [Fact]
    public void The_probe_returns_null_for_the_cpu_and_the_gpu_serializes_with_the_orchestrators_snake_case_names()
    {
        Assert.Null(new HeartMuLa.Server.Worker.NvidiaSmiGpuProbe().Read("cpu"));
        var json = JsonSerializer.Serialize(new HeartMuLa.Server.Api.GpuInfo("RTX", 100, 200));
        Assert.Equal("{\"name\":\"RTX\",\"free_mb\":100,\"total_mb\":200}", json);
    }

    [Fact]
    public async Task Health_reports_the_running_job()
    {
        using var factory = new ApiFactory();
        var release = new ManualResetEventSlim();
        factory.Generator.OnStep = (_, step) => { if (step == 1) release.Wait(5000); };
        using var client = factory.CreateClient();
        var post = await client.PostAsJsonAsync("/songs", new { tags = "a", lyrics = "b" });
        var id = Json(await post.Content.ReadAsStringAsync()).GetProperty("id").GetString();
        await Eventually(() => Json(client.GetStringAsync("/health").Result).TryGetProperty("runningJob", out var r) && r.GetString() == id);
        release.Set();
    }
}

public class ApiKeyTests
{
    private const string Key = "s3cret";

    [Theory]
    [InlineData(null)]
    [InlineData("wrong")]
    [InlineData("s3cre")]
    [InlineData("s3cret-and-more")]
    public async Task Without_the_right_key_every_song_endpoint_is_401_and_no_job_is_created(string? given)
    {
        using var factory = new ApiFactory(("Security:ApiKey", Key));
        using var client = factory.CreateClient();
        if (given is not null) client.DefaultRequestHeaders.Add("X-Api-Key", given);

        var post = await client.PostAsJsonAsync("/songs", new { tags = "a", lyrics = "b" });
        Assert.Equal(HttpStatusCode.Unauthorized, post.StatusCode);
        Assert.Equal("application/problem+json", post.Content.Headers.ContentType!.MediaType);
        Assert.Equal(HttpStatusCode.Unauthorized, (await client.GetAsync("/songs/" + new string('0', 32))).StatusCode);
        Assert.Equal(HttpStatusCode.Unauthorized, (await client.DeleteAsync("/songs/" + new string('0', 32))).StatusCode);
        Assert.Empty(factory.Services.GetRequiredService<HeartMuLa.Server.Jobs.JobStore>().All());
    }

    [Fact]
    public async Task With_the_right_key_requests_work_and_health_never_needs_it()
    {
        using var factory = new ApiFactory(("Security:ApiKey", Key));
        using var anonymous = factory.CreateClient();
        await Eventually(() => anonymous.GetAsync("/health").Result.StatusCode == HttpStatusCode.OK);   // open without a key

        using var client = factory.CreateClient();
        client.DefaultRequestHeaders.Add("X-Api-Key", Key);
        Assert.Equal(HttpStatusCode.Accepted, (await client.PostAsJsonAsync("/songs", new { tags = "a", lyrics = "b" })).StatusCode);
    }

    [Fact]
    public async Task The_key_is_also_accepted_as_a_bearer_token()
    {
        using var factory = new ApiFactory(("Security:ApiKey", Key));
        using var client = factory.CreateClient();
        client.DefaultRequestHeaders.Add("Authorization", "Bearer " + Key);
        Assert.Equal(HttpStatusCode.Accepted, (await client.PostAsJsonAsync("/songs", new { tags = "a", lyrics = "b" })).StatusCode);
        using var wrong = factory.CreateClient();
        wrong.DefaultRequestHeaders.Add("Authorization", "Bearer nope");
        Assert.Equal(HttpStatusCode.Unauthorized, (await wrong.PostAsJsonAsync("/songs", new { tags = "a", lyrics = "b" })).StatusCode);
    }

    [Fact]
    public async Task Without_a_configured_key_no_authentication_is_applied()
    {
        using var factory = new ApiFactory();
        using var client = factory.CreateClient();
        Assert.Equal(HttpStatusCode.Accepted, (await client.PostAsJsonAsync("/songs", new { tags = "a", lyrics = "b" })).StatusCode);
    }

    [Fact]
    public void The_default_listening_address_is_loopback_only()
    {
        using var factory = new ApiFactory();
        var urls = factory.Services.GetRequiredService<IConfiguration>()["Urls"];
        Assert.Equal("http://127.0.0.1:5080", urls);
        Assert.Equal(HeartMuLa.Server.Options.ServiceOptions.DefaultUrl, urls);
    }
}
