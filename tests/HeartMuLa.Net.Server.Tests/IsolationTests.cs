using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HeartMuLa.Server.Options;
using HeartMuLa.Server.Worker;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using static HeartMuLa.Server.Tests.WorkerHarness;

namespace HeartMuLa.Server.Tests;

public class IsolationSettingTests
{
    private static IConfiguration Cfg(params (string, string?)[] v) =>
        new ConfigurationBuilder().AddInMemoryCollection(v.Select(x => new KeyValuePair<string, string?>(x.Item1, x.Item2))).Build();

    [Fact]
    public void Codec_is_the_default() => Assert.Equal(IsolationMode.Codec, ServiceOptions.From(Cfg()).IsolationMode);

    [Theory]
    [InlineData("process", IsolationMode.Process)]
    [InlineData("Codec", IsolationMode.Codec)]
    [InlineData("INPROCESS", IsolationMode.InProcess)]
    public void The_three_values_are_understood_whatever_their_case(string text, IsolationMode expected)
        => Assert.Equal(expected, ServiceOptions.From(Cfg(("Service:Isolation", text))).IsolationMode);

    [Fact]
    public void The_environment_alias_is_a_fallback_and_the_services_own_setting_wins()
    {
        Assert.Equal(IsolationMode.InProcess, ServiceOptions.From(Cfg(("HEARTMULA_ISOLATION", "inprocess"))).IsolationMode);
        Assert.Equal(IsolationMode.Codec, ServiceOptions.From(Cfg(("Service:Isolation", "codec"), ("HEARTMULA_ISOLATION", "inprocess"))).IsolationMode);
        Assert.Equal(IsolationMode.InProcess, ServiceOptions.From(Cfg(("Service:Isolation", ""), ("HEARTMULA_ISOLATION", "inprocess"))).IsolationMode);   // appsettings may ship it empty
    }

    [Fact]
    public void An_unknown_value_stops_startup_with_a_message_naming_where_it_came_from()
    {
        Assert.Contains("Service:Isolation", Assert.Throws<ConfigurationException>(() => ServiceOptions.From(Cfg(("Service:Isolation", "thread")))).Message);
        var alias = Assert.Throws<ConfigurationException>(() => ServiceOptions.From(Cfg(("HEARTMULA_ISOLATION", "thread"))));
        Assert.Contains("HEARTMULA_ISOLATION", alias.Message);
        Assert.Contains("process, codec, inprocess", alias.Message);
        Assert.Throws<ConfigurationException>(() => ServiceHost.Build(new[] { "--HEARTMULA_ISOLATION=thread" }));
    }
}

/// <summary>The service in <c>process</c> isolation with a stub worker script in place of the real one: no model, no GPU.</summary>
public sealed class ProcessModeServiceTests : IDisposable
{
    private readonly string dir = Path.Combine(Path.GetTempPath(), "heartmula-iso-" + Guid.NewGuid().ToString("N"));
    private readonly List<ApiFactory> factories = new();

    public ProcessModeServiceTests()
    {
        Directory.CreateDirectory(Path.Combine(dir, "ckpt", "HeartCodec-oss"));
        Directory.CreateDirectory(Path.Combine(dir, "ckpt", "HeartMuLa-oss-3B"));
        foreach (var f in new[] { "gen_config.json", "tokenizer.json" }) File.WriteAllText(Path.Combine(dir, "ckpt", f), "{}");
        File.WriteAllBytes(Path.Combine(dir, "fixture.wav"), FakeSongGenerator.WavBytes(300));
    }

    public void Dispose() { foreach (var f in factories) f.Dispose(); if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }

    private ApiFactory Service(string workerScript, int cards = 2)
    {
        var f = new ApiFactory(("Service:Isolation", "process"), ("Model:Checkpoint", Path.Combine(dir, "ckpt")), ("Model:Device", "cuda:0"), ("Codec:Device", "cuda:1"))
        {
            UseFakeGenerator = false,
            Configure = services =>
            {
                services.RemoveAll<IGpuProbe>();
                services.AddSingleton<IGpuProbe>(new FixedGpuProbe { Count = cards });
                services.RemoveAll<ISongGenerator>();
                services.AddSingleton<ISongGenerator>(sp => new ProcessSongGenerator(sp.GetRequiredService<ServiceOptions>(), sp.GetRequiredService<IGpuProbe>(),
                    (mode, wav) => ("/bin/sh", new[] { "-c", workerScript, "stub", mode, wav })));
            },
        };
        factories.Add(f);
        return f;
    }

    private static JsonElement Parse(string s) => JsonDocument.Parse(s).RootElement.Clone();

    [ShellFact]
    public async Task Health_is_ready_at_once_without_loading_anything_and_says_the_model_is_on_demand()
    {
        using var client = Service("exit 0").CreateClient();
        var h = Parse(await client.GetStringAsync("/health"));
        Assert.Equal((true, "on-demand", "ok"), (h.GetProperty("ready").GetBoolean(), h.GetProperty("model").GetString(), h.GetProperty("status").GetString()));
        Assert.Empty(h.GetProperty("missing").EnumerateArray());
        Assert.Equal(("cuda:0", "cuda:1"), (h.GetProperty("modelDevice").GetString(), h.GetProperty("codecDevice").GetString()));
    }

    [ShellFact]
    public async Task A_missing_checkpoint_file_makes_it_not_ready_and_names_the_file()
    {
        File.Delete(Path.Combine(dir, "ckpt", "tokenizer.json"));
        using var client = Service("exit 0").CreateClient();
        var h = Parse(await client.GetStringAsync("/health"));
        Assert.False(h.GetProperty("ready").GetBoolean());
        Assert.Contains("tokenizer.json", h.GetProperty("missing").EnumerateArray().Select(e => e.GetString()));
        Assert.Contains("tokenizer.json", h.GetProperty("modelError").GetString());
    }

    [ShellFact]
    public async Task A_device_that_does_not_exist_makes_it_not_ready()
    {
        using var client = Service("exit 0", cards: 1).CreateClient();                    // Codec:Device is cuda:1
        var h = Parse(await client.GetStringAsync("/health"));
        Assert.False(h.GetProperty("ready").GetBoolean());
        Assert.Contains("Codec:Device", h.GetProperty("modelError").GetString());
    }

    [ShellFact]
    public async Task A_job_shows_the_loading_stage_in_both_apis_then_generating_and_ends_with_the_workers_wav()
    {
        var gate = Path.Combine(dir, "gate");
        var script = $"read req; echo '{{\"stage\":\"loading\",\"current\":0,\"total\":0}}'; while [ ! -e '{gate}' ]; do sleep 0.05; done; " +
                     $"echo '{{\"stage\":\"generating\",\"current\":1,\"total\":2}}'; cp '{Path.Combine(dir, "fixture.wav")}' \"$2\"";
        using var client = Service(script).CreateClient();

        var r = await client.PostAsJsonAsync("/generate", new { lyrics = "[verse]\nla", tags = "piano" });
        var id = Parse(await r.Content.ReadAsStringAsync()).GetProperty("id").GetString()!;

        await Eventually(() => Parse(client.GetStringAsync($"/songs/{id}").Result).GetProperty("progress").GetProperty("stage").GetString() == "loading");
        var compat = Parse(await client.GetStringAsync($"/jobs/{id}"));
        Assert.Equal(("running", 0, 0), (compat.GetProperty("state").GetString(), compat.GetProperty("progress").GetProperty("done").GetInt32(), compat.GetProperty("progress").GetProperty("total").GetInt32()));
        Assert.Equal("loading", Parse(await client.GetStringAsync($"/songs/{id}")).GetProperty("progress").GetProperty("stage").GetString());

        File.WriteAllText(gate, "go");
        await Eventually(() => Parse(client.GetStringAsync($"/jobs/{id}").Result).GetProperty("state").GetString() == "done");
        var wav = await client.GetByteArrayAsync($"/jobs/{id}/audio");
        Assert.Equal(File.ReadAllBytes(Path.Combine(dir, "fixture.wav")), wav);
        Assert.InRange(Parse(await client.GetStringAsync($"/songs/{id}")).GetProperty("audioDurationMs").GetInt32(), 299, 300);
    }

    [ShellFact]
    public async Task A_worker_that_dies_fails_the_job_with_a_clear_error_and_the_next_job_runs()
    {
        var marker = Path.Combine(dir, "second");
        // the first song's worker is killed; the second one (tags "ok") succeeds
        var script = $"read req; case \"$req\" in *'\"tags\":\"ok\"'*) cp '{Path.Combine(dir, "fixture.wav")}' \"$2\"; exit 0;; esac; kill -9 $$";
        using var client = Service(script).CreateClient();
        var bad = Parse(await (await client.PostAsJsonAsync("/generate", new { lyrics = "x", tags = "bad" })).Content.ReadAsStringAsync()).GetProperty("id").GetString();
        var good = Parse(await (await client.PostAsJsonAsync("/generate", new { lyrics = "x", tags = "ok" })).Content.ReadAsStringAsync()).GetProperty("id").GetString();

        await Eventually(() => Parse(client.GetStringAsync($"/jobs/{good}").Result).GetProperty("state").GetString() == "done");
        var failed = Parse(await client.GetStringAsync($"/jobs/{bad}"));
        Assert.Equal("failed", failed.GetProperty("state").GetString());
        Assert.Contains("ended unexpectedly", failed.GetProperty("error").GetString());
        Assert.False(File.Exists(marker));
    }

    [ShellFact]
    public async Task Forgetting_a_running_job_kills_its_worker_at_once_and_the_next_job_starts()
    {
        var script = "read req; case \"$req\" in *'\"tags\":\"next\"'*) cp '" + Path.Combine(dir, "fixture.wav") + "' \"$2\"; exit 0;; esac; echo '{\"stage\":\"loading\",\"current\":0,\"total\":0}'; sleep 60";
        using var client = Service(script).CreateClient();
        var slow = Parse(await (await client.PostAsJsonAsync("/generate", new { lyrics = "x", tags = "slow" })).Content.ReadAsStringAsync()).GetProperty("id").GetString();
        var next = Parse(await (await client.PostAsJsonAsync("/generate", new { lyrics = "x", tags = "next" })).Content.ReadAsStringAsync()).GetProperty("id").GetString();
        await Eventually(() => Parse(client.GetStringAsync($"/songs/{slow}").Result).GetProperty("progress").GetProperty("stage").GetString() == "loading");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        Assert.Equal(HttpStatusCode.OK, (await client.DeleteAsync($"/jobs/{slow}")).StatusCode);
        await Eventually(() => Parse(client.GetStringAsync($"/jobs/{next}").Result).GetProperty("state").GetString() == "done", timeoutMs: 3000);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(3), $"took {sw.Elapsed}");
        Assert.Equal(HttpStatusCode.NotFound, (await client.GetAsync($"/jobs/{slow}")).StatusCode);
    }

    [Fact]
    public void By_default_the_model_stays_in_the_service_and_the_codec_decodes_in_a_worker_and_process_mode_registers_the_process_generator()
    {
        using var byDefault = new ApiFactory(("Model:Checkpoint", Path.Combine(dir, "ckpt")), ("Model:Device", "cpu"), ("Codec:Device", "cpu")) { UseFakeGenerator = false };
        var codec = Assert.IsType<HeartMuLaSongGenerator>(byDefault.Services.GetRequiredService<ISongGenerator>());
        Assert.True(codec.DecodesInWorker);

        using var process = new ApiFactory(("Service:Isolation", "process"), ("Model:Checkpoint", Path.Combine(dir, "ckpt"))) { UseFakeGenerator = false };
        Assert.IsType<ProcessSongGenerator>(process.Services.GetRequiredService<ISongGenerator>());
    }
}
