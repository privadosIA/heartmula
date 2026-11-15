using System.Diagnostics;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using MusicGen.Core.Testing;

namespace HeartMuLa.Server.Tests;

/// <summary>
/// The service against the real checkpoint. Heavy: it loads HeartMuLa (≈8 GB) and runs the CLI once. Skipped unless HEARTMULA_CKPT is set.
/// Devices: HEARTMULA_DEVICE for the model and HEARTMULA_CODEC_DEVICE for the codec (both cpu by default). Build with the matching
/// -p:TorchBackend (cuda-linux for GPUs); the CLI child process runs from the same output folder.
/// </summary>
public static class RealModel
{
    public static readonly CheckpointEnvironment Env = new("HEARTMULA_CKPT", "HEARTMULA_AUDIO_DIR", "HEARTMULA_DEVICE");
    public static string CodecDevice => Environment.GetEnvironmentVariable("HEARTMULA_CODEC_DEVICE") is { Length: > 0 } d ? d : Env.DeviceName;
}

public sealed class RequiresRealModelAttribute : RequiresCheckpointFactAttribute
{
    public RequiresRealModelAttribute() : base(RealModel.Env) { }
}

/// <summary>The reference audio: what <c>generate --wav</c> writes, produced once per test run and shared by every real-model test class.</summary>
public static class RealCli
{
    public const string Tags = "piano,happy";
    public const string Lyrics = "[verse]\nla la la";
    public const long Seed = 1234;
    private static readonly SemaphoreSlim Gate = new(1, 1);
    private static (byte[] Wav, byte[]? Compat, string Output)? cached;

    /// <summary>The CLI's WAVs for the native defaults (2000 ms, guidance 1.5) and for what <c>/generate</c> defaults to (5000 ms, guidance 3.0).</summary>
    public static async Task<(byte[] Wav, byte[]? Compat, string Output)> GetAsync(string ckpt, bool withCompat = false)
    {
        await Gate.WaitAsync();
        try
        {
            if (cached is { } c && (!withCompat || c.Compat is not null)) return c;
            var dir = Path.Combine(Path.GetTempPath(), "heartmula-cli-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                File.WriteAllText(Path.Combine(dir, "tags.txt"), Tags);
                File.WriteAllText(Path.Combine(dir, "lyrics.txt"), Lyrics);
                var sb = new System.Text.StringBuilder();
                var wav = cached?.Wav ?? await Run(ckpt, dir, "cli.wav", 2000, 1.5, sb);
                var compat = withCompat ? await Run(ckpt, dir, "cli-compat.wav", 5000, 3.0, sb) : null;     // only the tests of /generate need it
                return (cached = (wav, compat, sb.ToString())).Value;
            }
            finally { Directory.Delete(dir, recursive: true); }
        }
        finally { Gate.Release(); }
    }

    private static async Task<byte[]> Run(string ckpt, string dir, string wavName, int ms, double cfg, System.Text.StringBuilder log)
    {
        var wav = Path.Combine(dir, wavName);
        var psi = new ProcessStartInfo("dotnet") { RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var a in new[]
        {
            Path.Combine(AppContext.BaseDirectory, "HeartMuLa.Net.dll"), "generate", "--ckpt", ckpt,
            "--tags", Path.Combine(dir, "tags.txt"), "--lyrics", Path.Combine(dir, "lyrics.txt"), "--ms", ms.ToString(), "--seed", Seed.ToString(),
            "--cfg", cfg.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--device", RealModel.Env.DeviceName, "--codec-device", RealModel.CodecDevice, "--out", Path.Combine(dir, "frames.safetensors"), "--wav", wav,
        }) psi.ArgumentList.Add(a);
        using var cli = Process.Start(psi)!;
        var output = await cli.StandardOutput.ReadToEndAsync() + await cli.StandardError.ReadToEndAsync();
        await cli.WaitForExitAsync();
        log.AppendLine(output);
        if (cli.ExitCode != 0) throw new InvalidOperationException($"the CLI failed ({cli.ExitCode}):\n{output}");
        return await File.ReadAllBytesAsync(wav);
    }
}

/// <summary>Video memory in use, as <c>nvidia-smi</c> reports it, and a results file the heavy runs append their numbers to.</summary>
public static class Gpu
{
    public static int UsedMiB(int index)
    {
        var psi = new ProcessStartInfo("nvidia-smi", $"--query-gpu=memory.used --format=csv,noheader,nounits -i {index}") { RedirectStandardOutput = true };
        using var p = Process.Start(psi)!;
        var text = p.StandardOutput.ReadToEnd().Trim();
        p.WaitForExit();
        return int.Parse(text, System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>The card number of a device name such as <c>cuda:1</c>.</summary>
    public static int Index(string device) => device.Contains(':') ? int.Parse(device.Split(':')[1]) : 0;

    public static void Record(string line)
    {
        if (Environment.GetEnvironmentVariable("HEARTMULA_RESULTS_FILE") is { Length: > 0 } f) File.AppendAllText(f, line + Environment.NewLine);
    }
}

public sealed class RealServiceFixture : IAsyncLifetime
{
    public const string Tags = "piano,happy";
    public const string Lyrics = "[verse]\nla la la";
    public const int Ms = 2000;
    public const long Seed = 1234;

    private WebApplicationFactory<Program>? factory;
    private string? dir;
    public HttpClient Client { get; private set; } = null!;
    public byte[]? CliWav { get; private set; }
    /// <summary>The CLI with what <c>POST /generate</c> defaults to (cfg 3.0) and its minimum length (5000 ms).</summary>
    public byte[]? CliWavCompat { get; private set; }
    public string? CliOutput { get; private set; }

    public async Task InitializeAsync()
    {
        if (RealModel.Env.Dir is not { } ckpt) return;                              // every test is skipped
        dir = Path.Combine(Path.GetTempPath(), "heartmula-real-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        // 1. The CLI first, in its own process: it exits and frees its memory before the service loads the model on the same device.
        (CliWav, CliWavCompat, CliOutput) = await RealCli.GetAsync(ckpt, withCompat: true);

        // 2. The service, with the real generator, on the same devices. These tests are about a model that stays loaded between songs: in-process isolation.
        factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
        {
            b.UseSetting("Service:Isolation", "inprocess");
            b.UseSetting("Model:Checkpoint", ckpt);
            b.UseSetting("Model:Device", RealModel.Env.DeviceName);
            b.UseSetting("Codec:Device", RealModel.CodecDevice);
            b.UseSetting("Storage:Directory", Path.Combine(dir, "songs"));
        });
        Client = factory.CreateClient();
        Client.Timeout = TimeSpan.FromMinutes(5);
        var until = DateTime.UtcNow.AddMinutes(5);
        while (!JsonDocument.Parse(await Client.GetStringAsync("/health")).RootElement.GetProperty("ready").GetBoolean())
        {
            if (DateTime.UtcNow > until) throw new TimeoutException("the model did not become ready: " + await Client.GetStringAsync("/health"));
            await Task.Delay(500);
        }
    }

    public Task DisposeAsync()
    {
        Client?.Dispose();
        factory?.Dispose();
        if (dir is not null && Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        return Task.CompletedTask;
    }

    public async Task<string> Submit(object body)
    {
        var r = await Client.PostAsJsonAsync("/songs", body);
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        return JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement.GetProperty("id").GetString()!;
    }

    public async Task<string> WaitFinal(string id, int timeoutSeconds = 300)
    {
        var until = DateTime.UtcNow.AddSeconds(timeoutSeconds);
        while (true)
        {
            var state = JsonDocument.Parse(await Client.GetStringAsync($"/songs/{id}")).RootElement.GetProperty("state").GetString()!;
            if (state is "succeeded" or "failed" or "cancelled") return state;
            if (DateTime.UtcNow > until) throw new TimeoutException($"job {id} still {state}");
            await Task.Delay(250);
        }
    }

    public Task<byte[]> Audio(string id) => Client.GetByteArrayAsync($"/songs/{id}/audio");
}

[Collection("RealGpu")]                              // the real-model classes use the same GPUs: one at a time
public class RealModelServiceTests : IClassFixture<RealServiceFixture>
{
    private readonly RealServiceFixture f;
    public RealModelServiceTests(RealServiceFixture f) => this.f = f;

    private object Song(string? lyrics = null, int ms = RealServiceFixture.Ms) =>
        new { tags = RealServiceFixture.Tags, lyrics = lyrics ?? RealServiceFixture.Lyrics, durationMs = ms, seed = RealServiceFixture.Seed };

    [RequiresRealModel]
    public async Task A_job_gives_byte_for_byte_the_wav_of_the_cli_for_the_same_seed_and_devices()
    {
        var id = await f.Submit(Song());
        Assert.Equal("succeeded", await f.WaitFinal(id));
        Assert.True(f.CliWav!.SequenceEqual(await f.Audio(id)), "the service's WAV differs from generate --wav\n" + f.CliOutput);
    }

    [RequiresRealModel]
    public async Task The_orchestrators_generate_route_gives_byte_for_byte_the_wav_of_the_cli_with_the_same_values()
    {
        var r = await f.Client.PostAsJsonAsync("/generate", new { lyrics = RealServiceFixture.Lyrics, tags = RealServiceFixture.Tags, max_ms = 5000, seed = RealServiceFixture.Seed });
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        var id = JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement.GetProperty("id").GetString()!;
        Assert.Equal("succeeded", await f.WaitFinal(id));
        var wav = await f.Client.GetByteArrayAsync($"/jobs/{id}/audio");
        Assert.True(f.CliWavCompat!.SequenceEqual(wav), "/generate differs from generate --wav --cfg 3 --ms 5000\n" + f.CliOutput);
        Assert.Equal(HttpStatusCode.OK, (await f.Client.DeleteAsync($"/jobs/{id}")).StatusCode);
    }

    [RequiresRealModel]
    public async Task Two_songs_in_a_row_on_one_loaded_model_both_succeed_and_the_same_seed_twice_is_identical()
    {
        var first = await f.Submit(Song());
        var other = await f.Submit(new { tags = "rock,guitar", lyrics = "[verse]\nhey hey", durationMs = RealServiceFixture.Ms, seed = 7 });
        var again = await f.Submit(Song());
        Assert.Equal("succeeded", await f.WaitFinal(first));
        Assert.Equal("succeeded", await f.WaitFinal(other));
        Assert.Equal("succeeded", await f.WaitFinal(again));

        var a = await f.Audio(first); var b = await f.Audio(again); var o = await f.Audio(other);
        Assert.True(a.SequenceEqual(b), "same seed on a reused model gave a different WAV");
        Assert.False(a.SequenceEqual(o), "a different song came out identical");
        Assert.True(a.SequenceEqual(f.CliWav!), "the reused model drifted from the CLI");
    }

    [RequiresRealModel]
    public async Task Cancelling_a_running_job_leaves_the_worker_able_to_run_the_next_one_identically()
    {
        var long_ = await f.Submit(Song(ms: 60_000));
        var next = await f.Submit(Song());
        // wait until it is really generating, then cancel
        var until = DateTime.UtcNow.AddSeconds(120);
        while (JsonDocument.Parse(await f.Client.GetStringAsync($"/songs/{long_}")).RootElement.GetProperty("progress").GetProperty("current").GetInt32() < 3)
        {
            if (DateTime.UtcNow > until) throw new TimeoutException("the long job never started generating");
            await Task.Delay(100);
        }
        Assert.Equal(HttpStatusCode.Accepted, (await f.Client.DeleteAsync($"/songs/{long_}")).StatusCode);

        Assert.Equal("cancelled", await f.WaitFinal(long_));
        Assert.Equal("succeeded", await f.WaitFinal(next));
        Assert.True((await f.Audio(next)).SequenceEqual(f.CliWav!), "after a cancellation the next song differs from the CLI's");
    }
}
