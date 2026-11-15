using System.Diagnostics;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;

namespace HeartMuLa.Server.Tests;

/// <summary>
/// The isolation modes against the real checkpoint on the GPUs: the same WAV as the command line, video memory returned to its baseline, and the cost
/// of each mode. Heavy (about 8 GB per model load, several minutes); skipped unless HEARTMULA_CKPT is set. HEARTMULA_RESULTS_FILE, when set, receives the numbers.
/// </summary>
[Collection("RealGpu")]
public class RealIsolationTests
{
    private const int SlackMiB = 600;                                              // what the service process alone may hold on a card

    private static async Task<(WebApplicationFactory<Program> Factory, HttpClient Client, string Dir)> Start(string mode, string ckpt)
    {
        var dir = Path.Combine(Path.GetTempPath(), "heartmula-iso-real-" + Guid.NewGuid().ToString("N"));
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
        {
            b.UseSetting("Service:Isolation", mode);
            b.UseSetting("Model:Checkpoint", ckpt);
            b.UseSetting("Model:Device", RealModel.Env.DeviceName);
            b.UseSetting("Codec:Device", RealModel.CodecDevice);
            b.UseSetting("Storage:Directory", dir);
        });
        var client = factory.CreateClient();
        client.Timeout = TimeSpan.FromMinutes(5);
        var until = DateTime.UtcNow.AddMinutes(5);
        while (!JsonDocument.Parse(await client.GetStringAsync("/health")).RootElement.GetProperty("ready").GetBoolean())
        {
            if (DateTime.UtcNow > until) throw new TimeoutException("not ready: " + await client.GetStringAsync("/health"));
            await Task.Delay(300);
        }
        return (factory, client, dir);
    }

    private static async Task<string> Submit(HttpClient c, int ms = 2000)
    {
        var r = await c.PostAsJsonAsync("/songs", new { tags = RealCli.Tags, lyrics = RealCli.Lyrics, durationMs = ms, seed = RealCli.Seed, cfgScale = 1.5 });
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        return JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement.GetProperty("id").GetString()!;
    }

    private static async Task<(string State, double Seconds)> WaitFinal(HttpClient c, string id)
    {
        var until = DateTime.UtcNow.AddMinutes(8);
        while (true)
        {
            var j = JsonDocument.Parse(await c.GetStringAsync($"/songs/{id}")).RootElement;
            var state = j.GetProperty("state").GetString()!;
            if (state is "succeeded" or "failed" or "cancelled")
            {
                var secs = j.TryGetProperty("endedAt", out var e) && j.TryGetProperty("startedAt", out var s) ? (e.GetDateTimeOffset() - s.GetDateTimeOffset()).TotalSeconds : 0;
                if (state == "failed") throw new InvalidOperationException("the job failed: " + j.GetProperty("error").GetString());
                return (state, secs);
            }
            if (DateTime.UtcNow > until) throw new TimeoutException($"job {id} still {state}");
            await Task.Delay(200);
        }
    }

    private static async Task<int> SettleMiB(int card)
    {
        await Task.Delay(1500);                                                  // the driver releases a dead process's memory a moment after it exits
        return Gpu.UsedMiB(card);
    }

    [RequiresRealModel]
    public async Task Process_mode_gives_the_cli_wav_and_returns_both_cards_to_their_baseline()
    {
        var ckpt = RealModel.Env.Dir!;
        var cli = await RealCli.GetAsync(ckpt);
        int mc = Gpu.Index(RealModel.Env.DeviceName), cc = Gpu.Index(RealModel.CodecDevice);
        int base0 = Gpu.UsedMiB(mc), base1 = Gpu.UsedMiB(cc);
        var (factory, client, dir) = await Start("process", ckpt);
        using (factory) using (client)
        {
            var id = await Submit(client);
            var (_, secs) = await WaitFinal(client, id);
            Assert.True(cli.Wav.SequenceEqual(await client.GetByteArrayAsync($"/songs/{id}/audio")), "process mode differs from generate --wav\n" + cli.Output);

            int after0 = await SettleMiB(mc), after1 = cc == mc ? after0 : await SettleMiB(cc);
            Gpu.Record($"process: song of 2 s took {secs:F1} s (model load included); model card {mc}: {base0} -> {after0} MiB, codec card {cc}: {base1} -> {after1} MiB");
            Assert.InRange(after0, 0, base0 + SlackMiB);
            Assert.InRange(after1, 0, base1 + SlackMiB);

            // a second song costs the same again: nothing stayed loaded
            var second = await Submit(client);
            var (_, secs2) = await WaitFinal(client, second);
            Gpu.Record($"process: second song took {secs2:F1} s");
        }
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }

    [RequiresRealModel]
    public async Task Inprocess_mode_gives_the_cli_wav_and_keeps_the_cards_occupied()
    {
        var ckpt = RealModel.Env.Dir!;
        var cli = await RealCli.GetAsync(ckpt);
        int mc = Gpu.Index(RealModel.Env.DeviceName), cc = Gpu.Index(RealModel.CodecDevice);
        int base0 = Gpu.UsedMiB(mc), base1 = Gpu.UsedMiB(cc);
        var (factory, client, dir) = await Start("inprocess", ckpt);
        using (factory) using (client)
        {
            var id = await Submit(client);
            var (_, secs) = await WaitFinal(client, id);
            Assert.True(cli.Wav.SequenceEqual(await client.GetByteArrayAsync($"/songs/{id}/audio")), "inprocess mode differs from generate --wav\n" + cli.Output);
            var second = await Submit(client);
            var (_, secs2) = await WaitFinal(client, second);
            int after0 = await SettleMiB(mc), after1 = cc == mc ? after0 : await SettleMiB(cc);
            Gpu.Record($"inprocess: first song {secs:F1} s, second song {secs2:F1} s (model stays loaded); model card {mc}: {base0} -> {after0} MiB, codec card {cc}: {base1} -> {after1} MiB");
            Assert.True(after0 > base0 + 5000, "the model should still be on its card");
        }
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }

    [RequiresRealModel]
    public async Task Codec_mode_is_deterministic_frees_the_codec_card_and_keeps_the_model_on_its_own()
    {
        var ckpt = RealModel.Env.Dir!;
        var cli = await RealCli.GetAsync(ckpt);
        int mc = Gpu.Index(RealModel.Env.DeviceName), cc = Gpu.Index(RealModel.CodecDevice);
        int base0 = Gpu.UsedMiB(mc), base1 = Gpu.UsedMiB(cc);
        var (factory, client, dir) = await Start("codec", ckpt);
        using (factory) using (client)
        {
            var first = await Submit(client);
            var (_, secs) = await WaitFinal(client, first);
            var wav1 = await client.GetByteArrayAsync($"/songs/{first}/audio");
            var second = await Submit(client);
            var (_, secs2) = await WaitFinal(client, second);
            var wav2 = await client.GetByteArrayAsync($"/songs/{second}/audio");

            bool sameAsCli = cli.Wav.SequenceEqual(wav1);
            int after0 = await SettleMiB(mc), after1 = cc == mc ? after0 : await SettleMiB(cc);
            Gpu.Record($"codec: first song {secs:F1} s, second {secs2:F1} s; equal to the CLI's WAV: {sameAsCli} (model {mc}, codec {cc}); model card: {base0} -> {after0} MiB, codec card: {base1} -> {after1} MiB");
            Assert.True(wav1.SequenceEqual(wav2), "codec mode is not deterministic");
            if (mc != cc)
            {
                Assert.True(sameAsCli, "with the model and the codec on different devices the WAV should equal the CLI's (see the spec); finding recorded in the results file");
                Assert.InRange(after1, 0, base1 + SlackMiB);                         // the codec's card is free again
                Assert.True(after0 > base0 + 5000, "the model should still be on its card");
            }
        }
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }

    [RequiresRealModel]
    public async Task Killing_a_running_job_returns_the_memory_at_once_and_the_next_song_is_the_clis()
    {
        var ckpt = RealModel.Env.Dir!;
        var cli = await RealCli.GetAsync(ckpt);
        int mc = Gpu.Index(RealModel.Env.DeviceName);
        int base0 = Gpu.UsedMiB(mc);
        var (factory, client, dir) = await Start("process", ckpt);
        using (factory) using (client)
        {
            var longJob = await Submit(client, 120_000);
            var until = DateTime.UtcNow.AddMinutes(3);
            while (JsonDocument.Parse(await client.GetStringAsync($"/songs/{longJob}")).RootElement.GetProperty("progress").GetProperty("current").GetInt32() < 5)
            {
                if (DateTime.UtcNow > until) throw new TimeoutException("the long job never started generating");
                await Task.Delay(100);
            }
            int during = Gpu.UsedMiB(mc);
            Assert.True(during > base0 + 5000, "the worker should be holding the model");

            var next = await Submit(client);
            var sw = Stopwatch.StartNew();
            Assert.Equal(HttpStatusCode.Accepted, (await client.DeleteAsync($"/songs/{longJob}")).StatusCode);
            int back;
            do { back = Gpu.UsedMiB(mc); if (back <= base0 + SlackMiB) break; await Task.Delay(50); } while (sw.Elapsed < TimeSpan.FromSeconds(10));
            var freed = sw.Elapsed;
            Gpu.Record($"process: killing a running job returned the model card {mc} from {during} to {back} MiB in {freed.TotalSeconds:F2} s (baseline {base0})");
            Assert.True(freed < TimeSpan.FromSeconds(3), $"the memory took {freed} to come back");

            var (_, _) = await WaitFinal(client, next);
            Assert.True(cli.Wav.SequenceEqual(await client.GetByteArrayAsync($"/songs/{next}/audio")), "the song after a kill differs from the CLI's");
        }
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }
}
