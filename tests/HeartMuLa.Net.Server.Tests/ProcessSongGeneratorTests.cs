using System.Diagnostics;
using HeartMuLa.Server.Api;
using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;
using HeartMuLa.Server.Worker;

namespace HeartMuLa.Server.Tests;

/// <summary>[Fact] for tests whose stub worker is a shell script.</summary>
public sealed class ShellFactAttribute : FactAttribute
{
    public ShellFactAttribute() { if (OperatingSystem.IsWindows()) Skip = "the stub worker is a shell script"; }
}

public sealed class FixedGpuProbe : IGpuProbe
{
    public int Count { get; set; } = 2;
    public GpuInfo? Read(string device) => null;
    public int DeviceCount() => Count;
}

public sealed class ProcessSongGeneratorTests : IDisposable
{
    private readonly string dir = Path.Combine(Path.GetTempPath(), "heartmula-proc-" + Guid.NewGuid().ToString("N"));
    private string Wav => Path.Combine(dir, "song.wav.tmp");
    private string Fixture => Path.Combine(dir, "fixture.wav");

    public ProcessSongGeneratorTests()
    {
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Fixture, FakeSongGenerator.WavBytes(250));
    }
    public void Dispose() { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }

    private const string Loading = "{\"stage\":\"loading\",\"current\":0,\"total\":0}";
    private static string Gen(int i, int n) => $"{{\"stage\":\"generating\",\"current\":{i},\"total\":{n}}}";

    private ProcessSongGenerator Make(string script, ServiceOptions? options = null)
    {
        var o = options ?? Options();
        return new ProcessSongGenerator(o, new FixedGpuProbe(), (mode, wav) => ("/bin/sh", new[] { "-c", script, "stub", mode, wav }));
    }

    private ServiceOptions Options(string? checkpoint = null) =>
        new() { Model = new ModelOptions { Checkpoint = checkpoint ?? dir, Device = "cuda:0" }, Codec = new CodecOptions { Device = "cuda:1" } };

    private static SongRequest Req() => new("piano", "la", 5000, 7, 3.0, null, 1.0, 50);

    private List<(JobStage, int, int)> Run(ProcessSongGenerator g, out GeneratedSong song, CancellationToken ct = default)
    {
        var seen = new List<(JobStage, int, int)>();
        song = g.Generate(Req(), Wav, (s, c, t) => { lock (seen) seen.Add((s, c, t)); }, ct);
        return seen;
    }

    [ShellFact]
    public void A_worker_that_makes_the_song_gives_its_progress_in_order_and_a_wav_whose_length_is_read_from_the_file()
    {
        var script = $"read req; echo '{Loading}'; echo '{Gen(1, 2)}'; echo '{Gen(2, 2)}'; echo '{{\"stage\":\"decoding\",\"current\":1,\"total\":1}}'; cp '{Fixture}' \"$2\"";
        var seen = Run(Make(script), out var song);

        Assert.Equal(new[] { (JobStage.Loading, 0, 0), (JobStage.Generating, 1, 2), (JobStage.Generating, 2, 2), (JobStage.Decoding, 1, 1) }, seen);
        Assert.True(File.Exists(Wav));
        Assert.InRange(song.AudioDurationMs, 249, 250);
    }

    [ShellFact]
    public void The_worker_is_started_with_the_mode_the_wav_path_and_a_request_line_that_carries_the_resolved_settings()
    {
        var capture = Path.Combine(dir, "request.txt");
        var script = $"read req; echo \"$req\" > '{capture}'; echo \"$1 $2\" > '{capture}.args'; cp '{Fixture}' \"$2\"";
        Run(Make(script), out _);

        var request = WorkerProtocol.DecodeRequest(File.ReadAllText(capture).Trim());
        Assert.Equal((WorkerProtocol.SongMode, "piano", 5000, 7L, 3.0), (request.Mode, request.Tags, request.DurationMs, request.Seed, request.CfgScale));
        Assert.Equal(("cuda:0", "cuda:1", dir), (request.Settings.ModelDevice, request.Settings.CodecDevice, request.Settings.Checkpoint));
        Assert.Equal($"{WorkerProtocol.SongMode} {Wav}", File.ReadAllText(capture + ".args").Trim());
    }

    [ShellFact]
    public void Lines_that_are_not_progress_lines_are_ignored()
    {
        var script = $"read req; echo 'warning: something from a native library'; echo '{{\"warning\":1}}'; echo '{Gen(1, 1)}'; echo ''; cp '{Fixture}' \"$2\"";
        Assert.Equal(new[] { (JobStage.Generating, 1, 1) }, Run(Make(script), out _));
    }

    [ShellFact]
    public void A_failure_reports_the_last_line_of_standard_error_as_the_cause_and_leaves_no_file()
    {
        var script = "read req; echo 'some diagnostics' >&2; echo 'InvalidOperationException: device cuda:3 does not exist' >&2; printf partial > \"$2\"; exit 1";
        var e = Assert.Throws<SongProcessException>(() => Run(Make(script), out _));
        Assert.Equal("InvalidOperationException: device cuda:3 does not exist", e.Message);
        Assert.False(File.Exists(Wav));
    }

    [ShellFact]
    public void A_worker_killed_by_a_signal_is_an_unexpected_end_with_its_exit_code()
    {
        var e = Assert.Throws<SongProcessException>(() => Run(Make("read req; echo 'about to die' >&2; kill -9 $$"), out _));
        Assert.Contains("ended unexpectedly", e.Message);
        Assert.Contains("exit code 137", e.Message);
        Assert.Contains("about to die", e.Message);
    }

    [ShellFact]
    public void A_crash_with_another_exit_code_is_an_unexpected_end_too()
    {
        var e = Assert.Throws<SongProcessException>(() => Run(Make("read req; echo boom >&2; exit 7"), out _));
        Assert.Contains("exit code 7", e.Message);
    }

    [ShellFact]
    public void Exit_0_without_a_wav_is_a_failure()
    {
        var e = Assert.Throws<SongProcessException>(() => Run(Make("read req; exit 0"), out _));
        Assert.Contains("without writing the song", e.Message);
    }

    [ShellFact]
    public void Cancelling_ends_the_whole_process_tree_within_a_second_and_leaves_no_file()
    {
        // `sleep` is a child of the shell: killing only the shell would leave it running.
        // 61 s, not 60: other test classes run `sleep 60` in parallel and IsOurSleep must only see this one.
        var script = $"read req; echo '{Loading}'; printf partial > \"$2\"; sleep 61";
        using var cts = new CancellationTokenSource();
        var g = Make(script);
        var started = new ManualResetEventSlim();
        var task = Task.Run(() => g.Generate(Req(), Wav, (s, c, t) => started.Set(), cts.Token));
        Assert.True(started.Wait(5000));

        var sw = Stopwatch.StartNew();
        cts.Cancel();
        var e = Assert.ThrowsAny<OperationCanceledException>(() => task.GetAwaiter().GetResult());
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(1), $"took {sw.Elapsed}");
        Assert.False(File.Exists(Wav));
        Assert.DoesNotContain(Process.GetProcessesByName("sleep"), IsOurSleep);
    }

    private static bool IsOurSleep(Process p)
    {
        try { return File.ReadAllText($"/proc/{p.Id}/cmdline").Contains("61"); } catch { return false; }
    }

    [ShellFact]
    public void A_cancellation_before_the_start_does_not_start_anything()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => Run(Make("exit 0"), out _, cts.Token));
    }

    [Fact]
    public void A_program_that_cannot_be_started_is_a_clear_failure()
    {
        var g = new ProcessSongGenerator(Options(), new FixedGpuProbe(), (_, _) => ("/definitely/not/a/program", Array.Empty<string>()));
        var e = Assert.Throws<SongProcessException>(() => Run(g, out _));
        Assert.Contains("could not be started", e.Message);
    }

    [Fact]
    public void The_generator_is_ready_without_loading_anything_when_the_files_and_devices_are_there()
    {
        foreach (var f in new[] { "gen_config.json", "tokenizer.json" }) File.WriteAllText(Path.Combine(dir, f), "{}");
        Directory.CreateDirectory(Path.Combine(dir, "HeartCodec-oss"));
        Directory.CreateDirectory(Path.Combine(dir, "HeartMuLa-oss-3B"));
        var g = Make("exit 0");
        g.Load();
        Assert.Equal((ModelStatus.Ready, null, 0), (g.Status, g.LoadError, g.Missing.Count));
        Assert.Equal(new DeviceInfo("cuda:0", "cuda:1"), g.Devices);
    }

    [Fact]
    public void Missing_files_and_non_existent_devices_make_it_not_ready_with_the_reasons()
    {
        File.WriteAllText(Path.Combine(dir, "gen_config.json"), "{}");               // everything else is absent
        var g = Make("exit 0");
        Assert.Equal(ModelStatus.Failed, g.Status);
        Assert.Contains("tokenizer.json", g.Missing);
        Assert.Contains("tokenizer.json", g.LoadError);

        var noGpu = new ProcessSongGenerator(Options(), new FixedGpuProbe { Count = 1 }, (_, _) => ("true", Array.Empty<string>()));
        Assert.Contains("Codec:Device", noGpu.LoadError);                            // cuda:1 on a machine with one card
    }

    [Fact]
    public void Devices_default_from_the_cards_the_machine_has()
    {
        var o = new ServiceOptions { Model = new ModelOptions { Checkpoint = dir } };
        Assert.Equal(new DeviceInfo("cuda:0", "cuda:0"), new ProcessSongGenerator(o, new FixedGpuProbe { Count = 2 }).Devices);
        Assert.Equal(new DeviceInfo("cpu", "cpu"), new ProcessSongGenerator(o, new FixedGpuProbe { Count = 0 }).Devices);
    }

    [Theory]
    [InlineData("tiny", 250)]
    public void WavInfo_reads_the_length_from_the_header(string name, int ms)
    {
        var path = Path.Combine(dir, name + ".wav");
        File.WriteAllBytes(path, FakeSongGenerator.WavBytes(ms));
        Assert.InRange(WavInfo.DurationMs(path), ms - 1, ms);
        File.WriteAllText(Path.Combine(dir, "bad.wav"), "not a wav at all");
        Assert.Throws<InvalidDataException>(() => WavInfo.DurationMs(Path.Combine(dir, "bad.wav")));
    }
}
