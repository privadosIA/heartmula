using System.Collections.Concurrent;
using System.Diagnostics;
using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Worker;
using TorchSharp;
using static TorchSharp.torch;
using ServerWorker = HeartMuLa.Server.Worker.Worker;

namespace HeartMuLa.Server.Tests;

/// <summary>A stand-in for the worker's standard input: lines go in, and <see cref="Close"/> is the end of the stream (the service disappearing).</summary>
public sealed class ControlledReader : TextReader
{
    private readonly BlockingCollection<string?> lines = new();
    public void Send(string line) => lines.Add(line);
    public new void Close() => lines.Add(null);
    public override string? ReadLine() => lines.Take();
}

public sealed class FakeSteps : IWorkerSteps
{
    public ConcurrentQueue<string> Calls { get; } = new();
    public Func<SongRequest, Action<JobStage, int, int>, Tensor>? OnGenerate { get; set; }
    public Exception? FailWith { get; set; }
    public bool Disposed { get; private set; }

    public void LoadModel() { Calls.Enqueue("load"); if (FailWith is not null) throw FailWith; }

    public Tensor GenerateFrames(SongRequest request, Action<JobStage, int, int> progress)
    {
        Calls.Enqueue(FormattableString.Invariant($"generate:{request.Tags}:{request.Seed}:{request.CfgScale}:{request.Temperature}:{request.TopK}:{request.DurationMs}"));
        if (OnGenerate is not null) return OnGenerate(request, progress);
        for (int i = 1; i <= 3; i++) progress(JobStage.Generating, i, 3);
        return zeros(new long[] { 8, 3 }, ScalarType.Int64);
    }

    public Tensor LoadFrames(string path) { Calls.Enqueue("loadframes:" + path); return zeros(new long[] { 8, 3 }, ScalarType.Int64); }

    public int DecodeFrames(Tensor frames, string wavPath, Action<JobStage, int, int> progress)
    {
        Calls.Enqueue("decode");
        progress(JobStage.Decoding, 1, 2);
        File.WriteAllBytes(wavPath, FakeSongGenerator.WavBytes(100));            // partly written...
        progress(JobStage.Decoding, 2, 2);
        return 100;
    }

    public void Dispose() => Disposed = true;
}

public sealed class WorkerTests : IDisposable
{
    private static readonly WorkerSettings Settings = new("/ckpt", "3B", "cpu", "cpu", false);
    private readonly string dir = Path.Combine(Path.GetTempPath(), "heartmula-worker-proc-" + Guid.NewGuid().ToString("N"));
    private readonly StringWriter stdout = new(), stderr = new();
    private readonly ControlledReader stdin = new();
    private readonly FakeSteps steps = new();
    private string Wav => Path.Combine(dir, "song.wav.tmp");

    public WorkerTests() => Directory.CreateDirectory(dir);
    public void Dispose() { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }

    private static WorkerRequest Song(string mode = WorkerProtocol.SongMode, string? frames = null) =>
        new(mode, "piano", "la", 5000, 99, 3.0, 0.7, 40, frames, Settings);

    private int Run(string mode, WorkerRequest request)
    {
        stdin.Send(WorkerProtocol.EncodeRequest(request));
        return ServerWorker.Run(new[] { WorkerProtocol.Argument, mode, Wav }, stdin, stdout, stderr, _ => steps);
    }

    private (JobStage Stage, int Current, int Total)[] Progress() =>
        stdout.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => { Assert.True(WorkerProtocol.TryDecodeProgress(l.Trim(), out var p), l); return p; }).ToArray();

    [Fact]
    public void A_song_loads_generates_and_decodes_in_order_and_reports_every_step()
    {
        Assert.Equal(WorkerProtocol.Done, Run(WorkerProtocol.SongMode, Song()));

        Assert.Equal(new[] { "load", "generate:piano:99:3:0.7:40:5000", "decode" }, steps.Calls.ToArray());
        Assert.Equal(new[]
        {
            (JobStage.Loading, 0, 0), (JobStage.Generating, 1, 3), (JobStage.Generating, 2, 3), (JobStage.Generating, 3, 3), (JobStage.Decoding, 1, 2), (JobStage.Decoding, 2, 2),
        }, Progress());
        Assert.True(File.Exists(Wav));
        Assert.True(steps.Disposed);
        Assert.Equal("", stderr.ToString());
    }

    [Fact]
    public void A_codec_request_only_decodes_the_frames_it_is_given()
    {
        Assert.Equal(WorkerProtocol.Done, Run(WorkerProtocol.CodecMode, Song(WorkerProtocol.CodecMode, "/tmp/f.safetensors")));
        Assert.Equal(new[] { "loadframes:/tmp/f.safetensors", "decode" }, steps.Calls.ToArray());          // no model is loaded
        Assert.Equal(new[] { (JobStage.Decoding, 1, 2), (JobStage.Decoding, 2, 2) }, Progress());
    }

    [Fact]
    public void A_failure_is_the_last_line_of_standard_error_with_exit_1_and_no_partial_file()
    {
        steps.FailWith = new InvalidOperationException("device cuda:3 does not exist");
        File.WriteAllBytes(Wav, new byte[] { 1 });                                  // a leftover partial file

        Assert.Equal(WorkerProtocol.Failed, Run(WorkerProtocol.SongMode, Song()));

        Assert.Equal("InvalidOperationException: device cuda:3 does not exist", stderr.ToString().TrimEnd().Split('\n').Last().Trim());
        Assert.False(File.Exists(Wav));
        Assert.Equal(new[] { (JobStage.Loading, 0, 0) }, Progress());                // standard output has progress lines only
    }

    [Fact]
    public async Task Standard_input_closing_before_the_song_ends_cancels_it_removes_the_partial_file_and_exits_3()
    {
        var inLoop = new ManualResetEventSlim();
        steps.OnGenerate = (_, progress) =>
        {
            File.WriteAllBytes(Wav, new byte[] { 1, 2, 3 });                        // partial output on disk
            inLoop.Set();
            for (int i = 1; i < 100_000; i++) { progress(JobStage.Generating, i, 100_000); Thread.Sleep(2); }
            return zeros(new long[] { 8, 3 }, ScalarType.Int64);
        };
        stdin.Send(WorkerProtocol.EncodeRequest(Song()));
        var run = Task.Run(() => ServerWorker.Run(new[] { WorkerProtocol.Argument, WorkerProtocol.SongMode, Wav }, stdin, stdout, stderr, _ => steps));
        Assert.True(inLoop.Wait(5000));

        var sw = Stopwatch.StartNew();
        stdin.Close();                                                              // the service disappeared
        Assert.Equal(WorkerProtocol.Cancelled, await run.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(3));
        Assert.False(File.Exists(Wav));
        Assert.DoesNotContain("decode", steps.Calls);
    }

    public static TheoryData<string[]> BadArguments => new()
    {
        Array.Empty<string>(), new[] { "--worker" }, new[] { "--worker", "banana", "x.wav" }, new[] { "--nope", "song", "x.wav" },
    };

    [Fact]
    public void A_multiline_failure_ends_standard_error_with_its_first_line_not_with_a_stack_frame()
    {
        steps.FailWith = new InvalidOperationException("CUDA out of memory. Tried to allocate 216.00 MiB.\nException raised from malloc\nframe #26: [0x7f3917b9d9ca]");
        Assert.Equal(WorkerProtocol.Failed, Run(WorkerProtocol.SongMode, Song()));
        var last = stderr.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => l.TrimEnd('\r')).Last();
        Assert.Equal("InvalidOperationException: CUDA out of memory. Tried to allocate 216.00 MiB.", last);
    }

    [Theory]
    [InlineData("cuda:0", "cuda:0", true)]
    [InlineData("cuda:0", "cuda:1", false)]
    [InlineData("cpu", "cpu", false)]
    public void The_model_leaves_before_the_decode_only_when_it_shares_a_cuda_card_with_the_codec(string model, string codec, bool shares)
    {
        using var pipeline = new SongPipeline(new WorkerSettings("/ckpt", "3B", model, codec, false));
        Assert.Equal(shares, pipeline.SharesDevice);
    }

    [Theory]
    [MemberData(nameof(BadArguments))]
    public void Bad_arguments_print_the_usage_and_exit_1(string[] args)
    {
        Assert.Equal(WorkerProtocol.Failed, ServerWorker.Run(args, stdin, stdout, stderr, _ => steps));
        Assert.Contains("usage", stderr.ToString());
        Assert.Equal("", stdout.ToString());
    }

    [Fact]
    public void No_request_or_a_request_for_another_mode_is_a_failure_with_a_clear_cause()
    {
        stdin.Close();
        Assert.Equal(WorkerProtocol.Failed, ServerWorker.Run(new[] { WorkerProtocol.Argument, WorkerProtocol.SongMode, Wav }, stdin, stdout, stderr, _ => steps));
        Assert.Contains("no request", stderr.ToString());

        var other = new ControlledReader();
        other.Send(WorkerProtocol.EncodeRequest(Song(WorkerProtocol.CodecMode, "/f")));
        var err = new StringWriter();
        Assert.Equal(WorkerProtocol.Failed, ServerWorker.Run(new[] { WorkerProtocol.Argument, WorkerProtocol.SongMode, Wav }, other, new StringWriter(), err, _ => steps));
        Assert.Contains("mode", err.ToString());
    }

    [Fact]
    public async Task The_real_executable_as_a_worker_keeps_standard_output_for_the_protocol_and_reports_the_cause_on_standard_error()
    {
        // No model is needed: the checkpoint does not exist, so loading fails. This runs the real entry point (Console redirection included) in a real process.
        var dll = Path.Combine(AppContext.BaseDirectory, "HeartMuLa.Net.Server.dll");
        var psi = new ProcessStartInfo("dotnet") { RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var a in new[] { dll, WorkerProtocol.Argument, WorkerProtocol.SongMode, Wav }) psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        await p.StandardInput.WriteLineAsync(WorkerProtocol.EncodeRequest(Song() with { Settings = Settings with { Checkpoint = Path.Combine(dir, "no-such-checkpoint") } }));
        await p.StandardInput.FlushAsync();
        var outText = await p.StandardOutput.ReadToEndAsync();
        var errText = await p.StandardError.ReadToEndAsync();
        await p.WaitForExitAsync();

        Assert.Equal(WorkerProtocol.Failed, p.ExitCode);
        Assert.All(outText.Split('\n', StringSplitOptions.RemoveEmptyEntries), l => Assert.True(WorkerProtocol.TryDecodeProgress(l.Trim(), out _), "not a progress line: " + l));
        Assert.Contains("loading", outText);
        Assert.Matches(@"(FileNotFound|DirectoryNotFound)Exception", errText.TrimEnd().Split('\n').Last());
    }
}
