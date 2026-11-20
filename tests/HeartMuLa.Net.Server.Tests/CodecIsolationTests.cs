using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;
using HeartMuLa.Server.Storage;
using HeartMuLa.Server.Worker;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Server.Tests;

public sealed class CodecIsolationTests : IDisposable
{
    private readonly string dir = Path.Combine(Path.GetTempPath(), "heartmula-codec-" + Guid.NewGuid().ToString("N"));
    private string Wav => Path.Combine(dir, "abc.wav.tmp");
    private string Fixture => Path.Combine(dir, "fixture.wav");
    private static readonly WorkerSettings Settings = new("/ckpt", "3B", "cpu", "cpu", false);

    public CodecIsolationTests()
    {
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Fixture, FakeSongGenerator.WavBytes(250));
    }
    public void Dispose() { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }

    private static WorkerFrameDecoder Decoder(string script) => new(Settings, (mode, wav) => ("/bin/sh", new[] { "-c", script, "stub", mode, wav }));
    private static SongRequest Req() => new("piano", "la", 5000, 7, 3.0, null, 1.0, 50);
    private static Tensor Frames() => arange(0, 8 * 5, dtype: ScalarType.Int64).reshape(8, 5);
    private static string FramesFile(string wav) => WorkerFrameDecoder.FramesPath(wav);

    [Fact]
    public void The_frames_file_sits_next_to_the_wav_and_is_swept_with_the_other_temporary_files()
    {
        Assert.Equal(Path.Combine(dir, "abc.frames.tmp"), WorkerFrameDecoder.FramesPath(Wav));
        Assert.EndsWith(".tmp", WorkerFrameDecoder.FramesPath(Wav));
    }

    [ShellFact]
    public void The_worker_gets_the_codec_mode_and_a_readable_frames_file_which_is_gone_when_the_song_is_done()
    {
        var copy = Path.Combine(dir, "seen.safetensors");
        var script = $"read req; echo \"$req\" > '{dir}/req.json'; f=$(sed -n 's/.*\"framesPath\":\"\\([^\"]*\\)\".*/\\1/p' '{dir}/req.json'); cp \"$f\" '{copy}'; " +
                     $"echo '{{\"stage\":\"decoding\",\"current\":1,\"total\":1}}'; cp '{Fixture}' \"$2\"";
        var seen = new List<(JobStage, int, int)>();
        int ms;
        using (var frames = Frames()) ms = Decoder(script).Decode(frames, Req(), Wav, (s, c, t) => seen.Add((s, c, t)), CancellationToken.None);

        var request = WorkerProtocol.DecodeRequest(File.ReadAllText(Path.Combine(dir, "req.json")).Trim());
        Assert.Equal((WorkerProtocol.CodecMode, FramesFile(Wav), 7L), (request.Mode, request.FramesPath, request.Seed));
        using var read = SafeTensors.LoadFile(copy)["frames"];                       // what the worker could read
        Assert.Equal(new long[] { 8, 5 }, read.shape);
        Assert.Equal(Enumerable.Range(0, 40).Select(i => (long)i), read.flatten().data<long>().ToArray());
        Assert.False(File.Exists(FramesFile(Wav)));
        Assert.Equal(new[] { (JobStage.Decoding, 1, 1) }, seen);
        Assert.InRange(ms, 249, 250);
    }

    [ShellFact]
    public void The_frames_file_is_gone_after_a_failure_a_kill_and_a_cancellation()
    {
        foreach (var script in new[] { "read req; echo 'RuntimeError: boom' >&2; exit 1", "read req; kill -9 $$" })
        {
            using var frames = Frames();
            Assert.Throws<SongProcessException>(() => Decoder(script).Decode(frames, Req(), Wav, (_, _, _) => { }, CancellationToken.None));
            Assert.False(File.Exists(FramesFile(Wav)));
            Assert.False(File.Exists(Wav));
        }

        using var cts = new CancellationTokenSource();
        var started = new ManualResetEventSlim();
        using var frames2 = Frames();
        var task = Task.Run(() => Decoder("read req; echo '{\"stage\":\"decoding\",\"current\":1,\"total\":9}'; sleep 60")
            .Decode(frames2, Req(), Wav, (_, _, _) => started.Set(), cts.Token));
        Assert.True(started.Wait(5000));
        Assert.True(File.Exists(FramesFile(Wav)));                                   // present while the worker runs
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => task.GetAwaiter().GetResult());
        Assert.False(File.Exists(FramesFile(Wav)));
    }

    [Fact]
    public void A_leftover_frames_file_of_a_crashed_run_is_removed_at_startup()
    {
        var store = new ResultStore(new StorageOptions { Directory = dir }, TimeProvider.System);
        File.WriteAllText(Path.Combine(dir, "dead.frames.tmp"), "x");
        File.WriteAllText(Path.Combine(dir, "dead.wav.tmp"), "x");
        Assert.Empty(store.LoadAll());
        Assert.Empty(Directory.GetFiles(dir, "*.tmp"));
    }

    [ShellFact]
    public void In_codec_isolation_the_model_steps_run_here_and_only_the_decode_goes_to_the_worker()
    {
        var steps = new FakeSteps();
        var options = new ServiceOptions { Model = new ModelOptions { Checkpoint = dir, Device = "cpu" }, Codec = new CodecOptions { Device = "cpu" } };
        var script = $"read req; echo '{{\"stage\":\"decoding\",\"current\":1,\"total\":2}}'; echo '{{\"stage\":\"decoding\",\"current\":2,\"total\":2}}'; cp '{Fixture}' \"$2\"";
        var generator = new HeartMuLaSongGenerator(options, NullLogger<HeartMuLaSongGenerator>.Instance, Decoder(script), steps);

        var seen = new List<(JobStage, int, int)>();
        var song = generator.Generate(Req(), Wav, (s, c, t) => seen.Add((s, c, t)), CancellationToken.None);

        Assert.Equal(new[] { "generate:piano:7:3:1:50:5000" }, steps.Calls.ToArray());           // no decode here
        Assert.Equal(new[] { (JobStage.Generating, 1, 3), (JobStage.Generating, 2, 3), (JobStage.Generating, 3, 3), (JobStage.Decoding, 1, 2), (JobStage.Decoding, 2, 2) }, seen);
        Assert.InRange(song.AudioDurationMs, 249, 250);
        Assert.Equal(("cpu", "cpu"), (generator.Settings.ModelDevice, generator.Settings.CodecDevice));
    }

    [Fact]
    public void The_codec_setting_builds_the_in_process_generator_with_a_worker_decoder_and_inprocess_builds_it_without()
    {
        Directory.CreateDirectory(Path.Combine(dir, "ckpt"));
        foreach (var (mode, inWorker) in new[] { ("codec", true), ("inprocess", false) })
        {
            using var factory = new ApiFactory(("Service:Isolation", mode), ("Model:Checkpoint", Path.Combine(dir, "ckpt")), ("Model:Device", "cpu"), ("Codec:Device", "cpu")) { UseFakeGenerator = false };
            var generator = Assert.IsType<HeartMuLaSongGenerator>(factory.Services.GetRequiredService<ISongGenerator>());
            Assert.Equal(inWorker, generator.DecodesInWorker);
            Assert.False(((ISongGenerator)generator).LoadsPerSong);                         // the model stays resident in both
        }
    }
}
