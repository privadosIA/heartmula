using System.Text;
using HeartMuLa.Server.Jobs;
using MusicGen.Core;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Server.Worker;

/// <summary>The model-touching steps a worker runs (implemented by <see cref="SongPipeline"/>; a fake in tests).</summary>
public interface IWorkerSteps : IDisposable
{
    /// <summary>Reads the model weights and the tokenizer (throws on failure).</summary>
    void LoadModel();
    /// <summary>Generates the codec frames <c>[8, T]</c> for a request, reporting progress.</summary>
    Tensor GenerateFrames(SongRequest request, Action<JobStage, int, int> progress);
    /// <summary>Reads the frames saved at <paramref name="path"/> (the file a <c>codec</c> worker is given).</summary>
    Tensor LoadFrames(string path);
    /// <summary>Decodes the frames with the codec into a WAV at <paramref name="wavPath"/> and returns its length in milliseconds.</summary>
    int DecodeFrames(Tensor frames, string wavPath, Action<JobStage, int, int> progress);
}

/// <summary>
/// The worker process: makes one song (or only decodes its frames) and exits, so that everything it used, video memory included, goes back to the system.
/// Standard input carries the request line and then stays open: when it closes before the song is done the service is gone and the worker stops.
/// Standard output carries only progress lines; the cause of a failure is the last line of standard error. See <see cref="WorkerProtocol"/>.
/// </summary>
public static class Worker
{
    /// <summary>Runs the worker for <c>--worker &lt;song|codec&gt; &lt;wavPath&gt;</c> and returns its exit code.</summary>
    public static int Run(string[] args, TextReader stdin, TextWriter stdout, TextWriter stderr, Func<WorkerSettings, IWorkerSteps> createSteps)
    {
        if (args.Length < 3 || args[0] != WorkerProtocol.Argument || args[1] is not (WorkerProtocol.SongMode or WorkerProtocol.CodecMode))
        {
            stderr.WriteLine($"usage: {WorkerProtocol.Argument} <{WorkerProtocol.SongMode}|{WorkerProtocol.CodecMode}> <wav path>");
            return WorkerProtocol.Failed;
        }
        string mode = args[1], wavPath = args[2];

        using var cancel = new CancellationTokenSource();
        try
        {
            var line = stdin.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) throw new InvalidOperationException("no request on standard input");
            var request = WorkerProtocol.DecodeRequest(line);
            if (request.Mode != mode) throw new InvalidOperationException($"the request is for mode '{request.Mode}' but the worker was started in mode '{mode}'");

            // The service keeps standard input open while it waits; its end means the service disappeared.
            _ = Task.Run(() =>
            {
                try { while (stdin.ReadLine() is not null) { } } catch (IOException) { }
                cancel.Cancel();
            });

            var ct = cancel.Token;
            void Report(JobStage stage, int current, int total)
            {
                ct.ThrowIfCancellationRequested();
                lock (stdout) { stdout.WriteLine(WorkerProtocol.EncodeProgress(stage, current, total)); stdout.Flush(); }
            }

            using var steps = createSteps(request.Settings);
            var song = new SongRequest(request.Tags, request.Lyrics, request.DurationMs, request.Seed, request.CfgScale, null, request.Temperature, request.TopK);
            if (mode == WorkerProtocol.SongMode)
            {
                Report(JobStage.Loading, 0, 0);
                steps.LoadModel();
                ct.ThrowIfCancellationRequested();
                using var frames = steps.GenerateFrames(song, Report);
                ct.ThrowIfCancellationRequested();
                steps.DecodeFrames(frames, wavPath, Report);
            }
            else
            {
                if (request.FramesPath is null) throw new InvalidOperationException("the codec request has no framesPath");
                using var frames = steps.LoadFrames(request.FramesPath);
                Seeding.Apply(request.Seed);                       // the decode noise comes from the seeded generator of the codec's device
                steps.DecodeFrames(frames, wavPath, Report);
            }
            ct.ThrowIfCancellationRequested();
            return WorkerProtocol.Done;
        }
        catch (OperationCanceledException)
        {
            RemovePartial(wavPath);
            return WorkerProtocol.Cancelled;
        }
        catch (Exception e)
        {
            RemovePartial(wavPath);
            stderr.WriteLine(e.ToString());                                  // the full detail first (libtorch messages span many lines and end in stack frames)...
            stderr.WriteLine($"{e.GetType().Name}: {FirstLine(e.Message)}");  // ...and the cause as the last line of standard error
            stderr.Flush();
            return WorkerProtocol.Failed;
        }
    }

    private static string FirstLine(string message)
    {
        var line = message.Split('\n', 2)[0].Trim();
        return line.Length > 0 ? line : "(no message)";
    }

    private static void RemovePartial(string wavPath)
    {
        foreach (var path in new[] { wavPath, wavPath + ".tmp" })
            try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>The real entry of a worker process (called from <c>Program.cs</c> when started with <c>--worker</c>): standard output is captured for the protocol and <c>Console.Out</c> is pointed at standard error, because the pipeline prints human text.</summary>
    public static int Execute(string[] args)
    {
        var utf8 = new UTF8Encoding(false);
        var protocolOut = new StreamWriter(Console.OpenStandardOutput(), utf8) { AutoFlush = true };
        var stderr = new StreamWriter(Console.OpenStandardError(), utf8) { AutoFlush = true };
        Console.SetOut(stderr);
        Console.SetError(stderr);
        var stdin = new StreamReader(Console.OpenStandardInput(), utf8);
        return Run(args, stdin, protocolOut, stderr, settings => new SongPipeline(settings));
    }
}
