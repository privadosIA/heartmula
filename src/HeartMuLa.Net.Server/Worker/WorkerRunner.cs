using System.Diagnostics;
using System.Text;
using HeartMuLa.Server.Jobs;

namespace HeartMuLa.Server.Worker;

/// <summary>Builds the command that starts a worker: the program and its arguments, for a mode and a WAV path.</summary>
public delegate (string Program, IReadOnlyList<string> Arguments) WorkerCommand(string mode, string wavPath);

/// <summary>The song process ended without making the song; the message is what the job's <c>error</c> will say.</summary>
public sealed class SongProcessException : Exception
{
    /// <summary>Creates the exception with the message the job's <c>error</c> will carry.</summary>
    public SongProcessException(string message) : base(message) { }
}

/// <summary>
/// Starts a worker process, speaks <see cref="WorkerProtocol"/> with it and turns its end into a result: the song is written, or an exception says why not.
/// Cancelling kills the whole process tree at once, which is what returns all of its memory (video memory included) to the system.
/// </summary>
public static class WorkerRunner
{
    private const int StderrLines = 20;

    /// <summary>This executable as a worker: <c>dotnet service.dll --worker mode wav</c> when run through the dotnet host, the program itself otherwise.</summary>
    public static (string Program, IReadOnlyList<string> Arguments) CurrentExecutable(string mode, string wavPath)
    {
        var path = Environment.ProcessPath ?? throw new InvalidOperationException("the path of this program is not known");
        bool host = Path.GetFileNameWithoutExtension(path).Equals("dotnet", StringComparison.OrdinalIgnoreCase);
        return host
            ? (path, new[] { typeof(WorkerRunner).Assembly.Location, WorkerProtocol.Argument, mode, wavPath })
            : (path, new[] { WorkerProtocol.Argument, mode, wavPath });
    }

    /// <summary>
    /// Runs one worker to its end. Returns when it exited with <see cref="WorkerProtocol.Done"/> and wrote the WAV; throws
    /// <see cref="OperationCanceledException"/> when <paramref name="ct"/> was cancelled and <see cref="SongProcessException"/> for any other end.
    /// A partial WAV is never left behind.
    /// </summary>
    public static void Run(WorkerCommand command, WorkerRequest request, string wavPath, Action<JobStage, int, int> progress, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var (program, arguments) = command(request.Mode, wavPath);
        var utf8 = new UTF8Encoding(false);
        var info = new ProcessStartInfo(program)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            StandardInputEncoding = utf8,
            StandardOutputEncoding = utf8,
            StandardErrorEncoding = utf8,
        };
        foreach (var a in arguments) info.ArgumentList.Add(a);

        Process process;
        try { process = Process.Start(info) ?? throw new SongProcessException("the song process could not be started"); }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException)
        { throw new SongProcessException($"the song process could not be started: {e.Message}"); }

        using (process)
        using (ct.Register(() => Kill(process)))
        {
            try
            {
                // One line, and standard input then stays open until the process ends: that is how the worker knows the service is still there.
                process.StandardInput.WriteLine(WorkerProtocol.EncodeRequest(request));
                process.StandardInput.Flush();
            }
            catch (IOException) { /* the worker already ended; its exit code says why */ }

            var tail = new Queue<string>();
            var stdout = Task.Run(() =>
            {
                string? line;
                while ((line = process.StandardOutput.ReadLine()) is not null)
                {
                    if (!WorkerProtocol.TryDecodeProgress(line, out var p)) continue;           // anything else is noise and is ignored
                    try { progress(p.Stage, p.Current, p.Total); }
                    catch (Exception) { /* a cancelled job refuses progress; the kill is already on its way */ }
                }
                // No token on purpose: cancelling kills the process, which closes the stream and ends this loop.
            }, CancellationToken.None);
            var stderr = Task.Run(() =>
            {
                string? line;
                while ((line = process.StandardError.ReadLine()) is not null)
                {
                    if (line.Length == 0) continue;
                    lock (tail) { tail.Enqueue(line); if (tail.Count > StderrLines) tail.Dequeue(); }
                }
            }, CancellationToken.None);

            process.WaitForExit();
            // Both readers end when the process does (a cancel kills it), so there is nothing to cancel here.
            Task.WaitAll(new[] { stdout, stderr }, CancellationToken.None);

            if (ct.IsCancellationRequested)
            {
                Remove(wavPath);
                throw new OperationCanceledException(ct);
            }
            string? last;
            lock (tail) last = tail.Count > 0 ? tail.Last() : null;
            int code = process.ExitCode;
            if (code == WorkerProtocol.Done)
            {
                if (!File.Exists(wavPath)) throw new SongProcessException("the song process ended without writing the song");
                return;
            }
            Remove(wavPath);
            if (code == WorkerProtocol.Failed && last is not null) throw new SongProcessException(last);
            throw new SongProcessException($"the song process ended unexpectedly (exit code {code})" + (last is null ? "" : ": " + last));
        }
    }

    private static void Kill(Process process)
    {
        try { process.Kill(entireProcessTree: true); }
        catch (InvalidOperationException) { }
        catch (System.ComponentModel.Win32Exception) { }
    }

    private static void Remove(string wavPath)
    {
        foreach (var path in new[] { wavPath, wavPath + ".tmp" })
            try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }
}
