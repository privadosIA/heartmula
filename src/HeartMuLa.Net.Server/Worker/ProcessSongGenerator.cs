using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;

namespace HeartMuLa.Server.Worker;

/// <summary>
/// <c>process</c> isolation: every song, model load included, is made by a short-lived worker process, so both GPUs are free between songs. The
/// service process itself loads no model and creates no GPU context (it asks <c>nvidia-smi</c> how many devices there are). There is no loading phase:
/// the generator is ready when the checkpoint files are there and the configured devices exist.
/// </summary>
public sealed class ProcessSongGenerator : ISongGenerator
{
    private static readonly string[] RequiredEntries = { "gen_config.json", "tokenizer.json", "HeartCodec-oss" };

    private readonly ServiceOptions options;
    private readonly IGpuProbe probe;
    private readonly WorkerCommand command;

    public ProcessSongGenerator(ServiceOptions options, IGpuProbe probe, WorkerCommand? command = null)
    {
        this.options = options;
        this.probe = probe;
        this.command = command ?? WorkerRunner.CurrentExecutable;
    }

    public bool LoadsPerSong => true;

    private int CudaCount => probe.DeviceCount();

    public DeviceInfo Devices
    {
        get
        {
            bool cuda = CudaCount > 0;
            return new(options.EffectiveModelDevice(cuda), options.EffectiveCodecDevice(cuda));
        }
    }

    public IReadOnlyList<string> Missing
    {
        get
        {
            var ckpt = options.Model.Checkpoint ?? "";
            return RequiredEntries.Append($"HeartMuLa-oss-{options.Model.Version}")
                .Where(e => !File.Exists(Path.Combine(ckpt, e)) && !Directory.Exists(Path.Combine(ckpt, e))).ToList();
        }
    }

    private IReadOnlyList<string> Problems() => options.Validate(CudaCount);

    public ModelStatus Status => Problems().Count == 0 && Missing.Count == 0 ? ModelStatus.Ready : ModelStatus.Failed;

    public string? LoadError
    {
        get
        {
            var problems = Problems().ToList();
            if (Missing.Count > 0) problems.Add("the checkpoint lacks " + string.Join(", ", Missing));
            return problems.Count == 0 ? null : string.Join("; ", problems);
        }
    }

    /// <summary>Nothing is loaded: the model belongs to the worker of each song.</summary>
    public void Load() { }

    public GeneratedSong Generate(SongRequest request, string wavPath, Action<JobStage, int, int> progress, CancellationToken ct)
    {
        var devices = Devices;
        var settings = new WorkerSettings(options.Model.Checkpoint ?? "", options.Model.Version, devices.Model, devices.Codec, options.Codec.Tf32);
        var worker = new WorkerRequest(WorkerProtocol.SongMode, request.Tags, request.Lyrics, request.DurationMs, request.Seed, request.CfgScale,
                                       request.Temperature, request.TopK, null, settings);
        WorkerRunner.Run(command, worker, wavPath, progress, ct);
        return new GeneratedSong(WavInfo.DurationMs(wavPath));
    }
}
