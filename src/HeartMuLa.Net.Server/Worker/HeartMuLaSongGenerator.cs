using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Server.Worker;

/// <summary>
/// The real generator. It repeats, in the CLI's order, exactly what <c>generate --wav</c> does (preprocess → seed → generate → frames to CPU →
/// release the generation → codec decode), so the same checkpoint, inputs, seed and devices give the same WAV. The model stays loaded between jobs;
/// only the caches and the song's tensors are released.
/// </summary>
public sealed class HeartMuLaSongGenerator : ISongGenerator, IDisposable
{
    private readonly ServiceOptions options;
    private readonly ILogger<HeartMuLaSongGenerator> log;
    private readonly Device modelDevice, codecDevice;
    private readonly IWorkerSteps steps;
    private readonly IFrameDecoder? decoder;
    private volatile ModelStatus status = ModelStatus.Loading;
    private volatile string? loadError;

    /// <summary>Creates the generator; the model is not read until <see cref="Load"/> is called.</summary>
    /// <param name="options">The service settings.</param>
    /// <param name="log">Logger for the loading messages.</param>
    /// <param name="decoder">Where the frames are decoded; null decodes in this process (<c>inprocess</c> isolation), a <see cref="WorkerFrameDecoder"/> in a worker (<c>codec</c>).</param>
    /// <param name="steps">The model-touching steps; null uses a <see cref="SongPipeline"/> for the resolved settings (tests pass a stand-in).</param>
    public HeartMuLaSongGenerator(ServiceOptions options, ILogger<HeartMuLaSongGenerator> log, IFrameDecoder? decoder = null, IWorkerSteps? steps = null)
    {
        this.decoder = decoder;
        this.options = options;
        this.log = log;
        Settings = ResolveSettings(options);
        modelDevice = new Device(Settings.ModelDevice);
        codecDevice = new Device(Settings.CodecDevice);
        this.steps = steps ?? new SongPipeline(Settings);
    }

    /// <summary>The settings a song needs with the devices resolved (cuda:0 by default when CUDA is available, otherwise cpu).</summary>
    public static WorkerSettings ResolveSettings(ServiceOptions options)
    {
        bool cuda = torch.cuda.is_available();
        return new WorkerSettings(options.Model.Checkpoint ?? "", options.Model.Version, options.EffectiveModelDevice(cuda), options.EffectiveCodecDevice(cuda), options.Codec.Tf32);
    }

    /// <summary>The resolved settings (devices included), as a worker receives them.</summary>
    public WorkerSettings Settings { get; }

    /// <summary>True in <c>codec</c> isolation: the frames are decoded by a worker process instead of here.</summary>
    public bool DecodesInWorker => decoder is not null;

    /// <inheritdoc/>
    public ModelStatus Status => status;
    /// <inheritdoc/>
    public string? LoadError => loadError;
    /// <inheritdoc/>
    public DeviceInfo Devices => new(modelDevice.ToString(), codecDevice.ToString());

    private static readonly string[] RequiredEntries = { "gen_config.json", "tokenizer.json", "HeartCodec-oss" };

    /// <inheritdoc/>
    public IReadOnlyList<string> Missing
    {
        get
        {
            var ckpt = options.Model.Checkpoint ?? "";
            return RequiredEntries.Append($"HeartMuLa-oss-{options.Model.Version}").Where(e => !File.Exists(Path.Combine(ckpt, e)) && !Directory.Exists(Path.Combine(ckpt, e))).ToList();
        }
    }

    /// <summary>Checks the settings that need the machine (checkpoint, devices) and returns the problems, naming each setting.</summary>
    public static IReadOnlyList<string> ValidateSettings(ServiceOptions options)
    {
        int cudaCount = torch.cuda.is_available() ? (int)torch.cuda.device_count() : 0;
        return options.Validate(cudaCount);
    }

    /// <inheritdoc/>
    public void Load()
    {
        try
        {
            log.LogInformation("Loading HeartMuLa from {Dir} on {Device}", Path.Combine(options.Model.Checkpoint!, $"HeartMuLa-oss-{options.Model.Version}"), modelDevice);
            steps.LoadModel();
            status = ModelStatus.Ready;
            log.LogInformation("HeartMuLa ready (model on {Model}, codec on {Codec})", modelDevice, codecDevice);
        }
        catch (Exception e)
        {
            loadError = e.Message;
            status = ModelStatus.Failed;
            log.LogError(e, "Loading the model failed");
        }
    }

    /// <inheritdoc/>
    public GeneratedSong Generate(SongRequest request, string wavPath, Action<JobStage, int, int> progress, CancellationToken ct)
    {
        using var frames = steps.GenerateFrames(request, progress);
        // On one shared card the weights must leave before the codec arrives, and a worker could not use the memory this process keeps cached: decode here.
        bool inWorker = decoder is not null && !(steps is SongPipeline { SharesDevice: true });
        return new GeneratedSong(!inWorker ? steps.DecodeFrames(frames, wavPath, progress) : decoder!.Decode(frames, request, wavPath, progress, ct));
    }

    /// <summary>Releases the model and the tokenizer.</summary>
    public void Dispose() => steps.Dispose();
}
