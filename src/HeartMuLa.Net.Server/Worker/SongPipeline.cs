using HeartMuLa.Model;
using HeartMuLa.Pipeline;
using HeartMuLa.Server.Jobs;
using MusicGen.Core;
using MusicGen.Core.Audio;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Server.Worker;

/// <summary>
/// The steps of a song that touch the model and the codec, in the order the CLI runs them (preprocess → seed → generate → frames to CPU → release the
/// generation → codec decode). They know nothing about where they run: the service process calls them in sequence for <c>inprocess</c> isolation, a
/// worker process does for <c>process</c>, and a codec worker calls only <see cref="DecodeFrames"/>.
/// </summary>
public sealed class SongPipeline : IWorkerSteps
{
    private readonly WorkerSettings settings;
    private readonly torch.Device modelDevice, codecDevice;
    private HeartMuLaModel? model;
    private HeartMuLaTokenizer? tokenizer;
    private MusicGenerationPipeline? pipeline;

    public SongPipeline(WorkerSettings settings)
    {
        this.settings = settings;
        modelDevice = new torch.Device(settings.ModelDevice);
        codecDevice = new torch.Device(settings.CodecDevice);
    }

    public bool IsLoaded => model is not null;

    /// <summary>True when the model and the codec share a CUDA device: they do not fit together, so (like the CLI) the weights go before the decode and come back for the next song.</summary>
    public bool SharesDevice => modelDevice.type == DeviceType.CUDA && modelDevice.ToString() == codecDevice.ToString();

    private bool unloaded;

    /// <summary>Reads the HeartMuLa weights and the tokenizer onto the model device (throws on failure).</summary>
    public void LoadModel()
    {
        var ckpt = settings.Checkpoint;
        var dtype = ScalarType.BFloat16;
        var mulaDir = Path.Combine(ckpt, $"HeartMuLa-oss-{settings.Version}");
        var m = HeartMuLaModel.Create(HeartMuLaConfig.FromFile(Path.Combine(mulaDir, "config.json")), dtype);
        m.eval();
        m.LoadWeights(mulaDir, dtype, modelDevice);
        tokenizer = new HeartMuLaTokenizer(Path.Combine(ckpt, "tokenizer.json"));
        var gen = GenConfig.FromFile(Path.Combine(ckpt, "gen_config.json"));
        pipeline = new MusicGenerationPipeline(m, tokenizer, gen, modelDevice, dtype);
        model = m;
    }

    /// <summary>Preprocess, seed, generate and bring the frames <c>[8, T]</c> to the CPU; the caches and prompt tensors are released (the weights stay).</summary>
    public Tensor GenerateFrames(SongRequest request, Action<JobStage, int, int> progress)
    {
        if (unloaded) { LoadModel(); unloaded = false; }
        if (model is null || pipeline is null) throw new InvalidOperationException("the model is not loaded");

        Tensor? frames = null;
        PromptInputs? inputs = null;
        try
        {
            inputs = pipeline.Preprocess(request.Tags, request.Lyrics, request.CfgScale);
            Seeding.Apply(request.Seed);                              // right before Generate, as the CLI does
            frames = pipeline.Generate(inputs, maxAudioLengthMs: request.DurationMs, cfgScale: request.CfgScale, temperature: request.Temperature, topk: request.TopK,
                progress: (i, n) => progress(JobStage.Generating, i, n));
            return frames.cpu().clone();                              // clone: on a CPU run cpu() is the same tensor and the release disposes it
        }
        finally
        {
            // Keeps the weights; frees the caches, masks and prompt tensors (also when generation was cancelled or failed).
            AudioStage.ReleaseGeneration(model, frames, inputs?.Tokens, inputs?.TokensMask, inputs?.MuqEmbed, inputs?.Pos);
        }
    }

    /// <summary>Reads the frames <c>[8, T]</c> a service handed over in a safetensors file (the codec worker's input).</summary>
    public Tensor LoadFrames(string path) => SafeTensors.LoadFile(path)["frames"].to(ScalarType.Int64);

    /// <summary>Decodes the frames with HeartCodec on the codec device into a PCM16 WAV at <paramref name="wavPath"/>; returns the length in milliseconds.</summary>
    public int DecodeFrames(Tensor framesCpu, string wavPath, Action<JobStage, int, int> progress)
    {
        if (SharesDevice && model is not null) { Dispose(); unloaded = true; }
        var audio = new AudioOptions(wavPath, WavFormat.Pcm16, codecDevice.ToString(), settings.CodecTf32);
        long samples = AudioStage.DecodeToWav(framesCpu, settings.Checkpoint, codecDevice, audio, log: null,
            progress: (w, n) => progress(JobStage.Decoding, w + 1, n));
        return (int)(samples * 1000 / HeartMuLa.Codec.DetokenizePlan.SampleRate);
    }

    public void Dispose()
    {
        model?.Dispose();
        tokenizer?.Dispose();
        model = null; tokenizer = null; pipeline = null;
    }
}
