using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Server.Options;

/// <summary>A setting (or the environment it is read from) is wrong; the message names it. The host prints it and exits.</summary>
public sealed class ConfigurationException : Exception
{
    public ConfigurationException(string message) : base(message) { }
}

public sealed class ModelOptions
{
    public const string Section = "Model";
    /// <summary>Checkpoint directory (the one with HeartMuLa-oss-3B/, HeartCodec-oss/ and tokenizer.json). Falls back to HEARTMULA_CKPT.</summary>
    public string? Checkpoint { get; set; }
    /// <summary>Device of the language model, e.g. cuda:0. Empty = cuda:0 when CUDA is available, otherwise cpu.</summary>
    public string? Device { get; set; }
    public string Version { get; set; } = "3B";
}

public sealed class CodecOptions
{
    public const string Section = "Codec";
    /// <summary>Device of the codec. Empty = the model's device.</summary>
    public string? Device { get; set; }
    /// <summary>cuDNN TF32 for the codec. Off by default so GPU audio matches the CPU's (see the README).</summary>
    public bool Tf32 { get; set; }
}

public sealed class StorageOptions
{
    public const string Section = "Storage";
    public string Directory { get; set; } = "songs";
    public double RetentionHours { get; set; } = 24;
    public double SweepMinutes { get; set; } = 10;
}

public sealed class JobsOptions
{
    public const string Section = "Jobs";
    public int DefaultDurationMs { get; set; } = 30_000;
    /// <summary>The backbone has 8192 positions; 360 s are 4500 frames, inside with the prompt. It is also the ceiling the orchestrator's service accepts.</summary>
    public int MaxDurationMs { get; set; } = 360_000;
    public int MaxQueued { get; set; } = 16;
    public double DefaultCfgScale { get; set; } = 1.5;
}

/// <summary>Where the work of a song runs. See <c>song-worker-process</c>.</summary>
public enum IsolationMode
{
    /// <summary>The whole song, model load included, in a worker process: both GPUs are free between songs.</summary>
    Process,
    /// <summary>The model stays loaded in the service; only the codec decode runs in a worker process.</summary>
    Codec,
    /// <summary>Everything in the service process (nothing is returned to the GPU until it exits).</summary>
    InProcess,
}

public sealed class ServiceInfoOptions
{
    public const string Section = "Service";
    /// <summary><c>codec</c> (default), <c>process</c> or <c>inprocess</c>. Alias: HEARTMULA_ISOLATION.</summary>
    public string? Isolation { get; set; }
    /// <summary>Version string reported by /health (the orchestrator compares it with the one it was made for).</summary>
    public string Version { get; set; } = "1";
}

public sealed class SecurityOptions
{
    public const string Section = "Security";
    /// <summary>Empty = no authentication (the service then listens on loopback unless told otherwise).</summary>
    public string? ApiKey { get; set; }
}

public sealed class WebhookOptions
{
    public const string Section = "Webhook";
    /// <summary>When set, every notification carries X-HeartMuLa-Signature: sha256=HMAC-SHA256(body).</summary>
    public string? Secret { get; set; }
    /// <summary>Hosts a callbackUrl may point to. Empty = any.</summary>
    public string[] AllowedHosts { get; set; } = Array.Empty<string>();
    public int TimeoutSeconds { get; set; } = 10;
    /// <summary>
    /// Waits before each retry; the delivery makes 1 + length attempts. Null (not set) means <see cref="DefaultRetryDelaysSeconds"/>. The default is not
    /// the property's initial value because the configuration binder appends to an array that already has elements instead of replacing it.
    /// </summary>
    public int[]? RetryDelaysSeconds { get; set; }
    public static readonly int[] DefaultRetryDelaysSeconds = { 1, 5, 30 };
    public int[] EffectiveRetryDelaysSeconds => RetryDelaysSeconds ?? DefaultRetryDelaysSeconds;
}

public sealed class ServiceOptions
{
    public ModelOptions Model { get; init; } = new();
    public CodecOptions Codec { get; init; } = new();
    public StorageOptions Storage { get; init; } = new();
    public JobsOptions Jobs { get; init; } = new();
    public SecurityOptions Security { get; init; } = new();
    public ServiceInfoOptions Service { get; init; } = new();

    /// <summary>The isolation mode, parsed (the default is <see cref="IsolationMode.Codec"/>). <see cref="From"/> has already checked the text.</summary>
    public IsolationMode IsolationMode => ParseIsolation(Service.Isolation, "Service:Isolation") ?? IsolationMode.Codec;
    public WebhookOptions Webhook { get; init; } = new();

    public const string DefaultUrl = "http://127.0.0.1:5080";

    public static ServiceOptions From(IConfiguration c)
    {
        var o = new ServiceOptions
        {
            Model = c.GetSection(ModelOptions.Section).Get<ModelOptions>() ?? new(),
            Codec = c.GetSection(CodecOptions.Section).Get<CodecOptions>() ?? new(),
            Storage = c.GetSection(StorageOptions.Section).Get<StorageOptions>() ?? new(),
            Jobs = c.GetSection(JobsOptions.Section).Get<JobsOptions>() ?? new(),
            Security = c.GetSection(SecurityOptions.Section).Get<SecurityOptions>() ?? new(),
            Service = c.GetSection(ServiceInfoOptions.Section).Get<ServiceInfoOptions>() ?? new(),
            Webhook = c.GetSection(WebhookOptions.Section).Get<WebhookOptions>() ?? new(),
        };
        o.Model.Checkpoint = string.IsNullOrWhiteSpace(o.Model.Checkpoint) ? c["HEARTMULA_CKPT"] : o.Model.Checkpoint;

        // The variables the container service uses: each is only a fallback, a setting given in the service's own form wins.
        if (string.IsNullOrEmpty(o.Security.ApiKey)) o.Security.ApiKey = c["HEARTMULA_TOKEN"];
        if (string.IsNullOrEmpty(c["Storage:Directory"]) && !string.IsNullOrEmpty(c["HEARTMULA_JOBS"])) o.Storage.Directory = c["HEARTMULA_JOBS"]!;
        if (string.IsNullOrEmpty(c["Service:Version"]) && !string.IsNullOrEmpty(c["HEARTMULA_VERSION"])) o.Service.Version = c["HEARTMULA_VERSION"]!;
        if (string.IsNullOrEmpty(o.Service.Isolation) && !string.IsNullOrEmpty(c["HEARTMULA_ISOLATION"])) o.Service.Isolation = c["HEARTMULA_ISOLATION"];
        ParseIsolation(o.Service.Isolation, string.IsNullOrEmpty(c["Service:Isolation"]) && !string.IsNullOrEmpty(c["HEARTMULA_ISOLATION"]) ? "HEARTMULA_ISOLATION" : "Service:Isolation");
        return o;
    }

    /// <summary>Null for an empty text (the default applies); throws <see cref="ConfigurationException"/> naming <paramref name="setting"/> for an unknown value.</summary>
    public static IsolationMode? ParseIsolation(string? text, string setting)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;
        return text.Trim().ToLowerInvariant() switch
        {
            "process" => IsolationMode.Process,
            "codec" => IsolationMode.Codec,
            "inprocess" => IsolationMode.InProcess,
            _ => throw new ConfigurationException($"{setting} '{text}' is not one of: process, codec, inprocess"),
        };
    }

    /// <summary>
    /// The address to listen on when <c>HEARTMULA_PORT</c> is set: all interfaces on that port, like the container. Null when it is not set or when
    /// <c>Urls</c> was changed from its default (the service's own setting wins). Throws <see cref="ConfigurationException"/> for a bad port.
    /// </summary>
    public static string? UrlFromPortAlias(IConfiguration c)
    {
        var port = c["HEARTMULA_PORT"];
        if (string.IsNullOrWhiteSpace(port)) return null;
        if (!int.TryParse(port, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out var p) || p < 1 || p > 65535)
            throw new ConfigurationException($"HEARTMULA_PORT '{port}' is not a port number (1 to 65535)");
        var urls = c["Urls"];
        return string.IsNullOrEmpty(urls) || urls == DefaultUrl ? $"http://0.0.0.0:{p}" : null;
    }

    /// <summary>Device of the model after applying the default (cuda:0 when CUDA is available).</summary>
    public string EffectiveModelDevice(bool cudaAvailable) => string.IsNullOrWhiteSpace(Model.Device) ? (cudaAvailable ? "cuda:0" : "cpu") : Model.Device!;
    public string EffectiveCodecDevice(bool cudaAvailable) => string.IsNullOrWhiteSpace(Codec.Device) ? EffectiveModelDevice(cudaAvailable) : Codec.Device!;

    /// <summary>Every problem found, each naming the setting. <paramref name="cudaDeviceCount"/> is the number of CUDA devices (0 without CUDA).</summary>
    public IReadOnlyList<string> Validate(int cudaDeviceCount, bool requireCheckpoint = true)
    {
        var errors = new List<string>();
        if (requireCheckpoint)
        {
            if (string.IsNullOrWhiteSpace(Model.Checkpoint)) errors.Add("Model:Checkpoint is required (or set HEARTMULA_CKPT)");
            else if (!Directory.Exists(Model.Checkpoint)) errors.Add($"Model:Checkpoint '{Model.Checkpoint}' is not an existing directory");
        }
        CheckDevice("Model:Device", EffectiveModelDevice(cudaDeviceCount > 0), cudaDeviceCount, errors);
        CheckDevice("Codec:Device", EffectiveCodecDevice(cudaDeviceCount > 0), cudaDeviceCount, errors);

        if (string.IsNullOrWhiteSpace(Storage.Directory)) errors.Add("Storage:Directory must not be empty");
        if (!(Storage.RetentionHours > 0)) errors.Add("Storage:RetentionHours must be greater than 0");
        if (!(Storage.SweepMinutes > 0)) errors.Add("Storage:SweepMinutes must be greater than 0");

        if (Jobs.DefaultDurationMs < 1000) errors.Add("Jobs:DefaultDurationMs must be at least 1000");
        if (Jobs.MaxDurationMs < Jobs.DefaultDurationMs) errors.Add("Jobs:MaxDurationMs must not be smaller than Jobs:DefaultDurationMs");
        if (Jobs.MaxQueued < 1) errors.Add("Jobs:MaxQueued must be at least 1");
        if (!(Jobs.DefaultCfgScale > 0)) errors.Add("Jobs:DefaultCfgScale must be greater than 0");

        if (Webhook.TimeoutSeconds < 1) errors.Add("Webhook:TimeoutSeconds must be at least 1");
        if (Webhook.EffectiveRetryDelaysSeconds.Any(s => s < 0)) errors.Add("Webhook:RetryDelaysSeconds must not contain negative values");
        foreach (var host in Webhook.AllowedHosts)
            if (string.IsNullOrWhiteSpace(host) || host.Contains("://") || host.Contains('/') || host.Contains(' '))
                errors.Add($"Webhook:AllowedHosts entry '{host}' must be a bare host name such as 'orchestrator.local'");
        return errors;
    }

    private static void CheckDevice(string setting, string device, int cudaDeviceCount, List<string> errors)
    {
        torch.Device parsed;
        try { parsed = new torch.Device(device); }
        catch (Exception e) when (e is ArgumentException or InvalidOperationException or FormatException)
        {
            errors.Add($"{setting} '{device}' is not a valid device: {e.Message}");
            return;
        }
        if (parsed.type == DeviceType.CUDA)
        {
            if (cudaDeviceCount == 0) errors.Add($"{setting} '{device}' needs CUDA, which is not available");
            else if (parsed.index >= cudaDeviceCount) errors.Add($"{setting} '{device}' does not exist: only {cudaDeviceCount} CUDA device(s)");
        }
    }

    /// <summary>True when the host name is allowed by <c>Webhook:AllowedHosts</c> (an empty list allows any).</summary>
    public bool IsWebhookHostAllowed(string host) =>
        Webhook.AllowedHosts.Length == 0 || Webhook.AllowedHosts.Contains(host, StringComparer.OrdinalIgnoreCase);
}
