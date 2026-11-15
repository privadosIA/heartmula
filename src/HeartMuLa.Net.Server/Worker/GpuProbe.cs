using System.Diagnostics;
using HeartMuLa.Server.Api;

namespace HeartMuLa.Server.Worker;

/// <summary>Reads the name and memory of the GPU the model runs on, for /health. TorchSharp 0.107 has no <c>mem_get_info</c>, so the real probe asks nvidia-smi.</summary>
public interface IGpuProbe
{
    /// <summary>Null when <paramref name="device"/> is not a CUDA device or the numbers cannot be read.</summary>
    GpuInfo? Read(string device);

    /// <summary>How many CUDA devices the machine has (0 when there is no driver or none can be read), without creating a context.</summary>
    int DeviceCount();
}

public sealed class NvidiaSmiGpuProbe : IGpuProbe
{
    private static readonly TimeSpan Ttl = TimeSpan.FromSeconds(5);
    private readonly object gate = new();
    private DateTime readAt = DateTime.MinValue;
    private List<GpuInfo> cache = new();

    public GpuInfo? Read(string device)
    {
        if (!device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase)) return null;
        var parts = device.Split(':');
        int index = parts.Length > 1 && int.TryParse(parts[1], out var i) ? i : 0;
        lock (gate)
        {
            if (DateTime.UtcNow - readAt > Ttl) { cache = Query(); readAt = DateTime.UtcNow; }
            return index < cache.Count ? cache[index] : null;
        }
    }

    public int DeviceCount()
    {
        lock (gate)
        {
            if (DateTime.UtcNow - readAt > Ttl) { cache = Query(); readAt = DateTime.UtcNow; }
            return cache.Count;
        }
    }

    /// <summary>One entry per line of <c>nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits</c>.</summary>
    public static List<GpuInfo> Parse(string output)
    {
        var list = new List<GpuInfo>();
        foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var f = line.Split(',', StringSplitOptions.TrimEntries);
            if (f.Length < 3) { list.Add(new GpuInfo(f.FirstOrDefault(), null, null)); continue; }
            list.Add(new GpuInfo(f[0], long.TryParse(f[1], out var free) ? free : null, long.TryParse(f[2], out var total) ? total : null));
        }
        return list;
    }

    private static List<GpuInfo> Query()
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", "--query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits")
            { RedirectStandardOutput = true, RedirectStandardError = true };
            using var p = Process.Start(psi);
            if (p is null) return new();
            var text = p.StandardOutput.ReadToEnd();
            if (!p.WaitForExit(3000)) { p.Kill(); return new(); }
            return p.ExitCode == 0 ? Parse(text) : new();
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or IOException or InvalidOperationException) { return new(); }
    }
}

public sealed class NoGpuProbe : IGpuProbe
{
    public GpuInfo? Read(string device) => null;
    public int DeviceCount() => 0;
}
