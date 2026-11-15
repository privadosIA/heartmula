using HeartMuLa.Server.Worker;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace HeartMuLa.Server.Tests;

/// <summary>The whole service in memory, with the model replaced by <see cref="FakeSongGenerator"/> and results in a temp directory.</summary>
public sealed class ApiFactory : WebApplicationFactory<Program>
{
    public string Dir { get; set; } = Path.Combine(Path.GetTempPath(), "heartmula-api-" + Guid.NewGuid().ToString("N"));
    public FakeSongGenerator Generator { get; } = new();
    /// <summary>When false the host keeps the generator it registers for the configured isolation (tests then replace pieces through <see cref="Configure"/>).</summary>
    public bool UseFakeGenerator { get; init; } = true;
    public Action<IServiceCollection>? Configure { get; init; }
    private readonly Dictionary<string, string> settings;

    public ApiFactory(params (string Key, string Value)[] settings) : this(null, settings) { }

    public ApiFactory(string? dir, params (string Key, string Value)[] settings)
    {
        if (dir is not null) Dir = dir;
        this.settings = settings.ToDictionary(s => s.Key, s => s.Value);
    }

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseSetting("Storage:Directory", Dir);
        foreach (var (k, v) in settings) builder.UseSetting(k, v);
        builder.ConfigureTestServices(services =>
        {
            if (UseFakeGenerator)
            {
                services.RemoveAll<ISongGenerator>();
                services.AddSingleton<ISongGenerator>(Generator);
            }
            Configure?.Invoke(services);
        });
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (disposing && Directory.Exists(Dir)) Directory.Delete(Dir, recursive: true);
    }
}
