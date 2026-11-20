using System.Collections.Concurrent;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace HeartMuLa.Server.Tests;

public sealed record Received(string Path, string Body, Dictionary<string, string> Headers);

/// <summary>A real HTTP receiver on loopback that records what it is sent and answers as a test scripts.</summary>
public sealed class FakeReceiver : IAsyncDisposable
{
    private readonly WebApplication app;
    public string Url { get; private set; } = "";
    public ConcurrentQueue<Received> Requests { get; } = new();

    /// <summary>Called with the request and its 1-based number; the default answers 200.</summary>
    public Func<HttpContext, int, Task> Behavior { get; set; } = (ctx, _) => { ctx.Response.StatusCode = 200; return Task.CompletedTask; };

    public FakeReceiver()
    {
        var builder = WebApplication.CreateBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        app = builder.Build();
        app.Run(async ctx =>
        {
            using var reader = new StreamReader(ctx.Request.Body);
            var body = await reader.ReadToEndAsync();
            Requests.Enqueue(new Received(ctx.Request.Path, body, ctx.Request.Headers.ToDictionary(h => h.Key, h => h.Value.ToString(), StringComparer.OrdinalIgnoreCase)));
            await Behavior(ctx, Requests.Count);
        });
    }

    public async Task<FakeReceiver> StartAsync()
    {
        await app.StartAsync();
        Url = app.Urls.First();
        return this;
    }

    public async ValueTask DisposeAsync()
    {
        await app.StopAsync();
        await app.DisposeAsync();
    }
}
