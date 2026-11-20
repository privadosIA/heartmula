using System.Text.Json.Serialization;
using HeartMuLa.Server.Api;
using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;
using HeartMuLa.Server.Storage;
using HeartMuLa.Server.Webhook;
using HeartMuLa.Server.Worker;
using Microsoft.OpenApi.Models;

namespace HeartMuLa.Server;

/// <summary>Builds the web application: registers the stores, the generator for the configured isolation mode, the background services and Swagger, and maps the endpoints.</summary>
public static class ServiceHost
{
    /// <summary>
    /// Builds the service. <paramref name="configureServices"/> runs after the standard registrations, so a caller can replace parts of it (for example
    /// <see cref="ISongGenerator"/> with a stand-in that needs no GPU).
    /// </summary>
    public static WebApplication Build(string[] args, Action<IServiceCollection>? configureServices = null)
    {
        var builder = WebApplication.CreateBuilder(args);
        if (ServiceOptions.UrlFromPortAlias(builder.Configuration) is { } url) builder.WebHost.UseUrls(url);
        var options = ServiceOptions.From(builder.Configuration);
        builder.Services.AddSingleton(options);
        builder.Services.AddSingleton(TimeProvider.System);
        builder.Services.AddSingleton<JobStore>();
        builder.Services.AddSingleton(sp => new ResultStore(options.Storage, sp.GetRequiredService<TimeProvider>()));
        builder.Services.AddSingleton<WebhookDispatcher>();
        builder.Services.AddSingleton<IJobEndedSink>(sp => sp.GetRequiredService<WebhookDispatcher>());
        builder.Services.AddSingleton<IGpuProbe, NvidiaSmiGpuProbe>();
        builder.Services.AddSingleton<SongService>();
        builder.Services.AddSingleton<ISongGenerator>(sp => options.IsolationMode switch
        {
            // The service process loads no model and creates no GPU context: each song (model load included) is made by a worker process.
            IsolationMode.Process => new ProcessSongGenerator(options, sp.GetRequiredService<IGpuProbe>()),
            IsolationMode.InProcess => CreateInProcessGenerator(sp, options, codecInWorker: false),
            _ => CreateInProcessGenerator(sp, options, codecInWorker: true),                  // IsolationMode.Codec
        });
        builder.Services.AddHostedService<Maintenance>();                                    // first: restores the jobs on disk
        builder.Services.AddHostedService(sp => sp.GetRequiredService<WebhookDispatcher>());   // second: resumes pending deliveries
        builder.Services.AddHostedService<GenerationWorker>();

        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen(c =>
        {
            c.SwaggerDoc("v1", new OpenApiInfo
            {
                Title = "HeartMuLa song service",
                Version = "v1",
                Description = "Asks the HeartMuLa music model for songs. A song takes seconds to minutes, so a request creates a job: poll it, "
                            + "download the WAV when it has succeeded, or give a callbackUrl to be notified. One song is generated at a time.",
            });
            c.IncludeXmlComments(Path.Combine(AppContext.BaseDirectory, $"{typeof(Program).Assembly.GetName().Name}.xml"));
            c.AddSecurityDefinition("ApiKey", new OpenApiSecurityScheme
            {
                Type = SecuritySchemeType.ApiKey,
                In = ParameterLocation.Header,
                Name = ApiKeyMiddleware.HeaderName,
                Description = "Only needed when the service is configured with Security:ApiKey. /health and this page are open.",
            });
            c.AddSecurityRequirement(new OpenApiSecurityRequirement
            {
                [new OpenApiSecurityScheme { Reference = new OpenApiReference { Type = ReferenceType.SecurityScheme, Id = "ApiKey" } }] = Array.Empty<string>(),
            });
            c.SchemaFilter<LowerCaseEnumSchemaFilter>();
        });
        builder.Services.AddProblemDetails();
        builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull);

        configureServices?.Invoke(builder.Services);

        var app = builder.Build();
        app.UseExceptionHandler();
        app.UseStatusCodePages();
        app.UseMiddleware<ApiKeyMiddleware>();
        app.UseSwagger();
        app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "HeartMuLa song service v1"));
        app.MapGet("/", () => Results.Redirect("/swagger")).ExcludeFromDescription();
        app.MapSongEndpoints();
        app.MapCompatEndpoints();
        return app;
    }

    /// <summary>The generator that keeps the model in this process; with <paramref name="codecInWorker"/> the codec decode runs in a worker process.</summary>
    private static HeartMuLaSongGenerator CreateInProcessGenerator(IServiceProvider sp, ServiceOptions options, bool codecInWorker)
    {
        var problems = HeartMuLaSongGenerator.ValidateSettings(options);
        if (problems.Count > 0) throw new InvalidOperationException("Invalid settings:" + string.Concat(problems.Select(p => "\n  - " + p)));
        var log = sp.GetRequiredService<ILogger<HeartMuLaSongGenerator>>();
        if (!codecInWorker) return new HeartMuLaSongGenerator(options, log);
        return new HeartMuLaSongGenerator(options, log, new WorkerFrameDecoder(HeartMuLaSongGenerator.ResolveSettings(options)));
    }
}

