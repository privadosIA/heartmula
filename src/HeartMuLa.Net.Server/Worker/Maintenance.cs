using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;

namespace HeartMuLa.Server.Worker;

/// <summary>At startup restores the jobs found on disk; afterwards deletes expired results every <c>Storage:SweepMinutes</c>.</summary>
public sealed class Maintenance : BackgroundService
{
    private readonly SongService songs;
    private readonly ServiceOptions options;
    private readonly ILogger<Maintenance> log;

    public Maintenance(SongService songs, ServiceOptions options, ILogger<Maintenance> log)
    {
        this.songs = songs; this.options = options; this.log = log;
    }

    public override Task StartAsync(CancellationToken cancellationToken)
    {
        foreach (var job in songs.Results.LoadAll()) songs.Jobs.Restore(job);
        log.LogInformation("Restored {Count} job(s) from {Dir}", songs.Jobs.All().Count, songs.Results.DirectoryPath);
        return base.StartAsync(cancellationToken);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var timer = new PeriodicTimer(TimeSpan.FromMinutes(options.Storage.SweepMinutes), songs.Time);
        try
        {
            do { SweepOnce(); } while (await timer.WaitForNextTickAsync(stoppingToken));
        }
        catch (OperationCanceledException) { }
    }

    public void SweepOnce()
    {
        foreach (var id in songs.Results.Sweep(songs.Jobs.All())) songs.Jobs.Remove(id);
    }
}
