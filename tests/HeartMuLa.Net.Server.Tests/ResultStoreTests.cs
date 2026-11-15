using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Options;
using HeartMuLa.Server.Storage;
using Microsoft.Extensions.Time.Testing;

namespace HeartMuLa.Server.Tests;

public sealed class ResultStoreTests : IDisposable
{
    private readonly string dir = Path.Combine(Path.GetTempPath(), "heartmula-store-" + Guid.NewGuid().ToString("N"));
    private readonly FakeTimeProvider clock = new(new DateTimeOffset(2026, 10, 22, 12, 0, 0, TimeSpan.Zero));

    private ResultStore Open(double retentionHours = 24) => new(new StorageOptions { Directory = dir, RetentionHours = retentionHours }, clock);
    private Job NewJob(string? callback = null) => new(JobStore.NewId(), new SongRequest("piano", "la", 10_000, 7, 1.5, callback), clock.GetUtcNow());

    public void Dispose() { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }

    private void Finish(ResultStore store, Job job, bool withWav = true)
    {
        job.TryStart(clock.GetUtcNow());
        if (withWav) { File.WriteAllBytes(store.WavTempPath(job.Id), new byte[] { 1, 2, 3 }); store.CommitWav(job.Id); }
        job.Succeed(1000, clock.GetUtcNow());
        store.Save(job.View());
    }

    [Fact]
    public void A_finished_job_survives_a_restart()
    {
        var first = Open();
        var job = NewJob("http://r/h");
        Finish(first, job);

        var restored = Assert.Single(Open().LoadAll());
        Assert.Equal(job.View(), restored.View());
        Assert.True(first.HasWav(job.Id));
    }

    [Fact]
    public void Unfinished_jobs_come_back_failed_with_the_restart_reason_and_the_sidecar_is_rewritten()
    {
        var store = Open();
        var queued = NewJob(); var running = NewJob();
        store.Save(queued.View());
        running.TryStart(clock.GetUtcNow());
        store.Save(running.View());

        var loaded = Open().LoadAll();
        Assert.All(loaded, j => Assert.Equal(("interrupted by restart", JobState.Failed), (j.View().Error, j.State)));
        Assert.All(Open().LoadAll(), j => Assert.Equal(JobState.Failed, j.State));      // already persisted as failed
    }

    [Fact]
    public void A_half_written_wav_is_never_shown_and_is_cleaned_up()
    {
        var store = Open();
        var job = NewJob();
        job.TryStart(clock.GetUtcNow());
        File.WriteAllBytes(store.WavTempPath(job.Id), new byte[] { 1 });                  // crash before the rename
        store.Save(job.View());

        var restored = Assert.Single(Open().LoadAll());
        Assert.Equal(JobState.Failed, restored.State);
        Assert.False(store.HasWav(job.Id));
        Assert.False(File.Exists(store.WavTempPath(job.Id)));
    }

    [Fact]
    public void A_succeeded_job_whose_wav_disappeared_is_dropped()
    {
        var store = Open();
        var job = NewJob();
        Finish(store, job);
        File.Delete(store.WavPath(job.Id));
        Assert.Empty(Open().LoadAll());
        Assert.False(File.Exists(Path.Combine(dir, job.Id + ".json")));
    }

    [Fact]
    public void Results_older_than_the_retention_are_deleted_and_recent_ones_kept()
    {
        var store = Open(retentionHours: 24);
        var old = NewJob(); var recent = NewJob(); var unfinished = NewJob();
        Finish(store, old);
        clock.Advance(TimeSpan.FromHours(23));
        Finish(store, recent);
        store.Save(unfinished.View());
        clock.Advance(TimeSpan.FromHours(2));                                              // old: 25 h, recent: 2 h

        var expired = store.Sweep(new[] { old, recent, unfinished });

        Assert.Equal(new[] { old.Id }, expired);
        Assert.False(store.HasWav(old.Id));
        Assert.True(store.HasWav(recent.Id));
    }

    [Fact]
    public void Unreadable_or_misnamed_sidecars_are_ignored()
    {
        var store = Open();
        File.WriteAllText(Path.Combine(dir, "garbage.json"), "{ not json");
        var job = NewJob();
        Finish(store, job);
        File.Copy(Path.Combine(dir, job.Id + ".json"), Path.Combine(dir, new string('a', 32) + ".json"));  // id inside does not match the file name
        Assert.Single(Open().LoadAll());
    }

    [Fact]
    public void A_sidecar_written_before_temperature_and_topk_existed_loads_with_the_defaults()
    {
        var store = Open();
        var id = JobStore.NewId();
        File.WriteAllText(Path.Combine(dir, id + ".json"), $$"""
            {"id":"{{id}}","state":"failed","request":{"tags":"a","lyrics":"b","durationMs":1000,"seed":5,"cfgScale":1.5},
             "createdAt":"2026-10-22T12:00:00+00:00","progress":{"stage":"queued","current":0,"total":0},"error":"x"}
            """);
        var job = Assert.Single(store.LoadAll());
        Assert.Equal((1.0, 50, 5L), (job.Request.Temperature, job.Request.TopK, job.Request.Seed));
    }

    [Fact]
    public void Delete_removes_every_file_of_the_job()
    {
        var store = Open();
        var job = NewJob();
        Finish(store, job);
        store.Delete(job.Id);
        Assert.Empty(Directory.GetFiles(dir));
    }
}
