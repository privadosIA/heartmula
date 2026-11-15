using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Worker;
using static HeartMuLa.Server.Tests.WorkerHarness;

namespace HeartMuLa.Server.Tests;

public class GenerationWorkerTests
{
    [Fact]
    public async Task Jobs_run_one_at_a_time_in_arrival_order()
    {
        await using var h = new WorkerHarness();
        var release = new ManualResetEventSlim();
        h.Generator.OnStep = (r, step) => { if (r.Tags == "first" && step == 2) release.Wait(5000); };
        await h.StartAsync();

        var first = h.Submit("first");
        var second = h.Submit("second");
        await Eventually(() => first.View().Progress.Current == 1);

        Assert.Equal(JobState.Running, first.State);
        Assert.Equal(JobState.Queued, second.State);                      // waits while the first runs
        Assert.Equal(new[] { "first" }, h.Generator.Started.ToArray());

        release.Set();
        await Eventually(() => second.State == JobState.Succeeded);
        Assert.Equal(new[] { "first", "second" }, h.Generator.Started.ToArray());
        Assert.True(first.View().EndedAt <= second.View().StartedAt);
    }

    [Fact]
    public async Task Progress_is_visible_while_the_job_runs()
    {
        await using var h = new WorkerHarness();
        var release = new ManualResetEventSlim();
        h.Generator.Steps = 5;
        h.Generator.OnStep = (_, step) => { if (step == 3) release.Wait(5000); };
        await h.StartAsync();

        var job = h.Submit();
        await Eventually(() => job.View().Progress.Current == 2);
        var v = job.View();
        Assert.Equal((JobState.Running, JobStage.Generating, 5), (v.State, v.Progress.Stage, v.Progress.Total));
        release.Set();
        await Eventually(() => job.State == JobState.Succeeded);
        Assert.Equal(JobStage.Done, job.View().Progress.Stage);
    }

    [Fact]
    public async Task A_successful_job_stores_the_wav_and_the_sidecar()
    {
        await using var h = new WorkerHarness();
        await h.StartAsync();
        var job = h.Submit(durationMs: 300);
        await Eventually(() => job.State == JobState.Succeeded);

        Assert.True(h.Songs.Results.HasWav(job.Id));
        Assert.False(File.Exists(h.Songs.Results.WavTempPath(job.Id)));
        Assert.Equal(300, job.View().AudioDurationMs);
        Assert.Single(h.Songs.Results.LoadAll(), j => j.Id == job.Id && j.State == JobState.Succeeded);
    }

    [Fact]
    public async Task The_model_is_loaded_once_for_any_number_of_jobs()
    {
        await using var h = new WorkerHarness();
        await h.StartAsync();
        var a = h.Submit("a"); var b = h.Submit("b"); var c = h.Submit("c");
        await Eventually(() => c.State == JobState.Succeeded);
        Assert.Equal(1, h.Generator.LoadCalls);
    }

    [Fact]
    public async Task Cancelling_a_queued_job_removes_it_at_once_and_it_never_runs()
    {
        await using var h = new WorkerHarness();
        var release = new ManualResetEventSlim();
        h.Generator.OnStep = (r, step) => { if (r.Tags == "a" && step == 1) release.Wait(5000); };
        await h.StartAsync();

        var a = h.Submit("a"); var b = h.Submit("b");
        await Eventually(() => a.State == JobState.Running);
        Assert.Equal(CancelOutcome.Cancelled, h.Songs.CancelOrRemove(b.Id));
        Assert.Equal(JobState.Cancelled, b.State);                        // immediately, while a still runs

        release.Set();
        await Eventually(() => a.State == JobState.Succeeded);
        Assert.Equal(new[] { "a" }, h.Generator.Started.ToArray());
        Assert.False(h.Songs.Results.HasWav(b.Id));
    }

    [Fact]
    public async Task Cancelling_a_running_job_ends_it_cancelled_without_a_wav_and_the_next_job_runs()
    {
        await using var h = new WorkerHarness();
        var inLoop = new ManualResetEventSlim();
        h.Generator.Steps = 1_000_000;
        h.Generator.OnStep = (r, step) => { if (r.Tags == "long") { inLoop.Set(); Thread.Sleep(2); } };
        await h.StartAsync();

        var running = h.Submit("long");
        var next = h.Submit("short");
        inLoop.Wait(5000);
        h.Generator.Steps = 4;                                            // the next job is short
        Assert.Equal(CancelOutcome.Cancelling, h.Songs.CancelOrRemove(running.Id));

        await Eventually(() => running.State == JobState.Cancelled);
        await Eventually(() => next.State == JobState.Succeeded);
        Assert.False(h.Songs.Results.HasWav(running.Id));
        Assert.False(File.Exists(h.Songs.Results.WavTempPath(running.Id)));
    }

    [Fact]
    public async Task A_generator_exception_fails_that_job_only()
    {
        await using var h = new WorkerHarness();
        h.Generator.OnStep = (r, _) => { if (r.Tags == "bad") throw new InvalidOperationException("CUDA out of memory"); };
        await h.StartAsync();

        var bad = h.Submit("bad"); var good = h.Submit("good");
        await Eventually(() => good.State == JobState.Succeeded);
        Assert.Equal((JobState.Failed, "CUDA out of memory"), (bad.State, bad.View().Error));
        Assert.False(h.Songs.Results.HasWav(bad.Id));
    }

    [Fact]
    public async Task A_model_that_failed_to_load_fails_the_jobs_with_the_reason_and_the_worker_keeps_going()
    {
        await using var h = new WorkerHarness();
        h.Generator.FailLoad = true;
        await h.StartAsync();
        var a = h.Submit(); var b = h.Submit();
        await Eventually(() => b.State == JobState.Failed);
        Assert.Contains("cannot read the weights", a.View().Error);
    }

    [Fact]
    public async Task Stopping_the_service_cancels_the_running_job_without_waiting_for_the_song()
    {
        await using var h = new WorkerHarness();
        var inLoop = new ManualResetEventSlim();
        h.Generator.Steps = 1_000_000;
        h.Generator.OnStep = (_, _) => { inLoop.Set(); Thread.Sleep(2); };
        await h.StartAsync();
        var job = h.Submit();
        inLoop.Wait(5000);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        await h.Worker.StopAsync(new CancellationTokenSource(5000).Token);

        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(3));
        Assert.Equal(JobState.Cancelled, job.State);
    }

    [Fact]
    public async Task The_sink_hears_every_job_that_ends_exactly_once_whatever_the_outcome()
    {
        await using var h = new WorkerHarness();
        var release = new ManualResetEventSlim();
        h.Generator.OnStep = (r, step) =>
        {
            if (r.Tags == "ok" && step == 1) release.Wait(5000);
            if (r.Tags == "bad") throw new InvalidOperationException("boom");
        };
        await h.StartAsync();

        var ok = h.Submit("ok"); var bad = h.Submit("bad"); var gone = h.Submit("gone");
        await Eventually(() => ok.State == JobState.Running);
        h.Songs.CancelOrRemove(gone.Id);
        release.Set();
        await Eventually(() => bad.State == JobState.Failed && ok.State == JobState.Succeeded);
        await Eventually(() => h.Sink.Ended.Count == 3);

        Assert.Equal(new[] { JobState.Cancelled, JobState.Succeeded, JobState.Failed }.OrderBy(x => x),
                     h.Sink.Ended.Select(e => e.State).OrderBy(x => x));
        Assert.Equal(3, h.Sink.Ended.Select(e => e.Id).Distinct().Count());
    }

    [Fact]
    public async Task Submit_returns_null_when_the_queue_is_full()
    {
        await using var h = new WorkerHarness(maxQueued: 2);                 // worker not started: nothing leaves the queue
        Assert.NotNull(h.Songs.Submit(new SongRequest("a", "x", 1000, 1, 1.5, null)));
        Assert.NotNull(h.Songs.Submit(new SongRequest("b", "x", 1000, 1, 1.5, null)));
        Assert.Null(h.Songs.Submit(new SongRequest("c", "x", 1000, 1, 1.5, null)));
    }

    [Fact]
    public async Task Deleting_a_finished_job_removes_its_files_and_unknown_ids_are_not_found()
    {
        await using var h = new WorkerHarness();
        await h.StartAsync();
        var job = h.Submit();
        await Eventually(() => job.State == JobState.Succeeded);

        Assert.Equal(CancelOutcome.Removed, h.Songs.CancelOrRemove(job.Id));
        Assert.Null(h.Songs.Jobs.Get(job.Id));
        Assert.False(h.Songs.Results.HasWav(job.Id));
        Assert.Equal(CancelOutcome.NotFound, h.Songs.CancelOrRemove(job.Id));
        Assert.Equal(CancelOutcome.NotFound, h.Songs.CancelOrRemove("../../etc/passwd"));
    }
}

public class ForgetTests
{
    [Fact]
    public async Task Forgetting_a_running_job_stops_it_and_leaves_no_file_no_store_entry_and_no_event()
    {
        await using var h = new WorkerHarness();
        var inLoop = new ManualResetEventSlim();
        h.Generator.Steps = 1_000_000;
        h.Generator.OnStep = (_, _) => { inLoop.Set(); Thread.Sleep(2); };
        await h.StartAsync();
        var job = h.Submit();
        inLoop.Wait(5000);

        Assert.True(h.Songs.Forget(job.Id));
        Assert.Null(h.Songs.Jobs.Get(job.Id));
        await Eventually(() => job.State == JobState.Cancelled);                            // the generation really stopped
        await Task.Delay(100);

        Assert.Empty(Directory.GetFiles(h.Dir));
        Assert.Empty(h.Sink.Ended);
        Assert.Empty(h.Songs.Results.LoadAll());                                            // and it does not come back after a restart
    }

    [Fact]
    public async Task Forgetting_a_queued_or_finished_job_deletes_it_and_an_unknown_id_is_false()
    {
        await using var h = new WorkerHarness();
        var release = new ManualResetEventSlim();
        h.Generator.OnStep = (r, step) => { if (r.Tags == "first" && step == 1) release.Wait(5000); };
        await h.StartAsync();
        var first = h.Submit("first"); var queued = h.Submit("queued");
        await Eventually(() => first.State == JobState.Running);

        Assert.True(h.Songs.Forget(queued.Id));
        release.Set();
        await Eventually(() => first.State == JobState.Succeeded);
        Assert.True(h.Songs.Forget(first.Id));

        Assert.Empty(Directory.GetFiles(h.Dir));
        Assert.False(h.Songs.Forget(first.Id));
        Assert.False(h.Songs.Forget("zz"));
        Assert.DoesNotContain("queued", h.Generator.Started);                               // the forgotten queued job never ran
    }
}

public class PublishTests
{
    [Fact]
    public async Task A_job_forgotten_before_the_worker_publishes_it_leaves_no_file_and_no_event()
    {
        await using var h = new WorkerHarness();                                              // no worker running: the steps are driven by hand
        var job = h.Submit();
        job.TryStart(DateTimeOffset.UtcNow);
        File.WriteAllBytes(h.Songs.Results.WavTempPath(job.Id), FakeSongGenerator.WavBytes(100));
        h.Songs.Results.CommitWav(job.Id);
        job.Succeed(100, DateTimeOffset.UtcNow);

        Assert.True(h.Songs.Forget(job.Id));              // the forget wins the race with the end of the job...
        h.Songs.Publish(job);                              // ...and the worker's last step must not bring anything back

        Assert.Empty(Directory.GetFiles(h.Dir));
        Assert.Empty(h.Sink.Ended);
    }

    [Fact]
    public async Task A_job_published_before_it_is_forgotten_is_stored_announced_and_then_removed_by_the_forget()
    {
        await using var h = new WorkerHarness();
        var job = h.Submit();
        job.TryStart(DateTimeOffset.UtcNow);
        job.Succeed(100, DateTimeOffset.UtcNow);

        h.Songs.Publish(job);
        Assert.Single(Directory.GetFiles(h.Dir), f => f.EndsWith(job.Id + ".json"));
        Assert.Single(h.Sink.Ended);

        Assert.True(h.Songs.Forget(job.Id));
        Assert.Empty(Directory.GetFiles(h.Dir));
    }
}

public class MaintenanceTests
{
    [Fact]
    public async Task Startup_restores_jobs_from_disk_and_a_sweep_removes_the_expired_ones()
    {
        await using var h = new WorkerHarness();
        await h.StartAsync();
        var job = h.Submit();
        await Eventually(() => job.State == JobState.Succeeded);

        var fresh = new SongService(new JobStore(), h.Songs.Results, h.Sink, h.Options, h.Songs.Time);
        var maintenance = new Maintenance(fresh, h.Options, Microsoft.Extensions.Logging.Abstractions.NullLogger<Maintenance>.Instance);
        await maintenance.StartAsync(CancellationToken.None);
        Assert.Equal(JobState.Succeeded, fresh.Jobs.Get(job.Id)!.State);                     // restored after a "restart"

        var clock = new Microsoft.Extensions.Time.Testing.FakeTimeProvider(DateTimeOffset.UtcNow.AddHours(25));
        var late = new Maintenance(new SongService(fresh.Jobs, new HeartMuLa.Server.Storage.ResultStore(h.Options.Storage, clock), h.Sink, h.Options, clock),
                                   h.Options, Microsoft.Extensions.Logging.Abstractions.NullLogger<Maintenance>.Instance);
        late.SweepOnce();
        Assert.Null(fresh.Jobs.Get(job.Id));
        Assert.False(h.Songs.Results.HasWav(job.Id));
        await maintenance.StopAsync(CancellationToken.None);
    }
}
