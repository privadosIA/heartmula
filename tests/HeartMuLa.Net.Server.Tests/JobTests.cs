using HeartMuLa.Server.Jobs;

namespace HeartMuLa.Server.Tests;

public class JobTests
{
    private static readonly DateTimeOffset T0 = new(2026, 10, 22, 12, 0, 0, TimeSpan.Zero);
    private static SongRequest Req(string? callback = null) => new("piano", "la", 10_000, 1234, 1.5, callback);
    private static Job NewJob(string? callback = null) => new(JobStore.NewId(), Req(callback), T0);

    [Fact]
    public void A_new_job_is_queued_with_no_progress_and_no_webhook()
    {
        var v = NewJob().View();
        Assert.Equal(JobState.Queued, v.State);
        Assert.Equal(new JobProgress(JobStage.Queued, 0, 0), v.Progress);
        Assert.Null(v.Webhook);
        Assert.Null(v.StartedAt);
    }

    [Fact]
    public void A_job_with_a_callback_starts_with_a_pending_webhook()
        => Assert.Equal(new WebhookStatus(WebhookState.Pending, 0, null), NewJob("http://x/y").View().Webhook);

    [Fact]
    public void Queued_to_running_to_succeeded()
    {
        var j = NewJob();
        Assert.True(j.TryStart(T0.AddSeconds(1)));
        j.ReportProgress(JobStage.Generating, 5, 100);
        Assert.Equal(new JobProgress(JobStage.Generating, 5, 100), j.View().Progress);
        j.Succeed(9_920, T0.AddSeconds(20));
        var v = j.View();
        Assert.Equal(JobState.Succeeded, v.State);
        Assert.Equal(JobStage.Done, v.Progress.Stage);
        Assert.Equal(9_920, v.AudioDurationMs);
        Assert.Equal(T0.AddSeconds(20), v.EndedAt);
    }

    [Fact]
    public void Running_to_failed_keeps_the_message()
    {
        var j = NewJob();
        j.TryStart(T0);
        j.Fail("CUDA out of memory", T0.AddSeconds(3));
        Assert.Equal((JobState.Failed, "CUDA out of memory"), (j.State, j.View().Error));
    }

    [Fact]
    public void Cancelling_a_queued_job_is_immediate_and_it_can_no_longer_start()
    {
        var j = NewJob();
        Assert.True(j.RequestCancel(T0));
        Assert.Equal(JobState.Cancelled, j.State);
        Assert.False(j.TryStart(T0));
    }

    [Fact]
    public void Cancelling_a_running_job_only_signals_it_and_the_worker_ends_it()
    {
        var j = NewJob();
        j.TryStart(T0);
        Assert.True(j.RequestCancel(T0.AddSeconds(1)));
        Assert.Equal(JobState.Running, j.State);                 // only the worker moves a started job
        Assert.True(j.Cancellation.IsCancellationRequested);
        j.MarkCancelled(T0.AddSeconds(2));
        Assert.Equal(JobState.Cancelled, j.State);
    }

    [Fact]
    public void Final_jobs_cannot_be_cancelled_or_moved_again()
    {
        var j = NewJob();
        j.TryStart(T0);
        j.Succeed(1000, T0);
        Assert.False(j.RequestCancel(T0));
        Assert.Throws<InvalidOperationException>(() => j.Fail("x", T0));
        Assert.Throws<InvalidOperationException>(() => j.Succeed(1, T0));
        Assert.Throws<InvalidOperationException>(() => j.MarkCancelled(T0));
        Assert.False(j.TryStart(T0));
    }

    [Fact]
    public void A_queued_job_cannot_succeed_fail_or_report_progress_without_starting()
    {
        var j = NewJob();
        Assert.Throws<InvalidOperationException>(() => j.Succeed(1, T0));
        Assert.Throws<InvalidOperationException>(() => j.Fail("x", T0));
        Assert.Throws<InvalidOperationException>(() => j.ReportProgress(JobStage.Generating, 1, 2));
    }

    [Fact]
    public void Interrupt_fails_an_unfinished_job_with_the_restart_reason()
    {
        var j = NewJob();
        j.TryStart(T0);
        j.Interrupt(T0.AddMinutes(1));
        Assert.Equal((JobState.Failed, "interrupted by restart"), (j.State, j.View().Error));
    }

    [Fact]
    public void Webhook_state_can_only_be_set_when_there_is_a_callback()
    {
        Assert.Throws<InvalidOperationException>(() => NewJob().SetWebhook(new(WebhookState.Delivered, 1, null)));
        var j = NewJob("http://x/y");
        j.SetWebhook(new(WebhookState.Delivered, 1, null));
        Assert.Equal(WebhookState.Delivered, j.View().Webhook!.State);
    }

    [Fact]
    public void The_store_enforces_the_queue_limit_counting_only_queued_jobs()
    {
        var store = new JobStore();
        var a = NewJob(); var b = NewJob(); var c = NewJob();
        Assert.True(store.TryAdd(a, 2));
        Assert.True(store.TryAdd(b, 2));
        Assert.False(store.TryAdd(c, 2));                         // two are waiting
        a.TryStart(T0);                                            // one starts running: room for another
        Assert.True(store.TryAdd(c, 2));
        Assert.Equal(2, store.QueuedCount);
        Assert.Same(a, store.Running);
    }

    [Theory]
    [InlineData("0123456789abcdef0123456789abcdef", true)]
    [InlineData("0123456789ABCDEF0123456789ABCDEF", false)]
    [InlineData("../../etc/passwd", false)]
    [InlineData("", false)]
    public void Only_32_hex_characters_are_valid_ids(string id, bool valid)
    {
        Assert.Equal(valid, JobStore.IsValidId(id));
        Assert.Null(new JobStore().Get(id));
    }
}
