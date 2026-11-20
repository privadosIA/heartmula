using HeartMuLa.Server.Jobs;

namespace HeartMuLa.Server.Webhook;

/// <summary>Told when a job reaches a final state. Must return at once: it is called from the generation worker.</summary>
public interface IJobEndedSink
{
    /// <summary>Called once when <paramref name="job"/> reaches a final state.</summary>
    /// <param name="job">The job that ended.</param>
    void JobEnded(Job job);
}

/// <summary>A sink that ignores every notification.</summary>
public sealed class NullJobEndedSink : IJobEndedSink
{
    /// <summary>Does nothing.</summary>
    /// <param name="job">The job that ended.</param>
    public void JobEnded(Job job) { }
}
