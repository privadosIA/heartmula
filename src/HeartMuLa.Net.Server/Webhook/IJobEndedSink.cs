using HeartMuLa.Server.Jobs;

namespace HeartMuLa.Server.Webhook;

/// <summary>Told when a job reaches a final state. Must return at once: it is called from the generation worker.</summary>
public interface IJobEndedSink
{
    void JobEnded(Job job);
}

public sealed class NullJobEndedSink : IJobEndedSink
{
    public void JobEnded(Job job) { }
}
