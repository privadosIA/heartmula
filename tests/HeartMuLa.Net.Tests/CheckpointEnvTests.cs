namespace HeartMuLa.Tests;

public class CheckpointEnvTests
{
    [RequiresCheckpoint]
    public void CheckpointIsReachable() => Assert.NotNull(Checkpoint.Dir);
}
