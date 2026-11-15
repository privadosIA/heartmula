namespace HeartMuLa.Server.Tests;

public class ScaffoldTests
{
    [Fact]
    public void The_host_builds_with_the_fake_generator()
    {
        using var factory = new ApiFactory();
        Assert.NotNull(factory.Services);
    }

    [Fact]
    public async Task Without_a_usable_checkpoint_the_default_mode_stops_startup_naming_the_setting()
    {
        // The default keeps the model in the service, so the settings are checked when it starts (not on the first song).
        // The service is started directly: through WebApplicationFactory a failing startup races with the disposal of the host
        // and, on a loaded machine, surfaces as an ObjectDisposedException instead of the real error.
        var dir = Path.Combine(Path.GetTempPath(), "heartmula-scaffold-" + Guid.NewGuid().ToString("N"));
        try
        {
            await using var app = HeartMuLa.Server.ServiceHost.Build(new[]
            {
                "--urls", "http://127.0.0.1:0", "--Storage:Directory", dir,
                "--Model:Checkpoint", "/definitely/not/here", "--Model:Device", "cpu", "--Codec:Device", "cpu",
            });
            var e = await Assert.ThrowsAnyAsync<Exception>(() => app.StartAsync());
            Assert.True(e.ToString().Contains("Model:Checkpoint"), e.ToString());
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }
}
