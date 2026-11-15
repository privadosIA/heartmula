using HeartMuLa.Server.Jobs;
using HeartMuLa.Server.Worker;

namespace HeartMuLa.Server.Tests;

public class WorkerProtocolTests
{
    private static readonly WorkerSettings Settings = new("/ckpt", "3B", "cuda:0", "cuda:1", false);

    [Fact]
    public void The_request_survives_the_round_trip_in_both_modes()
    {
        var song = new WorkerRequest(WorkerProtocol.SongMode, "piano,happy", "[verse]\nla \"la\" la", 30_000, 1234567890123, 3.0, 0.7, 40, null, Settings);
        var codec = song with { Mode = WorkerProtocol.CodecMode, FramesPath = "/tmp/x.frames.safetensors" };
        foreach (var r in new[] { song, codec })
        {
            var line = WorkerProtocol.EncodeRequest(r);
            Assert.DoesNotContain('\n', line);                                   // lyrics have newlines: JSON escapes them, so it is one line
            Assert.Equal(r, WorkerProtocol.DecodeRequest(line));
        }
    }

    [Theory]
    [InlineData(JobStage.Loading, 0, 0)]
    [InlineData(JobStage.Generating, 57, 375)]
    [InlineData(JobStage.Decoding, 2, 2)]
    public void A_progress_line_survives_the_round_trip(JobStage stage, int current, int total)
    {
        var line = WorkerProtocol.EncodeProgress(stage, current, total);
        Assert.True(WorkerProtocol.TryDecodeProgress(line, out var p));
        Assert.Equal((stage, current, total), p);
        Assert.Contains($"\"stage\":\"{stage.ToString().ToLowerInvariant()}\"", line);          // the words the API uses
    }

    [Theory]
    [InlineData("")]
    [InlineData("Loading HeartMuLa weights from /ckpt ...")]
    [InlineData("  OK")]
    [InlineData("{ not json")]
    [InlineData("{\"warning\":\"from a native library\"}")]
    [InlineData("[1,2,3]")]
    [InlineData("{\"stage\":\"cooking\",\"current\":1,\"total\":2}")]
    public void Anything_that_is_not_a_progress_line_is_ignored(string line) => Assert.False(WorkerProtocol.TryDecodeProgress(line, out _));

    [Fact]
    public void The_constants_are_the_ones_the_spec_and_the_sister_service_use()
    {
        Assert.Equal("--worker", WorkerProtocol.Argument);
        Assert.Equal((0, 1, 3), (WorkerProtocol.Done, WorkerProtocol.Failed, WorkerProtocol.Cancelled));
    }
}
