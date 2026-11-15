using System.Text.RegularExpressions;

namespace HeartMuLa.Tests;

/// <summary>The core and this repository must resolve the same TorchSharp (managed API and native package).</summary>
public class CoreVersionTests
{
    [Fact]
    public void BothDirectoryBuildProps_CarryTheSameTorchSharpVersion()
    {
        var mine = Path.Combine(RepoRoot(), "Directory.Build.props");
        var coreDir = Environment.GetEnvironmentVariable("MusicGenCoreDir");
        if (string.IsNullOrEmpty(coreDir))
        {
            var vendored = Path.Combine(RepoRoot(), "third_party", "musicgen-core");
            coreDir = Directory.Exists(vendored) ? vendored : Path.GetFullPath(Path.Combine(RepoRoot(), "..", "musicgen-core"));
        }
        var core = Path.Combine(coreDir, "Directory.Build.props");

        Assert.True(File.Exists(core), $"core Directory.Build.props not found at {core}");
        Assert.Equal(Version(core), Version(mine));
    }

    [Fact]
    public void TheCore_IsBuiltWithTheSameConfigurationAsThisPort()
    {
        // A solution build compiles project references that are not in the solution with the default (Debug) configuration, even for -c Release,
        // unless ShouldUnsetParentConfigurationAndPlatform is false in both Directory.Build.props.
        static string? Config(System.Reflection.Assembly a) =>
            (Attribute.GetCustomAttribute(a, typeof(System.Reflection.AssemblyConfigurationAttribute)) as System.Reflection.AssemblyConfigurationAttribute)?.Configuration;

        Assert.Equal(Config(typeof(CoreVersionTests).Assembly), Config(typeof(MusicGen.Core.Seeding).Assembly));
        Assert.Equal(Config(typeof(CoreVersionTests).Assembly), Config(typeof(HeartMuLa.Codec.CodecWeights).Assembly));
    }

    private static string Version(string props)
    {
        var m = Regex.Match(File.ReadAllText(props), @"<TorchSharpVersion>([^<]+)</TorchSharpVersion>");
        Assert.True(m.Success, $"no TorchSharpVersion in {props}");
        return m.Groups[1].Value;
    }

    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "HeartMuLa.Net.sln"))) dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("HeartMuLa.Net.sln not found above the test binaries");
    }
}
