using MusicGen.Core.Testing;
using TorchSharp;

namespace HeartMuLa.Tests;

/// <summary>Access to tests/fixtures (Python golden run, versioned in the repo), with checksum verification from MANIFEST.md.</summary>
public static class Fixtures
{
    private static readonly FixtureDirectory Directory_ = FixtureDirectory.Find();

    public static string Dir => Directory_.Dir;

    public static Dictionary<string, torch.Tensor> Load(string file) => Directory_.Load(file);
}
