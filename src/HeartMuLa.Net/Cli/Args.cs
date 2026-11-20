namespace HeartMuLa.Cli;

/// <summary>Minimal argument parser: --k v and positionals.</summary>
internal sealed class Args
{
    public List<string> Positional { get; } = new();
    private readonly Dictionary<string, string> named = new();
    public string? Get(string k) => named.TryGetValue(k, out var v) ? v : null;

    public static Args Parse(string[] argv)
    {
        var a = new Args();
        for (int i = 0; i < argv.Length; i++)
        {
            if (argv[i].StartsWith("--"))
            {
                var key = argv[i][2..];
                var val = i + 1 < argv.Length && !argv[i + 1].StartsWith("--") ? argv[++i] : "true";
                a.named[key] = val;
            }
            else a.Positional.Add(argv[i]);
        }
        return a;
    }
}
