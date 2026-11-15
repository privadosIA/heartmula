using MusicGen.Core.Text;
using System.Text.Json;
using HeartMuLa.Pipeline;
using Xunit.Abstractions;

namespace HeartMuLa.Tests;

/// <summary>
/// Real tokenizer against the 12 cases of tokenizer_cases.json, which tools/dump_reference.py generated with the Python
/// HuggingFace `tokenizers` library. Needs the checkpoint's tokenizer.json (9 MB, not in the repo).
/// </summary>
public sealed class HeartMuLaTokenizerTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly HeartMuLaTokenizer? _tok =
        Checkpoint.Dir is { } dir ? new HeartMuLaTokenizer(Path.Combine(dir, "tokenizer.json")) : null;

    public HeartMuLaTokenizerTests(ITestOutputHelper output) => _out = output;

    public void Dispose() => _tok?.Dispose();

    private static List<(string name, string text, long[] ids)> Cases()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures.Dir, "tokenizer_cases.json")));
        return doc.RootElement.GetProperty("cases").EnumerateArray()
            .Select(c => (c.GetProperty("name").GetString()!, c.GetProperty("text").GetString()!,
                          c.GetProperty("ids").EnumerateArray().Select(x => x.GetInt64()).ToArray()))
            .ToList();
    }

    [RequiresCheckpoint]
    public void ReferenceCases_GiveExactlyPythonsIds()
    {
        var cases = Cases();
        Assert.Equal(12, cases.Count);

        foreach (var (name, text, expected) in cases)
        {
            var got = _tok!.Encode(text);
            Assert.True(expected.SequenceEqual(got),
                $"{name}: ids differ (C# {got.Count}, Python {expected.Length}); first C# [{string.Join(",", got.Take(6))}] Python [{string.Join(",", expected.Take(6))}]");
            Assert.Equal(128000, got[0]);              // the BOS is prepended by the post-processor
            Assert.NotEqual(128001, got[^1]);          // and no EOS is added
        }
    }

    [RequiresCheckpoint]
    public void Encode_DependsOnTheText_SoTheComparisonCanFail()
    {
        var a = _tok!.Encode("hello world");
        var b = _tok.Encode("hello  world");           // one more space

        Assert.False(a.SequenceEqual(b));
        Assert.False(a.Select(i => i + 1).SequenceEqual(Cases().Single(c => c.name == "ascii_simple").ids), "an altered id should not match the expected one");
    }

    /// <summary>
    /// End to end, as in heartlib: file bytes → Python read → tokenizer without lowercasing (ids_raw) → lowercase → tokenizer (ids).
    /// The 28 cases of text_cases.json, generated with the `tokenizers` library and Python's `str.lower()`.
    /// </summary>
    [RequiresCheckpoint]
    public void TextCases_GiveExactlyPythonsLowercaseAndIds()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures.Dir, "text_cases.json")));
        var cases = doc.RootElement.GetProperty("cases").EnumerateArray().ToList();
        Assert.Equal(28, cases.Count);

        long[] Ids(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt64()).ToArray();
        foreach (var c in cases)
        {
            var name = c.GetProperty("name").GetString()!;
            var text = PythonText.DecodeText(Convert.FromBase64String(c.GetProperty("bytes_b64").GetString()!));
            var lower = PythonText.Lower(text);

            Assert.True(c.GetProperty("text").GetString() == text, $"{name}: text read differs");
            Assert.True(c.GetProperty("lower").GetString() == lower, $"{name}: lowercase differs");
            Assert.True(Ids(c.GetProperty("ids_raw")).SequenceEqual(_tok!.Encode(text)), $"{name}: ids without lowercasing differ");
            Assert.True(Ids(c.GetProperty("ids")).SequenceEqual(_tok.Encode(lower)), $"{name}: ids with lowercasing differ");
        }
    }

    /// <summary>Why PythonText.Lower exists: with ToLowerInvariant the prompt would change in these cases (ids different from Python's).</summary>
    [RequiresCheckpoint]
    public void ToLowerInvariant_WouldGiveDifferentIds_InTheSigmaAndDottedICases()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures.Dir, "text_cases.json")));
        var differing = new List<string>();
        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var pythonIds = c.GetProperty("ids").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            if (!pythonIds.SequenceEqual(_tok!.Encode(text.ToLowerInvariant()))) differing.Add(c.GetProperty("name").GetString()!);
        }

        _out.WriteLine($"with ToLowerInvariant {differing.Count} of 28 cases would give ids different from Python: {string.Join(", ", differing)}");
        Assert.Contains("sigma_final_words", differing);
        Assert.Contains("turkish_dotted_i", differing);
    }

    [RequiresCheckpoint]
    public void EmptyText_GivesOnlyTheBos_AndSpecialTokenLookalikesAreParsedAsSpecial()
    {
        Assert.Equal(new long[] { 128000 }, _tok!.Encode(""));

        var end = _tok.Encode("hello <|end_of_text|> world");
        Assert.Contains(128001L, end);                                   // the text is interpreted as the special EOS token
        Assert.Equal(1, end.Count(i => i == 128001));

        var begin = _tok.Encode("<|begin_of_text|>hello");
        Assert.Equal(2, begin.Count(i => i == 128000));                  // the post-processor BOS and the one in the text
    }

    [Fact]
    public void MissingTokenizerJson_FailsWithAClearMessage()
    {
        var ex = Assert.Throws<FileNotFoundException>(() => new HeartMuLaTokenizer("/no/existe/tokenizer.json"));

        Assert.Contains("/no/existe/tokenizer.json", ex.Message);
    }
}
