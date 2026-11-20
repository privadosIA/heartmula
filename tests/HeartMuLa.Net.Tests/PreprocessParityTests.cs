using MusicGen.Core.Text;
using System.Globalization;
using System.Text.Json;
using HeartMuLa.Model;
using HeartMuLa.Pipeline;
using TorchSharp;
using static TorchSharp.torch;

namespace HeartMuLa.Tests;

/// <summary>Toy tokenizer that remembers the texts it receives and returns only the BOS, to see what text Preprocess builds.</summary>
internal sealed class CapturingTokenizer : ITextTokenizer
{
    public List<string> Texts { get; } = new();
    public IReadOnlyList<long> Encode(string text) { Texts.Add(text); return new long[] { 128000 }; }
}

/// <summary>
/// Preprocess (reading, Python lowercasing, ordinal &lt;tag&gt; wrapping, BOS/EOS, mask, MuQ) against heartlib. The model only provides its
/// real configuration (8 codebooks, MuQ 512) with a tiny architecture, so no weights need to be loaded.
/// </summary>
public class PreprocessParityTests
{
    private static HeartMuLaModel ShapeOnlyModel() => new(
        new HeartMuLaConfig { TextVocabSize = 20, AudioVocabSize = 8197, AudioNumCodebooks = 8, MuqDim = 512 },
        TinyModel.Backbone, TinyModel.Decoder);

    private static MusicGenerationPipeline Pipeline(ITextTokenizer tok, GenConfig? gen = null) =>
        new(ShapeOnlyModel(), tok, gen ?? new GenConfig(), CPU, ScalarType.BFloat16);

    // ------------------------------------------------------------------ with the real tokenizer (needs the checkpoint)

    private static HeartMuLaTokenizer RealTokenizer() => new(Path.Combine(Checkpoint.Dir!, "tokenizer.json"));
    private static GenConfig RealGenConfig() => GenConfig.FromFile(Path.Combine(Checkpoint.Dir!, "gen_config.json"));

    [RequiresCheckpoint]
    public void ReferenceInputs_WithTheRealTokenizer_BuildTheReferencePrompt()
    {
        using var tok = RealTokenizer();
        var reference = Fixtures.Load("reference.safetensors");
        var inputs = Path.Combine(Fixtures.Dir, "inputs");

        var prompt = Pipeline(tok, RealGenConfig()).Preprocess(
            PythonText.ReadAllText(Path.Combine(inputs, "tags.txt")), PythonText.ReadAllText(Path.Combine(inputs, "lyrics.txt")), 1.5);

        Assert.Equal(new long[] { 2, 227, 9 }, prompt.Tokens.shape);
        Assert.True(prompt.Tokens.Equals(reference["tokens"]), "tokens differ from reference.safetensors");
        Assert.True(prompt.TokensMask.Equals(reference["tokens_mask"]), "mask differs");
        Assert.True(prompt.Pos.Equals(reference["pos"]), "positions differ");
        Assert.Equal(new[] { (int)reference["muq_idx"].item<long>(), (int)reference["muq_idx"].item<long>() }, prompt.MuqIdx);
        Assert.True(prompt.MuqEmbed.to(ScalarType.Float32).Equals(reference["muq_embed"]), "MuQ embedding differs");
    }

    [RequiresCheckpoint]
    public void TagLyricsPairs_GiveHeartlibsFinalIdsWithBosAndEos()
    {
        using var tok = RealTokenizer();
        var pipe = Pipeline(tok, RealGenConfig());
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures.Dir, "text_cases.json")));
        var prompts = doc.RootElement.GetProperty("prompts").EnumerateArray().ToList();
        Assert.Equal(8, prompts.Count);

        foreach (var p in prompts)
        {
            var name = p.GetProperty("name").GetString()!;
            long[] Ids(string key) => p.GetProperty(key).EnumerateArray().Select(x => x.GetInt64()).ToArray();
            var tagsRaw = Convert.FromBase64String(p.GetProperty("tags_b64").GetString()!);
            var lyricsRaw = Convert.FromBase64String(p.GetProperty("lyrics_b64").GetString()!);
            var expectedTags = Ids("tags_ids");
            var expectedLyrics = Ids("lyrics_ids");

            var prompt = pipe.Preprocess(PythonText.DecodeText(tagsRaw), PythonText.DecodeText(lyricsRaw), 1.5);

            long[] TextColumn(int from, int count) => Enumerable.Range(from, count).Select(i => prompt.Tokens[0, i, 8].item<long>()).ToArray();
            Assert.Equal(expectedTags.Length + 1 + expectedLyrics.Length, prompt.Tokens.shape[1]);
            Assert.Equal(expectedTags.Length, prompt.MuqIdx[0]);
            Assert.True(expectedTags.SequenceEqual(TextColumn(0, expectedTags.Length)), $"{name}: tag ids differ");
            Assert.True(expectedLyrics.SequenceEqual(TextColumn(expectedTags.Length + 1, expectedLyrics.Length)), $"{name}: lyrics ids differ");
            Assert.Equal(0L, prompt.Tokens[0, expectedTags.Length, 8].item<long>());           // the MuQ position carries no text
        }
    }

    // ------------------------------------------------------------------ without checkpoint

    [Theory]
    [InlineData("piano,happy", "<tag>piano,happy</tag>")]
    [InlineData("<tag>rock</tag>", "<tag>rock</tag>")]                      // already wrapped: not duplicated
    [InlineData("<tag>rock", "<tag>rock</tag>")]
    [InlineData("rock</tag>", "<tag>rock</tag>")]
    [InlineData("PIANO", "<tag>piano</tag>")]
    [InlineData("ΕΛΛΗΝΙΚΆ ΑΣ", "<tag>ελληνικά ας</tag>")]
    [InlineData("​<tag>rock", "<tag>​<tag>rock</tag>")]           // with a ZWSP in front it does NOT start with <tag>: heartlib wraps it
    [InlineData("­<tag>rock</tag>", "<tag>­<tag>rock</tag>")]
    public void TagEnvelope_IsOrdinal_AndCultureIndependent(string tags, string expectedWrapped)
    {
        var original = CultureInfo.CurrentCulture;
        try
        {
            foreach (var culture in new[] { "es-ES", "en-US", "tr-TR", "lt-LT", "" })
            {
                CultureInfo named;
                try { named = new CultureInfo(culture); }
                catch (CultureNotFoundException) { continue; }     // a runner with invariant globalization (no ICU) has no named cultures
                CultureInfo.CurrentCulture = named;
                var tok = new CapturingTokenizer();

                Pipeline(tok).Preprocess(tags, "la", 1.5);

                Assert.True(expectedWrapped == tok.Texts[0], $"culture '{culture}': tokenized {tok.Texts[0]!.Replace("​", "<ZWSP>")}");
                Assert.Equal("la", tok.Texts[1]);
            }
        }
        finally { CultureInfo.CurrentCulture = original; }
    }

    [Fact]
    public void Lyrics_AreLowercasedLikePython_NotLikeToLowerInvariant()
    {
        var tok = new CapturingTokenizer();

        Pipeline(tok).Preprocess("x", "ΟΔΥΣΣΕΥΣ İstanbul", 1.5);

        Assert.Equal("οδυσσευς i̇stanbul", tok.Texts[1]);
    }
}
