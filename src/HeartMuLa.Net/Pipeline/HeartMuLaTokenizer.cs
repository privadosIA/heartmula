using Tokenizers.DotNet;

namespace HeartMuLa.Pipeline;

/// <summary>
/// The real HeartMuLa tokenizer: loads the checkpoint's <c>tokenizer.json</c> (Llama 3-style BPE) with the HuggingFace library
/// (<c>Tokenizers.DotNet</c> wraps the same Rust library Python uses), so the ids are exactly the Python ones.
/// As in HuggingFace, the post-processor prepends BOS and does not add EOS (the pipeline does that).
/// </summary>
public sealed class HeartMuLaTokenizer : ITextTokenizer, IDisposable
{
    private readonly Tokenizer tokenizer;
    private readonly object gate = new(); // the native API does not document whether it is thread-safe

    /// <summary>Loads the tokenizer from <paramref name="tokenizerJsonPath"/>; throws <see cref="FileNotFoundException"/> if the file does not exist.</summary>
    public HeartMuLaTokenizer(string tokenizerJsonPath)
    {
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException($"The checkpoint tokenizer was not found at {tokenizerJsonPath}", tokenizerJsonPath);
        tokenizer = new Tokenizer(tokenizerJsonPath);
    }

    /// <summary>Encodes <paramref name="text"/> into token ids (with BOS, without EOS).</summary>
    public IReadOnlyList<long> Encode(string text)
    {
        uint[] ids;
        lock (gate) ids = tokenizer.Encode(text);
        return ids.Select(i => (long)i).ToArray();
    }

    /// <summary>Releases the native tokenizer.</summary>
    public void Dispose() => tokenizer.Dispose();
}
