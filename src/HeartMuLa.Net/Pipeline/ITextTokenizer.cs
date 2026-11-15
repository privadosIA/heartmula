namespace HeartMuLa.Pipeline;

/// <summary>
/// The checkpoint ships a HuggingFace tokenizer.json (Llama 3-style BPE with added tokens
/// &lt;tag&gt;, &lt;/tag&gt;, etc.). Options in .NET, from most to least convenient:
///
///  1. NuGet package "Tokenizers.DotNet" (native bindings to the Rust `tokenizers` library):
///     loads tokenizer.json as is. This is what I would use to get started.
///  2. Microsoft.ML.Tokenizers: has Tiktoken/Llama but no generic tokenizer.json loader;
///     you would have to export the merges/vocab and build a BpeTokenizer. More work.
///  3. For the golden run you do not even need it: the Python script dumps the already tokenized ids to
///     reference/prompt_ids.json and <see cref="JsonIdsTokenizer"/> serves them.
/// </summary>
public interface ITextTokenizer
{
    IReadOnlyList<long> Encode(string text);
}

/// <summary>
/// "Fake" tokenizer for tests: returns ids precomputed by the Python script
/// (reference/prompt_ids.json → {"&lt;text&gt;": [ids...]}). It lets you validate the model
/// without having solved the tokenizer in .NET yet.
/// </summary>
public sealed class JsonIdsTokenizer : ITextTokenizer
{
    private readonly Dictionary<string, long[]> table;

    public JsonIdsTokenizer(string jsonPath)
    {
        table = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, long[]>>(File.ReadAllText(jsonPath))
                ?? new Dictionary<string, long[]>();
    }

    public IReadOnlyList<long> Encode(string text)
    {
        if (table.TryGetValue(text, out var ids)) return ids;
        throw new KeyNotFoundException($"No precomputed ids for the text (len={text.Length}). Regenerate prompt_ids.json with dump_reference.py");
    }
}
