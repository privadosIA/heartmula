using System.Net;
using System.Text.Json;

namespace HeartMuLa.Server.Tests;

public class OpenApiTests
{
    private static async Task<JsonElement> Document(ApiFactory factory)
    {
        using var client = factory.CreateClient();
        var r = await client.GetAsync("/swagger/v1/swagger.json");
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        return JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement.Clone();
    }

    [Fact]
    public async Task The_document_describes_the_five_operations_with_their_status_codes()
    {
        using var factory = new ApiFactory();
        var paths = (await Document(factory)).GetProperty("paths");

        Assert.Equal(new[] { "202", "400", "401", "429" }, paths.GetProperty("/songs").GetProperty("post").GetProperty("responses").EnumerateObject().Select(p => p.Name).Order());
        Assert.Equal(new[] { "200", "401", "404" }, paths.GetProperty("/songs/{id}").GetProperty("get").GetProperty("responses").EnumerateObject().Select(p => p.Name).Order());
        var audio = paths.GetProperty("/songs/{id}/audio").GetProperty("get").GetProperty("responses");
        Assert.Equal(new[] { "200", "401", "404", "409" }, audio.EnumerateObject().Select(p => p.Name).Order());
        Assert.True(audio.GetProperty("200").GetProperty("content").TryGetProperty("audio/wav", out _));
        Assert.Equal(new[] { "202", "204", "401" }, paths.GetProperty("/songs/{id}").GetProperty("delete").GetProperty("responses").EnumerateObject().Select(p => p.Name).Order());
        Assert.Equal(new[] { "200" }, paths.GetProperty("/health").GetProperty("get").GetProperty("responses").EnumerateObject().Select(p => p.Name).Order());
    }

    [Fact]
    public async Task The_request_and_response_schemas_and_the_api_key_scheme_are_published()
    {
        using var factory = new ApiFactory();
        var doc = await Document(factory);
        var schemas = doc.GetProperty("components").GetProperty("schemas");

        var request = schemas.GetProperty("CreateSongRequest").GetProperty("properties");
        foreach (var name in new[] { "tags", "lyrics", "durationMs", "seed", "cfgScale", "callbackUrl" }) Assert.True(request.TryGetProperty(name, out _), name);
        Assert.Contains("http", request.GetProperty("callbackUrl").GetProperty("description").GetString() ?? "", StringComparison.OrdinalIgnoreCase);   // the XML comment reached the document

        var state = schemas.GetProperty("JobState");
        Assert.Equal("string", state.GetProperty("type").GetString());
        Assert.Equal(new[] { "queued", "running", "succeeded", "failed", "cancelled" }, state.GetProperty("enum").EnumerateArray().Select(e => e.GetString()));

        var key = doc.GetProperty("components").GetProperty("securitySchemes").GetProperty("ApiKey");
        Assert.Equal(("apiKey", "header", "X-Api-Key"), (key.GetProperty("type").GetString(), key.GetProperty("in").GetString(), key.GetProperty("name").GetString()));
    }

    [Fact]
    public async Task The_swagger_ui_is_served_and_stays_open_when_an_api_key_is_configured()
    {
        using var factory = new ApiFactory(("Security:ApiKey", "s3cret"));
        using var client = factory.CreateClient();
        var ui = await client.GetAsync("/swagger/index.html");
        Assert.Equal(HttpStatusCode.OK, ui.StatusCode);
        Assert.Contains("swagger-ui", await ui.Content.ReadAsStringAsync());
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/swagger/v1/swagger.json")).StatusCode);
        Assert.Equal(HttpStatusCode.Unauthorized, (await client.GetAsync("/songs/" + new string('0', 32))).StatusCode);
    }
}
