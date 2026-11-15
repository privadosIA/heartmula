using HeartMuLa.Server.Options;
using Microsoft.Extensions.Configuration;

namespace HeartMuLa.Server.Tests;

public class ServiceOptionsTests
{
    private static ServiceOptions Load(params (string, string?)[] values)
    {
        var cfg = new ConfigurationBuilder().AddInMemoryCollection(values.Select(v => new KeyValuePair<string, string?>(v.Item1, v.Item2))).Build();
        return ServiceOptions.From(cfg);
    }

    [Fact]
    public void Defaults_match_the_design()
    {
        var o = Load();
        Assert.Equal("http://127.0.0.1:5080", ServiceOptions.DefaultUrl);
        Assert.Equal(30_000, o.Jobs.DefaultDurationMs);
        Assert.Equal(360_000, o.Jobs.MaxDurationMs);
        Assert.Equal(1.5, o.Jobs.DefaultCfgScale);
        Assert.Equal(24, o.Storage.RetentionHours);
        Assert.False(o.Codec.Tf32);
        Assert.Equal(new[] { 1, 5, 30 }, o.Webhook.EffectiveRetryDelaysSeconds);
        Assert.Equal(10, o.Webhook.TimeoutSeconds);
        Assert.Empty(o.Webhook.AllowedHosts);
        Assert.True(string.IsNullOrEmpty(o.Security.ApiKey));
        Assert.Empty(o.Validate(cudaDeviceCount: 0, requireCheckpoint: false));
    }

    [Fact]
    public void Devices_default_to_cuda0_when_cuda_exists_and_the_codec_follows_the_model()
    {
        var o = Load();
        Assert.Equal("cuda:0", o.EffectiveModelDevice(cudaAvailable: true));
        Assert.Equal("cpu", o.EffectiveModelDevice(cudaAvailable: false));
        Assert.Equal("cuda:0", o.EffectiveCodecDevice(cudaAvailable: true));
        Assert.Equal("cuda:1", Load(("Codec:Device", "cuda:1")).EffectiveCodecDevice(true));
    }

    [Fact]
    public void Values_are_overridden_from_configuration_the_way_environment_variables_do()
    {
        var o = Load(("Jobs:MaxDurationMs", "60000"), ("Storage:RetentionHours", "2"), ("Webhook:AllowedHosts:0", "orchestrator.local"),
                     ("Webhook:RetryDelaysSeconds:0", "2"), ("Security:ApiKey", "k"), ("Model:Device", "cpu"));
        Assert.Equal(60_000, o.Jobs.MaxDurationMs);
        Assert.Equal(2, o.Storage.RetentionHours);
        Assert.Equal(new[] { "orchestrator.local" }, o.Webhook.AllowedHosts);
        Assert.Equal("k", o.Security.ApiKey);
        Assert.Equal("cpu", o.Model.Device);
    }

    [Fact]
    public void A_configured_retry_list_replaces_the_default_instead_of_extending_it()
    {
        Assert.Equal(new[] { 2 }, Load(("Webhook:RetryDelaysSeconds:0", "2")).Webhook.EffectiveRetryDelaysSeconds);
    }

    [Fact]
    public void The_checkpoint_falls_back_to_HEARTMULA_CKPT()
    {
        Assert.Equal("/x", Load(("HEARTMULA_CKPT", "/x")).Model.Checkpoint);
        Assert.Equal("/y", Load(("HEARTMULA_CKPT", "/x"), ("Model:Checkpoint", "/y")).Model.Checkpoint);
    }

    [Theory]
    [InlineData("Model:Checkpoint", "/definitely/not/here", "Model:Checkpoint")]
    [InlineData("Model:Device", "quantum:3", "Model:Device")]
    [InlineData("Model:Device", "cuda:0", "Model:Device")]            // no CUDA in this call
    [InlineData("Codec:Device", "cuda:1", "Codec:Device")]
    [InlineData("Storage:RetentionHours", "0", "Storage:RetentionHours")]
    [InlineData("Jobs:MaxQueued", "0", "Jobs:MaxQueued")]
    [InlineData("Jobs:DefaultCfgScale", "-1", "Jobs:DefaultCfgScale")]
    [InlineData("Jobs:MaxDurationMs", "500", "Jobs:MaxDurationMs")]
    [InlineData("Webhook:TimeoutSeconds", "0", "Webhook:TimeoutSeconds")]
    [InlineData("Webhook:RetryDelaysSeconds:0", "-3", "Webhook:RetryDelaysSeconds")]
    [InlineData("Webhook:AllowedHosts:0", "https://x.example/path", "Webhook:AllowedHosts")]
    public void Invalid_values_are_rejected_with_a_message_naming_the_setting(string key, string value, string setting)
    {
        var o = key == "Model:Checkpoint" ? Load((key, value)) : Load(("Model:Checkpoint", Path.GetTempPath()), (key, value));
        Assert.Contains(o.Validate(cudaDeviceCount: 0), e => e.Contains(setting));
    }

    [Fact]
    public void A_cuda_index_beyond_the_installed_gpus_is_rejected()
    {
        var o = Load(("Model:Checkpoint", Path.GetTempPath()), ("Codec:Device", "cuda:2"));
        Assert.Contains(o.Validate(cudaDeviceCount: 2), e => e.Contains("Codec:Device") && e.Contains("2 CUDA"));
        Assert.Empty(Load(("Model:Checkpoint", Path.GetTempPath()), ("Codec:Device", "cuda:1")).Validate(cudaDeviceCount: 2));
    }

    [Fact]
    public void An_empty_host_list_allows_any_host_and_a_list_is_case_insensitive()
    {
        Assert.True(Load().IsWebhookHostAllowed("anything.example"));
        var o = Load(("Webhook:AllowedHosts:0", "Orchestrator.Local"));
        Assert.True(o.IsWebhookHostAllowed("orchestrator.local"));
        Assert.False(o.IsWebhookHostAllowed("evil.example"));
    }
}

public class EnvironmentAliasTests
{
    private static IConfiguration Cfg(params (string, string?)[] v) =>
        new ConfigurationBuilder().AddInMemoryCollection(v.Select(x => new KeyValuePair<string, string?>(x.Item1, x.Item2))).Build();

    [Fact]
    public void The_containers_variables_are_used_when_the_services_own_settings_are_absent()
    {
        var o = ServiceOptions.From(Cfg(("HEARTMULA_TOKEN", "tok"), ("HEARTMULA_JOBS", "/var/jobs"), ("HEARTMULA_VERSION", "7"), ("HEARTMULA_CKPT", "/ckpt")));
        Assert.Equal(("tok", "/var/jobs", "7", "/ckpt"), (o.Security.ApiKey, o.Storage.Directory, o.Service.Version, o.Model.Checkpoint));
    }

    [Fact]
    public void The_services_own_settings_win_over_the_aliases()
    {
        var o = ServiceOptions.From(Cfg(("Security:ApiKey", "own"), ("HEARTMULA_TOKEN", "alias"), ("Storage:Directory", "mine"), ("HEARTMULA_JOBS", "/alias"),
                                        ("Service:Version", "own"), ("HEARTMULA_VERSION", "alias")));
        Assert.Equal(("own", "mine", "own"), (o.Security.ApiKey, o.Storage.Directory, o.Service.Version));
    }

    [Fact]
    public void An_empty_own_key_as_appsettings_ships_it_does_not_hide_the_alias()
        => Assert.Equal("alias", ServiceOptions.From(Cfg(("Security:ApiKey", ""), ("HEARTMULA_TOKEN", "alias"))).Security.ApiKey);

    [Fact]
    public void Without_aliases_the_defaults_stay()
    {
        var o = ServiceOptions.From(Cfg());
        Assert.Equal(("songs", "1"), (o.Storage.Directory, o.Service.Version));
        Assert.True(string.IsNullOrEmpty(o.Security.ApiKey));
        Assert.Null(ServiceOptions.UrlFromPortAlias(Cfg()));
    }

    [Fact]
    public void The_port_alias_listens_on_all_interfaces_like_the_container_unless_Urls_was_set()
    {
        Assert.Equal("http://0.0.0.0:8787", ServiceOptions.UrlFromPortAlias(Cfg(("HEARTMULA_PORT", "8787"))));
        Assert.Equal("http://0.0.0.0:8787", ServiceOptions.UrlFromPortAlias(Cfg(("HEARTMULA_PORT", "8787"), ("Urls", ServiceOptions.DefaultUrl))));    // appsettings' default
        Assert.Null(ServiceOptions.UrlFromPortAlias(Cfg(("HEARTMULA_PORT", "8787"), ("Urls", "http://127.0.0.1:9000"))));                                // own setting wins
    }

    [Theory]
    [InlineData("abc")]
    [InlineData("0")]
    [InlineData("70000")]
    [InlineData("-1")]
    [InlineData("80.5")]
    public void A_bad_port_stops_startup_with_a_message_naming_the_variable(string port)
    {
        var e = Assert.Throws<ConfigurationException>(() => ServiceOptions.UrlFromPortAlias(Cfg(("HEARTMULA_PORT", port))));
        Assert.Contains("HEARTMULA_PORT", e.Message);
        var host = Assert.Throws<ConfigurationException>(() => HeartMuLa.Server.ServiceHost.Build(new[] { "--HEARTMULA_PORT=" + port }));
        Assert.Contains("HEARTMULA_PORT", host.Message);
    }

    [Fact]
    public async Task The_version_alias_shows_in_health_and_the_token_alias_protects_the_api()
    {
        using var factory = new ApiFactory(("HEARTMULA_VERSION", "9"), ("HEARTMULA_TOKEN", "tok"));
        using var client = factory.CreateClient();
        var until = DateTime.UtcNow.AddSeconds(5);
        System.Text.Json.JsonElement h;
        do { h = System.Text.Json.JsonDocument.Parse(await client.GetStringAsync("/health")).RootElement; } while (!h.GetProperty("ready").GetBoolean() && DateTime.UtcNow < until);
        Assert.Equal("9", h.GetProperty("version").GetString());
        Assert.Equal(System.Net.HttpStatusCode.Unauthorized, (await client.PostAsync("/generate", new StringContent("{}", System.Text.Encoding.UTF8, "application/json"))).StatusCode);
        client.DefaultRequestHeaders.Add("Authorization", "Bearer tok");
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, (await client.PostAsync("/generate", new StringContent("{}", System.Text.Encoding.UTF8, "application/json"))).StatusCode);
    }
}
