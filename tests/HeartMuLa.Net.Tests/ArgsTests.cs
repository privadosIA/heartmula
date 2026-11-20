using HeartMuLa.Cli;

namespace HeartMuLa.Tests;

/// <summary>The CLI's minimal argument parser: <c>--key value</c>, bare <c>--flag</c> and positionals.</summary>
public class ArgsTests
{
    [Fact]
    public void Named_values_flags_and_positionals_are_separated()
    {
        var a = Args.Parse(new[] { "generate", "extra", "--ckpt", "/models", "--wav", "song.wav", "--codec-tf32" });

        Assert.Equal(new[] { "generate", "extra" }, a.Positional);
        Assert.Equal("/models", a.Get("ckpt"));
        Assert.Equal("song.wav", a.Get("wav"));
        Assert.Equal("true", a.Get("codec-tf32"));      // a flag with nothing after it is "true"
    }

    [Fact]
    public void A_flag_followed_by_another_option_is_true_and_does_not_swallow_it()
    {
        var a = Args.Parse(new[] { "--codec-tf32", "--seed", "42" });

        Assert.Equal("true", a.Get("codec-tf32"));
        Assert.Equal("42", a.Get("seed"));
        Assert.Empty(a.Positional);
    }

    [Fact]
    public void An_option_that_was_not_given_is_null_and_the_last_repetition_wins()
    {
        var a = Args.Parse(new[] { "--ms", "1000", "--ms", "2000" });

        Assert.Null(a.Get("seed"));
        Assert.Equal("2000", a.Get("ms"));
    }
}
