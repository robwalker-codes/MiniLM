using System;
using MiniLM.Infer;
using Xunit;

namespace MiniLM.Tests.Cli;

public sealed class InferCliArgsTests
{
    [Fact]
    public void ParseArgsUsesDefaults()
    {
        var args = new[]
        {
            "--checkpoint", "ckpt.json",
            "--prompt", "Hi",
            "--max-tokens", "5",
            "--temperature", "0.9",
            "--top-k", "2",
            "--seed", "11",
            "--no-sampler-determinism",
            "--verbose"
        };

        var config = Program.ParseArgs(args);
        Assert.Equal("ckpt.json", config.CheckpointPath);
        Assert.Equal("Hi", config.Prompt);
        Assert.Equal(5, config.MaxTokens);
        Assert.Equal(0.9f, config.Temperature);
        Assert.Equal(2, config.TopK);
        Assert.Equal(11, config.Seed);
        Assert.False(config.Deterministic);
        Assert.True(config.Verbose);
    }

    [Fact]
    public void MissingCheckpointThrows()
    {
        Assert.Throws<ArgumentException>(() => Program.ParseArgs(Array.Empty<string>()));
    }
}
