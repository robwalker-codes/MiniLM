using System;
using MiniLM.Train;
using Xunit;

namespace MiniLM.Tests.Cli;

public sealed class TrainCliArgsTests
{
    [Fact]
    public void ParseArgsPopulatesConfig()
    {
        var args = new[]
        {
            "--urls", "urls.txt",
            "--output", "out.json",
            "--epochs", "2",
            "--batch-size", "8",
            "--context-length", "16",
            "--lr", "0.001",
            "--seed", "7",
            "--model", "mlp",
            "--embedding-dim", "12",
            "--hidden-dim", "24",
            "--hidden-dim2", "24",
            "--optimiser", "sgd",
            "--disable-layer-norm",
            "--no-shuffle",
            "--verbose",
            "--corpus", "hello"
        };

        var config = Program.ParseArgs(args);
        Assert.Equal("urls.txt", config.UrlListPath);
        Assert.False(config.UseLayerNorm);
        Assert.False(config.Shuffle);
        Assert.True(config.Verbose);
        Assert.Equal("hello", config.CorpusOverride);
        Assert.Equal(2, config.Epochs);
        Assert.Equal(8, config.BatchSize);
        Assert.Equal(16, config.ContextLength);
    }

    [Fact]
    public void MissingRequiredArgThrows()
    {
        Assert.Throws<ArgumentException>(() => Program.ParseArgs(Array.Empty<string>()));
    }
}
