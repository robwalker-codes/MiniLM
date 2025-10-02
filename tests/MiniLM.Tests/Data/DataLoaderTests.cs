using System.Linq;
using MiniLM.Common.Data;
using MiniLM.Common.Tokens;
using Xunit;

namespace MiniLM.Tests.Data;

public sealed class DataLoaderTests
{
    [Fact]
    public void ProducesExpectedBatches()
    {
        var corpus = "abcd";
        var vocab = CharVocab.FromCorpus(corpus);
        var tokenizer = new CharTokenizer(vocab);
        var loader = new DataLoader(corpus, tokenizer, contextLength: 2, batchSize: 2, shuffle: false, seed: 1);

        var batches = loader.ToList();
        Assert.Equal(2, batches.Count);

        var first = batches[0];
        Assert.Equal(2, first.ValidCount);
        Assert.Equal(vocab.BosId, first.Inputs[0, 0]);
        Assert.Equal(vocab['a'], first.Inputs[0, 1]);
        Assert.Equal(vocab['b'], first.Targets[0]);
        Assert.Equal(vocab['a'], first.Inputs[1, 0]);
        Assert.Equal(vocab['b'], first.Inputs[1, 1]);
        Assert.Equal(vocab['c'], first.Targets[1]);

        var second = batches[1];
        Assert.Equal(2, second.ValidCount);
        Assert.Equal(vocab['b'], second.Inputs[0, 0]);
        Assert.Equal(vocab['c'], second.Inputs[0, 1]);
        Assert.Equal(vocab['d'], second.Targets[0]);
        Assert.Equal(vocab['c'], second.Inputs[1, 0]);
        Assert.Equal(vocab['d'], second.Inputs[1, 1]);
        Assert.Equal(vocab.EosId, second.Targets[1]);
    }

    [Fact]
    public void PadsIncompleteBatch()
    {
        var corpus = "a";
        var vocab = CharVocab.FromCorpus(corpus);
        var tokenizer = new CharTokenizer(vocab);
        var loader = new DataLoader(corpus, tokenizer, contextLength: 2, batchSize: 3, shuffle: false, seed: 1);

        var batch = loader.Single();
        Assert.Equal(1, batch.ValidCount);
        for (var row = batch.ValidCount; row < batch.BatchSize; row++)
        {
            Assert.Equal(vocab.PadId, batch.Targets[row]);
            Assert.All(Enumerable.Range(0, batch.ContextLength), col => Assert.Equal(vocab.PadId, batch.Inputs[row, col]));
        }
    }
}
