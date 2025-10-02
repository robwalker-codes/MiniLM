using System.Linq;
using MiniLM.Common.Data;
using MiniLM.Common.Math.Optimisers;
using MiniLM.Common.Tokens;
using MiniLM.Train.Model;
using MiniLM.Train.Model.Loss;
using Xunit;

namespace MiniLM.Tests.Model;

public sealed class MlpCharLMTests
{
    [Fact]
    public void ForwardProducesExpectedShape()
    {
        var vocab = CharVocab.FromCorpus("abcd");
        var tokenizer = new CharTokenizer(vocab);
        var loader = new DataLoader("abcd", tokenizer, contextLength: 2, batchSize: 1, shuffle: false, seed: 1);
        var batch = loader.First();
        var model = new MlpCharLM(vocab.Size, 2, embeddingDim: 4, hiddenDim: 8, hiddenDim2: 8, useLayerNorm: true, seed: 2);

        var logits = model.Forward(batch);
        Assert.Equal(1, logits.GetLength(0));
        Assert.Equal(vocab.Size, logits.GetLength(1));
    }

    [Fact]
    public void BackwardPopulatesGradients()
    {
        var vocab = CharVocab.FromCorpus("abcd");
        var tokenizer = new CharTokenizer(vocab);
        var loader = new DataLoader("abcd", tokenizer, contextLength: 2, batchSize: 1, shuffle: false, seed: 1);
        var batch = loader.First();
        var model = new MlpCharLM(vocab.Size, 2, embeddingDim: 4, hiddenDim: 8, hiddenDim2: 8, useLayerNorm: false, seed: 3);

        model.ZeroGradients();
        var logits = model.Forward(batch);
        var grad = new float[1, vocab.Size];
        grad[0, vocab['c']] = 1f;
        model.Backward(grad, batch);

        foreach (var parameter in model.Parameters)
        {
            Assert.All(parameter.Grad, value => Assert.False(float.IsNaN(value)));
        }
    }

    [Fact]
    public void ModelCanOverfitTinyCorpus()
    {
        var text = "hello hello";
        var vocab = CharVocab.FromCorpus(text);
        var tokenizer = new CharTokenizer(vocab);
        var model = new MlpCharLM(vocab.Size, contextLength: 3, embeddingDim: 8, hiddenDim: 16, hiddenDim2: 16, useLayerNorm: false, seed: 5);
        var optimiser = new Adam(0.05f);
        var loss = new CrossEntropyLoss();
        float finalLoss = float.MaxValue;

        for (var epoch = 0; epoch < 40; epoch++)
        {
            var loader = new DataLoader(text, tokenizer, contextLength: 3, batchSize: 2, shuffle: true, seed: epoch);
            foreach (var batch in loader)
            {
                model.ZeroGradients();
                var logits = model.Forward(batch);
                var (batchLoss, _, grad) = loss.Compute(logits, batch.Targets, batch.ValidCount);
                finalLoss = batchLoss;
                model.Backward(grad, batch);
                optimiser.Step(model.Parameters);
            }
        }

        Assert.True(finalLoss < 0.5f, $"Expected loss to fall below 0.5 but was {finalLoss:F4}");
    }
}
