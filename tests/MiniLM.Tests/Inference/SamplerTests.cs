using System;
using System.Collections.Generic;
using MiniLM.Common.Data;
using MiniLM.Common.Math;
using MiniLM.Common.Tokens;
using MiniLM.Infer.Inference;
using MiniLM.Train.Model;
using Xunit;

namespace MiniLM.Tests.Inference;

public sealed class SamplerTests
{
    [Fact]
    public void TopKRestrictsCandidates()
    {
        var vocab = CharVocab.FromCorpus("abc");
        var tokenizer = new CharTokenizer(vocab);
        var logits = new float[1, vocab.Size];
        logits[0, vocab['a']] = 5f;
        logits[0, vocab['b']] = 1f;

        var model = new FakeModel(vocab.Size, logits);
        var sampler = new Sampler(seed: 1, deterministic: true);
        var result = sampler.Generate(tokenizer, model, string.Empty, maxTokens: 1, temperature: 1f, topK: 1);

        Assert.Equal("a", result);
    }

    [Fact]
    public void TemperatureZeroReturnsArgMax()
    {
        var vocab = CharVocab.FromCorpus("abc");
        var tokenizer = new CharTokenizer(vocab);
        var logits = new float[1, vocab.Size];
        logits[0, vocab['b']] = 4f;
        logits[0, vocab['a']] = 1f;

        var model = new FakeModel(vocab.Size, logits);
        var sampler = new Sampler(seed: 1, deterministic: false);
        var result = sampler.Generate(tokenizer, model, string.Empty, maxTokens: 1, temperature: 0f, topK: 0);

        Assert.Equal("b", result);
    }

        private sealed class FakeModel : IModel
    {
        private readonly float[,] _logits;

        public FakeModel(int vocabSize, float[,] logits)
        {
            VocabSize = vocabSize;
            _logits = logits;
        }

        public int ContextLength => 2;
        public int VocabSize { get; }
        public IReadOnlyList<Tensor> Parameters => Array.Empty<Tensor>();

        public float[,] Forward(Batch batch)
        {
            return (float[,])_logits.Clone();
        }

        public void Backward(float[,] gradLogits, Batch batch) => throw new NotSupportedException();
        public void ZeroGradients() { }
        public IReadOnlyDictionary<string, Tensor> GetNamedParameters() => throw new NotSupportedException();
        public void LoadNamedParameters(IReadOnlyDictionary<string, float[]> data, IReadOnlyDictionary<string, int[]> shapes) => throw new NotSupportedException();
    }
}
