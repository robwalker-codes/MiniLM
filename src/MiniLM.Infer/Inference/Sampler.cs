using System;
using System.Collections.Generic;
using MiniLM.Common.Data;
using MiniLM.Common.Tokens;
using MiniLM.Train.Model;

namespace MiniLM.Infer.Inference;

public sealed class Sampler
{
    private readonly Random _rng;

    public Sampler(int seed, bool deterministic)
    {
        _rng = deterministic ? new Random(seed) : new Random();
    }

    public string Generate(CharTokenizer tokenizer, IModel model, string prompt, int maxTokens, float temperature, int topK)
    {
        if (temperature < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be non-negative.");
        }

        if (maxTokens <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTokens));
        }

        var sequence = new List<int> { tokenizer.Vocab.BosId };
        var promptTokens = tokenizer.Encode(prompt, addBosEos: false);
        sequence.AddRange(promptTokens);

        for (var step = 0; step < maxTokens; step++)
        {
            var context = BuildContext(sequence, model.ContextLength, tokenizer.Vocab.PadId);
            var batch = new Batch(context, new int[1], 1);
            var logits = model.Forward(batch);
            var nextToken = SampleToken(logits, temperature, topK, tokenizer.Vocab.UnkId);
            sequence.Add(nextToken);

            if (nextToken == tokenizer.Vocab.EosId)
            {
                break;
            }
        }

        return tokenizer.Decode(sequence);
    }

    private int[,] BuildContext(List<int> sequence, int contextLength, int padId)
    {
        var context = new int[1, contextLength];
        for (var i = 0; i < contextLength; i++)
        {
            context[0, i] = padId;
        }

        var count = Math.Min(contextLength, sequence.Count);
        for (var i = 0; i < count; i++)
        {
            context[0, contextLength - count + i] = sequence[sequence.Count - count + i];
        }

        return context;
    }

    private int SampleToken(float[,] logits, float temperature, int topK, int fallbackToken)
    {
        var (scores, maxLogit) = ExtractScores(logits);

        if (temperature == 0f)
        {
            return ArgMax(scores);
        }

        var candidates = GetCandidates(scores, topK);
        var (probabilities, sum) = ComputeProbabilities(scores, candidates, maxLogit, temperature);

        if (sum <= 0f || float.IsNaN(sum))
        {
            return fallbackToken;
        }

        return SampleFromDistribution(probabilities, candidates, sum, fallbackToken);
    }

    private static (float[] scores, float maxLogit) ExtractScores(float[,] logits)
    {
        var vocab = logits.GetLength(1);
        var scores = new float[vocab];
        var maxLogit = float.NegativeInfinity;

        for (var i = 0; i < vocab; i++)
        {
            var value = logits[0, i];
            if (float.IsNaN(value))
            {
                throw new InvalidOperationException("Logits contain NaN values.");
            }

            scores[i] = value;
            maxLogit = MathF.Max(maxLogit, value);
        }

        return (scores, maxLogit);
    }

    private static (float[] probabilities, float sum) ComputeProbabilities(
        float[] scores,
        IReadOnlyList<int> candidates,
        float maxLogit,
        float temperature)
    {
        var probabilities = new float[candidates.Count];
        var sum = 0f;
        var scale = MathF.Max(1e-5f, temperature);

        for (var i = 0; i < candidates.Count; i++)
        {
            var adjusted = (scores[candidates[i]] - maxLogit) / scale;
            var value = MathF.Exp(adjusted);
            probabilities[i] = value;
            sum += value;
        }

        return (probabilities, sum);
    }

    private int SampleFromDistribution(float[] probabilities, IReadOnlyList<int> candidates, float sum, int fallbackToken)
    {
        var threshold = (float)_rng.NextDouble() * sum;
        var cumulative = 0f;

        for (var i = 0; i < candidates.Count; i++)
        {
            cumulative += probabilities[i];
            if (threshold <= cumulative)
            {
                return candidates[i];
            }
        }

        return candidates.Count > 0 ? candidates[^1] : fallbackToken;
    }

    private static int ArgMax(float[] scores)
    {
        var bestIndex = 0;
        var bestValue = float.NegativeInfinity;
        for (var i = 0; i < scores.Length; i++)
        {
            if (scores[i] > bestValue)
            {
                bestValue = scores[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    private static List<int> GetCandidates(float[] scores, int topK)
    {
        var indices = new List<int>(scores.Length);
        for (var i = 0; i < scores.Length; i++)
        {
            indices.Add(i);
        }

        if (topK > 0 && topK < scores.Length)
        {
            indices.Sort((a, b) => scores[b].CompareTo(scores[a]));
            indices = indices.GetRange(0, topK);
        }

        return indices;
    }
}
