using System;

namespace MiniLM.Train.Model.Loss;

public sealed class CrossEntropyLoss
{
    public (float Loss, float Accuracy, float[,] Gradient) Compute(float[,] logits, int[] targets, int validCount)
    {
        var batch = logits.GetLength(0);
        var vocab = logits.GetLength(1);
        var gradient = new float[batch, vocab];
        var totalLoss = 0f;
        var correct = 0;
        var effectiveCount = Math.Max(1, validCount);

        for (var row = 0; row < validCount; row++)
        {
            var target = targets[row];
            if (!IsValidTarget(target, vocab))
            {
                continue;
            }

            var logStats = ComputeLogStatistics(logits, row, vocab);
            totalLoss += ComputeLossContribution(logits, row, target, logStats.LogSumExp);
            PopulateGradientRow(logits, row, target, logStats.LogSumExp, vocab, gradient);

            if (IsCorrectPrediction(logits, row, target))
            {
                correct++;
            }
        }

        var loss = totalLoss / effectiveCount;
        NormaliseGradient(gradient, validCount, vocab, effectiveCount);

        return (loss, validCount == 0 ? 0f : correct / (float)validCount, gradient);
    }

    private static bool IsValidTarget(int target, int vocab) => target >= 0 && target < vocab;

    private static (float MaxLogit, float LogSumExp) ComputeLogStatistics(float[,] logits, int row, int vocab)
    {
        var maxLogit = FindMaxLogit(logits, row, vocab);
        var sumExp = 0f;
        for (var col = 0; col < vocab; col++)
        {
            sumExp += MathF.Exp(logits[row, col] - maxLogit);
        }

        return (maxLogit, maxLogit + MathF.Log(sumExp));
    }

    private static float FindMaxLogit(float[,] logits, int row, int vocab)
    {
        var maxLogit = float.NegativeInfinity;
        for (var col = 0; col < vocab; col++)
        {
            maxLogit = MathF.Max(maxLogit, logits[row, col]);
        }

        return maxLogit;
    }

    private static float ComputeLossContribution(float[,] logits, int row, int target, float logSumExp) =>
        -logits[row, target] + logSumExp;

    private static void PopulateGradientRow(float[,] logits, int row, int target, float logSumExp, int vocab, float[,] gradient)
    {
        for (var col = 0; col < vocab; col++)
        {
            var softmax = MathF.Exp(logits[row, col] - logSumExp);
            gradient[row, col] = softmax;
        }

        gradient[row, target] -= 1f;
    }

    private static bool IsCorrectPrediction(float[,] logits, int row, int target) => ArgMax(logits, row) == target;

    private static void NormaliseGradient(float[,] gradient, int validCount, int vocab, int effectiveCount)
    {
        if (validCount == 0)
        {
            return;
        }

        for (var row = 0; row < validCount; row++)
        {
            for (var col = 0; col < vocab; col++)
            {
                gradient[row, col] /= effectiveCount;
            }
        }
    }

    private static int ArgMax(float[,] logits, int row)
    {
        var cols = logits.GetLength(1);
        var bestIndex = 0;
        var bestValue = float.NegativeInfinity;
        for (var col = 0; col < cols; col++)
        {
            var value = logits[row, col];
            if (value > bestValue)
            {
                bestValue = value;
                bestIndex = col;
            }
        }

        return bestIndex;
    }
}
