using System;
using MiniLM.Common.Math;

namespace MiniLM.Train.Model.Layers;

public sealed class LayerNorm
{
    private readonly int _features;
    private readonly float _epsilon;
    private float[,]? _normalized;
    private float[]? _invStd;

    public LayerNorm(int features, float epsilon = 1e-5f)
    {
        _features = features;
        _epsilon = epsilon;
        Gamma = new Tensor(features);
        Beta = new Tensor(features);
        for (var i = 0; i < features; i++)
        {
            Gamma.Data[i] = 1f;
            Beta.Data[i] = 0f;
        }
    }

    public Tensor Gamma { get; }
    public Tensor Beta { get; }

    public float[,] Forward(float[,] input)
    {
        var batch = input.GetLength(0);
        var output = new float[batch, _features];
        PrepareBuffers(batch);

        for (var row = 0; row < batch; row++)
        {
            var mean = CalculateMean(input, row);
            var invStd = CalculateInverseStd(input, row, mean);
            _invStd![row] = invStd;
            ApplyAffineTransform(input, output, row, mean, invStd);
        }

        return output;
    }

    private void PrepareBuffers(int batch)
    {
        _normalized = new float[batch, _features];
        _invStd = new float[batch];
    }

    private float CalculateMean(float[,] input, int row)
    {
        var sum = 0f;
        for (var col = 0; col < _features; col++)
        {
            sum += input[row, col];
        }

        return sum / _features;
    }

    private float CalculateInverseStd(float[,] input, int row, float mean)
    {
        var variance = 0f;
        for (var col = 0; col < _features; col++)
        {
            var diff = input[row, col] - mean;
            variance += diff * diff;
        }

        variance /= _features;
        return 1f / MathF.Sqrt(variance + _epsilon);
    }

    private void ApplyAffineTransform(float[,] input, float[,] output, int row, float mean, float invStd)
    {
        for (var col = 0; col < _features; col++)
        {
            var normalized = (input[row, col] - mean) * invStd;
            _normalized![row, col] = normalized;
            output[row, col] = normalized * Gamma.Data[col] + Beta.Data[col];
        }
    }

    public float[,] Backward(float[,] gradOutput)
    {
        EnsureForwardState();

        var batch = gradOutput.GetLength(0);
        var gradInput = CreateGradInput(batch);

        AccumulateParameterGradients(gradOutput, batch);
        ComputeGradInput(gradOutput, gradInput, batch);

        return gradInput;
    }

    private void EnsureForwardState()
    {
        if (_normalized is null || _invStd is null)
        {
            throw new InvalidOperationException("Forward must be called before Backward.");
        }
    }

    private float[,] CreateGradInput(int batch)
    {
        return new float[batch, _features];
    }

    private void AccumulateParameterGradients(float[,] gradOutput, int batch)
    {
        for (var col = 0; col < _features; col++)
        {
            var gradGamma = 0f;
            var gradBeta = 0f;

            for (var row = 0; row < batch; row++)
            {
                gradGamma += gradOutput[row, col] * _normalized![row, col];
                gradBeta += gradOutput[row, col];
            }

            Gamma.Grad[col] += gradGamma;
            Beta.Grad[col] += gradBeta;
        }
    }

    private void ComputeGradInput(float[,] gradOutput, float[,] gradInput, int batch)
    {
        for (var row = 0; row < batch; row++)
        {
            var (sumDy, sumDyNorm) = CalculateRowSums(gradOutput, row);
            ApplyRowGradient(gradOutput, gradInput, row, sumDy, sumDyNorm);
        }
    }

    private (float sumDy, float sumDyNorm) CalculateRowSums(float[,] gradOutput, int row)
    {
        var sumDy = 0f;
        var sumDyNorm = 0f;

        for (var col = 0; col < _features; col++)
        {
            var dy = gradOutput[row, col] * Gamma.Data[col];
            sumDy += dy;
            sumDyNorm += dy * _normalized![row, col];
        }

        return (sumDy, sumDyNorm);
    }

    private void ApplyRowGradient(float[,] gradOutput, float[,] gradInput, int row, float sumDy, float sumDyNorm)
    {
        for (var col = 0; col < _features; col++)
        {
            var dy = gradOutput[row, col] * Gamma.Data[col];
            gradInput[row, col] = (_invStd![row] / _features) * (_features * dy - sumDy - _normalized![row, col] * sumDyNorm);
        }
    }

    public void ZeroGradients()
    {
        Gamma.ZeroGrad();
        Beta.ZeroGrad();
    }
}
