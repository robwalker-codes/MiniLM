using System;
using MiniLM.Common.Math;

namespace MiniLM.Train.Model.Layers;

public sealed class Dense
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private float[,]? _lastInput;

    public Dense(int inputDim, int outputDim, Random rng)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        Weights = new Tensor(inputDim, outputDim);
        Bias = new Tensor(outputDim);
        Initialisers.XavierUniform(Weights, inputDim, outputDim, rng);
        Initialisers.FillConstant(Bias, 0f);
    }

    public Tensor Weights { get; }
    public Tensor Bias { get; }

    public float[,] Forward(float[,] input)
    {
        _lastInput = (float[,])input.Clone();
        var batch = input.GetLength(0);
        var output = new float[batch, _outputDim];

        for (var row = 0; row < batch; row++)
        {
            for (var col = 0; col < _outputDim; col++)
            {
                var sum = Bias.Data[col];
                for (var i = 0; i < _inputDim; i++)
                {
                    sum += input[row, i] * Weights.Data[i * _outputDim + col];
                }

                output[row, col] = sum;
            }
        }

        return output;
    }

    public float[,] Backward(float[,] gradOutput)
    {
        if (_lastInput is null)
        {
            throw new InvalidOperationException("Forward must be called before Backward.");
        }

        var batch = gradOutput.GetLength(0);
        var gradInput = new float[batch, _inputDim];

        for (var col = 0; col < _outputDim; col++)
        {
            Bias.Grad[col] += SumColumn(gradOutput, col);
        }

        for (var row = 0; row < batch; row++)
        {
            for (var col = 0; col < _outputDim; col++)
            {
                var grad = gradOutput[row, col];
                for (var i = 0; i < _inputDim; i++)
                {
                    Weights.Grad[i * _outputDim + col] += _lastInput[row, i] * grad;
                    gradInput[row, i] += Weights.Data[i * _outputDim + col] * grad;
                }
            }
        }

        return gradInput;
    }

    public void ZeroGradients()
    {
        Weights.ZeroGrad();
        Bias.ZeroGrad();
    }

    private static float SumColumn(float[,] matrix, int column)
    {
        var sum = 0f;
        var rows = matrix.GetLength(0);
        for (var row = 0; row < rows; row++)
        {
            sum += matrix[row, column];
        }

        return sum;
    }
}
