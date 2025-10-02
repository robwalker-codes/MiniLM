using System;

namespace MiniLM.Train.Model.Layers;

public sealed class ReluActivation
{
    private float[,]? _lastOutput;

    public float[,] Forward(float[,] input)
    {
        var rows = input.GetLength(0);
        var cols = input.GetLength(1);
        var output = new float[rows, cols];

        for (var row = 0; row < rows; row++)
        {
            for (var col = 0; col < cols; col++)
            {
                var value = input[row, col];
                output[row, col] = value > 0 ? value : 0f;
            }
        }

        _lastOutput = output;
        return output;
    }

    public float[,] Backward(float[,] gradOutput)
    {
        if (_lastOutput is null)
        {
            throw new InvalidOperationException("Forward must be called before Backward.");
        }

        var rows = gradOutput.GetLength(0);
        var cols = gradOutput.GetLength(1);
        var gradInput = new float[rows, cols];

        for (var row = 0; row < rows; row++)
        {
            for (var col = 0; col < cols; col++)
            {
                gradInput[row, col] = _lastOutput[row, col] > 0 ? gradOutput[row, col] : 0f;
            }
        }

        return gradInput;
    }
}
