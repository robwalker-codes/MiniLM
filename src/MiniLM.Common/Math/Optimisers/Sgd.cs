using System.Collections.Generic;
using MiniLM.Common.Math;

namespace MiniLM.Common.Math.Optimisers;

public sealed class Sgd : IOptimiser
{
    private readonly float _learningRate;

    public Sgd(float learningRate)
    {
        _learningRate = learningRate;
    }

    public void Step(IEnumerable<Tensor> parameters)
    {
        foreach (var parameter in parameters)
        {
            for (var i = 0; i < parameter.Data.Length; i++)
            {
                parameter.Data[i] -= _learningRate * parameter.Grad[i];
            }
        }
    }

    public void Reset()
    {
        // No-op for SGD.
    }
}
