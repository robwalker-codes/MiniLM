using System;
using System.Collections.Generic;
using MiniLM.Common.Math;

namespace MiniLM.Common.Math.Optimisers;

public sealed class Adam : IOptimiser
{
    private readonly float _learningRate;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly Dictionary<Tensor, AdamState> _states = new();
    private int _timeStep;

    public Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
    {
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
    }

    public void Step(IEnumerable<Tensor> parameters)
    {
        _timeStep++;
        foreach (var parameter in parameters)
        {
            if (!_states.TryGetValue(parameter, out var state))
            {
                state = new AdamState(parameter.Length);
                _states[parameter] = state;
            }

            for (var i = 0; i < parameter.Length; i++)
            {
                var grad = parameter.Grad[i];
                state.M[i] = _beta1 * state.M[i] + (1 - _beta1) * grad;
                state.V[i] = _beta2 * state.V[i] + (1 - _beta2) * grad * grad;

                var mHat = state.M[i] / (1 - MathF.Pow(_beta1, _timeStep));
                var vHat = state.V[i] / (1 - MathF.Pow(_beta2, _timeStep));

                parameter.Data[i] -= _learningRate * mHat / (MathF.Sqrt(vHat) + _epsilon);
            }
        }
    }

    public void Reset()
    {
        _states.Clear();
        _timeStep = 0;
    }

    private sealed class AdamState
    {
        public AdamState(int length)
        {
            M = new float[length];
            V = new float[length];
        }

        public float[] M { get; }
        public float[] V { get; }
    }
}
