using System;
using System.Collections.Generic;
using MiniLM.Common.Data;
using MiniLM.Common.Math;
using MiniLM.Train.Model.Layers;

namespace MiniLM.Train.Model;

public sealed class MlpCharLM : IModel
{
    private readonly Tensor _embedding;
    private readonly Dense _dense1;
    private readonly Dense _dense2;
    private readonly Dense _output;
    private readonly ReluActivation _relu1 = new();
    private readonly ReluActivation _relu2 = new();
    private readonly LayerNorm? _layerNorm;
    private readonly List<Tensor> _parameters;
    private readonly Dictionary<string, Tensor> _namedParameters;
    private readonly int _embeddingDim;
    private int[,]? _lastTokens;
    private int _lastValidCount;

    public MlpCharLM(int vocabSize, int contextLength, int embeddingDim, int hiddenDim, int hiddenDim2, bool useLayerNorm, int seed)
    {
        ValidateDimensions(vocabSize, contextLength, embeddingDim);

        VocabSize = vocabSize;
        ContextLength = contextLength;
        _embeddingDim = embeddingDim;

        var rng = new Random(seed);
        _embedding = CreateEmbedding(vocabSize, embeddingDim, rng);
        (_dense1, _dense2, _output) = CreateDenseStack(contextLength, embeddingDim, hiddenDim, hiddenDim2, vocabSize, rng);
        _layerNorm = CreateLayerNorm(useLayerNorm, hiddenDim2);

        _parameters = CollectParameters(_embedding, _dense1, _dense2, _output, _layerNorm);
        _namedParameters = BuildNamedParameters(_embedding, _dense1, _dense2, _output, _layerNorm);
    }

    private static void ValidateDimensions(int vocabSize, int contextLength, int embeddingDim)
    {
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (contextLength <= 0) throw new ArgumentOutOfRangeException(nameof(contextLength));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));
    }

    private static Tensor CreateEmbedding(int vocabSize, int embeddingDim, Random rng)
    {
        var embedding = new Tensor(vocabSize, embeddingDim);
        Initialisers.FillNormal(embedding, 0f, 0.02f, rng);
        return embedding;
    }

    private static (Dense dense1, Dense dense2, Dense output) CreateDenseStack(
        int contextLength,
        int embeddingDim,
        int hiddenDim,
        int hiddenDim2,
        int vocabSize,
        Random rng)
    {
        var inputDim = contextLength * embeddingDim;
        var dense1 = new Dense(inputDim, hiddenDim, rng);
        var dense2 = new Dense(hiddenDim, hiddenDim2, rng);
        var output = new Dense(hiddenDim2, vocabSize, rng);
        return (dense1, dense2, output);
    }

    private static LayerNorm? CreateLayerNorm(bool useLayerNorm, int hiddenDim2)
    {
        return useLayerNorm ? new LayerNorm(hiddenDim2) : null;
    }

    private static List<Tensor> CollectParameters(Tensor embedding, Dense dense1, Dense dense2, Dense output, LayerNorm? layerNorm)
    {
        var parameters = new List<Tensor>
        {
            embedding,
            dense1.Weights,
            dense1.Bias,
            dense2.Weights,
            dense2.Bias,
            output.Weights,
            output.Bias
        };

        if (layerNorm is not null)
        {
            parameters.Add(layerNorm.Gamma);
            parameters.Add(layerNorm.Beta);
        }

        return parameters;
    }

    private static Dictionary<string, Tensor> BuildNamedParameters(Tensor embedding, Dense dense1, Dense dense2, Dense output, LayerNorm? layerNorm)
    {
        var named = new Dictionary<string, Tensor>
        {
            ["embedding"] = embedding,
            ["dense1.weight"] = dense1.Weights,
            ["dense1.bias"] = dense1.Bias,
            ["dense2.weight"] = dense2.Weights,
            ["dense2.bias"] = dense2.Bias,
            ["output.weight"] = output.Weights,
            ["output.bias"] = output.Bias
        };

        if (layerNorm is not null)
        {
            named["layernorm.gamma"] = layerNorm.Gamma;
            named["layernorm.beta"] = layerNorm.Beta;
        }

        return named;
    }

    public int ContextLength { get; }
    public int VocabSize { get; }
    public IReadOnlyList<Tensor> Parameters => _parameters;

    public float[,] Forward(Batch batch)
    {
        _lastTokens = (int[,])batch.Inputs.Clone();
        _lastValidCount = batch.ValidCount;

        var embedded = Embed(_lastTokens);
        var hidden1 = _dense1.Forward(embedded);
        var activated1 = _relu1.Forward(hidden1);
        var hidden2 = _dense2.Forward(activated1);
        var activated2 = _relu2.Forward(hidden2);
        var normed = _layerNorm is null ? activated2 : _layerNorm.Forward(activated2);
        return _output.Forward(normed);
    }

    public void Backward(float[,] gradLogits, Batch batch)
    {
        var grad = _output.Backward(gradLogits);
        if (_layerNorm is not null)
        {
            grad = _layerNorm.Backward(grad);
        }

        grad = _relu2.Backward(grad);
        grad = _dense2.Backward(grad);
        grad = _relu1.Backward(grad);
        var gradEmbedding = _dense1.Backward(grad);
        UpdateEmbeddingGradients(gradEmbedding);
    }

    public void ZeroGradients()
    {
        foreach (var parameter in _parameters)
        {
            parameter.ZeroGrad();
        }

        _dense1.ZeroGradients();
        _dense2.ZeroGradients();
        _output.ZeroGradients();
        _layerNorm?.ZeroGradients();
    }

    public IReadOnlyDictionary<string, Tensor> GetNamedParameters() => _namedParameters;

    public void LoadNamedParameters(IReadOnlyDictionary<string, float[]> data, IReadOnlyDictionary<string, int[]> shapes)
    {
        foreach (var (name, tensor) in _namedParameters)
        {
            if (!data.TryGetValue(name, out var buffer))
            {
                throw new InvalidOperationException($"Checkpoint missing parameter '{name}'.");
            }

            if (!shapes.TryGetValue(name, out var shape))
            {
                throw new InvalidOperationException($"Checkpoint missing shape for '{name}'.");
            }

            if (Tensor.TotalLength(shape) != tensor.Length)
            {
                throw new InvalidOperationException($"Shape mismatch for '{name}'.");
            }

            Array.Copy(buffer, tensor.Data, tensor.Data.Length);
        }
    }

    private float[,] Embed(int[,] tokens)
    {
        var batch = tokens.GetLength(0);
        var output = new float[batch, ContextLength * _embeddingDim];

        for (var row = 0; row < batch; row++)
        {
            for (var pos = 0; pos < ContextLength; pos++)
            {
                var tokenId = tokens[row, pos];
                var baseIndex = tokenId * _embeddingDim;
                var dest = pos * _embeddingDim;
                for (var dim = 0; dim < _embeddingDim; dim++)
                {
                    output[row, dest + dim] = _embedding.Data[baseIndex + dim];
                }
            }
        }

        return output;
    }

    private void UpdateEmbeddingGradients(float[,] gradEmbedding)
    {
        if (_lastTokens is null)
        {
            throw new InvalidOperationException("Forward must be called before Backward.");
        }

        for (var i = 0; i < _embedding.Grad.Length; i++)
        {
            _embedding.Grad[i] = 0f;
        }

        var batch = _lastTokens.GetLength(0);
        for (var row = 0; row < Math.Min(batch, _lastValidCount); row++)
        {
            for (var pos = 0; pos < ContextLength; pos++)
            {
                var tokenId = _lastTokens[row, pos];
                var baseIndex = tokenId * _embeddingDim;
                var source = pos * _embeddingDim;
                for (var dim = 0; dim < _embeddingDim; dim++)
                {
                    _embedding.Grad[baseIndex + dim] += gradEmbedding[row, source + dim];
                }
            }
        }
    }
}
