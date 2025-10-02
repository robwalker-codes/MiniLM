using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using MiniLM.Common.Checkpoint;
using MiniLM.Common.Tokens;
using MiniLM.Train.Model;
using Xunit;

namespace MiniLM.Tests.Checkpoint;

public sealed class CheckpointIOTests
{
    [Fact]
    public async Task SaveAndLoadRoundTrip()
    {
        var vocab = CharVocab.FromCorpus("abc");
        var checkpoint = new CheckpointModel
        {
            ContextLength = 4,
            ModelType = "mlp",
            VocabSize = vocab.Size,
            Vocab = vocab.ToSerializable(),
            Parameters = new Dictionary<string, float[]>
            {
                ["embedding"] = new float[] { 0.1f, 0.2f, 0.3f, 0.4f },
                ["dense1.weight"] = new float[] { 0.1f, 0.2f },
                ["dense1.bias"] = new float[] { 0f },
                ["dense2.weight"] = new float[] { 0.1f },
                ["dense2.bias"] = new float[] { 0f },
                ["output.weight"] = new float[] { 0.1f, 0.2f, 0.3f },
                ["output.bias"] = new float[] { 0f, 0f, 0f }
            },
            Shapes = new Dictionary<string, int[]>
            {
                ["embedding"] = new[] { 1, 4 },
                ["dense1.weight"] = new[] { 1, 2 },
                ["dense1.bias"] = new[] { 1 },
                ["dense2.weight"] = new[] { 1, 1 },
                ["dense2.bias"] = new[] { 1 },
                ["output.weight"] = new[] { 1, 3 },
                ["output.bias"] = new[] { 3 }
            }
        };

        var path = Path.Combine(Path.GetTempPath(), $"checkpoint-{Guid.NewGuid():N}.json");
        await CheckpointIO.SaveAsync(path, checkpoint, CancellationToken.None);
        var loaded = await CheckpointIO.LoadAsync(path, CancellationToken.None);

        Assert.Equal(checkpoint.ContextLength, loaded.ContextLength);
        Assert.Equal(checkpoint.VocabSize, loaded.VocabSize);
        Assert.Equal(checkpoint.Parameters["embedding"], loaded.Parameters["embedding"]);
    }

    [Fact]
    public void LoadNamedParametersDetectsMismatch()
    {
        var vocab = CharVocab.FromCorpus("hello");
        var model = new MlpCharLM(vocab.Size, 2, embeddingDim: 4, hiddenDim: 8, hiddenDim2: 8, useLayerNorm: false, seed: 1);
        var parameters = new Dictionary<string, float[]>();
        var shapes = new Dictionary<string, int[]>();

        foreach (var (name, tensor) in model.GetNamedParameters())
        {
            parameters[name] = (float[])tensor.Data.Clone();
            shapes[name] = (int[])tensor.Shape.Clone();
        }

        shapes["embedding"] = new[] { 99, 99 };

        Assert.Throws<InvalidOperationException>(() => model.LoadNamedParameters(parameters, shapes));
    }
}
