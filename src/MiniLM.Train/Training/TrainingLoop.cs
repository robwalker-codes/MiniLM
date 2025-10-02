using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MiniLM.Common.Checkpoint;
using MiniLM.Common.Data;
using MiniLM.Common.Math;
using MiniLM.Common.Math.Optimisers;
using MiniLM.Common.Text;
using MiniLM.Common.Tokens;
using MiniLM.Common.Util;
using MiniLM.Train.Model;
using MiniLM.Train.Model.Loss;

namespace MiniLM.Train.Training;

public sealed class TrainingLoop
{
    private readonly CorpusBuilder _corpusBuilder;
    private readonly CrossEntropyLoss _loss = new();

    public TrainingLoop(CorpusBuilder corpusBuilder)
    {
        _corpusBuilder = corpusBuilder;
    }

    public async Task<CheckpointModel> TrainAsync(TrainingConfig config, CancellationToken cancellationToken)
    {
        Logging.SetVerbose(config.Verbose);
        Logging.Info("Starting training...");

        var corpus = await LoadCorpusAsync(config, cancellationToken).ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(corpus))
        {
            throw new InvalidOperationException("Corpus is empty. Provide URLs with textual content or a corpus override.");
        }

        var vocab = CharVocab.FromCorpus(corpus);
        var tokenizer = new CharTokenizer(vocab);

        var model = CreateModel(config, vocab.Size);
        var optimiser = CreateOptimiser(config);

        for (var epoch = 0; epoch < config.Epochs; epoch++)
        {
            Logging.Info($"Epoch {epoch + 1}/{config.Epochs}");
            var dataLoader = new DataLoader(corpus, tokenizer, config.ContextLength, config.BatchSize, config.Shuffle, config.Seed + epoch);
            var tracker = new MetricsTracker();
            foreach (var batch in dataLoader)
            {
                cancellationToken.ThrowIfCancellationRequested();
                model.ZeroGradients();
                var logits = model.Forward(batch);
                var (loss, accuracy, grad) = _loss.Compute(logits, batch.Targets, batch.ValidCount);
                model.Backward(grad, batch);
                optimiser.Step(model.Parameters);
                tracker.Update(loss, accuracy);
            }

            Logging.Info($"Epoch {epoch + 1} loss={tracker.Loss:F4} acc={tracker.Accuracy:P2}");
        }

        var checkpoint = new CheckpointModel
        {
            Version = "1.0",
            ModelType = config.ModelType,
            ContextLength = config.ContextLength,
            VocabSize = vocab.Size,
            Vocab = vocab.ToSerializable(),
            Parameters = ExtractParameterData(model.GetNamedParameters()),
            Shapes = ExtractShapes(model.GetNamedParameters()),
            Training = new CheckpointModel.TrainingMetadata
            {
                Epochs = config.Epochs,
                LearningRate = config.LearningRate,
                Optimiser = config.Optimiser
            }
        };

        await CheckpointIO.SaveAsync(config.OutputPath, checkpoint, cancellationToken).ConfigureAwait(false);
        Logging.Info($"Checkpoint saved to {config.OutputPath}");
        return checkpoint;
    }

    private async Task<string> LoadCorpusAsync(TrainingConfig config, CancellationToken cancellationToken)
    {
        if (!string.IsNullOrEmpty(config.CorpusOverride))
        {
            return config.CorpusOverride;
        }

        if (!File.Exists(config.UrlListPath))
        {
            throw new FileNotFoundException("URL list file not found", config.UrlListPath);
        }

        var urls = await File.ReadAllLinesAsync(config.UrlListPath, cancellationToken).ConfigureAwait(false);
        var filtered = urls.Select(u => u.Trim()).Where(u => !string.IsNullOrWhiteSpace(u)).ToArray();
        return await _corpusBuilder.BuildAsync(filtered, cancellationToken).ConfigureAwait(false);
    }

    private static IModel CreateModel(TrainingConfig config, int vocabSize)
    {
        if (!string.Equals(config.ModelType, "mlp", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"Unknown model type '{config.ModelType}'.");
        }

        return new MlpCharLM(vocabSize, config.ContextLength, config.EmbeddingDim, config.HiddenDim, config.HiddenDim2, config.UseLayerNorm, config.Seed);
    }

    private static IOptimiser CreateOptimiser(TrainingConfig config)
    {
        return config.Optimiser.ToLowerInvariant() switch
        {
            "sgd" => new Sgd(config.LearningRate),
            _ => new Adam(config.LearningRate)
        };
    }

    private static Dictionary<string, float[]> ExtractParameterData(IReadOnlyDictionary<string, Tensor> namedParameters)
    {
        var result = new Dictionary<string, float[]>();
        foreach (var (name, tensor) in namedParameters)
        {
            result[name] = (float[])tensor.Data.Clone();
        }

        return result;
    }

    private static Dictionary<string, int[]> ExtractShapes(IReadOnlyDictionary<string, Tensor> namedParameters)
    {
        var result = new Dictionary<string, int[]>();
        foreach (var (name, tensor) in namedParameters)
        {
            result[name] = (int[])tensor.Shape.Clone();
        }

        return result;
    }
}
