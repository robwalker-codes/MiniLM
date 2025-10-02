using System;
using System.Threading;
using System.Threading.Tasks;
using MiniLM.Common.Checkpoint;
using MiniLM.Common.Tokens;
using MiniLM.Common.Util;
using MiniLM.Infer.Inference;
using MiniLM.Train.Model;

namespace MiniLM.Infer;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        try
        {
            var config = ParseArgs(args);
            Logging.SetVerbose(config.Verbose);
            var checkpoint = await CheckpointIO.LoadAsync(config.CheckpointPath, CancellationToken.None).ConfigureAwait(false);
            var vocab = CharVocab.FromSerializable(checkpoint.Vocab);
            if (vocab.Size != checkpoint.VocabSize)
            {
                throw new InvalidOperationException("Vocabulary size mismatch in checkpoint.");
            }

            if (!string.Equals(checkpoint.ModelType, "mlp", StringComparison.OrdinalIgnoreCase))
            {
                throw new NotSupportedException($"Unsupported model type '{checkpoint.ModelType}'.");
            }

            var model = CreateModelFromCheckpoint(checkpoint, config.Seed);
            model.LoadNamedParameters(checkpoint.Parameters, checkpoint.Shapes);

            var tokenizer = new CharTokenizer(vocab);
            var sampler = new Sampler(config.Seed, config.Deterministic);
            Console.WriteLine("Press Ctrl+C or submit an empty prompt to exit.");
            RunInteractiveLoop(tokenizer, model, sampler, config);
            return 0;
        }
        catch (Exception ex)
        {
            Logging.Error(ex.Message);
            return 1;
        }
    }

    internal static InferenceConfig ParseArgs(string[] args)
    {
        var options = new[]
        {
            new ArgOption("checkpoint", true, Required: true),
            new ArgOption("max-tokens", true, Validator: v => int.TryParse(v, out var value) && value > 0, DefaultValue: "200"),
            new ArgOption("temperature", true, Validator: v => float.TryParse(v, out var value) && value >= 0, DefaultValue: "0.8"),
            new ArgOption("top-k", true, Validator: v => int.TryParse(v, out var value) && value >= 0, DefaultValue: "0"),
            new ArgOption("seed", true, Validator: v => int.TryParse(v, out _), DefaultValue: "42"),
            new ArgOption("no-sampler-determinism", HasValue: false),
            new ArgOption("verbose", HasValue: false)
        };

        var parsed = ArgParsing.Parse(args, options);

        return new InferenceConfig
        {
            CheckpointPath = parsed["checkpoint"] ?? string.Empty,
            MaxTokens = ArgParsing.GetInt(parsed, "max-tokens", 200),
            Temperature = ArgParsing.GetFloat(parsed, "temperature", 0.8f),
            TopK = ArgParsing.GetInt(parsed, "top-k", 0),
            Seed = ArgParsing.GetInt(parsed, "seed", 42),
            Deterministic = !ArgParsing.GetBool(parsed, "no-sampler-determinism", false),
            Verbose = ArgParsing.GetBool(parsed, "verbose", false)
        };
    }

    private static void RunInteractiveLoop(CharTokenizer tokenizer, IModel model, Sampler sampler, InferenceConfig config)
    {
        while (true)
        {
            Console.Write("What's on the agenda today? ");
            var prompt = Console.ReadLine();

            if (prompt is null)
            {
                Console.WriteLine();
                return;
            }

            if (string.IsNullOrWhiteSpace(prompt))
            {
                Console.WriteLine("Goodbye!");
                return;
            }

            var result = sampler.Generate(tokenizer, model, prompt, config.MaxTokens, config.Temperature, config.TopK);
            Console.WriteLine(result);
            Console.WriteLine();
        }
    }

    private static IModel CreateModelFromCheckpoint(CheckpointModel checkpoint, int seed)
    {
        if (!checkpoint.Shapes.TryGetValue("embedding", out var embeddingShape) || embeddingShape.Length != 2)
        {
            throw new InvalidOperationException("Checkpoint missing embedding shape.");
        }

        if (!checkpoint.Shapes.TryGetValue("dense1.weight", out var dense1Shape) || dense1Shape.Length != 2)
        {
            throw new InvalidOperationException("Checkpoint missing dense1 weight shape.");
        }

        if (!checkpoint.Shapes.TryGetValue("dense2.weight", out var dense2Shape) || dense2Shape.Length != 2)
        {
            throw new InvalidOperationException("Checkpoint missing dense2 weight shape.");
        }

        var embeddingDim = embeddingShape[1];
        var hiddenDim = dense1Shape[1];
        var hiddenDim2 = dense2Shape[1];
        var useLayerNorm = checkpoint.Shapes.ContainsKey("layernorm.gamma");

        return new MlpCharLM(checkpoint.VocabSize, checkpoint.ContextLength, embeddingDim, hiddenDim, hiddenDim2, useLayerNorm, seed);
    }
}
