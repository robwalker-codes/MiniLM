using System;
using System.Threading;
using System.Threading.Tasks;
using MiniLM.Common.Data;
using MiniLM.Common.Text;
using MiniLM.Common.Util;
using MiniLM.Train.Training;

namespace MiniLM.Train;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        try
        {
            var config = ParseArgs(args);
            var fetcher = new HtmlFetcher();
            var cleaner = new HtmlCleaner();
            var normaliser = new Normaliser();
            var builder = new CorpusBuilder(fetcher, cleaner, normaliser);
            var loop = new TrainingLoop(builder);
            await loop.TrainAsync(config, CancellationToken.None).ConfigureAwait(false);
            return 0;
        }
        catch (Exception ex)
        {
            Logging.Error(ex.Message);
            return 1;
        }
    }

    internal static TrainingConfig ParseArgs(string[] args)
    {
        var options = new[]
        {
            new ArgOption("urls", true, Required: true),
            new ArgOption("output", true, Required: true),
            new ArgOption("epochs", true, Validator: v => int.TryParse(v, out var value) && value > 0, DefaultValue: "3"),
            new ArgOption("batch-size", true, Validator: v => int.TryParse(v, out var value) && value > 0, DefaultValue: "64"),
            new ArgOption("context-length", true, Validator: v => int.TryParse(v, out var value) && value > 0, DefaultValue: "128"),
            new ArgOption("lr", true, Validator: v => float.TryParse(v, out var value) && value > 0, DefaultValue: "0.0003"),
            new ArgOption("seed", true, Validator: v => int.TryParse(v, out _), DefaultValue: "42"),
            new ArgOption("model", true, DefaultValue: "mlp"),
            new ArgOption("embedding-dim", true, Validator: v => int.TryParse(v, out var value) && value > 0, DefaultValue: "64"),
            new ArgOption("hidden-dim", true, Validator: v => int.TryParse(v, out var value) && value > 0, DefaultValue: "128"),
            new ArgOption("hidden-dim2", true, Validator: v => int.TryParse(v, out var value) && value > 0, DefaultValue: "128"),
            new ArgOption("optimiser", true, DefaultValue: "adam"),
            new ArgOption("disable-layer-norm", HasValue: false),
            new ArgOption("no-shuffle", HasValue: false),
            new ArgOption("verbose", HasValue: false),
            new ArgOption("corpus", true)
        };

        var parsed = ArgParsing.Parse(args, options);

        return new TrainingConfig
        {
            UrlListPath = parsed["urls"] ?? string.Empty,
            OutputPath = parsed["output"] ?? string.Empty,
            Epochs = ArgParsing.GetInt(parsed, "epochs", 3),
            BatchSize = ArgParsing.GetInt(parsed, "batch-size", 64),
            ContextLength = ArgParsing.GetInt(parsed, "context-length", 128),
            LearningRate = ArgParsing.GetFloat(parsed, "lr", 3e-4f),
            Seed = ArgParsing.GetInt(parsed, "seed", 42),
            ModelType = parsed.TryGetValue("model", out var model) ? model ?? "mlp" : "mlp",
            EmbeddingDim = ArgParsing.GetInt(parsed, "embedding-dim", 64),
            HiddenDim = ArgParsing.GetInt(parsed, "hidden-dim", 128),
            HiddenDim2 = ArgParsing.GetInt(parsed, "hidden-dim2", 128),
            Optimiser = parsed.TryGetValue("optimiser", out var opt) ? opt ?? "adam" : "adam",
            UseLayerNorm = !ArgParsing.GetBool(parsed, "disable-layer-norm", false),
            Shuffle = !ArgParsing.GetBool(parsed, "no-shuffle", false),
            Verbose = ArgParsing.GetBool(parsed, "verbose", false),
            CorpusOverride = parsed.TryGetValue("corpus", out var corpus) ? corpus : null
        };
    }
}
