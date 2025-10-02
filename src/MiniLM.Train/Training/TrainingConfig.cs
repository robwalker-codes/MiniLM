namespace MiniLM.Train.Training;

public sealed class TrainingConfig
{
    public string UrlListPath { get; set; } = string.Empty;
    public string OutputPath { get; set; } = string.Empty;
    public int Epochs { get; set; } = 3;
    public int BatchSize { get; set; } = 64;
    public int ContextLength { get; set; } = 128;
    public int EmbeddingDim { get; set; } = 64;
    public int HiddenDim { get; set; } = 128;
    public int HiddenDim2 { get; set; } = 128;
    public float LearningRate { get; set; } = 3e-4f;
    public int Seed { get; set; } = 42;
    public string Optimiser { get; set; } = "adam";
    public bool UseLayerNorm { get; set; } = true;
    public bool Shuffle { get; set; } = true;
    public bool Verbose { get; set; } = false;
    public string ModelType { get; set; } = "mlp";
    public string? CorpusOverride { get; set; }
}
