namespace MiniLM.Infer.Inference;

public sealed class InferenceConfig
{
    public string CheckpointPath { get; set; } = string.Empty;
    public int MaxTokens { get; set; } = 200;
    public float Temperature { get; set; } = 0.8f;
    public int TopK { get; set; } = 0;
    public int Seed { get; set; } = 42;
    public bool Deterministic { get; set; } = true;
    public bool Verbose { get; set; } = false;
}
