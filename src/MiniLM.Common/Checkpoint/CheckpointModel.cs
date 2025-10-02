using System.Collections.Generic;
using MiniLM.Common.Tokens;

namespace MiniLM.Common.Checkpoint;

public sealed class CheckpointModel
{
    public string Version { get; set; } = "1.0";
    public string ModelType { get; set; } = "mlp";
    public int ContextLength { get; set; }
    public int VocabSize { get; set; }
    public Dictionary<string, float[]> Parameters { get; set; } = new();
    public Dictionary<string, int[]> Shapes { get; set; } = new();
    public CharVocab.SerializableVocab Vocab { get; set; } = new();
    public TrainingMetadata Training { get; set; } = new();

    public sealed class TrainingMetadata
    {
        public int Epochs { get; set; }
        public float LearningRate { get; set; }
        public string Optimiser { get; set; } = string.Empty;
    }
}
