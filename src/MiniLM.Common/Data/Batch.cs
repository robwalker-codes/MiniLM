namespace MiniLM.Common.Data;

public sealed class Batch
{
    public Batch(int[,] inputs, int[] targets, int validCount)
    {
        Inputs = inputs;
        Targets = targets;
        ValidCount = validCount;
    }

    public int[,] Inputs { get; }
    public int[] Targets { get; }
    public int ValidCount { get; }
    public int BatchSize => Inputs.GetLength(0);
    public int ContextLength => Inputs.GetLength(1);
}
