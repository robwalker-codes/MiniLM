using System.Collections.Generic;
using MiniLM.Common.Data;
using MiniLM.Common.Math;

namespace MiniLM.Train.Model;

public interface IModel
{
    int ContextLength { get; }
    int VocabSize { get; }
    IReadOnlyList<Tensor> Parameters { get; }

    float[,] Forward(Batch batch);
    void Backward(float[,] gradLogits, Batch batch);
    void ZeroGradients();
    IReadOnlyDictionary<string, Tensor> GetNamedParameters();
    void LoadNamedParameters(IReadOnlyDictionary<string, float[]> data, IReadOnlyDictionary<string, int[]> shapes);
}
