using System.Collections.Generic;
using MiniLM.Common.Math;

namespace MiniLM.Common.Math.Optimisers;

public interface IOptimiser
{
    void Step(IEnumerable<Tensor> parameters);
    void Reset();
}
