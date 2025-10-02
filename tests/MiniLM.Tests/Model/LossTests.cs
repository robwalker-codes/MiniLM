using MiniLM.Train.Model.Loss;
using Xunit;

namespace MiniLM.Tests.Model;

public sealed class LossTests
{
    [Fact]
    public void CrossEntropyMatchesExpectedValue()
    {
        var loss = new CrossEntropyLoss();
        var logits = new float[1, 2]
        {
            { 0f, 1f }
        };

        var targets = new[] { 1 };
        var (value, accuracy, gradient) = loss.Compute(logits, targets, validCount: 1);

        Assert.InRange(value, 0.31f, 0.32f);
        Assert.Equal(1f, accuracy);
        Assert.InRange(gradient[0, 0] + gradient[0, 1], -1e-5f, 1e-5f);
    }
}
