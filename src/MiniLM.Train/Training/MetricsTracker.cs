namespace MiniLM.Train.Training;

public sealed class MetricsTracker
{
    private readonly float _momentum;
    private float? _loss;
    private float? _accuracy;

    public MetricsTracker(float momentum = 0.9f)
    {
        _momentum = momentum;
    }

    public void Update(float loss, float accuracy)
    {
        _loss = _loss is null ? loss : _momentum * _loss.Value + (1 - _momentum) * loss;
        _accuracy = _accuracy is null ? accuracy : _momentum * _accuracy.Value + (1 - _momentum) * accuracy;
    }

    public float Loss => _loss ?? 0f;
    public float Accuracy => _accuracy ?? 0f;
}
