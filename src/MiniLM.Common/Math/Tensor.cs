using System;
using System.Collections.Generic;

namespace MiniLM.Common.Math;

public sealed class Tensor
{
    public Tensor(params int[] shape)
    {
        Shape = shape;
        Data = new float[TotalLength(shape)];
        Grad = new float[Data.Length];
    }

    public Tensor(float[] data, params int[] shape)
    {
        if (TotalLength(shape) != data.Length)
        {
            throw new ArgumentException("Shape does not match data length.");
        }

        Shape = shape;
        Data = data;
        Grad = new float[data.Length];
    }

    public int[] Shape { get; }
    public float[] Data { get; }
    public float[] Grad { get; }

    public int Rank => Shape.Length;
    public int Length => Data.Length;

    public void ZeroGrad() => Array.Clear(Grad);

    public Tensor Clone()
    {
        return new Tensor((float[])Data.Clone(), (int[])Shape.Clone());
    }

    public static int TotalLength(IReadOnlyList<int> shape)
    {
        var total = 1;
        foreach (var dim in shape)
        {
            total *= dim;
        }

        return total;
    }
}
