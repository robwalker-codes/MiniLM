using System;

namespace MiniLM.Common.Math;

public static class Initialisers
{
    public static void XavierUniform(Tensor tensor, int fanIn, int fanOut, Random rng)
    {
        var limit = MathF.Sqrt(6f / (fanIn + fanOut));
        FillUniform(tensor, -limit, limit, rng);
    }

    public static void FillUniform(Tensor tensor, float min, float max, Random rng)
    {
        for (var i = 0; i < tensor.Data.Length; i++)
        {
            tensor.Data[i] = (float)(min + rng.NextDouble() * (max - min));
        }
    }

    public static void FillNormal(Tensor tensor, float mean, float stdDev, Random rng)
    {
        for (var i = 0; i < tensor.Data.Length; i += 2)
        {
            var u1 = 1.0 - rng.NextDouble();
            var u2 = 1.0 - rng.NextDouble();
            var randStdNormal = System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Sin(2.0 * System.Math.PI * u2);
            var randStdNormal2 = System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Cos(2.0 * System.Math.PI * u2);
            tensor.Data[i] = (float)(mean + stdDev * randStdNormal);
            if (i + 1 < tensor.Data.Length)
            {
                tensor.Data[i + 1] = (float)(mean + stdDev * randStdNormal2);
            }
        }
    }

    public static void FillConstant(Tensor tensor, float value)
    {
        Array.Fill(tensor.Data, value);
    }
}
