using System;
using System.Collections;
using System.Collections.Generic;
using MiniLM.Common.Tokens;

namespace MiniLM.Common.Data;

public sealed class DataLoader : IEnumerable<Batch>
{
    private readonly int[] _tokens;
    private readonly CharTokenizer _tokenizer;
    private readonly int _contextLength;
    private readonly int _batchSize;
    private readonly int[] _indices;

    public DataLoader(string corpus, CharTokenizer tokenizer, int contextLength, int batchSize, bool shuffle, int seed)
    {
        _tokenizer = tokenizer;
        _contextLength = contextLength;
        _batchSize = batchSize;

        var encoded = tokenizer.Encode(corpus, addBosEos: false);
        _tokens = new int[encoded.Length + 2];
        _tokens[0] = tokenizer.Vocab.BosId;
        Array.Copy(encoded, 0, _tokens, 1, encoded.Length);
        _tokens[^1] = tokenizer.Vocab.EosId;

        var sampleCount = System.Math.Max(0, _tokens.Length - contextLength);
        if (sampleCount <= 0)
        {
            _indices = Array.Empty<int>();
            return;
        }

        _indices = new int[sampleCount];
        for (var i = 0; i < sampleCount; i++)
        {
            _indices[i] = i;
        }

        if (shuffle)
        {
            Shuffle(_indices, seed);
        }
    }

    public IEnumerator<Batch> GetEnumerator()
    {
        if (_indices.Length == 0)
        {
            yield break;
        }

        var padId = _tokenizer.Vocab.PadId;
        for (var offset = 0; offset < _indices.Length; offset += _batchSize)
        {
            var validCount = System.Math.Min(_batchSize, _indices.Length - offset);
            var inputs = new int[_batchSize, _contextLength];
            var targets = new int[_batchSize];

            for (var row = 0; row < _batchSize; row++)
            {
                targets[row] = padId;
                for (var col = 0; col < _contextLength; col++)
                {
                    inputs[row, col] = padId;
                }
            }

            for (var batchRow = 0; batchRow < validCount; batchRow++)
            {
                var sequenceIndex = _indices[offset + batchRow];
                for (var i = 0; i < _contextLength; i++)
                {
                    inputs[batchRow, i] = _tokens[sequenceIndex + i];
                }

                targets[batchRow] = _tokens[sequenceIndex + _contextLength];
            }

            yield return new Batch(inputs, targets, validCount);
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    private static void Shuffle(int[] indices, int seed)
    {
        var rng = new Random(seed);
        for (var i = indices.Length - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
    }
}
