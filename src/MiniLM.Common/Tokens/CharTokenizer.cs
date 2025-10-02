using System.Collections.Generic;

namespace MiniLM.Common.Tokens;

public sealed class CharTokenizer
{
    private readonly CharVocab _vocab;

    public CharTokenizer(CharVocab vocab)
    {
        _vocab = vocab;
    }

    public CharVocab Vocab => _vocab;

    public int[] Encode(string text, bool addBosEos = true)
    {
        var tokens = new List<int>(text.Length + 2);
        if (addBosEos)
        {
            tokens.Add(_vocab.BosId);
        }

        foreach (var ch in text)
        {
            tokens.Add(_vocab[ch]);
        }

        if (addBosEos)
        {
            tokens.Add(_vocab.EosId);
        }

        return tokens.ToArray();
    }

    public string Decode(IEnumerable<int> tokenIds)
    {
        var chars = new List<char>();
        foreach (var tokenId in tokenIds)
        {
            if (tokenId is < 0)
            {
                continue;
            }

            if (tokenId == _vocab.BosId || tokenId == _vocab.EosId || tokenId == _vocab.PadId)
            {
                continue;
            }

            var ch = _vocab.ToChar(tokenId);
            if (ch.HasValue)
            {
                chars.Add(ch.Value);
            }
        }

        return new string(chars.ToArray());
    }
}
