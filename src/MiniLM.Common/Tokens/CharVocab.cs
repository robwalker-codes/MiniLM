using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace MiniLM.Common.Tokens;

public sealed class CharVocab
{
    public const string PadToken = "<pad>";
    public const string UnkToken = "<unk>";
    public const string BosToken = "<bos>";
    public const string EosToken = "<eos>";

    private readonly Dictionary<char, int> _charToId;
    private readonly Dictionary<int, char> _idToChar;
    private readonly List<string> _tokens;
    private readonly IReadOnlyList<string> _readOnlyTokens;

    private CharVocab(List<string> tokens, Dictionary<char, int> charToId, Dictionary<int, char> idToChar)
    {
        _tokens = tokens;
        _readOnlyTokens = new ReadOnlyCollection<string>(_tokens);
        _charToId = charToId;
        _idToChar = idToChar;
    }

    public int Size => _tokens.Count;
    public IReadOnlyList<string> Tokens => _readOnlyTokens;

    public int PadId => 0;
    public int UnkId => 1;
    public int BosId => 2;
    public int EosId => 3;

    public int this[char c] => _charToId.TryGetValue(c, out var id) ? id : UnkId;

    public char? ToChar(int id)
    {
        return _idToChar.TryGetValue(id, out var ch) ? ch : null;
    }

    public static CharVocab FromCorpus(string corpus)
    {
        var uniqueChars = new SortedSet<char>(corpus);
        var tokens = new List<string> { PadToken, UnkToken, BosToken, EosToken };
        var charToId = new Dictionary<char, int>();
        var idToChar = new Dictionary<int, char>();

        foreach (var ch in uniqueChars)
        {
            if (char.IsControl(ch))
            {
                continue;
            }

            var id = tokens.Count;
            tokens.Add(ch.ToString());
            charToId[ch] = id;
            idToChar[id] = ch;
        }

        return new CharVocab(tokens, charToId, idToChar);
    }

    public SerializableVocab ToSerializable()
    {
        return new SerializableVocab
        {
            Tokens = _tokens.ToArray()
        };
    }

    public static CharVocab FromSerializable(SerializableVocab serialisable)
    {
        var tokens = serialisable.Tokens.ToList();
        var charToId = new Dictionary<char, int>();
        var idToChar = new Dictionary<int, char>();

        for (var i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            if (token.Length == 1)
            {
                charToId[token[0]] = i;
                idToChar[i] = token[0];
            }
        }

        return new CharVocab(tokens, charToId, idToChar);
    }

    public sealed class SerializableVocab
    {
        public string[] Tokens { get; set; } = System.Array.Empty<string>();
    }
}
