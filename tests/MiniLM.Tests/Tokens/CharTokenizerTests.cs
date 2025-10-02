using MiniLM.Common.Tokens;
using Xunit;

namespace MiniLM.Tests.Tokens;

public sealed class CharTokenizerTests
{
    [Fact]
    public void EncodeDecodeRoundTrip()
    {
        var vocab = CharVocab.FromCorpus("hello world");
        var tokenizer = new CharTokenizer(vocab);
        var text = "hello";

        var tokens = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(tokens);

        Assert.Contains(vocab.BosId, tokens);
        Assert.Contains(vocab.EosId, tokens);
        Assert.Equal(text, decoded);
    }

    [Fact]
    public void UnknownCharactersMapToUnk()
    {
        var vocab = CharVocab.FromCorpus("abc");
        var tokenizer = new CharTokenizer(vocab);

        var tokens = tokenizer.Encode("z", addBosEos: false);
        Assert.All(tokens, t => Assert.Equal(vocab.UnkId, t));
    }

    [Fact]
    public void SerializableRoundTripPreservesTokens()
    {
        var vocab = CharVocab.FromCorpus("testing 123");
        var serialised = vocab.ToSerializable();
        var restored = CharVocab.FromSerializable(serialised);

        Assert.Equal(vocab.Size, restored.Size);
        Assert.Equal(vocab.Tokens, restored.Tokens);
    }
}
