using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace MiniLM.Common.Text;

public sealed class Normaliser
{
    private static readonly Regex CollapseWhitespace = new("\\s+", RegexOptions.Compiled);

    public string Normalise(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return string.Empty;
        }

        var normalized = input.Normalize(NormalizationForm.FormKC);
        normalized = CollapseWhitespace.Replace(normalized, " ").Trim();

        var builder = new StringBuilder(normalized.Length);
        foreach (var rune in normalized.EnumerateRunes())
        {
            if (IsAllowedRune(rune))
            {
                builder.Append(rune);
            }
        }

        return builder.ToString();
    }

    private static bool IsAllowedRune(Rune rune)
    {
        if (Rune.IsControl(rune))
        {
            return false;
        }

        if (rune.Value >= 0x20 && rune.Value <= 0x7E)
        {
            return true;
        }

        if (Rune.IsLetterOrDigit(rune) || Rune.IsWhiteSpace(rune) || Rune.IsPunctuation(rune))
        {
            return true;
        }

        return false;
    }
}
