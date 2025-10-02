using System;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using AngleSharp.Dom;
using AngleSharp.Html.Parser;

namespace MiniLM.Common.Text;

public sealed class HtmlCleaner
{
    private static readonly Regex CollapseWhitespace = new("\\s+", RegexOptions.Compiled);
    private readonly HtmlParser _parser = new();

    public async Task<string> CleanAsync(string html)
    {
        if (string.IsNullOrWhiteSpace(html))
        {
            return string.Empty;
        }

        var document = await _parser.ParseDocumentAsync(html).ConfigureAwait(false);
        RemoveNodes(document, "script", "style", "noscript", "template");
        var text = document.Body?.TextContent ?? string.Empty;
        text = CollapseWhitespace.Replace(text, " ").Trim();
        return text;
    }

    private static void RemoveNodes(IDocument document, params string[] selectors)
    {
        foreach (var selector in selectors)
        {
            foreach (var element in document.QuerySelectorAll(selector))
            {
                element.Remove();
            }
        }
    }
}
