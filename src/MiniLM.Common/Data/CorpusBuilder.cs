using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MiniLM.Common.Text;
using MiniLM.Common.Util;

namespace MiniLM.Common.Data;

public sealed class CorpusBuilder
{
    private readonly HtmlFetcher _fetcher;
    private readonly HtmlCleaner _cleaner;
    private readonly Normaliser _normaliser;
    private readonly string _separator;

    public CorpusBuilder(HtmlFetcher fetcher, HtmlCleaner cleaner, Normaliser normaliser, string? separator = null)
    {
        _fetcher = fetcher;
        _cleaner = cleaner;
        _normaliser = normaliser;
        _separator = separator ?? "\n\n——PAGE——\n\n";
    }

    public async Task<string> BuildAsync(IEnumerable<string> urls, CancellationToken cancellationToken)
    {
        var builder = new StringBuilder();
        var first = true;

        foreach (var url in urls)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var html = await _fetcher.FetchAsync(url, cancellationToken).ConfigureAwait(false);
            if (html is null)
            {
                Logging.Warn($"Skipping URL '{url}' because it could not be fetched.");
                continue;
            }

            var cleaned = await _cleaner.CleanAsync(html).ConfigureAwait(false);
            var normalised = _normaliser.Normalise(cleaned);
            if (string.IsNullOrWhiteSpace(normalised))
            {
                Logging.Warn($"Skipping URL '{url}' because it yielded no textual content.");
                continue;
            }

            if (!first)
            {
                builder.Append(_separator);
            }

            builder.Append(normalised);
            first = false;
        }

        return builder.ToString();
    }
}
