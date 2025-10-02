using System;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace MiniLM.Common.Text;

public sealed class HtmlFetcher
{
    private readonly HttpClient _httpClient;

    public HtmlFetcher(HttpClient? httpClient = null)
    {
        var handler = new SocketsHttpHandler
        {
            AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate | DecompressionMethods.Brotli,
            AllowAutoRedirect = true,
            PooledConnectionLifetime = TimeSpan.FromMinutes(5),
            ConnectTimeout = TimeSpan.FromSeconds(10)
        };

        _httpClient = httpClient ?? new HttpClient(handler);

        _httpClient.Timeout = TimeSpan.FromSeconds(10);

        _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd(
            "MiniLM-TrainingBot/0.1 (+https://github.com/robwalker-codes/MiniLM; contact: me@example.com)");

        _httpClient.DefaultRequestHeaders.Accept.ParseAdd("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
        _httpClient.DefaultRequestHeaders.AcceptLanguage.ParseAdd("en-GB,en;q=0.9");
    }

    public async Task<string?> FetchAsync(string url, CancellationToken cancellationToken)
    {
        try
        {
            using var response = await _httpClient.GetAsync(url, cancellationToken).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
            {
                return null;
            }

            return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (HttpRequestException)
        {
            return null;
        }
        catch (TaskCanceledException)
        {
            return null;
        }
    }
}
