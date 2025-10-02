using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using MiniLM.Common.Data;
using MiniLM.Common.Text;
using Xunit;

namespace MiniLM.Tests.Data;

public sealed class CorpusBuilderTests
{
    [Fact]
    public async Task CombinesFetchedDocuments()
    {
        var pages = new Dictionary<string, string>
        {
            ["https://example.com/a"] = "<html><body><h1>Title</h1><script>ignore</script><p>Content</p></body></html>",
            ["https://example.com/b"] = "<html><body><p>Second page</p></body></html>"
        };

        var handler = new FakeHandler(pages);
        var fetcher = new HtmlFetcher(new HttpClient(handler));
        var cleaner = new HtmlCleaner();
        var normaliser = new Normaliser();
        var builder = new CorpusBuilder(fetcher, cleaner, normaliser, separator: "|SEP|");

        var corpus = await builder.BuildAsync(pages.Keys, CancellationToken.None);

        Assert.Contains("Title", corpus);
        Assert.Contains("Content", corpus);
        Assert.Contains("Second page", corpus);
        Assert.Contains("|SEP|", corpus);
    }

    private sealed class FakeHandler : HttpMessageHandler
    {
        private readonly Dictionary<string, string> _responses;

        public FakeHandler(Dictionary<string, string> responses)
        {
            _responses = responses;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (_responses.TryGetValue(request.RequestUri!.ToString(), out var body))
            {
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(body)
                });
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
