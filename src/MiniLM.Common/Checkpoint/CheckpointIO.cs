using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace MiniLM.Common.Checkpoint;

public static class CheckpointIO
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true
    };

    public static async Task SaveAsync(string path, CheckpointModel model, CancellationToken cancellationToken)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        await using var stream = File.Create(path);
        await JsonSerializer.SerializeAsync(stream, model, Options, cancellationToken).ConfigureAwait(false);
    }

    public static async Task<CheckpointModel> LoadAsync(string path, CancellationToken cancellationToken)
    {
        await using var stream = File.OpenRead(path);
        var model = await JsonSerializer.DeserializeAsync<CheckpointModel>(stream, Options, cancellationToken).ConfigureAwait(false);
        if (model is null)
        {
            throw new InvalidDataException("Checkpoint file is empty or malformed.");
        }

        return model;
    }
}
