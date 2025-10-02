using System;

namespace MiniLM.Common.Util;

public enum LogLevel
{
    Info,
    Warning,
    Error,
    Debug
}

public static class Logging
{
    private static readonly object Sync = new();
    private static bool _verbose;

    public static void SetVerbose(bool verbose) => _verbose = verbose;

    public static void Info(string message) => Write(LogLevel.Info, message);

    public static void Warn(string message) => Write(LogLevel.Warning, message);

    public static void Error(string message) => Write(LogLevel.Error, message);

    public static void Debug(string message)
    {
        if (_verbose)
        {
            Write(LogLevel.Debug, message);
        }
    }

    private static void Write(LogLevel level, string message)
    {
        lock (Sync)
        {
            var prefix = level switch
            {
                LogLevel.Info => "[info]",
                LogLevel.Warning => "[warn]",
                LogLevel.Error => "[error]",
                LogLevel.Debug => "[debug]",
                _ => "[log]"
            };

            var line = $"{DateTime.UtcNow:O} {prefix} {message}";
            if (level == LogLevel.Error)
            {
                Console.Error.WriteLine(line);
            }
            else
            {
                Console.WriteLine(line);
            }
        }
    }
}
