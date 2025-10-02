using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace MiniLM.Common.Util;

public sealed record ArgOption(
    string Name,
    bool HasValue,
    bool Required = false,
    string? Description = null,
    Func<string, bool>? Validator = null,
    string? DefaultValue = null);

public static class ArgParsing
{
    public static IReadOnlyDictionary<string, string?> Parse(string[] args, IEnumerable<ArgOption> options)
    {
        var optionMap = options.ToDictionary(o => o.Name, StringComparer.OrdinalIgnoreCase);
        var values = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);

        var index = 0;
        while (index < args.Length)
        {
            var token = args[index];
            var (name, inlineValue) = SplitToken(token);
            var option = ResolveOption(optionMap, name);
            var (value, nextIndex) = ResolveValue(args, index, inlineValue, option);
            ValidateValue(option, value, name);
            values[name] = value;
            index = nextIndex;
        }

        ApplyDefaults(optionMap.Values, values);

        return values;
    }

    private static (string name, string? inlineValue) SplitToken(string token)
    {
        if (!token.StartsWith("--", StringComparison.Ordinal))
        {
            throw new ArgumentException($"Unexpected argument '{token}'. Options must start with --.");
        }

        var trimmed = token[2..];
        var eqIndex = trimmed.IndexOf('=');
        return eqIndex >= 0
            ? (trimmed[..eqIndex], trimmed[(eqIndex + 1)..])
            : (trimmed, null);
    }

    private static ArgOption ResolveOption(IReadOnlyDictionary<string, ArgOption> optionMap, string name)
    {
        if (!optionMap.TryGetValue(name, out var option))
        {
            throw new ArgumentException($"Unknown option '--{name}'.");
        }

        return option;
    }

    private static (string? value, int nextIndex) ResolveValue(
        string[] args,
        int currentIndex,
        string? inlineValue,
        ArgOption option)
    {
        if (!option.HasValue)
        {
            return ("true", currentIndex + 1);
        }

        if (inlineValue is not null)
        {
            return (inlineValue, currentIndex + 1);
        }

        if (currentIndex + 1 >= args.Length)
        {
            throw new ArgumentException($"Option '--{option.Name}' expects a value.");
        }

        return (args[currentIndex + 1], currentIndex + 2);
    }

    private static void ValidateValue(ArgOption option, string? value, string name)
    {
        if (option.Validator is null || value is null)
        {
            return;
        }

        if (!option.Validator(value))
        {
            throw new ArgumentException($"Value '{value}' for option '--{name}' is invalid.");
        }
    }

    private static void ApplyDefaults(IEnumerable<ArgOption> options, IDictionary<string, string?> values)
    {
        foreach (var option in options)
        {
            if (values.ContainsKey(option.Name))
            {
                continue;
            }

            if (option.Required)
            {
                throw new ArgumentException($"Missing required option '--{option.Name}'.");
            }

            if (option.DefaultValue is not null)
            {
                values[option.Name] = option.DefaultValue;
            }
            else if (!option.HasValue)
            {
                values[option.Name] = "false";
            }
        }
    }

    public static int GetInt(IReadOnlyDictionary<string, string?> parsed, string name, int defaultValue)
    {
        if (!parsed.TryGetValue(name, out var value) || string.IsNullOrWhiteSpace(value))
        {
            return defaultValue;
        }

        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var result))
        {
            throw new ArgumentException($"Value '{value}' for option '--{name}' is not a valid integer.");
        }

        return result;
    }

    public static float GetFloat(IReadOnlyDictionary<string, string?> parsed, string name, float defaultValue)
    {
        if (!parsed.TryGetValue(name, out var value) || string.IsNullOrWhiteSpace(value))
        {
            return defaultValue;
        }

        if (!float.TryParse(value, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var result))
        {
            throw new ArgumentException($"Value '{value}' for option '--{name}' is not a valid number.");
        }

        return result;
    }

    public static bool GetBool(IReadOnlyDictionary<string, string?> parsed, string name, bool defaultValue)
    {
        if (!parsed.TryGetValue(name, out var value) || string.IsNullOrWhiteSpace(value))
        {
            return defaultValue;
        }

        if (!bool.TryParse(value, out var result))
        {
            throw new ArgumentException($"Value '{value}' for option '--{name}' is not a valid boolean.");
        }

        return result;
    }
}
