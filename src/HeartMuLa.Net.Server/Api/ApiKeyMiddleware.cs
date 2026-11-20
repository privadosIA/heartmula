using System.Security.Cryptography;
using System.Text;

namespace HeartMuLa.Server.Api;

/// <summary>
/// When <c>Security:ApiKey</c> is set, every request needs it in <c>X-Api-Key</c> or as <c>Authorization: Bearer</c>, except <c>/health</c> and the Swagger pages. The comparison takes
/// the same time whatever the number of matching characters.
/// </summary>
public sealed class ApiKeyMiddleware
{
    /// <summary>Name of the request header that carries the API key.</summary>
    public const string HeaderName = "X-Api-Key";
    private readonly RequestDelegate next;
    private readonly byte[]? key;

    /// <summary>Creates the middleware; the key is read once from <c>Security:ApiKey</c> (no key means the check is off).</summary>
    public ApiKeyMiddleware(RequestDelegate next, Options.ServiceOptions options)
    {
        this.next = next;
        key = string.IsNullOrEmpty(options.Security.ApiKey) ? null : Encoding.UTF8.GetBytes(options.Security.ApiKey);
    }

    /// <summary>Lets the request through when no key is configured, the path is open or the key matches; otherwise answers 401.</summary>
    public Task InvokeAsync(HttpContext context)
    {
        if (key is null || IsOpen(context.Request.Path)) return next(context);

        if (Matches(context.Request.Headers[HeaderName].ToString()) || Matches(Bearer(context.Request.Headers.Authorization.ToString()))) return next(context);

        context.Response.StatusCode = StatusCodes.Status401Unauthorized;
        if (IsOrchestratorRoute(context.Request.Path))
            return context.Response.WriteAsJsonAsync(new { error = "a bearer token is required" }, context.RequestAborted);
        return context.Response.WriteAsJsonAsync(new Microsoft.AspNetCore.Mvc.ProblemDetails
        {
            Status = StatusCodes.Status401Unauthorized,
            Title = "Unauthorized",
            Detail = $"Send the service's API key in the {HeaderName} header.",
        }, options: null, contentType: "application/problem+json", cancellationToken: context.RequestAborted);
    }

    private bool Matches(string given) => given.Length > 0 && CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(given), key!);

    private static string Bearer(string authorization) =>
        authorization.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) ? authorization[7..].Trim() : "";

    /// <summary>The routes of the orchestrator's contract answer errors as <c>{"error": ...}</c>.</summary>
    public static bool IsOrchestratorRoute(PathString path) =>
        path.StartsWithSegments("/generate", StringComparison.OrdinalIgnoreCase) || path.StartsWithSegments("/jobs", StringComparison.OrdinalIgnoreCase);

    private static bool IsOpen(PathString path) =>
        path.Equals("/health", StringComparison.OrdinalIgnoreCase) || path.StartsWithSegments("/swagger", StringComparison.OrdinalIgnoreCase);
}
