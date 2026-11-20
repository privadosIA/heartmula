using Microsoft.OpenApi.Any;
using Microsoft.OpenApi.Models;
using Swashbuckle.AspNetCore.SwaggerGen;

namespace HeartMuLa.Server.Api;

/// <summary>Enums travel as lower-case words (see <c>LowerCaseEnumConverter</c>), so the document lists them that way instead of as integers.</summary>
public sealed class LowerCaseEnumSchemaFilter : ISchemaFilter
{
    /// <summary>Replaces the schema of an enum with a string schema listing its names in lower case; other types are left alone.</summary>
    public void Apply(OpenApiSchema schema, SchemaFilterContext context)
    {
        if (!context.Type.IsEnum) return;
        schema.Type = "string";
        schema.Format = null;
        schema.Enum = Enum.GetNames(context.Type).Select(n => (IOpenApiAny)new OpenApiString(n.ToLowerInvariant())).ToList();
    }
}
