// A worker process (see Worker/Worker.cs) is this same executable started with --worker: it makes one song and exits.
if (args.Length > 0 && args[0] == HeartMuLa.Server.Worker.WorkerProtocol.Argument)
    return HeartMuLa.Server.Worker.Worker.Execute(args);

try
{
    await HeartMuLa.Server.ServiceHost.Build(args).RunAsync();
}
catch (HeartMuLa.Server.Options.ConfigurationException e)
{
    await Console.Error.WriteLineAsync(e.Message);
    return 1;
}
return 0;

// Lets the test project reach the entry point through WebApplicationFactory<Program>.
/// <summary>Entry-point type, public so that the test project can start the service through <c>WebApplicationFactory&lt;Program&gt;</c>.</summary>
public partial class Program;
