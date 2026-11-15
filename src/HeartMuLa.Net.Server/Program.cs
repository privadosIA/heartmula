// A worker process (see Worker/Worker.cs) is this same executable started with --worker: it makes one song and exits.
if (args.Length > 0 && args[0] == HeartMuLa.Server.Worker.WorkerProtocol.Argument)
    return HeartMuLa.Server.Worker.Worker.Main(args);

try
{
    HeartMuLa.Server.ServiceHost.Build(args).Run();
}
catch (HeartMuLa.Server.Options.ConfigurationException e)
{
    Console.Error.WriteLine(e.Message);
    return 1;
}
return 0;

// Lets the test project reach the entry point through WebApplicationFactory<Program>.
public partial class Program;
