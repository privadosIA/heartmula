# Tasks

## 1. Structure and solution

- [x] 1.1 Initialize git (`git init`) and move the code to `src/HeartMuLa.Net/` (Weights, Model, Pipeline, Program.cs, csproj); verify with `ls src/HeartMuLa.Net` and that the root no longer contains `.cs` files
- [x] 1.2 Create `Directory.Build.props` with `TorchBackend` (default `cpu`) and the backend's conditional `PackageReference`s; verify that `dotnet build -p:TorchBackend=cpu` resolves the `TorchSharp-cpu` package
- [x] 1.3 Clean up the app's csproj: remove `PlatformTarget=x64` and `System.CommandLine` if unused, check the TorchSharp version on NuGet; verify with `dotnet restore`
- [x] 1.4 Create `HeartMuLa.Net.sln` with the app and the tests project; verify with `dotnet sln list`

## 2. Compile the skeleton

- [x] 2.1 Fix compilation errors against the TorchSharp API without changing the logic; verify with `dotnet build -c Release -p:TorchBackend=cpu` with no new errors or warnings, and list in this file every non-trivial fix (result: none, the existing code compiled unchanged)

## 3. Tests

- [x] 3.1 Create `tests/HeartMuLa.Net.Tests` (xUnit, reference to the app) and the `[RequiresCheckpoint]` attribute that reads `HEARTMULA_CKPT`; verify that without the variable the marked test shows up as Skipped
- [x] 3.2 Add a smoke test without a checkpoint (e.g. write and read a `.safetensors` with `SafeTensorsWriter`/`SafeTensors`, preserving dtype BF16); verify with `dotnet test -p:TorchBackend=cpu` green
- [x] 3.3 Create `tests/fixtures/README.md` stating that C1 populates it and adjust `.gitignore`; verify with `git status` that READMEs are not ignored and no checkpoints sneak in

## 4. CI and documentation

- [x] 4.1 Add `.github/workflows/ci.yml` (ubuntu-latest, .NET 8, build + test with `TorchBackend=cpu`); verify with `actionlint` if available or by running the same commands locally (actionlint not available: YAML parsed and commands run locally)
- [x] 4.2 Update `README.md` and paths in `tools/` to the new layout and to the default `cpu` backend; verify that the commands documented in the README run as written (`dotnet build`, `dotnet test`)
- [x] 4.3 Final integration check: `dotnet build` and `dotnet test` clean from a fresh clone on CPU, without environment variables

## Workflow follow-up

- Archive the change after review (`/opsx:archive`); being `skip_specs` it generates no specs.
