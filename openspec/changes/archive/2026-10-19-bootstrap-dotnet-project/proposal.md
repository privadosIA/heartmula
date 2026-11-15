# Proposal

## Why

The HeartMuLa.Net C# skeleton (≈1.300 lines, written on 2026-10-18) has never been compiled and has no project structure: a single `.csproj` at the root, no tests, no CI, no git. All the following changes (C1–C8) need a base that compiles, an xUnit test project to hang the parity tests on, and a switchable TorchSharp backend so that validation can run on CPU.

## What Changes

- Reorganize into `src/HeartMuLa.Net/` (current code: `Weights/`, `Model/`, `Pipeline/`, `Program.cs`) and `tests/HeartMuLa.Net.Tests/` (xUnit), with a `HeartMuLa.Net.sln` solution.
- Move the `TorchBackend` property (`cpu|cuda-windows|cuda-linux`) to `Directory.Build.props` so that app and tests share it; the default value becomes `cpu`.
- Remove `PlatformTarget=x64` (it breaks on Apple Silicon, where development happens) and unused dependencies.
- Make `dotnet build -p:TorchBackend=cpu` compile cleanly, fixing only what is necessary of the TorchSharp API in the existing code (without changing behavior).
- Add a `[RequiresCheckpoint]` attribute (xUnit, skipped if `HEARTMULA_CKPT` is missing) and a smoke test that does not need a checkpoint.
- Prepare `tests/fixtures/` (with a consistent `.gitignore`) and move `tools/` as is; the fixtures' content is C1.
- Add CI (GitHub Actions) that builds and runs tests with the `cpu` backend.
- Initialize git.

Out of scope: any change to the model logic, generating fixtures, tokenizer, HeartCodec.

## Capabilities

### New Capabilities
None. It is scaffolding; the change declares `skip_specs: true`.

### Modified Capabilities
None.

## Impact

- Numerical parity: does not apply (no behavior change); it is only required that it compiles and the smoke test passes.
- Code: all current paths change to `src/HeartMuLa.Net/...`; `README.md` and `tools/` are updated for paths.
- Dependencies: TorchSharp (+ a native backend), xUnit, `Microsoft.NET.Test.Sdk`.
- Systems: new workflow `.github/workflows/ci.yml`.
