# Design

## Context

The current code lives at the root with a single `HeartMuLa.Net.csproj` (net8.0, TorchSharp `0.105.*`, cuda-windows backend by default, `PlatformTarget=x64`). If the tests stayed under the root, the app project's compile glob would include them; that is why `src/` and `tests/` must be separated. The development machine is macOS (Darwin, probably Apple Silicon; `dotnet` is installed), where only the `cpu` backend works. See proposal.md for the motivation.

## Goals / Non-Goals

**Goals:**
- `dotnet build` and `dotnet test` green with `-p:TorchBackend=cpu` on macOS and on Linux CI.
- A single place where the backend is decided.
- A checkpoint-bound test pattern that other changes can reuse.

**Non-Goals:**
- Nothing is validated numerically (no fixtures yet). The logic of `Llama.cs`/`HeartMuLaModel.cs` is not touched beyond what the compiler forces.
- There is no CI with GPU and no checkpoint download.

## Decisions

1. **`src/` + `tests/` + `.sln`** instead of everything at the root. It prevents the app's glob from absorbing the tests and is the .NET convention. Alternative: exclude `tests/**` in the csproj; rejected as fragile.
2. **`TorchBackend` in `Directory.Build.props`**, default `cpu`. With `cuda-windows` as the default the build fails on macOS/Linux; the safe default is CPU and CUDA is requested explicitly (`-p:TorchBackend=cuda-linux`). The backend's conditional `PackageReference`s are moved there so that app and tests resolve the same native runtime.
3. **Remove `PlatformTarget=x64`**: TorchSharp-cpu supports osx-arm64; forcing x64 prevents running on Apple Silicon. `ServerGarbageCollection` is kept. We verify that version `0.105.*` exists on NuGet and, if not, pin the latest stable.
4. **`System.CommandLine`**: `Program.cs` uses its own `Args`; if the compiler confirms it is not used, the reference is removed (a pinned beta is free debt).
5. **`[RequiresCheckpoint]`**: an attribute derived from `FactAttribute` that sets `Skip` when `HEARTMULA_CKPT` is not defined or the directory does not exist. It exposes `Checkpoint.Dir` to the tests. It keeps the parity tests runnable in CI without weights.
6. **Fixtures in `tests/fixtures/`**: `*.safetensors` are ignored except the small ones agreed in C1; the concrete policy (git LFS or not) is decided in C1. In this change only the folder is created, with a `README.md`.
7. **Minimal CI**: `ubuntu-latest`, .NET 8, `dotnet build/test -p:TorchBackend=cpu`. No checkpoint cache.

On torchtune vs HuggingFace (RoPE, RMSNorm, mask): does not apply, attention/RoPE/norm are not modified.
Reference fixture: none in this change; C1 generates them with `tools/dump_reference.py`.

## Risks / Trade-offs

- [Never-compiled code may have many TorchSharp API errors] → fix only the minimum needed to compile and note in tasks every non-trivial fix; if it grows, open `/opsx:explore` before C3.
- [Compiling without errors does not prove correctness] → intentional; parity is C3.
- [The TorchSharp-cpu native package weighs hundreds of MB and slows down CI] → acceptable; NuGet cache in the workflow.
- [Moving files breaks paths in the README and in `tools/`] → update paths in the same task.
