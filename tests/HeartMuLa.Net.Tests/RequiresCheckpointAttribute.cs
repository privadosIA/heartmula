using MusicGen.Core.Testing;

namespace HeartMuLa.Tests;

/// <summary>Checkpoint directory, taken from HEARTMULA_CKPT (never inside the repo); audio directory from HEARTMULA_AUDIO_DIR; device from HEARTMULA_DEVICE (cpu by default, or cuda).</summary>
public static class Checkpoint
{
    public static readonly CheckpointEnvironment Env = new("HEARTMULA_CKPT", "HEARTMULA_AUDIO_DIR", "HEARTMULA_DEVICE");

    public static string? Dir => Env.Dir;

    /// <summary>Directory where the tests may leave audio to listen to. Without it, no test writes audio.</summary>
    public static string? AudioDir => Env.AudioDir;

    public static string DeviceName => Env.DeviceName;
}

/// <summary>[Fact] that is skipped if HEARTMULA_CKPT does not point to an existing directory.</summary>
public sealed class RequiresCheckpointAttribute : RequiresCheckpointFactAttribute
{
    public RequiresCheckpointAttribute() : base(Checkpoint.Env) { }
}

/// <summary>[Fact] that needs the checkpoint and only makes sense on CPU; skipped with an explicit reason if HEARTMULA_DEVICE is not cpu.</summary>
public sealed class RequiresCpuCheckpointAttribute : RequiresCpuCheckpointFactAttribute
{
    public RequiresCpuCheckpointAttribute() : base(Checkpoint.Env) { }
}

/// <summary>[Fact] that writes audio to be listened to: it needs the checkpoint and HEARTMULA_AUDIO_DIR.</summary>
public sealed class RequiresAudioDirAttribute : RequiresAudioDirFactAttribute
{
    public RequiresAudioDirAttribute() : base(Checkpoint.Env) { }
}
