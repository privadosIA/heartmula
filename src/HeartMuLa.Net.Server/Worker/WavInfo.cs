namespace HeartMuLa.Server.Worker;

/// <summary>Reads the length of a PCM WAV from its header (the song a worker wrote).</summary>
public static class WavInfo
{
    /// <summary>The audio length in whole milliseconds, from the <c>fmt </c> and <c>data</c> chunks.</summary>
    public static int DurationMs(string path)
    {
        using var f = File.OpenRead(path);
        using var r = new BinaryReader(f);
        if (f.Length < 12 || new string(r.ReadChars(4)) != "RIFF") throw new InvalidDataException("not a WAV file");
        r.ReadInt32();
        if (new string(r.ReadChars(4)) != "WAVE") throw new InvalidDataException("not a WAV file");
        int byteRate = 0;
        while (f.Position + 8 <= f.Length)
        {
            var id = new string(r.ReadChars(4));
            int size = r.ReadInt32();
            if (id == "fmt ")
            {
                r.ReadInt16(); r.ReadInt16(); r.ReadInt32();
                byteRate = r.ReadInt32();
                f.Position += size - 12;
            }
            else if (id == "data")
            {
                if (byteRate <= 0) throw new InvalidDataException("the WAV has no fmt chunk before its data");
                long bytes = size >= 0 && f.Position + size <= f.Length ? size : f.Length - f.Position;     // a streamed header may carry a wrong size
                return (int)(bytes * 1000 / byteRate);
            }
            else f.Position += size + (size & 1);
        }
        throw new InvalidDataException("the WAV has no data chunk");
    }
}
