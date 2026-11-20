# HeartMuLa.Net

C# port (.NET 8 + TorchSharp) of [heartlib](https://github.com/HeartMuLa/heartlib), the HeartMuLa music generation model.
This site is the API reference, generated from the XML documentation comments in the source.

## Where to start

| You want to | Look at |
| --- | --- |
| Generate a song from your own code | <xref:HeartMuLa.Pipeline.MusicGenerationPipeline> (tags and lyrics in, audio frames out), then <xref:HeartMuLa.Pipeline.AudioStage> (frames to WAV) |
| Run it as an HTTP service | the `HeartMuLa.Server` namespaces: `Api` (endpoints), `Jobs`, `Worker` (generation and isolation modes), `Options` |
| Understand the model | `HeartMuLa.Model` (backbone and codebook decoder) and `HeartMuLa.Codec` (HeartCodec: flow matching, vocoder) |

Setup, requirements, the service's HTTP contract and the memory figures are in the
[README](https://github.com/privadosIA/heartmula#readme).
