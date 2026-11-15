# song-worker-process Specification

## Purpose

Let the HTTP service make a song, or only its codec decode, in a short-lived worker process, so that the memory the song needed (host and video) goes back to the system when it ends, a song can be stopped instantly, and the GPUs can be shared with other programs between songs.

## Requirements

### Requirement: The isolation setting
The setting `Service:Isolation`, also readable from the environment variable `HEARTMULA_ISOLATION`, SHALL choose how songs are made: `process`, `codec` or `inprocess`. When it is not set it SHALL be `codec`. The service's own setting SHALL win over the variable. Any other value SHALL stop the service at startup with a message naming the setting. Only `inprocess` and `codec` keep a model loaded between songs; the requirement of the `generation-worker` capability that the model is loaded once applies to them alone.

#### Scenario: Default
- **WHEN** nothing sets the isolation
- **THEN** the model is loaded in the service process and the codec decode of each song is made in a worker process

#### Scenario: Invalid value
- **WHEN** `HEARTMULA_ISOLATION` is `thread`
- **THEN** the service exits at startup with a message naming the setting

### Requirement: A song per process
In `process` isolation the service SHALL make each song in a new worker process started from the service's own executable, and SHALL NOT load the model or create a GPU context in the process that serves the API. The worker SHALL load the model, generate, decode with HeartCodec on the codec device and write the WAV to the path it is given with the models, devices and settings of the service (inherited from its environment), and exit with code 0. The video memory the worker used, on both devices, SHALL be returned to the system when it exits.

#### Scenario: Memory returns after a song
- **WHEN** a song has been made on CUDA devices and the service is idle
- **THEN** the video memory in use on each card is what it was before the first song, plus at most a few hundred MiB for the service process itself

#### Scenario: The same song as the command line
- **WHEN** a song is made in `process` isolation with the same checkpoint, tags, lyrics, length, sampling values, seed and devices as `generate --wav`
- **THEN** the two WAV files are byte-identical, and so are those of `inprocess` isolation

#### Scenario: Loading is part of the song
- **WHEN** a job starts in `process` isolation
- **THEN** its progress stage is `loading` until the worker has read the model, then `generating` and `decoding`

### Requirement: The codec in a process
In `codec` isolation the service SHALL keep the HeartMuLa model loaded and generate in its own process, hand the generated frames to a worker through a file, and decode them in the worker, which loads only the codec on the codec device, writes the WAV and exits. The video memory the codec used SHALL be returned when the worker exits; the memory of the model stays in use by the service. The frames file SHALL be removed when the song ends, however it ends.

#### Scenario: Codec memory returns
- **WHEN** a song has been made and the service is idle
- **THEN** the codec device holds what it held before the first song, and the model device holds the model

#### Scenario: Same song when the devices differ
- **WHEN** the model device and the codec device are different and the same inputs and seed are used as `generate --wav`
- **THEN** the WAV is byte-identical to the command line's

#### Scenario: Always deterministic
- **WHEN** the same request is made twice, on any devices
- **THEN** the two WAV files are identical

### Requirement: One card for the model and the codec
When the model device and the codec device are the same CUDA device, the weights of the model and the codec do not fit together, so the model SHALL be unloaded before the codec decodes (as the command line does) and loaded again before the next song's generation, in every isolation mode. In `codec` isolation the decode SHALL then run in the service's own process, because a worker could not use the memory that process keeps cached. The job's progress, its result and the model's reported readiness SHALL NOT change.

#### Scenario: A song on a single card
- **WHEN** the service sees a single CUDA device (or both devices are set to the same one) and makes a song in `codec` or `process` isolation
- **THEN** the song is made, with no out-of-memory failure in the decode

#### Scenario: Songs one after another on a single card
- **WHEN** a second song follows the first on that card
- **THEN** it is made too, the model having been loaded again for it, and `/health` stays `ready`

#### Scenario: Two cards are not affected
- **WHEN** the model and the codec devices differ
- **THEN** the model stays loaded and the isolation mode behaves as in the requirements above

### Requirement: Request in, progress out
The worker SHALL read the song request as one JSON line on its standard input, and then keep reading standard input until it is closed. It SHALL write one JSON line on its standard output at each progress point (the stage `loading`; `generating` with frames done and expected; `decoding` with the window and the number of windows) and nothing else on standard output. Diagnostic text SHALL go to standard error. The service SHALL turn the progress lines into the job's progress and SHALL ignore any other line.

#### Scenario: Progress
- **WHEN** a worker makes a song of 25 frames and two codec windows
- **THEN** the job's progress goes through `loading`, `generating` 1 to 25 of 25 and `decoding` 1 to 2 of 2 as the lines arrive

#### Scenario: Stray output
- **WHEN** a native library writes a line that is not a progress line to standard output
- **THEN** it is ignored and the song is unaffected

### Requirement: Failures are reported with their cause
If the worker cannot make the song it SHALL exit with a non-zero code after writing the cause (the exception type and the first line of its message, so that a multi-line native message such as libtorch's does not end in a stack frame) as the last line of standard error, and the job SHALL be `failed` with that cause in `error`. If the worker ends without having written the song and without that line (killed by the system, out of memory, a crash) the job SHALL be `failed` with an error stating that the song process ended unexpectedly and its exit code. The service SHALL go on with the next job in both cases and SHALL NOT leave a partial file.

#### Scenario: A song that cannot be made
- **WHEN** the worker fails because a configured device does not exist
- **THEN** the job is `failed` and its error names the device problem

#### Scenario: A native failure with a long message
- **WHEN** the worker fails with a multi-line libtorch message (for example `CUDA out of memory`)
- **THEN** the job's error is the exception type and the first line of that message, never a stack frame

#### Scenario: A killed worker
- **WHEN** the worker is killed by a signal while it makes a song
- **THEN** the job is `failed` with an error saying the process ended unexpectedly, no file of the job remains, and the next job runs

### Requirement: Cancelling ends the worker at once
Cancelling or forgetting a running job, and stopping the service, SHALL end its worker process immediately, without waiting for a frame or a codec window, and the next job SHALL start as soon as the process is gone. The job SHALL end `cancelled` (or be forgotten) and no WAV SHALL remain.

#### Scenario: Deleting a running job
- **WHEN** a running job is deleted through either API
- **THEN** its worker process is gone within a second, the video memory is returned and the next queued job starts

#### Scenario: Stopping the service
- **WHEN** the service is stopped while a worker makes a song
- **THEN** the worker is ended and the service exits without waiting for the song

### Requirement: A worker that loses its service stops by itself
A worker whose standard input is closed before it has finished (because the service disappeared) SHALL stop within a few seconds, remove its partial files and exit.

#### Scenario: The service is killed
- **WHEN** the service process is killed while a worker makes a song
- **THEN** the worker ends by itself shortly after and leaves no partial audio file

### Requirement: The on-demand mode reports its state
In `process` isolation `GET /health` SHALL report `ready` when the checkpoint files the model needs are present and the configured devices exist, without loading any model, and SHALL show `model` as `on-demand`. `missing` SHALL list the absent files. The `gpu` object SHALL still describe the model's device. In the other modes health SHALL be as before.

#### Scenario: Ready without loading
- **WHEN** the service has just started in `process` isolation with a complete checkpoint
- **THEN** `ready` is true at once and no GPU memory is in use

#### Scenario: Missing file
- **WHEN** the checkpoint lacks `tokenizer.json`
- **THEN** `ready` is false and `missing` contains `tokenizer.json`

### Requirement: The service's contract does not change with the mode
Every endpoint of both HTTP APIs, the job states, cancel and forget, the webhook and the stored results SHALL behave the same in all three modes, apart from the extra `loading` progress stage and the health fields above.

#### Scenario: Same API, three modes
- **WHEN** the same job is submitted in each mode
- **THEN** it ends `succeeded` with a WAV in each, and `GET /songs/{id}` and `GET /jobs/{id}` show the same fields
