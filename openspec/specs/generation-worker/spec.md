# generation-worker Specification

## Purpose

Define how the service runs song jobs: one worker that owns the GPUs, keeps the model loaded between songs, reports progress, can be cancelled, survives failures and stores the results with a retention time.

## Requirements

### Requirement: Jobs run one at a time in arrival order
The service SHALL run at most one generation at any moment and SHALL start queued jobs in the order they were accepted.

#### Scenario: Two jobs submitted together
- **WHEN** two jobs are accepted while none is running
- **THEN** the first runs while the second stays `queued`, and the second starts only after the first has finished, failed or been cancelled

### Requirement: The model is loaded once and reused
The service SHALL read the HeartMuLa weights once, at startup, on the configured model device, and SHALL reuse the loaded model for every job, releasing only the caches and tensors of each song. The codec SHALL be loaded for the decode of each song on the configured codec device, in fp32 with TF32 off.

#### Scenario: Second song without reloading
- **WHEN** a second job runs after a first one has succeeded
- **THEN** the weights are not read again, the second job's generation starts without a model-loading phase and the first job's memory (caches, prompt tensors) has been released

#### Scenario: Separate devices
- **WHEN** the model device and the codec device differ
- **THEN** the model stays resident on its device while the codec is loaded and released on the other

#### Scenario: Invalid device
- **WHEN** a configured device does not exist
- **THEN** the service fails to start with a message naming the setting

### Requirement: The service output equals the CLI output
For the same checkpoint, tags, lyrics, duration, `cfgScale`, seed and devices, the WAV produced by a job SHALL be byte-identical to the one produced by `generate --wav`.

#### Scenario: Same seed as the CLI
- **WHEN** a job and `generate` run with the same inputs and seed on the same devices
- **THEN** the two WAV files are identical

#### Scenario: Same seed twice
- **WHEN** two jobs with the same inputs and seed run one after the other on the same loaded model
- **THEN** their WAV files are identical

### Requirement: Progress is reported
While a job runs, the service SHALL update its progress after each generated frame and after each decoded codec window, so that `GET /songs/{id}` shows the current stage and counters.

#### Scenario: Progress moves
- **WHEN** a job is polled twice during generation
- **THEN** the second answer shows a `progress.current` greater than or equal to the first

### Requirement: Cancellation stops the work and frees the GPU
Cancelling a queued job SHALL remove it from the queue at once. Cancelling a running job SHALL stop it at the next frame or codec window, release its caches and tensors, and leave the worker ready for the next job.

#### Scenario: Cancel while generating
- **WHEN** a running job is cancelled
- **THEN** it ends in state `cancelled` within one frame time plus one codec window, no WAV is stored, and the next queued job starts

#### Scenario: Shutdown
- **WHEN** the service is stopped while a job runs
- **THEN** the running job is cancelled and the process exits without waiting for the song to finish

### Requirement: A forgotten job leaves nothing behind
When a job is forgotten (deleted through the orchestrator's route) while it is running, the worker SHALL stop it, SHALL NOT store its sidecar or its WAV afterwards, SHALL delete any partial file, and SHALL NOT notify a webhook for it.

#### Scenario: Forget while generating
- **WHEN** a running job is forgotten
- **THEN** the generation stops within one frame, the storage directory holds no file of that job, and a receiver of its `callbackUrl` gets nothing

### Requirement: A failed job does not stop the service
An error during a job (including an out-of-memory error) SHALL end that job in state `failed` with a message, SHALL release what the job allocated, and SHALL leave the service accepting and running other jobs.

#### Scenario: Out of memory on one job
- **WHEN** a job fails with a device out-of-memory error
- **THEN** it is `failed` with an error message and the next queued job runs normally

### Requirement: Webhook delivery does not hold the worker
Notifications SHALL be sent by a delivery component independent of the generation worker, with a timeout per attempt, so that a slow or unreachable receiver never delays the next job. The delivery state SHALL be stored with the job, and deliveries that were still `pending` when the service stopped SHALL be resumed at startup for results that have not expired.

#### Scenario: Slow receiver
- **WHEN** a receiver takes longer than the timeout to answer while another job is queued
- **THEN** the next job starts as soon as the previous one has ended and the slow delivery is retried in the background

#### Scenario: Restart with a pending delivery
- **WHEN** the service restarts while a delivery of a finished job was pending
- **THEN** the delivery is attempted again after startup, and `GET /songs/{id}` shows the state of the delivery

### Requirement: Results are stored on disk with a retention time
Each finished job SHALL be stored as a WAV file and a small metadata file in the configured directory. The index of jobs SHALL be rebuilt from that directory at startup, jobs that were `queued` or `running` when the service stopped SHALL appear as `failed` with the reason "interrupted by restart", and results older than the configured retention SHALL be deleted.

#### Scenario: Restart keeps results
- **WHEN** the service restarts after a job succeeded
- **THEN** `GET /songs/{id}` and `GET /songs/{id}/audio` still answer for that job

#### Scenario: Retention
- **WHEN** a result is older than the retention time
- **THEN** it is deleted and its endpoints answer `404`

#### Scenario: Interrupted job
- **WHEN** the service restarts and finds the metadata of a job that never finished
- **THEN** that job is reported as `failed` with the reason "interrupted by restart"
