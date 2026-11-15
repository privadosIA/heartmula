# engine-compat-api Specification

## Purpose

Let the orchestrator's existing client for music services work against this service unchanged: the same routes, request limits, answers and errors as the HeartMuLa and DiffRhythm2 services it already uses, served over the same queue and worker as the service's own API.

## Requirements

### Requirement: The orchestrator's routes are served
The service SHALL answer these calls with JSON bodies (UTF-8) except for the audio: `GET /health`; `POST /generate`, which takes a song request and answers `202` with `{"id": "<job id>"}`; `GET /jobs/{id}`, which answers `{"id", "state", "progress": {"done", "total"}, "error", "seconds"}`; `GET /jobs/{id}/audio`, which answers the song; `DELETE /jobs/{id}`, which answers `{"ok": true}` and forgets the job. A trailing slash SHALL be accepted. A path or method that matches no route of the service SHALL answer `404` with `{"error": "not found"}`, and a lower-case hexadecimal job id the service does not know SHALL answer `404` with `{"error": "unknown job"}` (an id that is not lower-case hexadecimal matches no route and answers `not found`). Jobs are the same ones the service's own API shows: a job made by `POST /generate` can be read through `GET /songs/{id}` and the other way round.

#### Scenario: A song from request to file
- **WHEN** a valid request is posted to `/generate`, `/jobs/{id}` is polled until its state is `done` and its audio is fetched
- **THEN** the answers are `202` with an id, `200` with state `done`, and `200` with a WAV

#### Scenario: Unknown job
- **WHEN** an id that was never issued, or was deleted, is asked for
- **THEN** the answer is `404` with `{"error": "unknown job"}`

#### Scenario: Unknown route
- **WHEN** a path or method the orchestrator's contract does not have is called on its routes (for example `PUT /generate` or `POST /jobs/{id}`)
- **THEN** the answer is `404` with `{"error": "not found"}`

#### Scenario: One job, two views
- **WHEN** a job is created with `POST /generate`
- **THEN** `GET /songs/{id}` shows the same job and `GET /jobs/{id}` shows it with the orchestrator's field names

### Requirement: The song request keeps the orchestrator's limits
`POST /generate` SHALL take `lyrics` (required, non-empty text of at most 20,000 characters), `tags` (a text or a list of words; trimmed, lower-cased, with the spaces around commas removed and commas at either end trimmed; at least one style must remain) and the optional numbers `max_ms` (5,000 to 360,000, default 240,000), `temperature` (0.1 to 2.0, default 1.0), `topk` (1 to 500, default 50; a fractional `max_ms` or `topk` is truncated, as the container does), `cfg_scale` (1.0 to 10.0, default 3.0) and `seed` (a whole number, 0 to 2^63 − 1, default 0, so the same request always gives the same song). A body that is not a JSON object, malformed JSON, a boolean or text where a number is expected, or any value out of range SHALL answer `400` with `{"error": "<reason naming the field>"}` and no job SHALL be created. A body larger than 1,000,000 bytes SHALL answer `413` with `{"error": "the request is too large"}`. A full queue SHALL answer `429` with `{"error": "the queue is full"}` and a `Retry-After` header.

#### Scenario: Defaults
- **WHEN** only `lyrics` and `tags` are sent
- **THEN** the job runs with a 240 s ceiling, temperature 1.0, top-k 50, guidance 3.0 and seed 0

#### Scenario: Tags as a list
- **WHEN** `tags` is `["Piano", " Happy "]`
- **THEN** the job's tags are `piano,happy`

#### Scenario: Every invalid field is named
- **WHEN** `lyrics` is blank, `tags` is empty, `max_ms` is 4999, `topk` is 0, `cfg_scale` is 11, or `seed` is negative, fractional or `true`
- **THEN** the answer is `400` and the error names that field, and no job exists

#### Scenario: Not JSON, or too large
- **WHEN** the body is `not json`, or larger than 1,000,000 bytes
- **THEN** the answers are `400` with an error and `413`, and no job exists afterwards

#### Scenario: Same song as the CLI
- **WHEN** `/generate` is called with the same lyrics, tags, `max_ms`, `seed`, `temperature`, `topk` and `cfg_scale` that `generate --wav` is given
- **THEN** the WAV is byte-identical to the CLI's

### Requirement: The job view uses the orchestrator's words
`GET /jobs/{id}` SHALL report `state` as `queued`, `running`, `done` or `failed` (a succeeded job is `done`; a job cancelled through the service's own API is `failed` with the error `cancelled`), `progress` as `{"done", "total"}` counting the frames of the song while it is generated, `error` as the message of a failed job and an empty string otherwise, and `seconds` as the time the job has been running (until it ended, once it has), rounded to one decimal and 0 before it starts.

#### Scenario: Running job
- **WHEN** the job is generating
- **THEN** `state` is `running`, `progress.done` is the frames made and `progress.total` the frame limit

#### Scenario: Failed job
- **WHEN** the job fails
- **THEN** `state` is `failed` and `error` holds the reason

### Requirement: The audio answers as the orchestrator expects
`GET /jobs/{id}/audio` SHALL answer `200` with `Content-Type: audio/wav` and a `Content-Length` once the job is `done`; before that it SHALL answer `409` with `{"error": "the job is <state>"}` using the orchestrator's state names; and if the job is done but its file is no longer there it SHALL answer `410` with `{"error": "the audio is gone"}`.

#### Scenario: Not finished
- **WHEN** the audio of a queued or running job is requested
- **THEN** the answer is `409` naming the state

#### Scenario: File gone
- **WHEN** the job is done but its WAV has been deleted from the storage directory
- **THEN** the answer is `410`

### Requirement: Deleting through the orchestrator's route forgets the job
`DELETE /jobs/{id}` SHALL cancel the job if it is queued or running, remove it from the service and delete its files, whatever its state, and answer `{"ok": true}`. Afterwards the job SHALL be unknown through both APIs, and nothing a running generation finishes later SHALL bring it back (no stored result, no webhook).

#### Scenario: Forget a finished job
- **WHEN** a done job is deleted
- **THEN** the answer is `200` `{"ok": true}` and `GET /jobs/{id}` answers `404`

#### Scenario: Forget a running job
- **WHEN** a running job is deleted
- **THEN** the answer is `200`, the generation stops, no result file remains, `GET /jobs/{id}` answers `404`, and after a restart the job does not come back

### Requirement: The orchestrator's health shape
`GET /health` SHALL answer `200` with `{"ready", "version", "busy", "queued", "gpu", "missing"}` and, besides them, the service's own fields (`status`, `model`, `modelError`, `modelDevice`, `codecDevice`, `runningJob`). `ready` is true only when the model is loaded; `version` is the configured service version (default `1`); `busy` says whether a song is being made; `queued` counts the waiting jobs; `gpu` is `null` when the model runs on the CPU and, otherwise, an object with `name`, `free_mb` and `total_mb` for the model's device where the values can be known (an unknown number is `null`); `missing` lists the checkpoint files that are absent. Health SHALL answer promptly while a song is being made, and SHALL answer `200` whether or not the model is ready.

#### Scenario: Ready
- **WHEN** the model is loaded
- **THEN** `ready` is true, `missing` is empty and `busy` tells whether a song is running

#### Scenario: Missing files
- **WHEN** the checkpoint lacks `tokenizer.json`
- **THEN** `ready` is false and `missing` contains `tokenizer.json`

#### Scenario: While singing
- **WHEN** health is asked during a song
- **THEN** it answers at once with `busy` true

### Requirement: The bearer token
When an API key is configured, every route except `/health` and the Swagger pages SHALL accept it as `Authorization: Bearer <key>` (as well as `X-Api-Key`) and SHALL answer `401` otherwise. On the orchestrator's routes the answer body is `{"error": "a bearer token is required"}`. The comparison SHALL take the same time wherever the strings differ.

#### Scenario: Wrong or missing token
- **WHEN** a key is configured and `/generate` is called with none, or another one
- **THEN** the answer is `401` with `{"error": "a bearer token is required"}` and no job is created

#### Scenario: Bearer works everywhere
- **WHEN** the right key is sent as a bearer token to `/songs`
- **THEN** the call is served

### Requirement: The orchestrator's environment variables are understood
The settings SHALL also be readable from the variables the container service uses: `HEARTMULA_TOKEN` (the API key), `HEARTMULA_JOBS` (the storage directory), `HEARTMULA_VERSION` (the version reported by health) and `HEARTMULA_PORT` (listen on all interfaces on that port, as the container does). `HEARTMULA_CKPT` is the checkpoint directory as before. A setting given in the service's own form SHALL win over its alias, and an invalid value (a port that is not a number) SHALL stop the service at startup with a message naming the variable.

#### Scenario: Alias used
- **WHEN** only `HEARTMULA_TOKEN` is set
- **THEN** the API key is that value

#### Scenario: Own setting wins
- **WHEN** both `Security:ApiKey` and `HEARTMULA_TOKEN` are set
- **THEN** `Security:ApiKey` is used

#### Scenario: Bad port
- **WHEN** `HEARTMULA_PORT` is `abc`
- **THEN** the service exits at startup with a message naming `HEARTMULA_PORT`
