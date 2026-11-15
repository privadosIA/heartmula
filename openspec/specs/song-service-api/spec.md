# song-service-api Specification

## Purpose

Define the HTTP contract of the HeartMuLa service: how a client asks for a song, follows its progress, downloads the audio and cancels it, and how the contract is published as an OpenAPI document with a Swagger UI.

## Requirements

### Requirement: A song is requested with an asynchronous job
`POST /songs` SHALL accept a JSON body with `tags` and `lyrics` (required, non-empty text), `durationMs` (optional, 1000 up to the configured maximum, 360 s by default), `seed` (optional 64-bit integer), `cfgScale` (optional), `temperature` and `topK` (optional sampling parameters) and `callbackUrl` (optional absolute `http` or `https` URL), SHALL create a job and SHALL answer `202 Accepted` with the job id, its state `queued` and a `Location` header pointing at `/songs/{id}`, without waiting for the generation.

#### Scenario: Job accepted
- **WHEN** a valid request is posted
- **THEN** the answer is `202` with a body holding the job id and state `queued`, and `Location: /songs/{id}`

#### Scenario: Defaults
- **WHEN** `durationMs`, `seed` and `cfgScale` are omitted
- **THEN** the job uses the configured default duration, a seed chosen by the server, `cfgScale` 1.5, `temperature` 1.0 and `topK` 50, and all of them are reported back by `GET /songs/{id}`

#### Scenario: Invalid body
- **WHEN** `tags` or `lyrics` is missing or blank, `durationMs` is outside `1000..MaxDurationMs`, `cfgScale` is not positive, `temperature` is outside 0.1 to 2.0, `topK` outside 1 to 500, or `callbackUrl` is not an absolute http(s) URL or its host is not allowed by the configured list
- **THEN** the answer is `400` with a problem-details body naming the offending fields and no job is created

#### Scenario: Queue full
- **WHEN** the number of queued jobs has reached the configured limit
- **THEN** the answer is `429` with a `Retry-After` header and no job is created

### Requirement: A job can be inspected
`GET /songs/{id}` SHALL return the job's `state` (`queued`, `running`, `succeeded`, `failed` or `cancelled`), the request parameters including the seed actually used, creation, start and end timestamps, a `progress` object (`stage` = `queued | generating | decoding | done`, `current`, `total`), for a failed job an `error` message and, when a `callbackUrl` was given, a `webhook` object (`state` = `pending | delivered | failed`, `attempts`, `lastError`).

#### Scenario: Running job
- **WHEN** the job is generating
- **THEN** `state` is `running`, `progress.stage` is `generating` and `current`/`total` are the frames done and the frame limit

#### Scenario: Unknown job
- **WHEN** the id does not exist or its result has expired
- **THEN** the answer is `404` with a problem-details body

### Requirement: The audio is downloaded as WAV
`GET /songs/{id}/audio` SHALL return the finished song as `audio/wav` (stereo, 48 kHz, PCM16) once the job has succeeded.

#### Scenario: Finished job
- **WHEN** the job has succeeded
- **THEN** the answer is `200` with `Content-Type: audio/wav` and a valid WAV whose duration is within one frame (80 ms) of the generated length

#### Scenario: Job not finished
- **WHEN** the job is `queued`, `running`, `failed` or `cancelled`
- **THEN** the answer is `409` with a problem-details body that reports the state

### Requirement: A job can be cancelled or removed
`DELETE /songs/{id}` SHALL cancel a queued or running job, and SHALL delete the stored result of a finished job. It SHALL be idempotent for jobs that are already gone.

#### Scenario: Cancel a running job
- **WHEN** a running job is deleted
- **THEN** the answer is `202` and the job ends in state `cancelled` without producing audio

#### Scenario: Remove a finished job
- **WHEN** a succeeded job is deleted
- **THEN** the answer is `204` and both `GET /songs/{id}` and `GET /songs/{id}/audio` answer `404`

### Requirement: A client can be notified by webhook
When a job has a `callbackUrl`, the service SHALL send it one `POST` with a JSON body (`id`, `state`, `seed`, actual `durationMs` of the audio, `error` if any, creation, start and end timestamps and a relative `audioUrl`) and the header `X-HeartMuLa-Event: song.finished` when the job ends as `succeeded`, `failed` or `cancelled`. A `2xx` answer SHALL count as delivered. When a webhook secret is configured the request SHALL carry `X-HeartMuLa-Signature: sha256=<hex HMAC-SHA256 of the body>`. A failed delivery SHALL be retried and SHALL never change the state of the job. Deleting the stored result of a finished job SHALL not send an event.

#### Scenario: Notified on success
- **WHEN** a job with a `callbackUrl` succeeds and the receiver answers `200`
- **THEN** the receiver got one request with `state: "succeeded"` and the job's `webhook.state` is `delivered` with `attempts: 1`

#### Scenario: Notified on failure and on cancellation
- **WHEN** a job with a `callbackUrl` fails or is cancelled
- **THEN** the receiver gets an event with `state` `failed` (with `error`) or `cancelled` respectively

#### Scenario: Signed body
- **WHEN** a secret is configured
- **THEN** the signature header equals the HMAC-SHA256 of the exact bytes of the body, hex encoded, prefixed by `sha256=`

#### Scenario: Receiver down
- **WHEN** the receiver does not answer or answers a non-`2xx` status
- **THEN** the delivery is retried up to three more times with growing waits, then `webhook.state` becomes `failed` with `lastError`, and the job's own state is unchanged and its audio still downloadable

#### Scenario: Redirects and host list
- **WHEN** the receiver answers with a redirect, or a host list is configured and the `callbackUrl` host is not on it
- **THEN** the redirect is not followed (the attempt counts as failed), and a host that is not on the list is rejected at `POST /songs` with `400`

#### Scenario: No callback
- **WHEN** a job has no `callbackUrl`
- **THEN** no outgoing request is made and the job has no `webhook` object

### Requirement: The service reports its health
`GET /health` SHALL answer without authentication and always with `200`, reporting whether the model is `loading`, `ready` or `failed` (and why, when it failed), the devices in use, the queue length and the running job id if any. It carries besides these the fields the orchestrator's client reads (see the `engine-compat-api` capability). A model that is not ready is shown by the fields, not by the status code.

#### Scenario: Model still loading
- **WHEN** the service has just started and the weights are being read
- **THEN** `GET /health` answers `200` with `model: "loading"` and `ready: false`, and `POST /songs` still queues jobs

#### Scenario: Model failed to load
- **WHEN** the weights could not be read
- **THEN** `GET /health` answers `200` with `model: "failed"`, `ready: false` and the reason in `modelError`

### Requirement: The contract is published as OpenAPI
The service SHALL serve its OpenAPI 3 document at `/swagger/v1/swagger.json` and a Swagger UI at `/swagger`, describing every endpoint, request and response body, status code and the API-key scheme.

#### Scenario: Swagger UI available
- **WHEN** a browser opens `/swagger`
- **THEN** the UI lists `POST /songs`, `GET /songs/{id}`, `GET /songs/{id}/audio`, `DELETE /songs/{id}` and `GET /health`, and a request can be sent from it

### Requirement: Access can be restricted with an API key
When an API key is configured, every endpoint except `GET /health`, `/swagger` and the OpenAPI document SHALL require it in the `X-Api-Key` header or as `Authorization: Bearer <key>`. Without a configured key no authentication is applied, and the service SHALL listen on the loopback interface unless an address is configured explicitly.

#### Scenario: Missing or wrong key
- **WHEN** a key is configured and a request to `POST /songs` carries no key or a different one
- **THEN** the answer is `401` and no job is created

#### Scenario: Default binding
- **WHEN** the service starts with no address configured
- **THEN** it listens only on `127.0.0.1`
