# Senior Engineer Guidance

## Architecture

- Keep `audio` a leaf runtime subsystem. It may depend on `rodio`, logging, and small utility crates, but not renderer/window/editor/dogfood.
- Treat package/scene validation as authored-data validation, not playback initialization.
- Use `engine_events` as vocabulary only. Do not make `engine_events` call into `audio`.
- Keep the first audio facade boring: load/probe clip, create device-backed engine explicitly, play with options, control/drop handle.

## Identity

- Durable audio IDs are strings/newtypes such as `dogfood.audio.door_open`.
- Runtime playback handles are control objects only. They must not be serialized, used in package manifests, or treated as scene identity.
- Path hints may help loading, but path-only identity is not enough for package/scene contracts.

## Device Handling

- Unit tests should not call `OutputStream::try_default()` unless they are ignored/gated/manual.
- Prefer decode/probe tests using generated tiny WAV bytes or a small fixture.
- If a device smoke is run, use timeout-bound commands and report device availability separately from core validation.
- Be careful with background playback handles: stop/drop sinks in tests and avoid sleeps except tiny bounded waits in ignored/manual smoke.

## Packaging And Docs

- Extend existing validation paths. Do not fork CLI-only validation in `engine_pack`.
- Backward compatibility matters: old package/scene files without audio metadata should still validate.
- Docs should say exactly what landed and what did not. Avoid "spatial audio" wording unless the API and tests support it.

## Likely Failure Modes

- CI-like host has no default output device.
- `rodio::Decoder` accepts bytes in tests but runtime device creation fails.
- Package metadata accepts `kind = "audio"` but `scan-assets` or `add-asset` cannot generate it.
- Scene audio references validate syntactically but are not checked against known package IDs.
- Dogfood proof accidentally makes normal startup require an audio asset or device.
