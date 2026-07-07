# Target Design

## Design Goal

Provide a small alpha audio contract that separates authored identity, device-independent validation, and optional device-backed playback.

## Contract Shape

- Authored identity: durable clip IDs such as `audio.click.start`, not runtime handles and not raw file paths.
- Package asset record: an audio asset kind with path, optional display metadata, optional declared format, optional duration/sample metadata only if validation can keep it honest.
- Scene/app reference: a durable audio clip reference that can be validated against known package asset IDs. This may be scene-level metadata or node-associated metadata, but it must not imply editor placement UI unless UI work lands.
- Runtime facade: `audio` crate types for clip bytes/probed metadata, device-optional engine construction, playback options, playback control, and errors.
- Event bridge: app/audio integration translates started/stopped/finished/failed states into `engine_events::AudioEvent` with `AudioClipId`.

## Device Boundary

- Core tests should validate bytes, metadata, decoder behavior, API errors, and non-device control logic without opening an output stream.
- Device-backed playback should be initiated only by explicit runtime path, ignored/manual test, environment variable, CLI flag, or dogfood option.
- Reports must distinguish:
  - `core_audio_validated`: no device required;
  - `device_smoke_passed`: a device was present and playback path started safely;
  - `device_smoke_skipped`: not run by design;
  - `device_smoke_blocked`: attempted but host had no usable output device.

## Schema Guidance

Use the existing package/scene validation style:

```toml
[[assets]]
id = "dogfood.audio.pickup"
kind = "audio"
path = "audio/pickup.ogg"
display_name = "Pickup"

[assets.metadata.audio]
format = "ogg"
usage = "effect"
```

Possible scene/app reference shape:

```json
{
  "audio": [
    {
      "id": "scene.audio.pickup",
      "clip": { "id": "dogfood.audio.pickup" },
      "trigger": "startup",
      "volume": 0.5
    }
  ]
}
```

Workers may choose a smaller shape if it fits existing code better, but durable clip ID validation and backward compatibility are required.

## Event Bridge

- Keep `engine_events` independent.
- If existing `AudioEvent` is sufficient, add mapper/helper tests in the audio consumer layer instead of expanding the event enum.
- If an event field is missing and the gap is small, extend `engine_events` conservatively with tests.
- Emit failure events for missing/decode/device errors where the sample/dogfood path can observe them.

## Dogfood/Sample Proof

The proof should do the least invasive thing that demonstrates the contract:

- add packaged audio fixture metadata and a tiny valid audio file if feasible;
- add an opt-in flag/env/config path that loads the packaged clip and attempts playback;
- log or emit events for start/failure;
- keep normal dogfood startup unaffected when audio is not requested.

If this cannot be done safely, produce `reports/dogfood-audio-proof-debt.md` with exact blockers and future slices.

## Documentation Contract

Docs must say:

- supported alpha flow: package clip, validate, load/probe, optionally play;
- tests do not require speakers/audio device;
- runtime playback depends on host audio device and rodio backend;
- unsupported: production mixing, spatialization, streaming, editor placement, complete platform matrix.
