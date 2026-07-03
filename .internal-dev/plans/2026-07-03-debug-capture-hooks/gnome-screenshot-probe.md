# GNOME Screenshot Probe

Date: 2026-07-03

## Goal

Test whether `gnome-screenshot` can provide a noninteractive visual proof loop for the native Vulkan/editor window before building renderer-side capture hooks.

## Environment

```text
gnome-screenshot 41.0
XDG_SESSION_TYPE=wayland
WAYLAND_DISPLAY=wayland-0
DISPLAY=:0
```

## Commands Tested

```sh
gnome-screenshot -f .internal-dev/debug_reports/gnome-desktop-probe.png
```

Result: hung and wrote no file.

```sh
timeout 8s gnome-screenshot -f .internal-dev/debug_reports/gnome-desktop-probe-timeout.png
```

Result: timed out with exit code `124` and wrote no file.

Both attempts emitted:

```text
Unable to use GNOME Shell's builtin screenshot interface, resorting to fallback X11.
Unable to capture a screenshot of any window
```

The fallback path also emitted zero-width/zero-height GDK/GdkPixbuf assertions.

## Conclusion

`gnome-screenshot` is not viable as the primary visual proof loop in this current remote Wayland session. It fails before the engine is launched, so it cannot be trusted to prove a Vulkan/editor frame.

## Workflow Decision

Use a two-tier workflow:

1. Optional external screenshot probe
   - Run only with a timeout.
   - Treat success as a convenience artifact, not authoritative engine proof.
   - Command:
     ```sh
     timeout 8s gnome-screenshot -f .internal-dev/debug_reports/desktop-probe.png
     ```
   - Gate:
     - file exists;
     - `file` identifies it as PNG;
     - dimensions are nonzero.

2. Primary visual proof
   - Build renderer-side one-shot frame capture.
   - Capture final present image after ImGui/editor UI and before final `PRESENT_SRC_KHR` transition.
   - Write PNG and sidecar JSON under `.internal-dev/debug_reports/`.
   - Inspect with local image tooling.

## Future Re-Test Conditions

Re-test `gnome-screenshot` only if the session/compositor changes, for example:

- local interactive GNOME Shell session instead of remote/headless Wayland;
- portal permissions are confirmed working;
- `gnome-screenshot` can capture a plain desktop image with a bounded timeout.

