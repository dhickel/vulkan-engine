# Development Documentation

## Project State
The project is a Vulkan rendering engine written in Rust.
- **Core Rendering:** Implemented in `renderer` crate.
- **Vulkan Bindings:** Uses `ash`.
- **Math:** Uses `glam`.
- **Memory Management:** Uses `vk_mem`.
- **Windowing:** Uses `winit`.

## Recent Changes
- Fixed compiler warnings (unused variables, unsafe blocks).
- Improved portability by changing absolute shader/asset paths to relative paths.
- Restored original shader files to ensure rendering logic is preserved.

## Current Focal Point
- Code cleanup and portability improvements.
- Ensuring the build is stable and warning-free.

## Next Focal Point
- Implement missing shader logic (stubs were found in some places).
- Verify rendering output.
- Add tests for the rendering pipeline.
