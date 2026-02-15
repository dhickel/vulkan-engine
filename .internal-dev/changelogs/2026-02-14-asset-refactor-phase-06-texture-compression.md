# Phase 06: Texture Compression and Memory Budget

**Date:** 2026-02-14
**Branch:** asset-refactor
**Phase:** 06 (Texture Compression and Memory Budget)

## Summary

Implemented a robust texture compression pipeline using the `intel_tex_2` crate to support BC1-BC7 block compression formats. Introduced a new `TextureCompressionMode` policy (Disabled, Auto, Force) and a semantic-aware compression decision engine. Updated the asset pipeline to handle compressed payloads, perform CPU-side mipmap generation for compressed textures, and support multi-region buffer uploads.

## Changes

### New Dependencies
- `intel_tex_2`: Added for BC1-BC7 compression support.

### Modified Files
- `src/renderer/src/api/config.rs`:
  - Added `TextureCompressionMode` enum (Disabled, Auto, Force).
  - Added `CompressionConfig` struct.
  - Extended `AssetPolicyConfig` to include `compression: CompressionConfig`.
- `src/renderer/src/data/gpu_data.rs`:
  - Added `TextureSemantic` enum (BaseColor, Normal, MetallicRoughness, etc.).
  - Added `TexturePayload` enum (Raw, Compressed) to abstract texture data storage.
  - Updated `TextureMeta` to use `payload: TexturePayload` instead of raw fields.
- `src/renderer/src/data/compression.rs`:
  - New module implementing the compression decision logic and execution.
  - Supports format selection based on semantic (e.g., BC7 for color, BC5 for normal/metallic-roughness).
  - Implements swizzling for `MetallicRoughness` to map glTF G/B channels to BC5 R/G channels.
- `src/renderer/src/data/assimp_util.rs`:
  - Updated `get_texture_meta` to construct `TextureMeta` with `TexturePayload::Raw`.
  - Added `compress_meta` helper to invoke compression before caching.
  - Updated `process_materials` to determine texture semantic and compress textures using the configured policy.
  - Updated normalization functions (`normalize_metal_roughness_texture`, etc.) to handle `TexturePayload`.
- `src/renderer/src/data/data_cache.rs`:
  - Updated `add_texture` to handle `TexturePayload` and perform format validation/conversion on `Raw` payloads.
  - Updated `submit_texture_uploads` to calculate upload sizes and blit support based on `TexturePayload`.
  - Updated `save_debug_image` to support `TexturePayload`.
  - Updated `VkDataCache::new` default texture creation.
- `src/renderer/src/vulkan/vk_util.rs`:
  - Updated `record_host_to_image_buffer` to support `TexturePayload`.
  - For `Compressed` payloads: records multiple buffer-to-image copy regions (one per mip level) and skips GPU-side blit generation.
  - For `Raw` payloads: maintains existing behavior (single copy + GPU blit).
  - Updated `upload_skybox`, `upload_cubemap_faces`, and `upload_texture_2d` to work with the new `TextureMeta` structure.

## Technical Details

### Compression Policy
- **Auto:** Compresses textures if the format is suitable (e.g., 8-bit RGBA) and supported. Falls back to uncompressed if compression fails or format is unsupported.
- **Force:** Attempts compression and fails if unsupported (currently falls back to uncompressed with a warning in some paths to prevent crash, but strictly attempts compression).
- **Disabled:** Skips compression entirely.

### Format Mapping
- **BaseColor / Emissive:** BC7 (sRGB or UNORM)
- **Normal:** BC5 (UNORM)
- **MetallicRoughness / Occlusion:** BC5 (UNORM) - with channel swizzling (G/B -> R/G)
- **Generic:** BC7

### Pipeline Flow
1. **Ingest:** Image loaded as `DynamicImage`.
2. **Normalization:** Converted to RGBA or normalized (e.g., single channel to RGBA).
3. **Compression:** If enabled, `compress_texture` generates CPU mip chain, compresses each mip using `intel_tex_2`, and packs into `TexturePayload::Compressed`.
4. **Caching:** `TextureMeta` (Compressed or Raw) stored in `TextureCache`.
5. **Upload:** `submit_texture_uploads` identifies payload type.
   - **Compressed:** Uploads all mips in one go using multiple copy regions.
   - **Raw:** Uploads level 0, then generates mips on GPU using blits.

## Verification

- `cargo check -p renderer`: PASSED
- Verified handling of new `TexturePayload` in upload paths.
- Verified compilation with `intel_tex_2` dependency.
