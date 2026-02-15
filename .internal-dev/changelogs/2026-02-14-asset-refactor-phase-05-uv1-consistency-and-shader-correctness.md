# Phase 05: UV1 Consistency and Shader Correctness

**Date:** 2026-02-14
**Branch:** asset-refactor
**Phase:** 05 (UV1 Consistency and Shader Correctness)

## Summary

Extended the asset pipeline and shaders to support multiple UV sets (UV0 and UV1) consistently from ingest to sampling. Added deterministic fallback logic for meshes missing requested UV channels and updated PBR normal mapping to use correctly selected UV set derivatives.

## Changes

### Modified Files
- `src/renderer/src/api/assets.rs`:
  - Extended `ProceduralVertex` to include `uv1: glam::Vec2`.
  - Updated `validate_procedural_mesh` to validate finite values for `uv1`.
  - Updated `procedural_vertex_to_gpu` to map `uv1` to internal `Vertex` format.
  - `upload_procedural_mesh` now calculates `has_uv1` metadata based on vertex data.
  - Updated unit tests to include non-zero UV1 verification.
- `src/renderer/src/data/gpu_data.rs`:
  - Added `has_uv1: bool` metadata to `MeshMeta` and `VkMeshBuffers`.
  - Added `has_uv1: bool` to `RenderObject`.
  - Extended `VkModelPushConsts` to carry `has_uv1: u32` flag to shaders.
- `src/renderer/src/data/data_cache.rs`:
  - Added `requires_uv1: bool` to `VkLoadedMaterial`.
  - `MeshCache::allocate` now propagates `has_uv1` from `MeshMeta` to `VkMeshBuffers`.
  - `TextureCache::write_material_descriptors` now determines `requires_uv1` based on material UV set indices.
  - Initialized `skybox` mesh with `has_uv1: false`.
- `src/renderer/src/data/assimp_util.rs`:
  - `process_meshes` now correctly detects presence of UV1 channel from Assimp and populates `has_uv1`.
  - Upgraded UV set clamping log from `debug!` to `warn!` per policy rules.
- `src/renderer/src/vulkan/vk_render.rs`:
  - `VkRenderCore` now tracks `uv_fallback_warnings` via `Mutex<HashSet>` to avoid log spamming.
  - `resolve_submission_buckets` implements runtime validation: logs a warning once if a material requires UV1 but the mesh lacks it.
  - `record_geometry_draw_sequence` passes the `has_uv1` flag to GPU via push constants.
- `src/renderer/src/shaders/material_pbr.frag`:
  - Added `has_uv1` to push constant block.
  - Added `getUV(int)` helper function implementing the UV fallback policy.
  - Updated `getNormal` to use selected UV set for both texture sampling and `dFdx/dFdy` derivatives.
  - Updated all sampling points (base color, metallic-roughness, occlusion, emissive) to use `getUV`.
- `src/renderer/src/shaders/material_unlit.frag`:
  - Added `has_uv1` to push constants and integrated `getUV` helper for consistency.
- `apps/dungeon_dogfood/src/geometry.rs`:
  - Updated `make_vertex` to initialize `uv1` for procedural dungeon geometry.

## Public API Changes

```rust
// ProceduralVertex now supports two UV sets
pub struct ProceduralVertex {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub tangent: glam::Vec4,
    pub uv0: glam::Vec2,
    pub uv1: glam::Vec2, // NEW
    pub color: glam::Vec4,
}
```

## UV Policy Implementation

1. **Clamping:** If a material requests a UV set > 1, it is clamped to UV0 and a warning is emitted during ingest.
2. **Fallback:** If a material requests UV1 but the mesh only provides UV0, the shader automatically falls back to UV0. A warning is logged once per mesh/material pair at the first render submission.
3. **Correctness:** PBR tangent-space reconstruction now uses the partial derivatives of the *actually sampled* UV set, fixing distorted normal mapping on UV1-mapped surfaces.

## Tests

- New unit test `procedural_vertex_conversion_preserves_uv1` in `api::assets`.
- Verified all existing 109 tests PASS.

## Verification

- `cargo check`: PASSED
- `cargo check -p renderer`: PASSED
- `cargo test -p renderer --lib`: PASSED
- `cargo run -p renderer --example demo_pbr`: PASSED (smoke test)
- `cargo run -p renderer --example api_test`: PASSED (smoke test)
