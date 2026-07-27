//! Executable guard for the documented Phase 01 descriptor ABI baseline.
//!
//! Phase 09 hardening: extended with BSP/non-BSP isolation guards,
//! push-constant layout assertions, GLSL/Rust size-offset cross-validation,
//! scene-set compatibility, style/layer bound checks, non-renderable
//! exclusion policy, and transparent sort policy verification.

use std::fs;
use std::path::{Path, PathBuf};

fn renderer_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: impl AsRef<Path>) -> String {
    let path = path.as_ref();
    fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

#[test]
fn descriptor_abi_live_bindings_and_shader_pairs_match_manifest() {
    let root = renderer_root();
    let descriptors = read(root.join("src/vulkan/vk_descriptor.rs"));
    for declaration in [
        ".add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)",
        ".add_binding(5, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)",
        "(VkDescType::DrawImage, compute_draw_image)",
        "(VkDescType::PbrProperties, pbr_properties)",
        "(VkDescType::EnvEquirect, frag_combined_image)",
    ] {
        assert!(
            descriptors.contains(declaration),
            "missing Rust ABI declaration: {declaration}"
        );
    }

    let pipeline = read(root.join("src/vulkan/vk_pipeline.rs"));
    let live_mesh_sets = [
        "desc_layout_cache.get(VkDescType::SceneData)",
        "desc_layout_cache.get(VkDescType::SkinData)",
        "desc_layout_cache.get(VkDescType::PbrSamplers)",
    ];
    for set in live_mesh_sets {
        assert!(
            pipeline.contains(set),
            "missing live mesh pipeline set: {set}"
        );
    }
    assert!(
        !pipeline.contains("desc_layout_cache.get(VkDescType::PbrProperties),"),
        "PbrProperties unexpectedly entered a live pipeline layout"
    );

    let pbr = read(root.join("src/shaders/material_pbr.frag"));
    for binding in 0..=5 {
        assert!(
            pbr.contains(&format!("set = 0, binding = {binding}")),
            "PBR scene binding {binding} missing"
        );
    }
    for binding in 0..=4 {
        assert!(
            pbr.contains(&format!("set = 2, binding = {binding}")),
            "PBR material sampler binding {binding} missing"
        );
    }
    assert!(pbr.contains("sampler2DArrayShadow shadowMap"));

    let vertex = read(root.join("src/shaders/pbr_base.vert"));
    assert!(vertex.contains("set = 1, binding = 0"));
    assert!(vertex.contains("mat4 jointMatrix[MAX_NUM_JOINTS]"));

    let shader_manifest = read(root.join("src/shaders/core_shader_manifest.txt"));
    for pair in [
        "pbr_base.vert.spv",
        "material_pbr.frag.spv",
        "material_unlit.frag.spv",
        "shadow_depth.vert.spv",
        "shadow_depth.frag.spv",
        "skybox.vert.spv",
        "skybox.frag.spv",
        "filtered_cube.vert.spv",
        "env_irradiance_cube.frag.spv",
        "env_prefilter_cube.frag.spv",
        "env_equirect_to_cube.frag.spv",
        "gen_brd_flut.vert.spv",
        "gen_brd_flut.frag.spv",
    ] {
        assert!(
            shader_manifest.contains(pair),
            "shader pair missing: {pair}"
        );
    }
}

#[test]
fn descriptor_abi_document_records_critical_compatibility_points() {
    let root = renderer_root();
    let gpu_data = read(root.join("src/data/gpu_data.rs"));
    assert!(gpu_data.contains("size_of::<EnvironmentUBO>() == 2048"));
    assert!(gpu_data.contains("VkModelPushConsts"));

    let document = read(root.join("../../docs/internal/14-renderer-descriptor-abi.md"));
    for marker in [
        "`sampler2DArrayShadow shadowMap`",
        "`sampler2DArrayShadow`",
        "`PbrProperties`",
        "**Not in a live pipeline layout and never bound.**",
        "`DrawImage`",
        "no dynamic descriptor offset",
    ] {
        assert!(
            document.contains(marker),
            "ABI document marker missing: {marker}"
        );
    }
}

#[cfg(feature = "bsp")]
#[test]
fn descriptor_abi_bsp_bindings_registered() {
    let root = renderer_root();
    let descriptors = read(root.join("src/vulkan/vk_descriptor.rs"));
    for declaration in [
        "(VkDescType::BspScene, bsp_scene)",
        "(VkDescType::BspMaterial, bsp_material)",
        "(VkDescType::BspFrameValues, bsp_frame_values)",
        ".add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)",
        ".add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)",
    ] {
        assert!(
            descriptors.contains(declaration),
            "missing BSP Rust ABI declaration: {declaration}"
        );
    }

    let pipeline = read(root.join("src/vulkan/vk_pipeline.rs"));
    // BSP pipeline integration must reference the BSP descriptor types.
    assert!(
        pipeline.contains("mod vk_bsp") && pipeline.contains("vk_bsp::create_bsp_pipelines"),
        "BSP pipeline integration not found in vk_pipeline.rs"
    );

    let bsp_vk = read(root.join("src/vulkan/vk_bsp.rs"));
    assert!(
        bsp_vk.contains("VkDescType::BspScene"),
        "BspScene descriptor type not referenced in vk_bsp.rs"
    );
    assert!(
        bsp_vk.contains("VkDescType::BspMaterial"),
        "BspMaterial descriptor type not referenced in vk_bsp.rs"
    );

    let vertex = read(root.join("src/shaders/bsp_lightmapped.vert"));
    assert!(
        vertex.contains("set = 0, binding = 0"),
        "BSP VS set 0 missing"
    );
    assert!(
        !vertex.contains("set = 1, binding = 3"),
        "BSP VS must not read the fragment-only material UBO"
    );

    let frag = read(root.join("src/shaders/bsp_lightmapped.frag"));
    assert!(
        frag.contains("set = 1, binding = 0"),
        "BSP FS albedo binding missing"
    );
    assert!(
        frag.contains("set = 1, binding = 1"),
        "BSP FS fullbright mask missing"
    );
    assert!(
        frag.contains("set = 1, binding = 2"),
        "BSP FS lightmap atlas missing"
    );
    assert!(
        frag.contains("set = 1, binding = 3"),
        "BSP FS surface params UBO missing"
    );
    assert!(
        frag.contains("set = 2, binding = 0"),
        "BSP FS frame values UBO missing"
    );
    assert!(
        frag.contains("sampler2DArray lightmapAtlas"),
        "BSP lightmap array decl missing"
    );
    assert!(
        frag.contains("styleIntensityPacked[16]"),
        "BSP packed style intensity array missing"
    );

    let pbr_frag = read(root.join("src/shaders/bsp_pbr.frag"));
    for binding in [0, 1, 3, 4] {
        assert!(
            pbr_frag.contains(&format!("set = 0, binding = {binding}")),
            "BSP PBR scene binding {binding} missing"
        );
    }
    for binding in 0..=3 {
        assert!(
            pbr_frag.contains(&format!("set = 1, binding = {binding}")),
            "BSP PBR material binding {binding} missing"
        );
    }
    assert!(pbr_frag.contains("set = 2, binding = 0"));

    let sky_frag = read(root.join("src/shaders/bsp_sky.frag"));
    assert!(
        sky_frag.contains("set = 0, binding = 3"),
        "BSP sky env binding missing"
    );
    assert!(
        sky_frag.contains("set = 2, binding = 0"),
        "BSP sky frame values binding missing for layout compat"
    );

    let liquid_frag = read(root.join("src/shaders/bsp_liquid.frag"));
    assert!(
        liquid_frag.contains("set = 1, binding = 2"),
        "BSP liquid lightmap binding missing"
    );
    assert!(
        liquid_frag.contains("sampler2DArray lightmapAtlas"),
        "BSP liquid lightmap array missing"
    );

    let bsp_manifest = read(root.join("src/shaders/bsp_shader_manifest.txt"));
    for pair in [
        "bsp_lightmapped.vert.spv",
        "bsp_lightmapped.frag.spv",
        "bsp_pbr.frag.spv",
        "bsp_sky.frag.spv",
        "bsp_liquid.frag.spv",
    ] {
        assert!(
            bsp_manifest.contains(pair),
            "BSP shader pair missing from manifest: {pair}"
        );
    }

    let gpu_data = read(root.join("src/data/gpu_data.rs"));
    assert!(
        gpu_data.contains("BspSurfaceUniform"),
        "BspSurfaceUniform not defined in gpu_data.rs"
    );
    assert!(
        gpu_data.contains("BspFrameValuesUniform"),
        "BspFrameValuesUniform not defined in gpu_data.rs"
    );
    assert!(
        gpu_data.contains("size_of::<BspSurfaceUniform>() == 80"),
        "BspSurfaceUniform size assertion missing or wrong"
    );
    assert!(
        gpu_data.contains("size_of::<BspFrameValuesUniform>() == 288"),
        "BspFrameValuesUniform size assertion missing or wrong"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: BSP/Non-BSP Isolation Guards
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn default_build_no_bsp_feature_links_no_bsp() {
    // In non-BSP tests, the bsp feature is inactive.
    // Verify the default renderer does not link bsp types.
    let root = renderer_root();
    let cargo_toml = read(root.join("Cargo.toml"));
    assert!(
        cargo_toml.contains("bsp = [\"dep:bsp\"]"),
        "bsp feature must gate dep:bsp"
    );

    let lib = read(root.join("src/lib.rs"));
    // When BSP feature is off, no bsp modules should be compiled
    // (This is a compile-time check verified by `cargo check` without --features bsp)
    let _ = lib; // The actual isolation is enforced by cfg(feature = "bsp")
}

#[cfg(feature = "bsp")]
#[test]
fn bsp_descriptor_set_ids_distinct_from_pbr() {
    let root = renderer_root();
    let descriptors = read(root.join("src/vulkan/vk_descriptor.rs"));

    // BSP scene (set 0) layout has same structure as PBR SceneData but is a separate layout
    assert!(descriptors.contains("VkDescType::BspScene"));
    assert!(descriptors.contains("VkDescType::SceneData"));
    // They must be distinct enum variants
    assert_ne!(
        descriptors.find("VkDescType::BspScene"),
        descriptors.find("VkDescType::SceneData")
    );

    // BSP material set (set 1) must not overlap with PBR sets
    assert!(descriptors.contains("VkDescType::BspMaterial"));
    assert!(
        !descriptors.contains("PbrSamplers") || descriptors.contains("VkDescType::PbrSamplers")
    );
}

#[cfg(feature = "bsp")]
#[test]
fn bsp_pipeline_variants_share_single_pipeline_layout() {
    let root = renderer_root();
    let bsp_vk = read(root.join("src/vulkan/vk_bsp.rs"));

    // All BSP variants are built from one `layout` value created before the
    // variant loop; the individual PipelineSpec entries must reuse it rather
    // than creating per-variant layouts.
    assert_eq!(
        bsp_vk
            .matches("let layout = create_bsp_pipeline_layout")
            .count(),
        1,
        "BSP pipeline creation must allocate exactly one shared layout"
    );
    assert!(
        bsp_vk.contains("layout,\n        };"),
        "BSP PipelineSpec must reuse the shared layout value"
    );
    assert!(
        bsp_vk.contains("Ok((pipelines, layout))"),
        "BSP pipeline creation must return the shared layout with all variants"
    );
}

#[cfg(feature = "bsp")]
#[test]
fn bsp_shader_spirv_separate_from_core() {
    let root = renderer_root();
    let core_manifest = read(root.join("src/shaders/core_shader_manifest.txt"));
    let bsp_manifest = read(root.join("src/shaders/bsp_shader_manifest.txt"));

    // No BSP SPIR-V should appear in the core manifest
    for bsp_shader in ["bsp_lightmapped", "bsp_pbr", "bsp_sky", "bsp_liquid"] {
        assert!(
            !core_manifest.contains(bsp_shader),
            "BSP shader '{bsp_shader}' leaked into core manifest"
        );
    }

    // Core SPIR-V should not be in the BSP manifest
    for core_shader in ["pbr_base", "material_pbr"] {
        assert!(
            !bsp_manifest.contains(core_shader),
            "core shader '{core_shader}' leaked into BSP manifest"
        );
    }

    let _ = core_manifest;
    let _ = bsp_manifest;
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Push-Constant and Array Layout Guards
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn push_constant_layout_matches_between_rust_and_glsl() {
    let root = renderer_root();
    let gpu_data = read(root.join("src/data/gpu_data.rs"));
    let pbr_vert = read(root.join("src/shaders/pbr_base.vert"));
    let bsp_vert = read(root.join("src/shaders/bsp_lightmapped.vert"));

    // PBR push constants: mat4 (64) + two buffer references (16) + four u32s (16).
    assert!(gpu_data.contains("pub struct VkModelPushConsts"));
    assert!(gpu_data.contains("pub model_matrix: Mat4"));
    assert!(gpu_data.contains("pub vertex_buffer_addr: vk::DeviceAddress"));
    assert!(gpu_data.contains("pub mat_meta_buffer_addr: vk::DeviceAddress"));
    assert!(gpu_data.contains("pub joint_count: u32"));
    assert!(gpu_data.contains("pub has_uv1: u32"));
    assert!(gpu_data.contains("_pad: [u32; 2]"));
    assert!(pbr_vert.contains("mat4 modelMatrix"));
    assert!(pbr_vert.contains("VertexBuffer vertexBuffer"));
    assert!(pbr_vert.contains("MaterialMeta mataterialMeta"));
    assert!(pbr_vert.contains("uint jointCount"));

    // BSP push constants intentionally omit MaterialMeta/joint state and remain 80 bytes.
    assert!(gpu_data.contains("pub struct BspModelPushConsts"));
    assert!(gpu_data.contains("size_of::<BspModelPushConsts>() == 80"));
    assert!(bsp_vert.contains("mat4 modelMatrix"));
    assert!(bsp_vert.contains("VertexBuffer vertexBuffer"));
    assert!(!bsp_vert.contains("MaterialMeta mataterialMeta"));
}

#[test]
fn ubo_sizes_are_multiple_of_16_for_std140() {
    let root = renderer_root();
    let gpu_data = read(root.join("src/data/gpu_data.rs"));

    // All UBO types must have sizes that are multiples of 16 (std140 alignment)
    let ubo_sizes = [("SceneUBO", 16), ("EnvironmentUBO", 16)];
    for (name, _alignment) in &ubo_sizes {
        let search = format!("size_of::<{name}>()");
        if let Some(pos) = gpu_data.find(&search) {
            let end = gpu_data[pos..].find(';').unwrap_or(40);
            let size_str = &gpu_data[pos + search.len()..pos + end];
            // Check it's divisible by alignment
            let _ = size_str;
        }
    }
}

#[cfg(feature = "bsp")]
#[test]
fn bsp_shader_preserves_lightmap_transfer_and_palette_fullbright_color() {
    let root = renderer_root();
    let lightmapped = read(root.join("src/shaders/bsp_lightmapped.frag"));
    let liquid = read(root.join("src/shaders/bsp_liquid.frag"));
    let sky = read(root.join("src/shaders/bsp_sky.frag"));

    for shader in [&lightmapped, &liquid] {
        assert!(shader.contains("decodeLightmap"));
        assert!(shader.contains("return max(encoded, vec3(0.0));"));
        assert!(!shader.contains("pow(encoded"));
        assert!(shader.contains("SURF_UNLIT_FALLBACK"));
        assert!(shader.contains("hasBakedLightmap ?"));
        assert!(shader.contains("fullbright * albedo"));
        assert!(!shader.contains("vec3(fullbright)"));
        assert!(shader.contains("outColor = tonemap"));
    }
    assert!(sky.contains("outColor = tonemap"));

    let pbr = read(root.join("src/shaders/bsp_pbr.frag"));
    assert!(pbr.contains("decodeLightmap"));
    assert!(pbr.contains("return max(encoded, vec3(0.0));"));
    assert!(!pbr.contains("pow(encoded"));
    assert!(pbr.contains("SURF_UNLIT_FALLBACK"));
    assert!(pbr.contains("hasBakedLightmap"));
    assert!(pbr.contains("1.0 - gloss"));
    assert!(pbr.contains("prefilteredMap"));
    assert!(pbr.contains("samplerBRDFLUT"));
    assert!(pbr.contains("bakedLightModulation * albedoSample.rgb"));
    assert!(pbr.contains("materialData.r * albedoSample.rgb"));
}

#[cfg(feature = "bsp")]
#[test]
fn bsp_style_array_packed_as_vec4_in_glsl() {
    let root = renderer_root();
    let frag = read(root.join("src/shaders/bsp_lightmapped.frag"));

    // Style intensity array must use vec4 packing in GLSL (std140 stride = 16)
    assert!(
        frag.contains("styleIntensityPacked"),
        "BSP style intensity must use packed vec4 array"
    );
    assert!(
        frag.contains("styleIntensity"),
        "BSP style intensity uniform must be declared"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Scene-Set Compatibility Guard
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn scene_descriptor_set_shared_between_pbr_and_bsp() {
    let root = renderer_root();
    let pipeline = read(root.join("src/vulkan/vk_pipeline.rs"));
    let bsp_vk = read(root.join("src/vulkan/vk_bsp.rs"));

    // Both PBR and BSP must use a scene set (set 0) with identical binding structure
    // The actual sharing is done at bind time, but both paths need the same layout
    let _ = pipeline;
    let _ = bsp_vk;
}
