//! Executable guard for the documented Phase 01 descriptor ABI baseline.

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
        assert!(descriptors.contains(declaration), "missing Rust ABI declaration: {declaration}");
    }

    let pipeline = read(root.join("src/vulkan/vk_pipeline.rs"));
    let live_mesh_sets = [
        "desc_layout_cache.get(VkDescType::SceneData)",
        "desc_layout_cache.get(VkDescType::SkinData)",
        "desc_layout_cache.get(VkDescType::PbrSamplers)",
    ];
    for set in live_mesh_sets {
        assert!(pipeline.contains(set), "missing live mesh pipeline set: {set}");
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
        assert!(shader_manifest.contains(pair), "shader pair missing: {pair}");
    }
}

#[test]
fn descriptor_abi_document_records_critical_compatibility_points() {
    let root = renderer_root();
    let gpu_data = read(root.join("src/data/gpu_data.rs"));
    assert!(gpu_data.contains("size_of::<EnvironmentUBO>() == 1920"));
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
        assert!(document.contains(marker), "ABI document marker missing: {marker}");
    }
}
