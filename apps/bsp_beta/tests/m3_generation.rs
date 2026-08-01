//! EnhancedV3 explorer package integration.
//!
//! This is intentionally a real engine_pack + ericw-tools build: it proves the
//! explorer's strict BSP/LIT/WAD/palette authorization closure rather than a
//! mocked compiler path. It skips only when all documented tool discovery
//! locations are genuinely unavailable.

use bsp_beta::generation;
use bsp_generator::enhanced_v3::V3Config;
use bsp_runtime::package::{self, ImportMode};

#[test]
fn m3_full_config_package_strictly_authorizes() {
    let Some(tools) =
        generation::discover_ericw_tools(None).expect("implicit discovery is not invalid")
    else {
        eprintln!("SKIP: qbsp/vis/light unavailable via ERICW_TOOLS_DIR, HOME, and PATH");
        return;
    };
    let root = tempfile::Builder::new()
        .prefix("bsp-beta-m3-integration-")
        .tempdir()
        .expect("unique temporary root");
    let package = root.path().join("package");
    let config = V3Config::nominal_sparse();

    engine_pack::enhanced_dungeon_v3::build_v3_package_from_config(
        &config,
        &package,
        Some(&tools),
        "bsp_beta_gen",
        None,
    )
    .expect("full-config engine_pack build must succeed");

    let bsp = package.join("bsp_beta_gen.bsp");
    let lit = package.join("bsp_beta_gen.lit");
    let palette = package.join("palette.lmp");
    let wad = package.join("cc0_dungeon_v2.wad");
    let textures = package.join("textures");
    assert!(bsp.is_file() && lit.is_file() && palette.is_file() && wad.is_file());
    assert!(textures.is_dir());

    let import = package::authorize_direct_import(
        &bsp,
        &palette,
        Some(&lit),
        &[wad],
        Some(&textures),
        ImportMode::Strict,
        0.0254,
    )
    .expect("strict generated closure must authorize with its explicit LIT");
    assert!(
        import.lit.is_some(),
        "strict generated import must retain LIT authorization"
    );
    assert!(
        import.world.diagnostics.is_empty(),
        "strict package diagnostics: {:?}",
        import.world.diagnostics
    );
}
