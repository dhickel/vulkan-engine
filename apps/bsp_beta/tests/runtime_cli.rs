//! Runtime CLI integration tests for the BSP beta application.

use std::path::PathBuf;

#[path = "../src/cli.rs"]
mod cli;

#[test]
fn cli_default_args() {
    let args = cli::CliArgs::default();
    assert!(args.bsp_path.is_none());
    assert!(!args.headless);
    assert!(!args.mcp);
    assert_eq!(args.capture_frames, 0);
    assert!(!args.show_lights);
    assert!(args.import_mode.is_none());
    assert!((args.scale - 0.0254).abs() < 1e-6);
}

#[test]
fn cli_default_scale_is_standard_quake_to_engine() {
    let args = cli::CliArgs::default();
    assert!((args.scale - 0.0254).abs() < 1e-6);
}

#[test]
fn cli_parses_supported_space_separated_flags() {
    let args = cli::parse_from([
        "--strict",
        "--bsp",
        "maps/e1m1.bsp",
        "--scale",
        "0.03125",
        "--headless",
        "--mcp",
        "--capture-frames",
        "30",
        "--lights",
    ])
    .unwrap();

    assert_eq!(args.bsp_path, Some(PathBuf::from("maps/e1m1.bsp")));
    assert!((args.scale - 0.03125).abs() < 1e-6);
    assert!(args.headless);
    assert!(args.mcp);
    assert_eq!(args.capture_frames, 30);
    assert!(args.show_lights);
    assert_eq!(args.import_mode, Some(cli::ImportMode::Strict));
}

#[test]
fn cli_preserves_struct_values() {
    let original = cli::CliArgs {
        bsp_path: Some(PathBuf::from("data/test.bsp")),
        scale: 0.03125,
        headless: true,
        mcp: true,
        capture_frames: 120,
        show_lights: false,
        palette_path: None,
        lit_path: None,
        wad_path: None,
        import_mode: Some(cli::ImportMode::Development),
        textures_dir: None,
        stats: false,
        all_visible: false,
        corpus_identity: None,
        acceptance_camera: None,
    };

    let cloned = original.clone();
    assert_eq!(cloned.bsp_path, original.bsp_path);
    assert!((cloned.scale - original.scale).abs() < 1e-6);
    assert_eq!(cloned.headless, original.headless);
    assert_eq!(cloned.mcp, original.mcp);
    assert_eq!(cloned.capture_frames, original.capture_frames);
    assert_eq!(cloned.show_lights, original.show_lights);
    assert_eq!(cloned.import_mode, original.import_mode);
}

#[test]
fn cli_rejects_equals_forms_and_unknown_flags() {
    assert_eq!(
        cli::parse_from(["--strict", "--bsp=maps/e1m1.bsp"]).unwrap_err(),
        cli::CliError::UnknownArgument("--bsp=maps/e1m1.bsp".to_string())
    );
    assert_eq!(
        cli::parse_from(["--strict", "--capture-frames=5"]).unwrap_err(),
        cli::CliError::UnknownArgument("--capture-frames=5".to_string())
    );
    assert_eq!(
        cli::parse_from(["--strict", "--bogus"]).unwrap_err(),
        cli::CliError::UnknownArgument("--bogus".to_string())
    );
}

#[test]
fn cli_rejects_missing_or_malformed_values() {
    assert_eq!(
        cli::parse_from(["--strict", "--bsp", "--headless"]).unwrap_err(),
        cli::CliError::MissingValue("--bsp")
    );
    assert!(matches!(
        cli::parse_from(["--development", "--scale", "inf"]).unwrap_err(),
        cli::CliError::NonFiniteScale(_)
    ));
    assert!(matches!(
        cli::parse_from(["--development", "--capture-frames", "bad"]).unwrap_err(),
        cli::CliError::InvalidCaptureFrames(_)
    ));
}

#[test]
fn cli_rejects_conflicting_modes() {
    assert_eq!(
        cli::parse_from(["--strict", "--development"]).unwrap_err(),
        cli::CliError::ConflictingImportMode
    );
}

#[test]
fn cli_requires_import_mode() {
    let args = cli::CliArgs::default();
    assert_eq!(
        args.require_import_mode().unwrap_err(),
        cli::CliError::NoImportMode
    );
}

#[test]
fn cli_rejects_bsp_launch_without_import_mode() {
    assert_eq!(
        cli::parse_from(["--bsp", "maps/e1m1.bsp"]).unwrap_err(),
        cli::CliError::NoImportMode
    );
}

#[test]
fn cli_accepts_textures_dir() {
    let args = cli::parse_from([
        "--strict",
        "--textures",
        "src/bsp_generator/themes/cc0_stone_beta/textures",
    ])
    .unwrap();
    assert_eq!(
        args.textures_dir,
        Some(PathBuf::from(
            "src/bsp_generator/themes/cc0_stone_beta/textures"
        ))
    );
}

#[test]
fn cli_leaves_declared_resources_for_runtime_authorization() {
    let args = cli::parse_from([
        "--strict",
        "--palette",
        "/tmp/not-authorized-palette.lmp",
        "--lit",
        "/tmp/not-authorized.lit",
        "--wad",
        "/tmp/not-authorized.wad",
    ])
    .unwrap();

    assert_eq!(
        args.resolve_palette_path().unwrap(),
        PathBuf::from("/tmp/not-authorized-palette.lmp")
    );
    assert_eq!(
        args.resolve_lit_path(),
        Some(PathBuf::from("/tmp/not-authorized.lit"))
    );
    assert_eq!(
        args.resolve_wad_path().unwrap(),
        Some(PathBuf::from("/tmp/not-authorized.wad"))
    );
}
