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
        companion_dir: None,
        wad_path: None,
    };

    let cloned = original.clone();
    assert_eq!(cloned.bsp_path, original.bsp_path);
    assert!((cloned.scale - original.scale).abs() < 1e-6);
    assert_eq!(cloned.headless, original.headless);
    assert_eq!(cloned.mcp, original.mcp);
    assert_eq!(cloned.capture_frames, original.capture_frames);
    assert_eq!(cloned.show_lights, original.show_lights);
}

#[test]
fn cli_rejects_equals_forms_and_unknown_flags() {
    assert_eq!(
        cli::parse_from(["--bsp=maps/e1m1.bsp"]).unwrap_err(),
        cli::CliError::UnknownArgument("--bsp=maps/e1m1.bsp".to_string())
    );
    assert_eq!(
        cli::parse_from(["--capture-frames=5"]).unwrap_err(),
        cli::CliError::UnknownArgument("--capture-frames=5".to_string())
    );
    assert_eq!(
        cli::parse_from(["--bogus"]).unwrap_err(),
        cli::CliError::UnknownArgument("--bogus".to_string())
    );
}

#[test]
fn cli_rejects_missing_or_malformed_values() {
    assert_eq!(
        cli::parse_from(["--bsp", "--headless"]).unwrap_err(),
        cli::CliError::MissingValue("--bsp")
    );
    assert!(matches!(
        cli::parse_from(["--scale", "inf"]).unwrap_err(),
        cli::CliError::NonFiniteScale(_)
    ));
    assert!(matches!(
        cli::parse_from(["--capture-frames", "bad"]).unwrap_err(),
        cli::CliError::InvalidCaptureFrames(_)
    ));
}
